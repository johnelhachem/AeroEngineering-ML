from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

_EPOCH_ORIGIN = pd.Timestamp("1970-01-01", tz="UTC")

def _utc_epoch(ts) -> float:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return (t - _EPOCH_ORIGIN).total_seconds()

from trino.auth import OAuth2Authentication
from trino.dbapi import connect

@dataclass(frozen=True)
class TrinoConfig:
    """Connection settings for a browser-authenticated Trino session."""

    host: str
    user: str
    port: int = 443
    catalog: str = "minio"
    schema: str = "osky"
    http_scheme: str = "https"
    verify: bool | str = True
    source: str = "aero_fusion_step1"
    request_timeout_seconds: float = 300.0

@dataclass(frozen=True)
class ColumnSpec:
    """Resolved Trino column metadata used for SQL generation."""

    name: str
    trino_type: str

def get_connection(config: TrinoConfig):
    """Create a Trino DB-API connection using browser-based OAuth."""

    return connect(
        host=config.host,
        port=config.port,
        user=config.user,
        catalog=config.catalog,
        schema=config.schema,
        http_scheme=config.http_scheme,
        auth=OAuth2Authentication(),
        verify=config.verify,
        source=config.source,
        request_timeout=config.request_timeout_seconds,
    )

def fetch_dataframe(
    connection,
    query: str,
    *,
    chunk_size: int = 10000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Execute SQL and return a DataFrame, streaming rows to avoid huge in-memory fetchall calls."""

    cursor = connection.cursor()
    cursor.execute(query)
    columns = [item[0] for item in cursor.description]

    collected: list[list[Any]] = []
    fetched_total = 0
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break

        collected.extend(rows)
        fetched_total += len(rows)

        if max_rows is not None and fetched_total >= max_rows:
            collected = collected[:max_rows]
            break

    return pd.DataFrame(collected, columns=columns)

def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'

def qualify_table(config: TrinoConfig, table_name: str) -> str:
    return ".".join(
        [
            quote_identifier(config.catalog),
            quote_identifier(config.schema),
            quote_identifier(table_name),
        ]
    )

def describe_table(connection, config: TrinoConfig, table_name: str) -> pd.DataFrame:
    """Return normalized Trino column metadata for a table."""

    query = f"DESCRIBE {qualify_table(config, table_name)}"
    described = fetch_dataframe(connection, query)
    if described.empty:
        return pd.DataFrame(columns=["column_name", "trino_type"])

    normalized = described.rename(
        columns={
            described.columns[0]: "column_name",
            described.columns[1]: "trino_type",
        }
    )
    normalized["column_name"] = normalized["column_name"].astype(str)
    normalized["trino_type"] = normalized["trino_type"].astype(str)
    return normalized[["column_name", "trino_type"]]

def resolve_columns(
    available_columns: pd.DataFrame,
    candidates: Mapping[str, Sequence[str]],
    optional: Sequence[str] | None = None,
) -> dict[str, ColumnSpec | None]:
    """Map canonical field names to concrete Trino columns."""

    optional = set(optional or [])
    available_lookup = {
        row.column_name.lower(): ColumnSpec(name=row.column_name, trino_type=row.trino_type)
        for row in available_columns.itertuples(index=False)
    }

    resolved: dict[str, ColumnSpec | None] = {}
    missing_required: list[str] = []

    for canonical_name, aliases in candidates.items():
        match = next((available_lookup.get(alias.lower()) for alias in aliases), None)
        if match is None and canonical_name not in optional:
            missing_required.append(canonical_name)
        resolved[canonical_name] = match

    if missing_required:
        available = ", ".join(sorted(available_lookup))
        missing = ", ".join(missing_required)
        raise ValueError(
            f"Missing required columns for {missing}. Available columns: {available}"
        )

    return resolved

def is_numeric_time_type(trino_type: str) -> bool:
    lowered = trino_type.lower()
    numeric_markers = ("bigint", "integer", "smallint", "tinyint", "decimal", "double", "real")
    return any(marker in lowered for marker in numeric_markers)

def is_string_type(trino_type: str) -> bool:
    lowered = trino_type.lower()
    return "varchar" in lowered or "char" in lowered

def looks_like_epoch_string_time(column: ColumnSpec) -> bool:
    """OpenSky adsc.time is stored as VARCHAR but contains Unix epoch seconds, often with decimals like 1688735236 or 1754913657.879."""
    return is_string_type(column.trino_type) and column.name.lower() in {"time", "ts"}

def _parsed_text_timestamp_expression(identifier: str) -> str:
    normalized = f"replace(replace(replace({identifier}, 'T', ' '), 'Z', ''), '/', '-')"
    return (
        "coalesce("
        f"try(CAST(from_iso8601_timestamp({identifier}) AS timestamp)), "
        f"try(CAST({identifier} AS timestamp)), "
        f"try(date_parse({identifier}, '%Y-%m-%d %H:%i:%s')), "
        f"try(date_parse({normalized}, '%Y-%m-%d %H:%i:%s')), "
        f"try(date_parse({normalized}, '%Y-%m-%d %H:%i:%s.%f'))"
        ")"
    )

def time_expression(column: ColumnSpec) -> str:
    identifier = quote_identifier(column.name)

    if is_numeric_time_type(column.trino_type):
        return f"from_unixtime(CAST({identifier} AS double))"

    if looks_like_epoch_string_time(column):
        return f"try(CAST(from_unixtime(CAST({identifier} AS double)) AS timestamp))"

    if is_string_type(column.trino_type):
        return _parsed_text_timestamp_expression(identifier)

    return identifier

def time_window_predicate(
    column: ColumnSpec,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> str:
    identifier = quote_identifier(column.name)
    start_epoch = float(start_time.timestamp())
    end_epoch = float(end_time.timestamp())

    if is_numeric_time_type(column.trino_type):
        return f"{identifier} >= {start_epoch} AND {identifier} < {end_epoch}"

    if looks_like_epoch_string_time(column):
        numeric_expr = f"try(CAST({identifier} AS double))"
        return f"{numeric_expr} >= {start_epoch} AND {numeric_expr} < {end_epoch}"

    if is_string_type(column.trino_type):
        parsed_expression = _parsed_text_timestamp_expression(identifier)
        start_literal = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_literal = end_time.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{parsed_expression} >= TIMESTAMP '{start_literal}' "
            f"AND {parsed_expression} < TIMESTAMP '{end_literal}'"
        )

    start_literal = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_literal = end_time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{identifier} >= TIMESTAMP '{start_literal}' "
        f"AND {identifier} < TIMESTAMP '{end_literal}'"
    )

def literal_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"
