"""Ingest ADS-B and ADS-C data from OpenSky into fusion-ready segments."""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from .trino_io import (
    ColumnSpec,
    TrinoConfig,
    _EPOCH_ORIGIN,
    _utc_epoch,
    describe_table,
    fetch_dataframe,
    literal_value,
    qualify_table,
    quote_identifier,
    resolve_columns,
    time_expression,
    time_window_predicate,
    is_numeric_time_type,
)
from .validation import ValidationThresholds, haversine_km, validate_fusion_candidate

ADSC_COLUMN_CANDIDATES = {
    "icao24":    ["icao24", "icao"],
    "timestamp": ["time", "timestamp", "ts", "position_time", "msgtime", "event_time"],
    "latitude":  ["lat", "latitude"],
    "longitude": ["lon", "longitude", "lng"],
    "altitude_m": ["geoaltitude", "baroaltitude", "altitude", "alt_baro", "altitude_m"],
    "callsign":  ["callsign", "flight_id", "flightid"],
}

FLIGHT_COLUMN_CANDIDATES = {
    "icao24":              ["icao24", "icao"],
    "flight_start":        ["firstseen", "start_time", "departure_time", "time_start"],
    "flight_end":          ["lastseen",  "end_time",   "arrival_time",   "time_end"],
    "callsign":            ["callsign"],
    "estdepartureairport": ["estdepartureairport"],
    "estarrivalairport":   ["estarrivalairport"],
    "day":                 ["day"],
}

ADSB_COLUMN_CANDIDATES = {
    "icao24":       ["icao24", "icao"],
    "timestamp":    ["time", "timestamp", "ts", "event_time"],
    "latitude":     ["lat", "latitude"],
    "longitude":    ["lon", "longitude", "lng"],
    "velocity_mps": ["velocity", "groundspeed", "speed"],
    "heading_deg":  ["heading", "track", "true_track"],
    "geoaltitude_m":  ["geoaltitude", "altitude", "altitude_m"],
    "baroaltitude_m": ["baroaltitude"],
    "callsign":     ["callsign"],
    "hour":         ["hour"],
    "lastcontact":  ["lastcontact"],
    "onground":     ["onground"],
}

_NAT_LAT_MIN = 30.0
_NAT_LAT_MAX = 75.0
_NAT_LON_MIN = -70.0
_NAT_LON_MAX = 25.0

@dataclass(frozen=True)
class TrinoSources:
    adsc_table:    str = "adsc"
    flights_table: str = "flights_data4"
    adsb_table:    str = "state_vectors_data4"

@dataclass(frozen=True)
class RunWindow:
    start: str
    end: str

@dataclass(frozen=True)
class IngestConfig:
    run_window: RunWindow
    sources: TrinoSources = TrinoSources()
    output_root: Path = Path("artifacts/step1_tryout")
    adsc_segment_gap_minutes: int = 90
    min_adsc_points: int = 2
    processing_context_hours: int = 4
    adsb_context_minutes: int = 180
    validation_thresholds: ValidationThresholds = ValidationThresholds()
    icao_batch_size: int = 500
    sql_screen_segment_batch_size: int = 20
    sql_screen_icao_batch_size: int = 25
    debug_no_survivor_probe_count: int = 3
    max_segments_to_validate: int | None = None
    use_adsc_cache: bool = True
    min_gap_duration_minutes: float = 60.0
    min_anchor_distance_km: float = 800.0
    shanwick_bbox: tuple[float, float, float, float] | None = (35.0, 70.0, -65.0, 10.0)
    query_sleep_seconds: float = 2.0

    @property
    def start_date(self) -> date:
        return pd.Timestamp(self.run_window.start).date()

    @property
    def end_date(self) -> date:
        return pd.Timestamp(self.run_window.end).date()

@dataclass(frozen=True)
class SourceColumns:
    adsc:    dict[str, ColumnSpec | None]
    flights: dict[str, ColumnSpec | None]
    adsb:    dict[str, ColumnSpec | None]

def _utc_epoch_int(ts: pd.Timestamp) -> int:
    """Return a UTC Unix epoch integer for a timestamp."""
    return int(_utc_epoch(ts))


def _utc_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(None)

def _epoch_to_utc_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("float64"), unit="s", utc=True).dt.tz_convert(None)

def _force_datetime(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Re-assert datetime64 dtypes after merges and copies."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

def _iso_timestamp(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value

def _first_non_null(values: pd.Series) -> str | None:
    non_null = values.dropna()
    if non_null.empty:
        return None
    return str(non_null.iloc[0]).strip() or None

def _optional_select(alias: str, column: ColumnSpec | None, cast_to: str) -> str:
    if column is None:
        return f"CAST(NULL AS {cast_to}) AS {quote_identifier(alias)}"
    return f"CAST({quote_identifier(column.name)} AS {cast_to}) AS {quote_identifier(alias)}"

def _in_clause(column_name: str, values: list[str]) -> str:
    quoted_values = ", ".join(literal_value(v) for v in sorted(set(values)))
    return f"{quote_identifier(column_name)} IN ({quoted_values})"

def _iter_days(start_day: date, end_day: date) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start_day, end=end_day, freq="D"))

def _hour_partition_values(window_start: pd.Timestamp, window_end: pd.Timestamp) -> list[int]:
    """Enumerate UTC partitions covering a window."""
    cur = pd.Timestamp(window_start).floor("h")
    end = pd.Timestamp(window_end).floor("h")
    values = []
    while cur <= end:
        values.append(_utc_epoch_int(cur))
        cur += pd.Timedelta(hours=1)
    return values

def _day_partition_values(window_start: pd.Timestamp, window_end: pd.Timestamp) -> list[int]:
    """Enumerate UTC partitions covering a window."""
    cur = pd.Timestamp(window_start).floor("D")
    end = pd.Timestamp(window_end).floor("D")
    values = []
    while cur <= end:
        values.append(_utc_epoch_int(cur))
        cur += pd.Timedelta(days=1)
    return values

def _batched(values: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), batch_size):
        yield values[i:i + batch_size]

def _prefilter_segments(
    segments: pd.DataFrame,
    adsc_segment_points: pd.DataFrame,
    config: IngestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if segments.empty:
        return segments.copy(), adsc_segment_points.copy()

    mask = pd.Series(True, index=segments.index)
    total = len(segments)

    if config.min_gap_duration_minutes > 0:
        mask &= segments["gap_duration_minutes"] >= config.min_gap_duration_minutes
        print(
            f"Pre-filter gap duration >= {config.min_gap_duration_minutes:.0f} min: "
            f"{mask.sum()}/{total} segments kept"
        )

    if config.min_anchor_distance_km > 0 and not adsc_segment_points.empty:
        surviving_ids = set(segments.loc[mask, "segment_id"])
        pts = adsc_segment_points[adsc_segment_points["segment_id"].isin(surviving_ids)].sort_values("timestamp")
        first_pts = pts.groupby("segment_id")[["latitude", "longitude"]].first()
        last_pts  = pts.groupby("segment_id")[["latitude", "longitude"]].last()
        anchors   = first_pts.join(last_pts, lsuffix="_first", rsuffix="_last")

        import numpy as np
        lat1 = np.radians(anchors["latitude_first"].values)
        lon1 = np.radians(anchors["longitude_first"].values)
        lat2 = np.radians(anchors["latitude_last"].values)
        lon2 = np.radians(anchors["longitude_last"].values)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        anchors["dist_km"] = 2 * 6371.0088 * np.arcsin(np.sqrt(a))

        dist_map  = anchors["dist_km"].to_dict()
        distances = segments["segment_id"].map(dist_map).fillna(0.0)
        mask &= distances >= config.min_anchor_distance_km
        print(
            f"Pre-filter anchor distance >= {config.min_anchor_distance_km:.0f} km: "
            f"{mask.sum()}/{total} segments kept"
        )

    if config.shanwick_bbox is not None and not adsc_segment_points.empty:
        lat_min, lat_max, lon_min, lon_max = config.shanwick_bbox
        surviving_ids = set(segments.loc[mask, "segment_id"])
        pts = adsc_segment_points[adsc_segment_points["segment_id"].isin(surviving_ids)]
        inside = pts[
            (pts["latitude"]  >= lat_min) & (pts["latitude"]  <= lat_max) &
            (pts["longitude"] >= lon_min) & (pts["longitude"] <= lon_max)
        ]["segment_id"].unique()
        mask &= segments["segment_id"].isin(set(inside))
        print(
            f"Pre-filter Shanwick bbox (lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max}): "
            f"{mask.sum()}/{total} segments kept"
        )

    filtered_segments = segments[mask].copy().reset_index(drop=True)
    surviving_ids     = set(filtered_segments["segment_id"])
    filtered_points   = adsc_segment_points[
        adsc_segment_points["segment_id"].isin(surviving_ids)
    ].copy().reset_index(drop=True)
    return filtered_segments, filtered_points

def _prefilter_segments_by_exactly_one_overlapping_flight(
    segments: pd.DataFrame,
    adsc_segment_points: pd.DataFrame,
    flights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if segments.empty:
        return segments.copy(), adsc_segment_points.copy()
    if flights.empty:
        print(f"Pre-filter exactly one overlapping flight: 0/{len(segments)} segments kept")
        return segments.iloc[0:0].copy(), adsc_segment_points.iloc[0:0].copy()

    overlap = segments[["segment_id", "icao24", "segment_start_time", "segment_end_time"]].merge(
        flights[["icao24", "flight_start_time", "flight_end_time"]], on="icao24", how="left"
    )
    overlap = overlap[
        (overlap["flight_start_time"] < overlap["segment_end_time"]) &
        (overlap["flight_end_time"]   > overlap["segment_start_time"])
    ]
    overlap_counts = overlap.groupby("segment_id").size()
    keep_mask = segments["segment_id"].map(overlap_counts).fillna(0).astype(int).eq(1)
    print(f"Pre-filter exactly one overlapping flight: {keep_mask.sum()}/{len(segments)} segments kept")

    filtered_segments = segments.loc[keep_mask].copy().reset_index(drop=True)
    surviving_ids     = set(filtered_segments["segment_id"])
    filtered_points   = adsc_segment_points[
        adsc_segment_points["segment_id"].isin(surviving_ids)
    ].copy().reset_index(drop=True)
    return filtered_segments, filtered_points

def inspect_source_columns(
    connection, trino_config: TrinoConfig, sources: TrinoSources = TrinoSources()
) -> SourceColumns:
    adsc_columns    = describe_table(connection, trino_config, sources.adsc_table)
    flights_columns = describe_table(connection, trino_config, sources.flights_table)
    adsb_columns    = describe_table(connection, trino_config, sources.adsb_table)

    return SourceColumns(
        adsc=resolve_columns(adsc_columns, ADSC_COLUMN_CANDIDATES, optional=["altitude_m", "callsign"]),
        flights=resolve_columns(
            flights_columns, FLIGHT_COLUMN_CANDIDATES,
            optional=["callsign", "estdepartureairport", "estarrivalairport"],
        ),
        adsb=resolve_columns(
            adsb_columns, ADSB_COLUMN_CANDIDATES,
            optional=["velocity_mps", "heading_deg", "geoaltitude_m", "baroaltitude_m",
                      "callsign", "hour", "lastcontact", "onground"],
        ),
    )

def _adsc_cache_path(ingest_config: IngestConfig) -> Path:
    start = pd.Timestamp(ingest_config.run_window.start).strftime("%Y%m%d")
    end   = pd.Timestamp(ingest_config.run_window.end).strftime("%Y%m%d")
    return ingest_config.output_root / "cache" / f"adsc_{start}_{end}.parquet"

def _fetch_adsc_points_for_window(
    connection,
    trino_config: TrinoConfig,
    ingest_config: IngestConfig,
    source_columns: SourceColumns,
) -> pd.DataFrame:
    cache_path = _adsc_cache_path(ingest_config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if ingest_config.use_adsc_cache and cache_path.exists():
        frame = pd.read_parquet(cache_path)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            print(f"ADS-C cache hit: {len(frame)} rows from {cache_path.name}")
            return frame
        else:
            print("ADS-C cache file exists but is empty; deleting and re-querying Trino")
            cache_path.unlink()

    run_start    = pd.Timestamp(ingest_config.run_window.start).normalize()
    run_end      = pd.Timestamp(ingest_config.run_window.end).normalize() + pd.Timedelta(days=1)
    context      = pd.Timedelta(hours=ingest_config.processing_context_hours)
    window_start = run_start - context
    window_end   = run_end   + context

    time_col = source_columns.adsc["timestamp"]
    assert time_col is not None

    adsc_lat_name = quote_identifier(source_columns.adsc["latitude"].name)
    adsc_lon_name = quote_identifier(source_columns.adsc["longitude"].name)

    query = f"""
    SELECT
        CAST({quote_identifier(source_columns.adsc["icao24"].name)} AS VARCHAR) AS icao24,
        TRY(CAST({quote_identifier(time_col.name)} AS DOUBLE)) AS ts_epoch,
        CAST({adsc_lat_name} AS DOUBLE) AS latitude,
        CAST({adsc_lon_name} AS DOUBLE) AS longitude,
        {_optional_select("altitude_m", source_columns.adsc["altitude_m"], "DOUBLE")},
        {_optional_select("callsign",   source_columns.adsc["callsign"],   "VARCHAR")}
    FROM {qualify_table(trino_config, ingest_config.sources.adsc_table)}
    WHERE {time_window_predicate(time_col, window_start, window_end)}
      AND {adsc_lat_name} IS NOT NULL
      AND {adsc_lon_name} IS NOT NULL
      AND {quote_identifier(source_columns.adsc["icao24"].name)} IS NOT NULL
      AND TRY(CAST({adsc_lat_name} AS DOUBLE)) BETWEEN {_NAT_LAT_MIN} AND {_NAT_LAT_MAX}
      AND TRY(CAST({adsc_lon_name} AS DOUBLE)) BETWEEN {_NAT_LON_MIN} AND {_NAT_LON_MAX}
    """
    frame = fetch_dataframe(connection, query)
    if frame.empty:
        return frame

    frame["timestamp"] = _epoch_to_utc_naive(frame.pop("ts_epoch"))
    frame["icao24"]    = frame["icao24"].astype(str).str.lower()
    frame["callsign"]  = frame["callsign"].astype("string").str.strip()
    frame = frame.sort_values(["icao24", "timestamp"]).reset_index(drop=True)

    if ingest_config.use_adsc_cache and not frame.empty:
        frame.to_parquet(cache_path, index=False)
        print(f"ADS-C cache saved: {len(frame)} rows to {cache_path.name}")

    return frame

def _build_adsc_segments(
    adsc_points: pd.DataFrame,
    ingest_config: IngestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if adsc_points.empty:
        return pd.DataFrame(columns=[
            "segment_id", "icao24", "segment_start_time", "segment_end_time",
            "gap_duration_minutes", "adsc_point_count", "segment_callsign",
        ]), adsc_points

    working = adsc_points.sort_values(["icao24", "timestamp"]).copy()
    working["delta_seconds"] = (
        working.groupby("icao24")["timestamp"].diff().dt.total_seconds().fillna(0)
    )
    working["new_segment"] = (
        (working["delta_seconds"] > ingest_config.adsc_segment_gap_minutes * 60) |
        working.groupby("icao24").cumcount().eq(0)
    )
    working["segment_sequence"] = working.groupby("icao24")["new_segment"].cumsum()

    segments = (
        working.groupby(["icao24", "segment_sequence"], as_index=False)
        .agg(
            segment_start_time=("timestamp", "min"),
            segment_end_time=("timestamp",   "max"),
            adsc_point_count=("timestamp",   "size"),
            segment_callsign=("callsign",    _first_non_null),
        )
    )
    segments["gap_duration_minutes"] = (
        (segments["segment_end_time"] - segments["segment_start_time"]).dt.total_seconds() / 60.0
    )
    segments = segments[
        (segments["adsc_point_count"] >= ingest_config.min_adsc_points) &
        (segments["segment_end_time"] > segments["segment_start_time"])
    ].copy()

    run_start = pd.Timestamp(ingest_config.run_window.start).normalize()
    run_end   = pd.Timestamp(ingest_config.run_window.end).normalize() + pd.Timedelta(days=1)
    segments  = segments[
        (segments["segment_start_time"] >= run_start) &
        (segments["segment_start_time"] <  run_end)
    ].copy()

    segments["segment_id"] = segments.apply(
        lambda r: (
            f"{r['segment_start_time']:%Y%m%d}_{r['icao24']}_"
            f"{r['segment_start_time']:%H%M%S}_{r['segment_end_time']:%H%M%S}"
        ),
        axis=1,
    )
    working = working.merge(
        segments[["icao24", "segment_sequence", "segment_id"]],
        on=["icao24", "segment_sequence"], how="inner",
    )
    return segments.reset_index(drop=True), working.reset_index(drop=True)

def _fetch_overlapping_flights_for_candidates(
    connection,
    trino_config: TrinoConfig,
    ingest_config: IngestConfig,
    source_columns: SourceColumns,
    candidate_icao24: list[str],
) -> pd.DataFrame:
    if not candidate_icao24:
        return pd.DataFrame()

    run_start    = pd.Timestamp(ingest_config.run_window.start).normalize()
    run_end      = pd.Timestamp(ingest_config.run_window.end).normalize() + pd.Timedelta(days=1)
    context      = pd.Timedelta(hours=ingest_config.processing_context_hours)
    window_start = run_start - context
    window_end   = run_end   + context

    start_col = source_columns.flights["flight_start"]
    end_col   = source_columns.flights["flight_end"]
    day_col   = source_columns.flights["day"]
    assert start_col is not None and end_col is not None and day_col is not None

    start_expr = time_expression(start_col)
    end_expr   = time_expression(end_col)

    if is_numeric_time_type(end_col.trino_type):
        overlap_predicate = (
            f"{quote_identifier(end_col.name)} >= {_utc_epoch_int(window_start)} "
            f"AND {quote_identifier(start_col.name)} < {_utc_epoch_int(window_end)}"
        )
    else:
        overlap_predicate = (
            f"{end_expr} >= TIMESTAMP '{window_start.strftime('%Y-%m-%d %H:%M:%S')}' "
            f"AND {start_expr} < TIMESTAMP '{window_end.strftime('%Y-%m-%d %H:%M:%S')}'"
        )

    day_values = _day_partition_values(window_start, window_end)
    if is_numeric_time_type(day_col.trino_type):
        day_predicate = f"{quote_identifier(day_col.name)} IN ({', '.join(str(v) for v in day_values)})"
    else:
        day_literals  = ", ".join(
            f"DATE '{pd.Timestamp.fromtimestamp(v).strftime('%Y-%m-%d')}'" for v in day_values
        )
        day_predicate = f"{quote_identifier(day_col.name)} IN ({day_literals})"

    pieces: list[pd.DataFrame] = []
    for batch in _batched(candidate_icao24, ingest_config.icao_batch_size):
        query = f"""
        SELECT
            CAST({quote_identifier(source_columns.flights["icao24"].name)} AS VARCHAR) AS icao24,
            CAST({start_expr} AS TIMESTAMP) AS flight_start_time,
            CAST({end_expr}   AS TIMESTAMP) AS flight_end_time,
            {_optional_select("callsign",            source_columns.flights["callsign"],            "VARCHAR")},
            {_optional_select("estdepartureairport", source_columns.flights["estdepartureairport"], "VARCHAR")},
            {_optional_select("estarrivalairport",   source_columns.flights["estarrivalairport"],   "VARCHAR")}
        FROM {qualify_table(trino_config, ingest_config.sources.flights_table)}
        WHERE {day_predicate}
          AND {overlap_predicate}
          AND {_in_clause(source_columns.flights["icao24"].name, batch)}
          AND {quote_identifier(source_columns.flights["icao24"].name)} IS NOT NULL
        """
        piece = fetch_dataframe(connection, query)
        if not piece.empty:
            pieces.append(piece)
        _time.sleep(ingest_config.query_sleep_seconds)

    if not pieces:
        return pd.DataFrame()

    flights = pd.concat(pieces, ignore_index=True)
    flights["flight_start_time"] = _utc_naive(flights["flight_start_time"])
    flights["flight_end_time"]   = _utc_naive(flights["flight_end_time"])
    flights["icao24"]   = flights["icao24"].astype(str).str.lower()
    flights["callsign"] = flights["callsign"].astype("string").str.strip()
    return flights.drop_duplicates().sort_values(["icao24", "flight_start_time"]).reset_index(drop=True)

def _merge_time_windows(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not windows:
        return []
    ordered = sorted((pd.Timestamp(s), pd.Timestamp(e)) for s, e in windows)
    merged: list[list[pd.Timestamp]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        cur = merged[-1]
        if start <= cur[1]:
            if end > cur[1]:
                cur[1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]

def _build_adsb_window_map_for_segments(
    segments: pd.DataFrame,
    context_minutes: int,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if segments.empty:
        return {}
    context = pd.Timedelta(minutes=context_minutes)
    raw: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for row in segments.itertuples(index=False):
        icao24 = str(row.icao24).lower()
        raw.setdefault(icao24, []).append((
            pd.Timestamp(row.segment_start_time) - context,
            pd.Timestamp(row.segment_end_time)   + context,
        ))
    return {icao24: _merge_time_windows(windows) for icao24, windows in raw.items()}

def _partition_values_for_window_map(
    window_map: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> tuple[list[int], list[int]]:
    day_values: set[int] = set()
    hour_values: set[int] = set()
    for windows in window_map.values():
        for ws, we in windows:
            day_values.update(_day_partition_values(ws, we))
            hour_values.update(_hour_partition_values(ws, we))
    return sorted(day_values), sorted(hour_values)

def _exact_window_predicate(
    time_col: ColumnSpec,
    icao24_column_name: str,
    window_map: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> str:
    or_clauses: list[str] = []
    quoted_icao_col = quote_identifier(icao24_column_name)
    for icao24, windows in window_map.items():
        time_clauses = [
            f"({time_window_predicate(time_col, ws, we)})"
            for ws, we in windows
        ]
        if not time_clauses:
            continue
        or_clauses.append(
            f"({quoted_icao_col} = {literal_value(icao24)} AND ({' OR '.join(time_clauses)}))"
        )
    if not or_clauses:
        return "FALSE"
    return f"({' OR '.join(or_clauses)})"

def _segment_values_sql(
    segments: pd.DataFrame,
    context_minutes: int,
) -> str:
    """Build VALUES rows with epoch doubles for SQL screening."""
    context_seconds = context_minutes * 60
    rows: list[str] = []
    for row in segments.itertuples(index=False):
        seg_start_ep = _utc_epoch(pd.Timestamp(row.segment_start_time))
        seg_end_ep   = _utc_epoch(pd.Timestamp(row.segment_end_time))
        rows.append(
            f"({literal_value(row.segment_id)}, "
            f"{literal_value(str(row.icao24).lower())}, "
            f"{seg_start_ep}, "
            f"{seg_end_ep}, "
            f"{seg_start_ep - context_seconds}, "
            f"{seg_end_ep + context_seconds})"
        )
    return ",\n                ".join(rows)

def _fetch_adsb_sql_screen_for_segments(
    connection,
    trino_config: TrinoConfig,
    ingest_config: IngestConfig,
    source_columns: SourceColumns,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """Count ADS-B hits per segment without fetching detail rows."""
    empty_result = pd.DataFrame(columns=[
        "segment_id", "before_count_sql", "during_count_sql", "after_count_sql",
        "last_before_timestamp_sql", "first_after_timestamp_sql",
    ])
    if segments.empty:
        return empty_result

    time_col = source_columns.adsb["timestamp"]
    assert time_col is not None
    raw_time_col = quote_identifier(time_col.name)

    lat_col  = source_columns.adsb["latitude"]
    lon_col  = source_columns.adsb["longitude"]
    lat_name = quote_identifier(lat_col.name) if lat_col else '"lat"'
    lon_name = quote_identifier(lon_col.name) if lon_col else '"lon"'

    geo_predicate = (
        f"{lat_name} BETWEEN {_NAT_LAT_MIN} AND {_NAT_LAT_MAX} "
        f"AND {lon_name} BETWEEN {_NAT_LON_MIN} AND {_NAT_LON_MAX}"
    )
    adsb_col_names   = {c.name.lower() for c in source_columns.adsb.values() if c is not None}
    has_day_partition = "day" in adsb_col_names
    onground_predicate = (
        f" AND COALESCE(CAST({quote_identifier(source_columns.adsb['onground'].name)} AS BOOLEAN), FALSE) = FALSE"
        if source_columns.adsb["onground"] is not None else ""
    )

    seg_batch_size = max(1, int(getattr(ingest_config, "sql_screen_segment_batch_size", 20)))
    pieces: list[pd.DataFrame] = []

    for start_idx in range(0, len(segments), seg_batch_size):
        seg_batch   = segments.iloc[start_idx: start_idx + seg_batch_size].copy().reset_index(drop=True)
        window_map  = _build_adsb_window_map_for_segments(seg_batch, ingest_config.adsb_context_minutes)
        if not window_map:
            continue

        day_values, hour_values = _partition_values_for_window_map(window_map)
        values_sql              = _segment_values_sql(seg_batch, ingest_config.adsb_context_minutes)

        where_clauses: list[str] = []
        if has_day_partition and day_values:
            where_clauses.append(f'"day" IN ({", ".join(str(v) for v in day_values)})')
        if source_columns.adsb["hour"] is not None and hour_values:
            where_clauses.append(
                f'{quote_identifier(source_columns.adsb["hour"].name)} IN ({", ".join(str(v) for v in hour_values)})'
            )
        where_clauses.extend([
            geo_predicate,
            f"{lat_name} IS NOT NULL",
            f"{lon_name} IS NOT NULL",
            _in_clause(source_columns.adsb["icao24"].name, sorted(window_map.keys())),
            _exact_window_predicate(time_col, source_columns.adsb["icao24"].name, window_map),
        ])

        query = f"""
        WITH candidates (segment_id, icao24, seg_start_ep, seg_end_ep, win_start_ep, win_end_ep) AS (
            VALUES
                {values_sql}
        ),
        adsb_raw AS (
            SELECT
                CAST({quote_identifier(source_columns.adsb["icao24"].name)} AS VARCHAR) AS icao24,
                CAST({raw_time_col} AS DOUBLE) AS ts_ep
            FROM {qualify_table(trino_config, ingest_config.sources.adsb_table)}
            WHERE {' AND '.join(where_clauses)}{onground_predicate}
        )
        SELECT
            c.segment_id,
            CAST(COUNT_IF(a.ts_ep <  c.seg_start_ep) AS BIGINT) AS before_count_sql,
            CAST(COUNT_IF(a.ts_ep >= c.seg_start_ep AND a.ts_ep <= c.seg_end_ep) AS BIGINT) AS during_count_sql,
            CAST(COUNT_IF(a.ts_ep >  c.seg_end_ep)   AS BIGINT) AS after_count_sql,
            MAX(CASE WHEN a.ts_ep <  c.seg_start_ep THEN a.ts_ep END) AS last_before_ep,
            MIN(CASE WHEN a.ts_ep >  c.seg_end_ep   THEN a.ts_ep END) AS first_after_ep
        FROM candidates c
        LEFT JOIN adsb_raw a
          ON a.icao24  = c.icao24
         AND a.ts_ep  >= c.win_start_ep
         AND a.ts_ep  <  c.win_end_ep
        GROUP BY c.segment_id
        """
        piece = fetch_dataframe(connection, query)
        if piece.empty:
            continue
        pieces.append(piece)
        _time.sleep(ingest_config.query_sleep_seconds)

    if not pieces:
        return empty_result

    metrics = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["segment_id"])
    for col in ["before_count_sql", "during_count_sql", "after_count_sql"]:
        metrics[col] = metrics[col].fillna(0).astype(int)

    for ep_col, ts_col in [("last_before_ep", "last_before_timestamp_sql"),
                            ("first_after_ep", "first_after_timestamp_sql")]:
        if ep_col in metrics.columns:
            metrics[ts_col] = (
                pd.to_datetime(metrics[ep_col].astype(float), unit="s", utc=True)
                .dt.tz_convert(None)
            )
            metrics.drop(columns=[ep_col], inplace=True)

    return metrics.sort_values("segment_id").reset_index(drop=True)

def _fetch_adsb_detail_for_segments(
    connection,
    trino_config: TrinoConfig,
    ingest_config: IngestConfig,
    source_columns: SourceColumns,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch ADS-B rows only for segments that passed screening."""
    if segments.empty:
        return pd.DataFrame()

    time_col = source_columns.adsb["timestamp"]
    assert time_col is not None

    lat_col  = source_columns.adsb["latitude"]
    lon_col  = source_columns.adsb["longitude"]
    lat_name = quote_identifier(lat_col.name) if lat_col else '"lat"'
    lon_name = quote_identifier(lon_col.name) if lon_col else '"lon"'

    geo_predicate     = (
        f"{lat_name} BETWEEN {_NAT_LAT_MIN} AND {_NAT_LAT_MAX} "
        f"AND {lon_name} BETWEEN {_NAT_LON_MIN} AND {_NAT_LON_MAX}"
    )
    adsb_col_names    = {c.name.lower() for c in source_columns.adsb.values() if c is not None}
    has_day_partition = "day" in adsb_col_names
    onground_predicate = (
        f" AND COALESCE(CAST({quote_identifier(source_columns.adsb['onground'].name)} AS BOOLEAN), FALSE) = FALSE"
        if source_columns.adsb["onground"] is not None else ""
    )

    window_map     = _build_adsb_window_map_for_segments(segments, ingest_config.adsb_context_minutes)
    if not window_map:
        return pd.DataFrame()

    sql_batch_size = max(1, min(int(getattr(ingest_config, "sql_screen_icao_batch_size", 25)), 25))
    pieces: list[pd.DataFrame] = []

    for batch in _batched(sorted(window_map.keys()), sql_batch_size):
        batch_map              = {icao24: window_map[icao24] for icao24 in batch}
        batch_day, batch_hour  = _partition_values_for_window_map(batch_map)

        where_clauses: list[str] = []
        if has_day_partition and batch_day:
            where_clauses.append(f'"day" IN ({", ".join(str(v) for v in batch_day)})')
        if source_columns.adsb["hour"] is not None and batch_hour:
            where_clauses.append(
                f'{quote_identifier(source_columns.adsb["hour"].name)} IN ({", ".join(str(v) for v in batch_hour)})'
            )
        where_clauses.extend([
            geo_predicate,
            f"{lat_name} IS NOT NULL",
            f"{lon_name} IS NOT NULL",
            _in_clause(source_columns.adsb["icao24"].name, sorted(batch_map.keys())),
            _exact_window_predicate(time_col, source_columns.adsb["icao24"].name, batch_map),
        ])

        query = f"""
        SELECT
            CAST({quote_identifier(source_columns.adsb["icao24"].name)} AS VARCHAR) AS icao24,
            CAST({quote_identifier(time_col.name)} AS DOUBLE) AS ts_epoch,
            CAST({quote_identifier(source_columns.adsb["latitude"].name)}  AS DOUBLE) AS latitude,
            CAST({quote_identifier(source_columns.adsb["longitude"].name)} AS DOUBLE) AS longitude,
            {_optional_select("velocity_mps",   source_columns.adsb["velocity_mps"],   "DOUBLE")},
            {_optional_select("heading_deg",    source_columns.adsb["heading_deg"],    "DOUBLE")},
            {_optional_select("geoaltitude_m",  source_columns.adsb["geoaltitude_m"],  "DOUBLE")},
            {_optional_select("baroaltitude_m", source_columns.adsb["baroaltitude_m"], "DOUBLE")},
            {_optional_select("callsign",       source_columns.adsb["callsign"],       "VARCHAR")},
            {_optional_select("lastcontact",    source_columns.adsb["lastcontact"],    "DOUBLE")},
            {_optional_select("onground",       source_columns.adsb["onground"],       "BOOLEAN")}
        FROM {qualify_table(trino_config, ingest_config.sources.adsb_table)}
        WHERE {' AND '.join(where_clauses)}{onground_predicate}
        """
        piece = fetch_dataframe(connection, query)
        _time.sleep(ingest_config.query_sleep_seconds)
        if piece.empty:
            continue

        piece["icao24"]    = piece["icao24"].astype(str).str.lower()
        piece["timestamp"] = _epoch_to_utc_naive(piece.pop("ts_epoch"))
        if "callsign" in piece.columns:
            piece["callsign"] = piece["callsign"].astype("string").str.strip()
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame()

    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates()
        .sort_values(["icao24", "timestamp"])
        .reset_index(drop=True)
    )

def _match_flight(segment_row: pd.Series, flights_df: pd.DataFrame) -> pd.DataFrame:
    if flights_df.empty:
        return flights_df
    return flights_df[
        (flights_df["icao24"]           == segment_row["icao24"]) &
        (flights_df["flight_start_time"] < segment_row["segment_end_time"]) &
        (flights_df["flight_end_time"]   > segment_row["segment_start_time"])
    ].drop_duplicates().reset_index(drop=True)

def _slice_adsb_for_segment(
    adsb_bulk: pd.DataFrame,
    segment_row: pd.Series,
    context_minutes: int,
) -> pd.DataFrame:
    if adsb_bulk.empty:
        return adsb_bulk.copy()
    window_start = segment_row["segment_start_time"] - pd.Timedelta(minutes=context_minutes)
    window_end   = segment_row["segment_end_time"]   + pd.Timedelta(minutes=context_minutes)
    mask = (
        (adsb_bulk["icao24"]    == str(segment_row["icao24"]).lower()) &
        (adsb_bulk["timestamp"] >= window_start) &
        (adsb_bulk["timestamp"] <  window_end)
    )
    return adsb_bulk[mask].copy().reset_index(drop=True)

def _partition_adsb_track(
    adsb_track: pd.DataFrame,
    segment_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if adsb_track.empty:
        empty = adsb_track.copy()
        return empty, empty, empty
    before = adsb_track[adsb_track["timestamp"] <  segment_row["segment_start_time"]].copy()
    during = adsb_track[
        (adsb_track["timestamp"] >= segment_row["segment_start_time"]) &
        (adsb_track["timestamp"] <= segment_row["segment_end_time"])
    ].copy()
    after  = adsb_track[adsb_track["timestamp"] >  segment_row["segment_end_time"]].copy()
    return before.reset_index(drop=True), during.reset_index(drop=True), after.reset_index(drop=True)

def _fetch_adsb_for_segment(
    connection,
    trino_config: TrinoConfig,
    ingest_config: IngestConfig,
    source_columns: SourceColumns,
    segment_row: pd.Series,
) -> pd.DataFrame:
    time_col = source_columns.adsb["timestamp"]
    assert time_col is not None

    window_start = segment_row["segment_start_time"] - pd.Timedelta(
        minutes=int(segment_row.get("adsb_context_minutes", ingest_config.adsb_context_minutes))
    )
    window_end = segment_row["segment_end_time"] + pd.Timedelta(
        minutes=int(segment_row.get("adsb_context_minutes", ingest_config.adsb_context_minutes))
    )

    hour_predicate = ""
    if source_columns.adsb["hour"] is not None:
        hv = _hour_partition_values(window_start, window_end)
        hour_predicate = f' AND {quote_identifier(source_columns.adsb["hour"].name)} IN ({", ".join(str(v) for v in hv)})'

    onground_predicate = ""
    if source_columns.adsb["onground"] is not None:
        onground_predicate = (
            f" AND COALESCE(CAST({quote_identifier(source_columns.adsb['onground'].name)} AS BOOLEAN), FALSE) = FALSE"
        )

    lat_col  = source_columns.adsb["latitude"]
    lon_col  = source_columns.adsb["longitude"]
    lat_name = quote_identifier(lat_col.name) if lat_col else '"lat"'
    lon_name = quote_identifier(lon_col.name) if lon_col else '"lon"'

    query = f"""
    SELECT
        CAST({quote_identifier(source_columns.adsb["icao24"].name)} AS VARCHAR) AS icao24,
        CAST({quote_identifier(time_col.name)} AS DOUBLE) AS ts_epoch,
        CAST({quote_identifier(source_columns.adsb["latitude"].name)}  AS DOUBLE) AS latitude,
        CAST({quote_identifier(source_columns.adsb["longitude"].name)} AS DOUBLE) AS longitude,
        {_optional_select("velocity_mps",   source_columns.adsb["velocity_mps"],   "DOUBLE")},
        {_optional_select("heading_deg",    source_columns.adsb["heading_deg"],    "DOUBLE")},
        {_optional_select("geoaltitude_m",  source_columns.adsb["geoaltitude_m"],  "DOUBLE")},
        {_optional_select("baroaltitude_m", source_columns.adsb["baroaltitude_m"], "DOUBLE")},
        {_optional_select("callsign",       source_columns.adsb["callsign"],       "VARCHAR")},
        {_optional_select("lastcontact",    source_columns.adsb["lastcontact"],    "DOUBLE")},
        {_optional_select("onground",       source_columns.adsb["onground"],       "BOOLEAN")}
    FROM {qualify_table(trino_config, ingest_config.sources.adsb_table)}
    WHERE {time_window_predicate(time_col, window_start, window_end)}
      {hour_predicate}
      AND {quote_identifier(source_columns.adsb["icao24"].name)} = {literal_value(segment_row["icao24"])}
      AND {lat_name} IS NOT NULL
      AND {lon_name} IS NOT NULL
      {onground_predicate}
    """
    adsb = fetch_dataframe(connection, query)
    if adsb.empty:
        return adsb
    adsb["timestamp"] = _epoch_to_utc_naive(adsb.pop("ts_epoch"))
    adsb["icao24"]    = adsb["icao24"].astype(str).str.lower()
    if "callsign" in adsb.columns:
        adsb["callsign"] = adsb["callsign"].astype("string").str.strip()
    return adsb.sort_values(["icao24", "timestamp"]).reset_index(drop=True)

def _build_stitched_minimal(
    segment_id: str,
    adsb_before: pd.DataFrame,
    adsc_points: pd.DataFrame,
    adsb_after: pd.DataFrame,
) -> pd.DataFrame:
    stitched = pd.concat([
        adsb_before.assign(phase="ADS-B before", source="adsb"),
        adsc_points.assign(phase="ADS-C gap",    source="adsc"),
        adsb_after.assign( phase="ADS-B after",  source="adsb"),
    ], ignore_index=True, sort=False).sort_values("timestamp")
    stitched["segment_id"] = segment_id
    preferred_columns = [
        "segment_id", "timestamp", "phase", "source", "icao24",
        "latitude", "longitude", "altitude_m", "geoaltitude_m",
        "baroaltitude_m", "velocity_mps", "heading_deg", "callsign",
    ]
    return stitched[[c for c in preferred_columns if c in stitched.columns]].reset_index(drop=True)

def _write_segment_artifacts(
    output_root: Path,
    segment_row: pd.Series,
    matched_flight: pd.Series,
    validation_result,
    adsc_points: pd.DataFrame,
    adsb_before: pd.DataFrame,
    adsb_after: pd.DataFrame,
) -> dict[str, str]:
    segment_dir = output_root / "flights" / str(segment_row["segment_id"])
    segment_dir.mkdir(parents=True, exist_ok=True)

    adsc_export = adsc_points.copy()
    if "altitude_m" not in adsc_export.columns:
        adsc_export["altitude_m"] = pd.NA

    adsb_before_path = segment_dir / "adsb_before.parquet"
    adsb_after_path  = segment_dir / "adsb_after.parquet"
    adsc_path        = segment_dir / "adsc.parquet"
    stitched_path    = segment_dir / "stitched_minimal.parquet"
    metadata_path    = segment_dir / "metadata.json"

    adsc_export.to_parquet(adsc_path,        index=False)
    adsb_before.to_parquet(adsb_before_path, index=False)
    adsb_after.to_parquet(  adsb_after_path,  index=False)
    _build_stitched_minimal(
        segment_row["segment_id"], adsb_before, adsc_export, adsb_after
    ).to_parquet(stitched_path, index=False)

    metadata = {
        "segment_id":        segment_row["segment_id"],
        "icao24":            segment_row["icao24"],
        "segment_start_time": _iso_timestamp(segment_row["segment_start_time"]),
        "segment_end_time":   _iso_timestamp(segment_row["segment_end_time"]),
        "gap_duration_minutes": float(segment_row["gap_duration_minutes"]),
        "adsc_point_count":  int(segment_row["adsc_point_count"]),
        "flight_match": {k: _iso_timestamp(v) for k, v in matched_flight.to_dict().items()},
        "validation":   {k: _iso_timestamp(v) for k, v in validation_result.as_dict().items()},
        "files": {
            "adsc":            str(adsc_path.as_posix()),
            "adsb_before":     str(adsb_before_path.as_posix()),
            "adsb_after":      str(adsb_after_path.as_posix()),
            "stitched_minimal": str(stitched_path.as_posix()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "artifact_dir":         str(segment_dir.as_posix()),
        "adsc_path":            str(adsc_path.as_posix()),
        "adsb_before_path":     str(adsb_before_path.as_posix()),
        "adsb_after_path":      str(adsb_after_path.as_posix()),
        "stitched_minimal_path": str(stitched_path.as_posix()),
        "metadata_path":        str(metadata_path.as_posix()),
    }

def _save_checkpoint(
    checkpoint_path: Path,
    completed_days: set[str],
    validated_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    checkpoint_path.write_text(
        json.dumps({"completed_days": sorted(completed_days)}, indent=2),
        encoding="utf-8",
    )
    summary_df = pd.DataFrame(validated_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=[
            "segment_id", "processing_day", "icao24",
            "segment_start_time", "segment_end_time",
            "gap_duration_minutes", "adsc_point_count",
        ])
    summary_df.sort_values(["processing_day", "segment_start_time"], inplace=True, ignore_index=True)
    summary_df.to_parquet(output_root / "validated_fusion_candidates.parquet", index=False)
    summary_df.to_csv(    output_root / "validated_fusion_candidates.csv",     index=False)

def _print_sql_screen_breakdown(screened: pd.DataFrame, day_str: str) -> None:
    if screened.empty:
        print(f"  SQL ADS-B breakdown: day {day_str} - no candidate segments")
        return
    before_missing   = int((screened["before_count_sql"] <= 0).sum())
    during_present   = int((screened["during_count_sql"] >  0).sum())
    after_missing    = int((screened["after_count_sql"]  <= 0).sum())
    no_before_after  = int(((screened["before_count_sql"] <= 0) & (screened["after_count_sql"] <= 0)).sum())
    print(
        f"  SQL ADS-B breakdown: day {day_str} - "
        f"before_missing={before_missing}, during_present={during_present}, "
        f"after_missing={after_missing}, no_before_no_after={no_before_after}"
    )
    cols   = ["segment_id", "icao24", "segment_start_time", "segment_end_time",
              "before_count_sql", "during_count_sql", "after_count_sql",
              "last_before_timestamp_sql", "first_after_timestamp_sql"]
    sample = screened.sort_values(
        ["during_count_sql", "before_count_sql", "after_count_sql"], ascending=[False, True, True]
    ).head(5)
    if not sample.empty:
        print("  SQL ADS-B sample failed candidates:")
        for row in sample[[c for c in cols if c in sample.columns]].itertuples(index=False):
            print(
                f"    {row.segment_id} | {row.icao24} | "
                f"before={row.before_count_sql} during={row.during_count_sql} after={row.after_count_sql} "
                f"| last_before={row.last_before_timestamp_sql} | first_after={row.first_after_timestamp_sql}"
            )

def _probe_no_survivor_segments(
    connection,
    trino_config: TrinoConfig,
    ingest_config: IngestConfig,
    source_columns: SourceColumns,
    flights: pd.DataFrame,
    adsc_segment_points: pd.DataFrame,
    screened: pd.DataFrame,
    day_str: str,
) -> None:
    probe_count = max(0, int(getattr(ingest_config, "debug_no_survivor_probe_count", 3)))
    if probe_count <= 0 or screened.empty:
        return
    probe = screened.sort_values(
        ["during_count_sql", "before_count_sql", "after_count_sql"], ascending=[True, False, False]
    ).head(probe_count)
    print(f"  Probing {len(probe)} failed segment(s) with direct fetch for day {day_str}...")
    for segment_row in probe.itertuples(index=False):
        segment_dict = segment_row._asdict()
        segment_dict.pop("_day_str", None)
        segment_dict["adsb_context_minutes"] = ingest_config.adsb_context_minutes
        segment_series = pd.Series(segment_dict)
        adsb_track = _fetch_adsb_for_segment(
            connection=connection, trino_config=trino_config,
            ingest_config=ingest_config, source_columns=source_columns,
            segment_row=segment_series,
        )
        adsb_before, adsb_during, adsb_after = _partition_adsb_track(adsb_track, segment_series)
        print(
            f"    Probe {segment_series['segment_id']} | "
            f"SQL(b={int(segment_series.get('before_count_sql', 0))}, "
            f"d={int(segment_series.get('during_count_sql', 0))}, "
            f"a={int(segment_series.get('after_count_sql', 0))}) "
            f"vs direct(b={len(adsb_before)}, d={len(adsb_during)}, a={len(adsb_after)})"
        )
        _time.sleep(ingest_config.query_sleep_seconds)

def run_step1_ingestion(
    connection, trino_config: TrinoConfig, ingest_config: IngestConfig
) -> pd.DataFrame:
    output_root = ingest_config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "flights").mkdir(parents=True, exist_ok=True)

    checkpoint_path  = output_root / "checkpoint_days.json"
    completed_days: set[str]         = set()
    validated_rows: list[dict[str, Any]] = []

    if checkpoint_path.exists():
        data           = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed_days = set(data.get("completed_days", []))
        existing_pq    = output_root / "validated_fusion_candidates.parquet"
        if existing_pq.exists():
            validated_rows = pd.read_parquet(existing_pq).to_dict("records")
        print(
            f"Resuming: {len(completed_days)} day(s) done, "
            f"{len(validated_rows)} segment(s) already saved."
        )

    source_columns = inspect_source_columns(connection, trino_config, ingest_config.sources)

    print("Fetching ADS-C once for the full run window...")
    adsc_points = _fetch_adsc_points_for_window(
        connection=connection, trino_config=trino_config,
        ingest_config=ingest_config, source_columns=source_columns,
    )
    print(f"ADS-C points fetched: {len(adsc_points)}")
    _time.sleep(ingest_config.query_sleep_seconds)

    segments, adsc_segment_points = _build_adsc_segments(adsc_points, ingest_config)
    print(f"Raw candidate segments: {len(segments)} | unique icao24: {segments['icao24'].nunique()}")

    segments, adsc_segment_points = _prefilter_segments(segments, adsc_segment_points, ingest_config)
    candidate_icao24 = sorted(segments["icao24"].dropna().astype(str).str.lower().unique().tolist())
    print(f"After pre-filter: {len(segments)} segments | {len(candidate_icao24)} unique icao24")

    print("Fetching overlapping flights for candidate aircraft...")
    flights = _fetch_overlapping_flights_for_candidates(
        connection=connection, trino_config=trino_config,
        ingest_config=ingest_config, source_columns=source_columns,
        candidate_icao24=candidate_icao24,
    )
    print(f"Flight rows fetched: {len(flights)}")

    segments, adsc_segment_points = _prefilter_segments_by_exactly_one_overlapping_flight(
        segments=segments, adsc_segment_points=adsc_segment_points, flights=flights,
    )
    candidate_icao24 = sorted(segments["icao24"].dropna().astype(str).str.lower().unique().tolist())
    print(f"After flight-overlap filter: {len(segments)} segments | {len(candidate_icao24)} unique icao24")

    if ingest_config.max_segments_to_validate is not None:
        segments = segments.sort_values(
            ["adsc_point_count", "gap_duration_minutes"],
            ascending=[False, True]
        ).head(int(ingest_config.max_segments_to_validate)).reset_index(drop=True)
        print(f"Limiting validation to top {len(segments)} candidate segments.")

    segments = _force_datetime(segments, "segment_start_time", "segment_end_time")
    segments["_day_str"] = segments["segment_start_time"].dt.date.astype(str)
    print("Screening ADS-B day by day (SQL count screen to detail fetch for survivors only)...")

    checked = 0
    for day_str in sorted(segments["_day_str"].unique().tolist()):
        if day_str in completed_days:
            print(f"  Day {day_str} skipped (already checkpointed)")
            continue

        day_segments = segments[segments["_day_str"] == day_str].copy().reset_index(drop=True)
        if day_segments.empty:
            continue

        metrics  = _fetch_adsb_sql_screen_for_segments(
            connection=connection, trino_config=trino_config,
            ingest_config=ingest_config, source_columns=source_columns,
            segments=day_segments,
        )
        screened = day_segments.merge(metrics, on="segment_id", how="left")
        for col in ["before_count_sql", "during_count_sql", "after_count_sql"]:
            screened[col] = screened[col].fillna(0).astype(int)

        survivors = screened[
            (screened["before_count_sql"] > 0) &
            (screened["during_count_sql"] == 0)
        ].copy().reset_index(drop=True)

        print(
            f"  SQL screen: day {day_str} - {len(survivors)}/{len(day_segments)} kept "
            f"(before>0, during=0)"
        )

        if survivors.empty:
            _print_sql_screen_breakdown(screened, day_str)
            _probe_no_survivor_segments(
                connection=connection, trino_config=trino_config,
                ingest_config=ingest_config, source_columns=source_columns,
                flights=flights, adsc_segment_points=adsc_segment_points,
                screened=screened, day_str=day_str,
            )
            completed_days.add(day_str)
            _save_checkpoint(checkpoint_path, completed_days, validated_rows, output_root)
            print(f"  Checkpoint saved after {day_str} | validated: {len(validated_rows)} | checked: {checked}")
            continue

        adsb_bulk = _fetch_adsb_detail_for_segments(
            connection=connection, trino_config=trino_config,
            ingest_config=ingest_config, source_columns=source_columns,
            segments=survivors,
        )
        print(f"  Detail fetch: day {day_str} - {len(adsb_bulk)} rows for {len(survivors)} survivors")

        for segment_row in survivors.itertuples(index=False):
            checked += 1
            segment_dict = segment_row._asdict()
            segment_dict.pop("_day_str", None)
            segment_dict["adsb_context_minutes"] = ingest_config.adsb_context_minutes
            segment_series = pd.Series(segment_dict)

            matched_flights        = _match_flight(segment_series, flights)
            adsc_points_for_seg    = (
                adsc_segment_points[adsc_segment_points["segment_id"] == segment_series["segment_id"]]
                .copy().reset_index(drop=True)
            )
            adsb_track             = _slice_adsb_for_segment(adsb_bulk, segment_series, ingest_config.adsb_context_minutes)
            adsb_before, adsb_during, adsb_after = _partition_adsb_track(adsb_track, segment_series)

            validation_result = validate_fusion_candidate(
                segment_row=segment_series,
                matched_flights=matched_flights,
                adsc_points=adsc_points_for_seg,
                adsb_before=adsb_before,
                adsb_during=adsb_during,
                adsb_after=adsb_after,
                thresholds=ingest_config.validation_thresholds,
            )

            if not validation_result.is_valid:
                print(f"  REJECTED {segment_series['segment_id']} | reasons={validation_result.reasons}")
                continue

            matched_flight = matched_flights.iloc[0]
            artifact_paths = _write_segment_artifacts(
                output_root=output_root,
                segment_row=segment_series,
                matched_flight=matched_flight,
                validation_result=validation_result,
                adsc_points=adsc_points_for_seg,
                adsb_before=adsb_before,
                adsb_after=adsb_after,
            )
            validated_rows.append({
                "segment_id":         segment_series["segment_id"],
                "processing_day":     pd.Timestamp(segment_series["segment_start_time"]).date().isoformat(),
                "icao24":             segment_series["icao24"],
                "segment_start_time": segment_series["segment_start_time"],
                "segment_end_time":   segment_series["segment_end_time"],
                "gap_duration_minutes": float(segment_series["gap_duration_minutes"]),
                "adsc_point_count":   int(segment_series["adsc_point_count"]),
                "segment_callsign":   segment_series["segment_callsign"],
                "flight_callsign":    matched_flight.get("callsign"),
                "flight_start_time":  matched_flight["flight_start_time"],
                "flight_end_time":    matched_flight["flight_end_time"],
                "estdepartureairport": matched_flight.get("estdepartureairport"),
                "estarrivalairport":  matched_flight.get("estarrivalairport"),
                **validation_result.as_dict(),
                **artifact_paths,
            })
            print(
                f"VALIDATED {len(validated_rows)} | {segment_series['segment_id']} | "
                f"{segment_series['icao24']} | "
                f"{segment_series['segment_start_time']} -> {segment_series['segment_end_time']}"
            )

        completed_days.add(day_str)
        _save_checkpoint(checkpoint_path, completed_days, validated_rows, output_root)
        print(f"  Checkpoint saved after {day_str} | validated: {len(validated_rows)} | checked: {checked}")

    print(f"Done. Checked {checked} segments | validated {len(validated_rows)}")
    return (
        pd.DataFrame(validated_rows) if validated_rows
        else pd.DataFrame(columns=[
            "segment_id", "processing_day", "icao24",
            "segment_start_time", "segment_end_time",
            "gap_duration_minutes", "adsc_point_count",
        ])
    )
