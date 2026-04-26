from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .step1_master import default_master_root
from .validation import implied_speed_kts

SECTION_FILE_NAMES = {
    "before": "adsb_before.parquet",
    "adsc": "adsc.parquet",
    "after": "adsb_after.parquet",
    "stitched": "stitched_minimal.parquet",
}

STANDARD_SECTION_COLUMNS = [
    "segment_id",
    "source_section",
    "timestamp",
    "icao24",
    "latitude",
    "longitude",
    "altitude_m",
    "geoaltitude_m",
    "baroaltitude_m",
    "velocity_mps",
    "heading_deg",
    "callsign",
    "phase",
    "source",
]

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "altitude_m",
    "geoaltitude_m",
    "baroaltitude_m",
    "velocity_mps",
    "heading_deg",
]

OUTLIER_SPEED_THRESHOLDS_KTS = {
    "before": 620.0,
    "adsc": 700.0,
    "after": 620.0,
    "stitched": 620.0,
}

RESAMPLE_SECONDS = 60
BOUNDARY_WINDOW_MINUTES = 15

@dataclass(frozen=True)
class Step2Config:
    """Filesystem/runtime configuration for rerunnable Step 2 cleaning."""

    step1_master_root: Path
    output_root: Path
    resample_seconds: int = RESAMPLE_SECONDS
    max_flights_to_process: int | None = None
    write_per_flight_outputs: bool = True
    progress_every: int = 10
    clean_existing_output: bool = True
    verbose: bool = True

def default_step2_config(repo_root: Path | None = None) -> Step2Config:
    """Return the default local Step 2 configuration for this repository."""

    project_root = repo_root or Path(__file__).resolve().parents[2]
    return Step2Config(
        step1_master_root=default_master_root(project_root),
        output_root=project_root / "artifacts" / "step2_clean",
    )

def _log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)

def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def _utc_naive(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce", utc=True)
    return values.dt.tz_convert(None)

def _infer_epoch_unit(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "s"

    median_abs = float(numeric.abs().median())
    if median_abs >= 1e17:
        return "ns"
    if median_abs >= 1e14:
        return "us"
    if median_abs >= 1e11:
        return "ms"
    return "s"

def _coerce_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return _utc_naive(series)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return _utc_naive(
            pd.to_datetime(
                numeric,
                errors="coerce",
                unit=_infer_epoch_unit(numeric),
                utc=True,
            )
        )

    parsed = _utc_naive(series)
    failed_mask = parsed.isna() & series.notna()
    if not failed_mask.any():
        return parsed

    numeric_fallback = pd.to_numeric(series[failed_mask], errors="coerce").dropna()
    if numeric_fallback.empty:
        return parsed

    reparsed = _utc_naive(
        pd.to_datetime(
            numeric_fallback,
            errors="coerce",
            unit=_infer_epoch_unit(numeric_fallback),
            utc=True,
        )
    )
    parsed.loc[reparsed.index] = reparsed
    return parsed

def _coerce_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()

def _ensure_columns(frame: pd.DataFrame, segment_id: str, source_section: str) -> pd.DataFrame:
    """Map a raw Step 1 section onto a consistent Step 2 schema."""

    working = frame.copy()

    if "segment_id" not in working.columns:
        working["segment_id"] = segment_id
    if "source_section" not in working.columns:
        working["source_section"] = source_section
    if "phase" not in working.columns:
        phase_map = {
            "before": "ADS-B before",
            "adsc": "ADS-C gap",
            "after": "ADS-B after",
            "stitched": "stitched",
        }
        working["phase"] = phase_map[source_section]
    if "source" not in working.columns:
        source_map = {
            "before": "adsb",
            "adsc": "adsc",
            "after": "adsb",
            "stitched": "fused",
        }
        working["source"] = source_map[source_section]

    for column in STANDARD_SECTION_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA

    working["segment_id"] = working["segment_id"].astype(str)
    working["source_section"] = working["source_section"].astype(str)
    working["timestamp"] = _coerce_timestamp(working["timestamp"])

    for column in ("icao24", "callsign", "phase", "source"):
        working[column] = _coerce_string(working[column])

    for column in NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    return working[STANDARD_SECTION_COLUMNS]

def _max_implied_speed_kts(frame: pd.DataFrame) -> float | None:
    """Compute the maximum pairwise implied speed in knots for a section."""

    ordered = (
        frame.dropna(subset=["timestamp", "latitude", "longitude"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(ordered) < 2:
        return None

    lat = np.radians(ordered["latitude"].to_numpy(dtype=float))
    lon = np.radians(ordered["longitude"].to_numpy(dtype=float))

    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    dist_nm = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * 3440.065

    dt = ordered["timestamp"].diff().dt.total_seconds().to_numpy(dtype=float)[1:]
    valid = np.isfinite(dt) & (dt > 0)
    if not np.any(valid):
        return None

    speeds_kts = dist_nm[valid] / (dt[valid] / 3600.0)
    if speeds_kts.size == 0:
        return None
    max_speed = np.nanmax(speeds_kts)
    return None if not np.isfinite(max_speed) else float(max_speed)

def _speed_within_threshold(speed_kts: float | None, threshold_kts: float) -> bool:
    return speed_kts is None or float(speed_kts) <= float(threshold_kts)

def _boundary_window(frame: pd.DataFrame, section: str, window_minutes: int = BOUNDARY_WINDOW_MINUTES) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ordered = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if ordered.empty:
        return ordered

    window = pd.Timedelta(minutes=window_minutes)
    if section == "before":
        boundary_end = ordered["timestamp"].max()
        boundary_start = boundary_end - window
        return ordered[ordered["timestamp"] >= boundary_start].reset_index(drop=True)
    if section == "after":
        boundary_start = ordered["timestamp"].min()
        boundary_end = boundary_start + window
        return ordered[ordered["timestamp"] <= boundary_end].reset_index(drop=True)
    return ordered.copy()

def _anchor_speed_kts(left_frame: pd.DataFrame, right_frame: pd.DataFrame) -> float | None:
    left = (
        left_frame.dropna(subset=["timestamp", "latitude", "longitude"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    right = (
        right_frame.dropna(subset=["timestamp", "latitude", "longitude"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if left.empty or right.empty:
        return None
    return implied_speed_kts(left.iloc[-1], right.iloc[0])

def _resolve_master_paths(cfg: Step2Config, segment_id: str, record: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve portable paths for step1_master assets."""

    catalog_flight_dir_text = str(record.get("master_flight_dir") or "").strip()
    catalog_metadata_path_text = str(record.get("master_metadata_path") or "").strip()
    catalog_flight_dir = None if not catalog_flight_dir_text else Path(catalog_flight_dir_text)
    catalog_metadata_path = None if not catalog_metadata_path_text else Path(catalog_metadata_path_text)

    fallback_flight_dir = cfg.step1_master_root / "flights" / segment_id
    fallback_metadata_path = fallback_flight_dir / "metadata.json"

    flight_dir = catalog_flight_dir if catalog_flight_dir is not None and catalog_flight_dir.exists() else fallback_flight_dir
    metadata_path = (
        catalog_metadata_path
        if catalog_metadata_path is not None and catalog_metadata_path.exists()
        else fallback_metadata_path
    )
    return flight_dir, metadata_path

def _active_sections(write_per_flight_outputs: bool) -> tuple[str, ...]:
    return ("before", "adsc", "after", "stitched") if write_per_flight_outputs else ("before", "adsc", "after")

def _drop_isolated_spikes(frame: pd.DataFrame, threshold_kts: float) -> tuple[pd.DataFrame, int]:
    """Remove a point only when both adjacent legs are implausible but the bridge is plausible."""

    if len(frame) < 3:
        return frame, 0

    working = frame.sort_values("timestamp").reset_index(drop=True).copy()
    removed = 0
    changed = True

    while changed and len(working) >= 3:
        changed = False
        for idx in range(1, len(working) - 1):
            prev_row = working.iloc[idx - 1]
            cur_row = working.iloc[idx]
            next_row = working.iloc[idx + 1]

            speed_prev = implied_speed_kts(prev_row, cur_row)
            speed_next = implied_speed_kts(cur_row, next_row)
            speed_bridge = implied_speed_kts(prev_row, next_row)

            if (
                speed_prev is not None
                and speed_next is not None
                and speed_bridge is not None
                and speed_prev > threshold_kts
                and speed_next > threshold_kts
                and speed_bridge <= threshold_kts
            ):
                working = working.drop(index=idx).reset_index(drop=True)
                removed += 1
                changed = True
                break

    return working, removed

def _resample_section(frame: pd.DataFrame, resample_seconds: int) -> pd.DataFrame:
    """Build a standardized section using time interpolation without extrapolation."""

    if frame.empty:
        return frame.copy()

    ordered = frame.sort_values("timestamp").reset_index(drop=True).copy()
    if len(ordered) < 2:
        return ordered

    start = ordered["timestamp"].min().floor(f"{resample_seconds}s")
    end = ordered["timestamp"].max().ceil(f"{resample_seconds}s")
    if start == end:
        return ordered

    target_index = pd.date_range(start=start, end=end, freq=f"{resample_seconds}s")
    numeric = (
        ordered.set_index("timestamp")[NUMERIC_COLUMNS]
        .sort_index()
        .reindex(target_index)
        .interpolate(method="time", limit_direction="forward", limit_area="inside")
    )

    resampled = numeric.reset_index().rename(columns={"index": "timestamp"})
    resampled = resampled.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if resampled.empty:
        return pd.DataFrame(columns=STANDARD_SECTION_COLUMNS)

    for column in ("segment_id", "icao24", "callsign"):
        non_null = ordered[column].dropna()
        value = non_null.iloc[0] if not non_null.empty else pd.NA
        resampled[column] = value

    resampled["source_section"] = ordered["source_section"].iloc[0]
    resampled["phase"] = ordered["phase"].iloc[0]
    resampled["source"] = ordered["source"].iloc[0]

    return resampled[STANDARD_SECTION_COLUMNS]

def _resample_stitched_full_track(frame: pd.DataFrame, resample_seconds: int) -> pd.DataFrame:
    """Resample the full stitched trajectory as a single time series."""

    if frame.empty:
        return frame.copy()

    ordered = (
        frame.dropna(subset=["timestamp", "latitude", "longitude"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(ordered) < 2:
        stitched = ordered.copy()
        if not stitched.empty:
            stitched["source_section"] = "stitched"
            stitched["phase"] = "stitched"
            stitched["source"] = "fused"
            return stitched[STANDARD_SECTION_COLUMNS]
        return pd.DataFrame(columns=STANDARD_SECTION_COLUMNS)

    start = ordered["timestamp"].min().floor(f"{resample_seconds}s")
    end = ordered["timestamp"].max().ceil(f"{resample_seconds}s")
    if start == end:
        stitched = ordered.copy()
        stitched["source_section"] = "stitched"
        stitched["phase"] = "stitched"
        stitched["source"] = "fused"
        return stitched[STANDARD_SECTION_COLUMNS]

    target_index = pd.date_range(start=start, end=end, freq=f"{resample_seconds}s")
    numeric = (
        ordered.set_index("timestamp")[NUMERIC_COLUMNS]
        .sort_index()
        .reindex(target_index)
        .interpolate(method="time", limit_direction="forward", limit_area="inside")
    )
    resampled = numeric.reset_index().rename(columns={"index": "timestamp"})
    resampled = resampled.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if resampled.empty:
        return pd.DataFrame(columns=STANDARD_SECTION_COLUMNS)

    for column in ("segment_id", "icao24", "callsign"):
        non_null = ordered[column].dropna()
        value = non_null.iloc[0] if not non_null.empty else pd.NA
        resampled[column] = value

    resampled["source_section"] = "stitched"
    resampled["phase"] = "stitched"
    resampled["source"] = "fused"
    return resampled[STANDARD_SECTION_COLUMNS]

def _clean_section(
    frame: pd.DataFrame,
    segment_id: str,
    source_section: str,
    resample_seconds: int,
    build_resampled: bool = True,
    fast_mode: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit and clean one section of a fused flight."""

    metrics: dict[str, Any] = {
        "section": source_section,
        "rows_raw": int(len(frame)),
    }

    working = _ensure_columns(frame, segment_id=segment_id, source_section=source_section)

    metrics["timestamp_null_count"] = int(working["timestamp"].isna().sum())
    metrics["invalid_coord_count"] = int(
        (
            working["latitude"].notna() & ~working["latitude"].between(-90.0, 90.0)
        ).sum()
        + (
            working["longitude"].notna() & ~working["longitude"].between(-180.0, 180.0)
        ).sum()
    )
    metrics["missing_key_value_count"] = int(
        working[["timestamp", "latitude", "longitude"]].isna().any(axis=1).sum()
    )
    metrics["was_unsorted"] = not working["timestamp"].is_monotonic_increasing

    working = working.dropna(subset=["timestamp", "latitude", "longitude"]).copy()
    working = working[
        working["latitude"].between(-90.0, 90.0)
        & working["longitude"].between(-180.0, 180.0)
    ].copy()
    working = working.sort_values("timestamp", kind="stable").reset_index(drop=True)

    metrics["duplicate_exact_count"] = int(working.duplicated().sum())
    if metrics["duplicate_exact_count"] > 0:
        working = working.drop_duplicates().reset_index(drop=True)

    metrics["duplicate_timestamp_count"] = int(working.duplicated(subset=["timestamp"]).sum())
    if metrics["duplicate_timestamp_count"] > 0:
        working = working.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    threshold = OUTLIER_SPEED_THRESHOLDS_KTS[source_section]
    speed_before = _max_implied_speed_kts(working)

    if source_section in {"before", "after"}:
        spike_removed_count = 0
        speed_after = speed_before
    else:
        working, spike_removed_count = _drop_isolated_spikes(working, threshold_kts=threshold)
        speed_after = _max_implied_speed_kts(working)

    metrics["isolated_spike_removed_count"] = int(spike_removed_count)
    metrics["max_implied_speed_kts_before_clean"] = speed_before
    metrics["max_implied_speed_kts_after_clean"] = speed_after
    metrics["speed_threshold_kts"] = float(threshold)
    metrics["speed_valid_after_clean"] = _speed_within_threshold(speed_after, threshold)
    metrics["rows_clean"] = int(len(working))
    metrics["has_minimum_rows_after_clean"] = bool(
        (source_section in {"before", "after"} and len(working) >= 1)
        or (source_section == "adsc" and len(working) >= 2)
        or (source_section == "stitched" and len(working) >= 1)
    )

    if build_resampled:
        if source_section == "stitched":
            resampled = _resample_stitched_full_track(working, resample_seconds=resample_seconds)
        else:
            resampled = _resample_section(working, resample_seconds=resample_seconds)
        metrics["rows_resampled"] = int(len(resampled))
    else:
        resampled = None
        metrics["rows_resampled"] = 0

    return working, {**metrics, "resampled_frame": resampled}

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_frame(frame: pd.DataFrame, base_path: Path) -> None:
    """Write a table to parquet and CSV with a shared base path."""

    base_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(base_path.with_suffix(".parquet"), index=False)
    frame.to_csv(base_path.with_suffix(".csv"), index=False)

def _build_stitched_clean(before_df: pd.DataFrame, adsc_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    stitched = (
        pd.concat([before_df, adsc_df, after_df], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return stitched[STANDARD_SECTION_COLUMNS]

def _build_stitched_standardized(
    stitched_clean: pd.DataFrame,
    resample_seconds: int,
) -> pd.DataFrame:
    """Resample the entire stitched clean trajectory on one uniform grid."""
    return _resample_stitched_full_track(stitched_clean, resample_seconds=resample_seconds)

def _quality_issue_rows(
    segment_id: str,
    source_run: str,
    metrics_by_section: dict[str, dict[str, Any]],
    keep_flight: bool,
    drop_reason: str | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for section, metrics in metrics_by_section.items():
        for field in (
            "timestamp_null_count",
            "invalid_coord_count",
            "missing_key_value_count",
            "duplicate_exact_count",
            "duplicate_timestamp_count",
            "isolated_spike_removed_count",
        ):
            value = int(metrics.get(field, 0) or 0)
            if value > 0:
                issues.append(
                    {
                        "segment_id": segment_id,
                        "source_run": source_run,
                        "section": section,
                        "issue_type": field,
                        "issue_count": value,
                        "keep_flight": keep_flight,
                        "drop_reason": drop_reason,
                    }
                )

        if bool(metrics.get("was_unsorted")):
            issues.append(
                {
                    "segment_id": segment_id,
                    "source_run": source_run,
                    "section": section,
                    "issue_type": "rows_were_unsorted",
                    "issue_count": 1,
                    "keep_flight": keep_flight,
                    "drop_reason": drop_reason,
                }
            )

        if not bool(metrics.get("speed_valid_after_clean", True)):
            issues.append(
                {
                    "segment_id": segment_id,
                    "source_run": source_run,
                    "section": section,
                    "issue_type": "implausible_section_speed_after_clean",
                    "issue_count": 1,
                    "issue_value_kts": metrics.get("max_implied_speed_kts_after_clean"),
                    "threshold_kts": metrics.get("speed_threshold_kts"),
                    "keep_flight": keep_flight,
                    "drop_reason": drop_reason,
                }
            )

        anchor_value = metrics.get("anchor_speed_kts_after_clean")
        if anchor_value is not None and not bool(metrics.get("anchor_speed_valid_after_clean", True)):
            issues.append(
                {
                    "segment_id": segment_id,
                    "source_run": source_run,
                    "section": section,
                    "issue_type": "implausible_anchor_speed_after_clean",
                    "issue_count": 1,
                    "issue_value_kts": anchor_value,
                    "threshold_kts": metrics.get("speed_threshold_kts"),
                    "keep_flight": keep_flight,
                    "drop_reason": drop_reason,
                }
            )

        if bool(metrics.get("boundary_too_sparse", False)):
            issues.append(
                {
                    "segment_id": segment_id,
                    "source_run": source_run,
                    "section": section,
                    "issue_type": "boundary_too_sparse",
                    "issue_count": 1,
                    "keep_flight": keep_flight,
                    "drop_reason": drop_reason,
                }
            )

    return issues

def _warn_about_step1_master(cfg: Step2Config) -> None:
    catalog_path = cfg.step1_master_root / "catalog" / "master_flights_catalog.parquet"
    _log("=" * 72, cfg.verbose)
    _log("STEP 2 will run from the CURRENT step1_master only.", cfg.verbose)
    _log(f"step1_master root: {cfg.step1_master_root.as_posix()}", cfg.verbose)
    _log(f"output root      : {cfg.output_root.as_posix()}", cfg.verbose)
    if catalog_path.exists():
        modified = pd.Timestamp(catalog_path.stat().st_mtime, unit="s")
        _log(f"master catalog   : {catalog_path.as_posix()}", cfg.verbose)
        _log(f"catalog modified : {modified}", cfg.verbose)
    else:
        _log("WARNING: master catalog does not exist yet.", cfg.verbose)
    _log(
        "If you added new Step 1 windows recently, rebuild step1_master before running Step 2.",
        cfg.verbose,
    )
    _log("=" * 72, cfg.verbose)

def build_step2_clean(config: Step2Config | None = None) -> dict[str, Any]:
    """Build the full Step 2 dataset from the Step 1 master catalog."""

    cfg = config or default_step2_config()
    _warn_about_step1_master(cfg)

    master_catalog_path = cfg.step1_master_root / "catalog" / "master_flights_catalog.parquet"
    if not master_catalog_path.exists():
        raise FileNotFoundError(f"Step 1 master catalog not found: {master_catalog_path}")

    if cfg.clean_existing_output and cfg.output_root.exists():
        _log(f"Removing existing output root: {cfg.output_root.as_posix()}", cfg.verbose)
        shutil.rmtree(cfg.output_root)

    (cfg.output_root / "catalog").mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "flights").mkdir(parents=True, exist_ok=True)

    master_catalog = pd.read_parquet(master_catalog_path).copy()
    master_catalog = master_catalog.sort_values("segment_id").reset_index(drop=True)

    total_master_flights = len(master_catalog)
    if cfg.max_flights_to_process is not None:
        master_catalog = master_catalog.head(cfg.max_flights_to_process).copy()
        _log(
            f"Preview mode enabled: processing first {len(master_catalog)} / {total_master_flights} flights.",
            cfg.verbose,
        )
    else:
        _log(f"Loaded {total_master_flights} Step 1 master flights.", cfg.verbose)

    _log(f"Per-flight outputs enabled: {cfg.write_per_flight_outputs}", cfg.verbose)
    fast_catalog_only = not cfg.write_per_flight_outputs
    _log(f"Fast catalog-only mode    : {fast_catalog_only}", cfg.verbose)
    if fast_catalog_only:
        _log(
            "Fast catalog-only mode active: skipping resampling, stitched file reads, and per-flight-only work.",
            cfg.verbose,
        )
    _log(f"Progress print frequency  : every {cfg.progress_every} flights", cfg.verbose)

    audit_rows: list[dict[str, Any]] = []
    quality_issue_records: list[dict[str, Any]] = []
    clean_catalog_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(master_catalog.to_dict(orient="records"), start=1):
        if cfg.progress_every > 0 and (idx == 1 or idx % cfg.progress_every == 0 or idx == len(master_catalog)):
            _log(f"[Step2] Processing flight {idx}/{len(master_catalog)}", cfg.verbose)

        segment_id = str(record["segment_id"])
        source_run = str(record["source_run"])
        flight_dir, metadata_path = _resolve_master_paths(cfg, segment_id, record)

        section_frames: dict[str, pd.DataFrame] = {}
        section_metrics: dict[str, dict[str, Any]] = {}

        required_files_exist = True
        readable = True

        metadata_exists = metadata_path.exists()
        metadata_error = None
        if metadata_exists:
            try:
                _ = _load_json(metadata_path)
            except Exception as exc:
                metadata_error = f"{type(exc).__name__}: {exc}"
        else:
            metadata_error = "missing metadata.json"

        expected_columns_ok = True

        for section in _active_sections(cfg.write_per_flight_outputs):
            file_name = SECTION_FILE_NAMES[section]
            path = flight_dir / file_name

            if not path.exists():
                required_files_exist = False
                readable = False
                expected_columns_ok = False
                section_frames[section] = pd.DataFrame(columns=STANDARD_SECTION_COLUMNS)
                section_metrics[section] = {
                    "section": section,
                    "rows_raw": 0,
                    "rows_clean": 0,
                    "rows_resampled": 0,
                    "timestamp_null_count": 0,
                    "invalid_coord_count": 0,
                    "missing_key_value_count": 0,
                    "duplicate_exact_count": 0,
                    "duplicate_timestamp_count": 0,
                    "isolated_spike_removed_count": 0,
                    "was_unsorted": False,
                    "max_implied_speed_kts_before_clean": None,
                    "max_implied_speed_kts_after_clean": None,
                    "speed_threshold_kts": float(OUTLIER_SPEED_THRESHOLDS_KTS[section]),
                    "speed_valid_after_clean": True,
                    "boundary_rows_clean": 0,
                    "boundary_max_speed_kts_after_clean": None,
                    "boundary_speed_valid_after_clean": True,
                    "anchor_speed_kts_after_clean": None,
                    "anchor_speed_valid_after_clean": True,
                    "full_context_speed_issue": False,
                    "boundary_too_sparse": False,
                    "has_minimum_rows_after_clean": False,
                    "resampled_frame": pd.DataFrame(columns=STANDARD_SECTION_COLUMNS),
                }
                continue

            try:
                raw_df = pd.read_parquet(path)
            except Exception as exc:
                readable = False
                expected_columns_ok = False
                section_frames[section] = pd.DataFrame(columns=STANDARD_SECTION_COLUMNS)
                section_metrics[section] = {
                    "section": section,
                    "rows_raw": 0,
                    "rows_clean": 0,
                    "rows_resampled": 0,
                    "timestamp_null_count": 0,
                    "invalid_coord_count": 0,
                    "missing_key_value_count": 0,
                    "duplicate_exact_count": 0,
                    "duplicate_timestamp_count": 0,
                    "isolated_spike_removed_count": 0,
                    "was_unsorted": False,
                    "max_implied_speed_kts_before_clean": None,
                    "max_implied_speed_kts_after_clean": None,
                    "speed_threshold_kts": float(OUTLIER_SPEED_THRESHOLDS_KTS[section]),
                    "speed_valid_after_clean": True,
                    "boundary_rows_clean": 0,
                    "boundary_max_speed_kts_after_clean": None,
                    "boundary_speed_valid_after_clean": True,
                    "anchor_speed_kts_after_clean": None,
                    "anchor_speed_valid_after_clean": True,
                    "full_context_speed_issue": False,
                    "boundary_too_sparse": False,
                    "has_minimum_rows_after_clean": False,
                    "load_error": f"{type(exc).__name__}: {exc}",
                    "resampled_frame": pd.DataFrame(columns=STANDARD_SECTION_COLUMNS),
                }
                continue

            cleaned_df, metrics = _clean_section(
                raw_df,
                segment_id=segment_id,
                source_section=section,
                resample_seconds=cfg.resample_seconds,
                build_resampled=cfg.write_per_flight_outputs,
                fast_mode=fast_catalog_only,
            )
            expected_columns_ok = expected_columns_ok and set(STANDARD_SECTION_COLUMNS).issubset(cleaned_df.columns)
            section_frames[section] = cleaned_df
            section_metrics[section] = metrics

        if "stitched" not in section_frames:
            section_frames["stitched"] = pd.DataFrame(columns=STANDARD_SECTION_COLUMNS)
            section_metrics["stitched"] = {
                "section": "stitched",
                "rows_raw": 0,
                "rows_clean": 0,
                "rows_resampled": 0,
                "timestamp_null_count": 0,
                "invalid_coord_count": 0,
                "missing_key_value_count": 0,
                "duplicate_exact_count": 0,
                "duplicate_timestamp_count": 0,
                "isolated_spike_removed_count": 0,
                "was_unsorted": False,
                "max_implied_speed_kts_before_clean": None,
                "max_implied_speed_kts_after_clean": None,
                "speed_threshold_kts": float(OUTLIER_SPEED_THRESHOLDS_KTS["stitched"]),
                "speed_valid_after_clean": True,
                "boundary_rows_clean": 0,
                "boundary_max_speed_kts_after_clean": None,
                "boundary_speed_valid_after_clean": True,
                "anchor_speed_kts_after_clean": None,
                "anchor_speed_valid_after_clean": True,
                "full_context_speed_issue": False,
                "boundary_too_sparse": False,
                "has_minimum_rows_after_clean": False,
                "resampled_frame": None,
            }

        time_order_valid = all(
            bool(
                section_metrics[section]["rows_clean"] == 0
                or section_frames[section]["timestamp"].is_monotonic_increasing
            )
            for section in section_frames
        )
        coord_values_valid = all(
            int(section_metrics[section]["invalid_coord_count"]) == 0
            for section in section_metrics
        )
        non_empty_sections = all(
            bool(section_metrics[section]["has_minimum_rows_after_clean"])
            for section in ("before", "adsc", "after")
        )

        before_boundary = _boundary_window(section_frames["before"], "before")
        after_boundary = _boundary_window(section_frames["after"], "after")

        before_boundary_max_speed = _max_implied_speed_kts(before_boundary)
        after_boundary_max_speed = _max_implied_speed_kts(after_boundary)
        before_anchor_speed = _anchor_speed_kts(before_boundary, section_frames["adsc"])
        after_anchor_speed = _anchor_speed_kts(section_frames["adsc"], after_boundary)
        before_anchor_speed_valid = _speed_within_threshold(before_anchor_speed, OUTLIER_SPEED_THRESHOLDS_KTS["before"])
        after_anchor_speed_valid = _speed_within_threshold(after_anchor_speed, OUTLIER_SPEED_THRESHOLDS_KTS["after"])

        section_metrics["before"]["boundary_rows_clean"] = int(len(before_boundary))
        section_metrics["before"]["boundary_max_speed_kts_after_clean"] = before_boundary_max_speed
        section_metrics["before"]["boundary_speed_valid_after_clean"] = _speed_within_threshold(before_boundary_max_speed, OUTLIER_SPEED_THRESHOLDS_KTS["before"])
        section_metrics["before"]["anchor_speed_kts_after_clean"] = before_anchor_speed
        section_metrics["before"]["anchor_speed_valid_after_clean"] = before_anchor_speed_valid
        section_metrics["before"]["full_context_speed_issue"] = not bool(section_metrics["before"]["speed_valid_after_clean"])
        section_metrics["before"]["boundary_too_sparse"] = bool(len(before_boundary) < 2)

        section_metrics["after"]["boundary_rows_clean"] = int(len(after_boundary))
        section_metrics["after"]["boundary_max_speed_kts_after_clean"] = after_boundary_max_speed
        section_metrics["after"]["boundary_speed_valid_after_clean"] = _speed_within_threshold(after_boundary_max_speed, OUTLIER_SPEED_THRESHOLDS_KTS["after"])
        section_metrics["after"]["anchor_speed_kts_after_clean"] = after_anchor_speed
        section_metrics["after"]["anchor_speed_valid_after_clean"] = after_anchor_speed_valid
        section_metrics["after"]["full_context_speed_issue"] = not bool(section_metrics["after"]["speed_valid_after_clean"])
        section_metrics["after"]["boundary_too_sparse"] = bool(len(after_boundary) < 2)

        section_metrics["adsc"]["boundary_rows_clean"] = int(section_metrics["adsc"]["rows_clean"])
        section_metrics["adsc"]["boundary_max_speed_kts_after_clean"] = section_metrics["adsc"]["max_implied_speed_kts_after_clean"]
        section_metrics["adsc"]["boundary_speed_valid_after_clean"] = bool(section_metrics["adsc"]["speed_valid_after_clean"])
        section_metrics["adsc"]["anchor_speed_kts_after_clean"] = None
        section_metrics["adsc"]["anchor_speed_valid_after_clean"] = True
        section_metrics["adsc"]["full_context_speed_issue"] = not bool(section_metrics["adsc"]["speed_valid_after_clean"])
        section_metrics["adsc"]["boundary_too_sparse"] = False

        section_speed_valid_after_clean = all(
            [
                bool(section_metrics["adsc"]["speed_valid_after_clean"]),
                bool(section_metrics["before"]["anchor_speed_valid_after_clean"]),
                bool(section_metrics["after"]["anchor_speed_valid_after_clean"]),
            ]
        )

        drop_reasons: list[str] = []
        if not required_files_exist:
            drop_reasons.append("missing_required_file")
        if not readable:
            drop_reasons.append("unreadable_input")
        if not expected_columns_ok:
            drop_reasons.append("schema_issue")
        if not non_empty_sections:
            drop_reasons.append("section_too_short_after_clean")
        if not time_order_valid:
            drop_reasons.append("time_order_invalid_after_clean")
        if not coord_values_valid:
            drop_reasons.append("invalid_coordinate_values")
        if not bool(section_metrics["adsc"]["speed_valid_after_clean"]):
            drop_reasons.append("implausible_section_speed_after_clean")
        if not bool(section_metrics["before"]["anchor_speed_valid_after_clean"]):
            drop_reasons.append("implausible_anchor_speed_after_clean")
        if not bool(section_metrics["after"]["anchor_speed_valid_after_clean"]):
            drop_reasons.append("implausible_anchor_speed_after_clean")

        keep_flight = len(drop_reasons) == 0
        drop_reason = None if keep_flight else ";".join(sorted(set(drop_reasons)))

        stitched_clean = None
        stitched_clean_rows = int(len(section_frames["before"])) + int(len(section_frames["adsc"])) + int(len(section_frames["after"]))
        stitched_standardized = None
        stitched_standardized_rows = 0

        if cfg.write_per_flight_outputs:
            stitched_clean = _build_stitched_clean(
                before_df=section_frames["before"],
                adsc_df=section_frames["adsc"],
                after_df=section_frames["after"],
            )
            stitched_clean_rows = int(len(stitched_clean))
            stitched_standardized = _build_stitched_standardized(
                stitched_clean=stitched_clean,
                resample_seconds=cfg.resample_seconds,
            )
            stitched_standardized_rows = int(len(stitched_standardized))

        output_dir = cfg.output_root / "flights" / segment_id
        if keep_flight and cfg.write_per_flight_outputs:
            output_dir.mkdir(parents=True, exist_ok=True)
            section_frames["before"].to_parquet(output_dir / "adsb_before_clean.parquet", index=False)
            section_frames["adsc"].to_parquet(output_dir / "adsc_clean.parquet", index=False)
            section_frames["after"].to_parquet(output_dir / "adsb_after_clean.parquet", index=False)
            stitched_clean.to_parquet(output_dir / "stitched_clean.parquet", index=False)
            stitched_standardized.to_parquet(
                output_dir / f"stitched_standardized_{cfg.resample_seconds}s.parquet",
                index=False,
            )
            cleaning_metadata = {
                "segment_id": segment_id,
                "source_run": source_run,
                "step2_keep": keep_flight,
                "drop_reason": drop_reason,
                "section_metrics": {
                    section: {key: value for key, value in metrics.items() if key != "resampled_frame"}
                    for section, metrics in section_metrics.items()
                },
                "source_metadata_path": str(metadata_path.as_posix()),
            }
            (output_dir / "cleaning_metadata.json").write_text(
                json.dumps(cleaning_metadata, indent=2, default=_json_default),
                encoding="utf-8",
            )

        quality_issue_records.extend(
            _quality_issue_rows(
                segment_id=segment_id,
                source_run=source_run,
                metrics_by_section=section_metrics,
                keep_flight=keep_flight,
                drop_reason=drop_reason,
            )
        )

        audit_rows.append(
            {
                "segment_id": segment_id,
                "source_run": source_run,
                "metadata_exists": metadata_exists,
                "metadata_error": metadata_error,
                "required_files_exist": required_files_exist,
                "files_readable": readable,
                "expected_columns_ok": expected_columns_ok,
                "time_order_valid_after_clean": time_order_valid,
                "coordinate_values_valid_after_clean": coord_values_valid,
                "section_speed_valid_after_clean": section_speed_valid_after_clean,
                "before_rows_raw": section_metrics["before"]["rows_raw"],
                "before_rows_clean": section_metrics["before"]["rows_clean"],
                "before_boundary_rows_clean": section_metrics["before"]["boundary_rows_clean"],
                "before_boundary_too_sparse": section_metrics["before"]["boundary_too_sparse"],
                "max_before_speed_kts_after_clean": section_metrics["before"]["max_implied_speed_kts_after_clean"],
                "before_boundary_max_speed_kts_after_clean": section_metrics["before"]["boundary_max_speed_kts_after_clean"],
                "before_boundary_speed_valid_after_clean": section_metrics["before"]["boundary_speed_valid_after_clean"],
                "before_anchor_speed_kts_after_clean": section_metrics["before"]["anchor_speed_kts_after_clean"],
                "before_anchor_speed_valid_after_clean": section_metrics["before"]["anchor_speed_valid_after_clean"],
                "before_full_context_speed_issue": section_metrics["before"]["full_context_speed_issue"],
                "adsc_rows_raw": section_metrics["adsc"]["rows_raw"],
                "adsc_rows_clean": section_metrics["adsc"]["rows_clean"],
                "max_adsc_speed_kts_after_clean": section_metrics["adsc"]["max_implied_speed_kts_after_clean"],
                "after_rows_raw": section_metrics["after"]["rows_raw"],
                "after_rows_clean": section_metrics["after"]["rows_clean"],
                "after_boundary_rows_clean": section_metrics["after"]["boundary_rows_clean"],
                "after_boundary_too_sparse": section_metrics["after"]["boundary_too_sparse"],
                "max_after_speed_kts_after_clean": section_metrics["after"]["max_implied_speed_kts_after_clean"],
                "after_boundary_max_speed_kts_after_clean": section_metrics["after"]["boundary_max_speed_kts_after_clean"],
                "after_boundary_speed_valid_after_clean": section_metrics["after"]["boundary_speed_valid_after_clean"],
                "after_anchor_speed_kts_after_clean": section_metrics["after"]["anchor_speed_kts_after_clean"],
                "after_anchor_speed_valid_after_clean": section_metrics["after"]["anchor_speed_valid_after_clean"],
                "after_full_context_speed_issue": section_metrics["after"]["full_context_speed_issue"],
                "stitched_rows_raw": section_metrics["stitched"]["rows_raw"],
                "stitched_rows_clean": stitched_clean_rows,
                "step2_keep": keep_flight,
                "drop_reason": drop_reason,
            }
        )

        clean_catalog_rows.append(
            {
                "segment_id": segment_id,
                "source_run": source_run,
                "icao24": record.get("icao24"),
                "flight_callsign": record.get("flight_callsign"),
                "segment_start_time": record.get("segment_start_time"),
                "segment_end_time": record.get("segment_end_time"),
                "flight_start_time": record.get("flight_start_time"),
                "flight_end_time": record.get("flight_end_time"),
                "gap_duration_minutes": record.get("gap_duration_minutes"),
                "adsc_point_count_step1": record.get("adsc_point_count"),
                "before_rows_raw": section_metrics["before"]["rows_raw"],
                "before_rows_clean": section_metrics["before"]["rows_clean"],
                "before_boundary_rows_clean": section_metrics["before"]["boundary_rows_clean"],
                "before_boundary_too_sparse": section_metrics["before"]["boundary_too_sparse"],
                "adsc_rows_raw": section_metrics["adsc"]["rows_raw"],
                "adsc_rows_clean": section_metrics["adsc"]["rows_clean"],
                "after_rows_raw": section_metrics["after"]["rows_raw"],
                "after_rows_clean": section_metrics["after"]["rows_clean"],
                "after_boundary_rows_clean": section_metrics["after"]["boundary_rows_clean"],
                "after_boundary_too_sparse": section_metrics["after"]["boundary_too_sparse"],
                "stitched_rows_clean": stitched_clean_rows,
                "stitched_rows_standardized": stitched_standardized_rows,
                "before_duplicate_exact_count": section_metrics["before"]["duplicate_exact_count"],
                "before_duplicate_timestamp_count": section_metrics["before"]["duplicate_timestamp_count"],
                "before_isolated_spike_removed_count": section_metrics["before"]["isolated_spike_removed_count"],
                "adsc_duplicate_exact_count": section_metrics["adsc"]["duplicate_exact_count"],
                "adsc_duplicate_timestamp_count": section_metrics["adsc"]["duplicate_timestamp_count"],
                "adsc_isolated_spike_removed_count": section_metrics["adsc"]["isolated_spike_removed_count"],
                "after_duplicate_exact_count": section_metrics["after"]["duplicate_exact_count"],
                "after_duplicate_timestamp_count": section_metrics["after"]["duplicate_timestamp_count"],
                "after_isolated_spike_removed_count": section_metrics["after"]["isolated_spike_removed_count"],
                "max_before_speed_kts_after_clean": section_metrics["before"]["max_implied_speed_kts_after_clean"],
                "before_boundary_max_speed_kts_after_clean": section_metrics["before"]["boundary_max_speed_kts_after_clean"],
                "before_boundary_speed_valid_after_clean": section_metrics["before"]["boundary_speed_valid_after_clean"],
                "before_anchor_speed_kts_after_clean": section_metrics["before"]["anchor_speed_kts_after_clean"],
                "before_anchor_speed_valid_after_clean": section_metrics["before"]["anchor_speed_valid_after_clean"],
                "before_full_context_speed_issue": section_metrics["before"]["full_context_speed_issue"],
                "max_adsc_speed_kts_after_clean": section_metrics["adsc"]["max_implied_speed_kts_after_clean"],
                "max_after_speed_kts_after_clean": section_metrics["after"]["max_implied_speed_kts_after_clean"],
                "after_boundary_max_speed_kts_after_clean": section_metrics["after"]["boundary_max_speed_kts_after_clean"],
                "after_boundary_speed_valid_after_clean": section_metrics["after"]["boundary_speed_valid_after_clean"],
                "after_anchor_speed_kts_after_clean": section_metrics["after"]["anchor_speed_kts_after_clean"],
                "after_anchor_speed_valid_after_clean": section_metrics["after"]["anchor_speed_valid_after_clean"],
                "after_full_context_speed_issue": section_metrics["after"]["full_context_speed_issue"],
                "section_speed_valid_after_clean": section_speed_valid_after_clean,
                "step2_keep": keep_flight,
                "drop_reason": drop_reason,
                "quality_flag_any_duplicates_removed": bool(
                    section_metrics["before"]["duplicate_exact_count"]
                    or section_metrics["before"]["duplicate_timestamp_count"]
                    or section_metrics["adsc"]["duplicate_exact_count"]
                    or section_metrics["adsc"]["duplicate_timestamp_count"]
                    or section_metrics["after"]["duplicate_exact_count"]
                    or section_metrics["after"]["duplicate_timestamp_count"]
                ),
                "quality_flag_any_spike_removed": bool(
                    section_metrics["before"]["isolated_spike_removed_count"]
                    or section_metrics["adsc"]["isolated_spike_removed_count"]
                    or section_metrics["after"]["isolated_spike_removed_count"]
                ),
                "quality_flag_any_missing_key_values_removed": bool(
                    section_metrics["before"]["missing_key_value_count"]
                    or section_metrics["adsc"]["missing_key_value_count"]
                    or section_metrics["after"]["missing_key_value_count"]
                ),
                "quality_flag_any_sort_fix": bool(
                    section_metrics["before"]["was_unsorted"]
                    or section_metrics["adsc"]["was_unsorted"]
                    or section_metrics["after"]["was_unsorted"]
                ),
                "clean_flight_dir": (
                    None
                    if (not keep_flight or not cfg.write_per_flight_outputs)
                    else str(output_dir.as_posix())
                ),
                "clean_adsb_before_path": (
                    None
                    if (not keep_flight or not cfg.write_per_flight_outputs)
                    else str((output_dir / "adsb_before_clean.parquet").as_posix())
                ),
                "clean_adsc_path": (
                    None
                    if (not keep_flight or not cfg.write_per_flight_outputs)
                    else str((output_dir / "adsc_clean.parquet").as_posix())
                ),
                "clean_adsb_after_path": (
                    None
                    if (not keep_flight or not cfg.write_per_flight_outputs)
                    else str((output_dir / "adsb_after_clean.parquet").as_posix())
                ),
                "clean_stitched_path": (
                    None
                    if (not keep_flight or not cfg.write_per_flight_outputs)
                    else str((output_dir / "stitched_clean.parquet").as_posix())
                ),
                "standardized_stitched_path": (
                    None
                    if (not keep_flight or not cfg.write_per_flight_outputs)
                    else str((output_dir / f"stitched_standardized_{cfg.resample_seconds}s.parquet").as_posix())
                ),
                "cleaning_metadata_path": (
                    None
                    if (not keep_flight or not cfg.write_per_flight_outputs)
                    else str((output_dir / "cleaning_metadata.json").as_posix())
                ),
                "source_master_metadata_path": str(metadata_path.as_posix()),
            }
        )

    audit_df = pd.DataFrame(audit_rows).sort_values("segment_id").reset_index(drop=True)
    clean_catalog_df = pd.DataFrame(clean_catalog_rows).sort_values("segment_id").reset_index(drop=True)

    quality_issues_df = pd.DataFrame(quality_issue_records)
    if quality_issues_df.empty:
        quality_issues_df = pd.DataFrame(
            columns=[
                "segment_id",
                "source_run",
                "section",
                "issue_type",
                "issue_count",
                "issue_value_kts",
                "threshold_kts",
                "keep_flight",
                "drop_reason",
            ]
        )

    drop_reasons_df = clean_catalog_df.loc[
        ~clean_catalog_df["step2_keep"], ["segment_id", "drop_reason"]
    ].reset_index(drop=True)
    if drop_reasons_df.empty:
        drop_reasons_df = pd.DataFrame(columns=["segment_id", "drop_reason"])

    _log("Writing Step 2 catalog outputs...", cfg.verbose)
    _write_frame(clean_catalog_df, cfg.output_root / "catalog" / "clean_flights_catalog")
    _write_frame(audit_df, cfg.output_root / "catalog" / "quality_audit")
    _write_frame(quality_issues_df, cfg.output_root / "catalog" / "quality_issues")
    _write_frame(drop_reasons_df, cfg.output_root / "catalog" / "drop_reasons")

    summary_payload = {
        "step1_master_flights_loaded": int(total_master_flights),
        "step2_flights_processed_this_run": int(len(master_catalog)),
        "step2_flights_kept": int(clean_catalog_df["step2_keep"].sum()),
        "step2_flights_dropped": int((~clean_catalog_df["step2_keep"]).sum()),
        "quality_issue_rows": int(len(quality_issues_df)),
        "resample_seconds": cfg.resample_seconds,
        "write_per_flight_outputs": bool(cfg.write_per_flight_outputs),
        "preview_mode": bool(cfg.max_flights_to_process is not None),
        "max_flights_to_process": cfg.max_flights_to_process,
    }

    (cfg.output_root / "audit_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    readme = f"""# Step 2 Clean Dataset

This folder contains the rerunnable local Step 2 cleaning outputs built from:

- `{cfg.step1_master_root.as_posix()}`

Source of truth for Step 2:
- `catalog/clean_flights_validated.parquet`

Key outputs:
- `catalog/clean_flights_validated.parquet`
- `catalog/quality_audit.parquet`
- `catalog/quality_issues.parquet`
- `catalog/drop_reasons.parquet`

Optional per-flight outputs:
- `flights/<segment_id>/adsb_before_clean.parquet`
- `flights/<segment_id>/adsc_clean.parquet`
- `flights/<segment_id>/adsb_after_clean.parquet`
- `flights/<segment_id>/stitched_clean.parquet`
- `flights/<segment_id>/stitched_standardized_{cfg.resample_seconds}s.parquet`

Cleaning rules applied:
- parse timestamps and sort rows by time
- coerce numeric trajectory columns
- remove rows with missing timestamp / latitude / longitude
- remove invalid coordinates outside valid latitude / longitude bounds
- remove exact duplicate rows
- remove duplicate timestamps within the same section
- remove isolated one-point speed spikes when the bridged path is plausible
- use anchor continuity near the gap for keep/drop
- resample the full stitched clean trajectory on one unified grid
- metadata issues are audited but do not automatically drop flights

Run configuration:
- write_per_flight_outputs: {cfg.write_per_flight_outputs}
- preview_mode: {cfg.max_flights_to_process is not None}
- max_flights_to_process: {cfg.max_flights_to_process}

Counts:
- Step 1 master flights loaded: {summary_payload["step1_master_flights_loaded"]}
- Flights processed this run: {summary_payload["step2_flights_processed_this_run"]}
- Step 2 flights kept: {summary_payload["step2_flights_kept"]}
- Step 2 flights dropped: {summary_payload["step2_flights_dropped"]}
"""
    (cfg.output_root / "README.md").write_text(readme, encoding="utf-8")

    _log("Step 2 build finished.", cfg.verbose)
    return summary_payload

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Step 2 clean dataset from step1_master.")
    parser.add_argument("--repo-root", type=str, default=None, help="Optional repo root override.")
    parser.add_argument("--max-flights", type=int, default=None, help="Preview mode: process only the first N flights.")
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Write only catalog/audit outputs, not per-flight cleaned parquet files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N flights.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete the existing step2_clean output root before writing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging.",
    )
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    repo_root = None if args.repo_root is None else Path(args.repo_root)
    base_cfg = default_step2_config(repo_root)

    cfg = Step2Config(
        step1_master_root=base_cfg.step1_master_root,
        output_root=base_cfg.output_root,
        resample_seconds=base_cfg.resample_seconds,
        max_flights_to_process=args.max_flights,
        write_per_flight_outputs=not args.catalog_only,
        progress_every=args.progress_every,
        clean_existing_output=not args.keep_output,
        verbose=not args.quiet,
    )

    summary = build_step2_clean(cfg)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
