
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

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_TIME_COLUMN = "timestamp"

@dataclass(frozen=True)
class Step3Config:
    """Filesystem/runtime configuration for Step 3 baseline reconstruction."""

    step2_root: Path
    output_root: Path
    write_per_flight_outputs: bool = True
    max_flights_to_process: int | None = None
    progress_every: int = 25
    clean_existing_output: bool = True
    verbose: bool = True

def default_step3_config(repo_root: Path | None = None) -> Step3Config:
    project_root = repo_root or Path(__file__).resolve().parents[2]
    return Step3Config(
        step2_root=project_root / "artifacts" / "step2_clean",
        output_root=project_root / "artifacts" / "step3_baseline",
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

def _write_frame(frame: pd.DataFrame, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(base_path.with_suffix(".parquet"), index=False)
    frame.to_csv(base_path.with_suffix(".csv"), index=False)

def _coerce_timestamp(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce", utc=True)
    return values.dt.tz_convert(None)

def _resolve_flight_paths(step2_root: Path, row: pd.Series) -> dict[str, Path]:
    segment_id = str(row["segment_id"])
    flight_dir = step2_root / "flights" / segment_id
    resolved: dict[str, Path] = {}
    path_columns = {
        "before": "clean_adsb_before_path",
        "adsc": "clean_adsc_path",
        "after": "clean_adsb_after_path",
        "stitched": "clean_stitched_path",
        "standardized": "standardized_stitched_path",
        "metadata": "cleaning_metadata_path",
    }
    default_names = {
        "before": "adsb_before_clean.parquet",
        "adsc": "adsc_clean.parquet",
        "after": "adsb_after_clean.parquet",
        "stitched": "stitched_clean.parquet",
        "standardized": "stitched_standardized_60s.parquet",
        "metadata": "cleaning_metadata.json",
    }
    for key, col in path_columns.items():
        raw = row.get(col)
        candidate = None if pd.isna(raw) else Path(str(raw))
        if candidate is not None and candidate.exists():
            resolved[key] = candidate
        else:
            resolved[key] = flight_dir / default_names[key]
    return resolved

def _load_track(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing track file: {path}")
    df = pd.read_parquet(path).copy()
    if DEFAULT_TIME_COLUMN not in df.columns:
        raise ValueError(f"{path} is missing required timestamp column.")
    df[DEFAULT_TIME_COLUMN] = _coerce_timestamp(df[DEFAULT_TIME_COLUMN])
    df = df.dropna(subset=[DEFAULT_TIME_COLUMN, "latitude", "longitude"]).sort_values(DEFAULT_TIME_COLUMN).reset_index(drop=True)
    return df

def _haversine_m(
    lat1_deg: np.ndarray | float,
    lon1_deg: np.ndarray | float,
    lat2_deg: np.ndarray | float,
    lon2_deg: np.ndarray | float,
) -> np.ndarray:
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_M * c

def _track_length_m(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    return float(
        _haversine_m(
            df["latitude"].to_numpy()[:-1],
            df["longitude"].to_numpy()[:-1],
            df["latitude"].to_numpy()[1:],
            df["longitude"].to_numpy()[1:],
        ).sum()
    )

def _to_unit_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    return np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ],
        dtype=float,
    )

def _xyz_to_latlon(xyz: np.ndarray) -> tuple[float, float]:
    x, y, z = xyz.tolist()
    norm = float(np.linalg.norm(xyz))
    if norm == 0:
        raise ValueError("Cannot convert zero-length vector to lat/lon.")
    x, y, z = x / norm, y / norm, z / norm
    lat = math.degrees(math.asin(np.clip(z, -1.0, 1.0)))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon

def _great_circle_points(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p0 = _to_unit_xyz(start_lat, start_lon)
    p1 = _to_unit_xyz(end_lat, end_lon)
    dot = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    omega = math.acos(dot)

    if np.isclose(omega, 0.0):
        lats = np.full_like(fractions, float(start_lat), dtype=float)
        lons = np.full_like(fractions, float(start_lon), dtype=float)
        return lats, lons

    sin_omega = math.sin(omega)
    points = []
    for frac in fractions:
        frac = float(np.clip(frac, 0.0, 1.0))
        w0 = math.sin((1.0 - frac) * omega) / sin_omega
        w1 = math.sin(frac * omega) / sin_omega
        xyz = w0 * p0 + w1 * p1
        lat, lon = _xyz_to_latlon(xyz)
        points.append((lat, lon))
    lats = np.array([p[0] for p in points], dtype=float)
    lons = np.array([p[1] for p in points], dtype=float)
    return lats, lons

def _linear_column(start_value: Any, end_value: Any, fractions: np.ndarray) -> np.ndarray:
    try:
        start_f = float(start_value)
        end_f = float(end_value)
    except Exception:
        return np.full_like(fractions, np.nan, dtype=float)
    return start_f + (end_f - start_f) * fractions

def _build_baseline_prediction(
    before_df: pd.DataFrame,
    adsc_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> pd.DataFrame:
    if before_df.empty or adsc_df.empty or after_df.empty:
        raise ValueError("before/adsc/after tracks must all be non-empty.")

    start_anchor = before_df.sort_values(DEFAULT_TIME_COLUMN).iloc[-1]
    end_anchor = after_df.sort_values(DEFAULT_TIME_COLUMN).iloc[0]
    adsc_sorted = adsc_df.sort_values(DEFAULT_TIME_COLUMN).reset_index(drop=True).copy()

    start_time = pd.Timestamp(start_anchor[DEFAULT_TIME_COLUMN])
    end_time = pd.Timestamp(end_anchor[DEFAULT_TIME_COLUMN])
    adsc_times = pd.to_datetime(adsc_sorted[DEFAULT_TIME_COLUMN])

    total_gap_seconds = float((end_time - start_time).total_seconds())
    if total_gap_seconds <= 0:
        raise ValueError("Anchor timestamps are invalid: end anchor is not after start anchor.")

    fractions = ((adsc_times - start_time).dt.total_seconds() / total_gap_seconds).to_numpy(dtype=float)
    fractions = np.clip(fractions, 0.0, 1.0)

    pred_lat, pred_lon = _great_circle_points(
        start_lat=float(start_anchor["latitude"]),
        start_lon=float(start_anchor["longitude"]),
        end_lat=float(end_anchor["latitude"]),
        end_lon=float(end_anchor["longitude"]),
        fractions=fractions,
    )

    prediction = pd.DataFrame(
        {
            "segment_id": adsc_sorted["segment_id"].astype(str),
            "timestamp": adsc_times,
            "fraction_of_gap": fractions,
            "pred_latitude": pred_lat,
            "pred_longitude": pred_lon,
            "true_latitude": adsc_sorted["latitude"].to_numpy(dtype=float),
            "true_longitude": adsc_sorted["longitude"].to_numpy(dtype=float),
            "anchor_start_time": start_time,
            "anchor_end_time": end_time,
            "anchor_start_latitude": float(start_anchor["latitude"]),
            "anchor_start_longitude": float(start_anchor["longitude"]),
            "anchor_end_latitude": float(end_anchor["latitude"]),
            "anchor_end_longitude": float(end_anchor["longitude"]),
        }
    )

    for source_col, pred_col in [
        ("altitude_m", "pred_altitude_m"),
        ("geoaltitude_m", "pred_geoaltitude_m"),
        ("baroaltitude_m", "pred_baroaltitude_m"),
        ("velocity_mps", "pred_velocity_mps"),
        ("heading_deg", "pred_heading_deg"),
    ]:
        start_val = start_anchor[source_col] if source_col in before_df.columns else np.nan
        end_val = end_anchor[source_col] if source_col in after_df.columns else np.nan
        prediction[pred_col] = _linear_column(start_val, end_val, fractions)
        if source_col in adsc_sorted.columns:
            prediction[f"true_{source_col}"] = pd.to_numeric(adsc_sorted[source_col], errors="coerce").to_numpy(dtype=float)

    prediction["point_error_m"] = _haversine_m(
        prediction["pred_latitude"].to_numpy(),
        prediction["pred_longitude"].to_numpy(),
        prediction["true_latitude"].to_numpy(),
        prediction["true_longitude"].to_numpy(),
    )

    return prediction

def _evaluate_prediction(
    catalog_row: pd.Series,
    prediction: pd.DataFrame,
    before_df: pd.DataFrame,
    adsc_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> dict[str, Any]:
    point_errors = prediction["point_error_m"].to_numpy(dtype=float)
    true_path_length_m = _track_length_m(adsc_df)
    pred_path_length_m = _track_length_m(
        prediction.rename(columns={"pred_latitude": "latitude", "pred_longitude": "longitude"})[
            ["timestamp", "latitude", "longitude"]
        ]
    )
    anchor_gap_m = float(
        _haversine_m(
            float(before_df.iloc[-1]["latitude"]),
            float(before_df.iloc[-1]["longitude"]),
            float(after_df.iloc[0]["latitude"]),
            float(after_df.iloc[0]["longitude"]),
        )
    )

    row = {
        "segment_id": str(catalog_row["segment_id"]),
        "source_run": catalog_row.get("source_run"),
        "icao24": catalog_row.get("icao24"),
        "flight_callsign": catalog_row.get("flight_callsign"),
        "step2_keep": bool(catalog_row.get("step2_keep", False)),
        "adsc_point_count_step2": int(len(adsc_df)),
        "baseline_method": "great_circle_time_aligned",
        "prediction_point_count": int(len(prediction)),
        "mean_error_m": float(np.mean(point_errors)),
        "median_error_m": float(np.median(point_errors)),
        "rmse_error_m": float(np.sqrt(np.mean(point_errors ** 2))),
        "max_error_m": float(np.max(point_errors)),
        "min_error_m": float(np.min(point_errors)),
        "p90_error_m": float(np.percentile(point_errors, 90)),
        "p95_error_m": float(np.percentile(point_errors, 95)),
        "anchor_gap_m": anchor_gap_m,
        "true_path_length_m": true_path_length_m,
        "pred_path_length_m": pred_path_length_m,
        "path_length_error_m": float(pred_path_length_m - true_path_length_m),
        "path_length_abs_error_m": float(abs(pred_path_length_m - true_path_length_m)),
        "gap_duration_minutes": float(catalog_row.get("gap_duration_minutes", np.nan)),
        "before_anchor_speed_kts_after_clean": catalog_row.get("before_anchor_speed_kts_after_clean"),
        "after_anchor_speed_kts_after_clean": catalog_row.get("after_anchor_speed_kts_after_clean"),
        "clean_adsb_before_path": catalog_row.get("clean_adsb_before_path"),
        "clean_adsc_path": catalog_row.get("clean_adsc_path"),
        "clean_adsb_after_path": catalog_row.get("clean_adsb_after_path"),
    }
    return row

def _build_overall_summary(metrics_df: pd.DataFrame) -> dict[str, Any]:
    if metrics_df.empty:
        return {
            "flights_evaluated": 0,
            "baseline_method": "great_circle_time_aligned",
        }

    return {
        "flights_evaluated": int(len(metrics_df)),
        "baseline_method": "great_circle_time_aligned",
        "mean_of_mean_error_m": float(metrics_df["mean_error_m"].mean()),
        "median_of_mean_error_m": float(metrics_df["mean_error_m"].median()),
        "mean_rmse_error_m": float(metrics_df["rmse_error_m"].mean()),
        "median_rmse_error_m": float(metrics_df["rmse_error_m"].median()),
        "mean_max_error_m": float(metrics_df["max_error_m"].mean()),
        "median_max_error_m": float(metrics_df["max_error_m"].median()),
        "mean_path_length_abs_error_m": float(metrics_df["path_length_abs_error_m"].mean()),
        "median_path_length_abs_error_m": float(metrics_df["path_length_abs_error_m"].median()),
    }

def _load_step2_catalog(step2_root: Path) -> pd.DataFrame:
    catalog_path = step2_root / "catalog" / "clean_flights_validated.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Step 2 clean catalog not found: {catalog_path}")
    df = pd.read_parquet(catalog_path).copy()
    if "step2_keep" not in df.columns:
        raise ValueError("Step 2 clean catalog is missing 'step2_keep'.")
    return df

def run_step3_baseline(config: Step3Config | None = None) -> dict[str, Any]:
    cfg = config or default_step3_config()

    if cfg.clean_existing_output and cfg.output_root.exists():
        _log(f"Removing existing Step 3 output root: {cfg.output_root.as_posix()}", cfg.verbose)
        shutil.rmtree(cfg.output_root)

    (cfg.output_root / "catalog").mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "flights").mkdir(parents=True, exist_ok=True)

    step2_catalog = _load_step2_catalog(cfg.step2_root)
    kept = step2_catalog.loc[step2_catalog["step2_keep"] == True].copy().reset_index(drop=True)
    total_kept = len(kept)

    if cfg.max_flights_to_process is not None:
        kept = kept.head(cfg.max_flights_to_process).copy()
        _log(
            f"Preview mode enabled: processing first {len(kept)} / {total_kept} kept Step 2 flights.",
            cfg.verbose,
        )
    else:
        _log(f"Loaded {total_kept} kept Step 2 flights.", cfg.verbose)

    metric_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for idx, row in kept.iterrows():
        if cfg.progress_every > 0 and (idx == 0 or (idx + 1) % cfg.progress_every == 0 or (idx + 1) == len(kept)):
            _log(f"[Step3] Processing flight {idx + 1}/{len(kept)}", cfg.verbose)

        segment_id = str(row["segment_id"])
        paths = _resolve_flight_paths(cfg.step2_root, row)
        try:
            before_df = _load_track(paths["before"])
            adsc_df = _load_track(paths["adsc"])
            after_df = _load_track(paths["after"])

            prediction = _build_baseline_prediction(before_df, adsc_df, after_df)
            metrics_row = _evaluate_prediction(row, prediction, before_df, adsc_df, after_df)
            metric_rows.append(metrics_row)

            if cfg.write_per_flight_outputs:
                flight_dir = cfg.output_root / "flights" / segment_id
                flight_dir.mkdir(parents=True, exist_ok=True)
                prediction.to_parquet(flight_dir / "baseline_prediction.parquet", index=False)
                prediction.to_csv(flight_dir / "baseline_prediction.csv", index=False)
                (flight_dir / "baseline_metadata.json").write_text(
                    json.dumps(
                        {
                            "segment_id": segment_id,
                            "baseline_method": "great_circle_time_aligned",
                            "source_step2_paths": {k: str(v.as_posix()) for k, v in paths.items()},
                            "metrics": metrics_row,
                        },
                        indent=2,
                        default=_json_default,
                    ),
                    encoding="utf-8",
                )
        except Exception as exc:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "issue_type": "baseline_build_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values("segment_id").reset_index(drop=True) if metric_rows else pd.DataFrame()
    issues_df = pd.DataFrame(issue_rows).sort_values("segment_id").reset_index(drop=True) if issue_rows else pd.DataFrame(
        columns=["segment_id", "issue_type", "detail"]
    )
    summary = _build_overall_summary(metrics_df)

    _write_frame(metrics_df, cfg.output_root / "catalog" / "baseline_metrics")
    _write_frame(issues_df, cfg.output_root / "catalog" / "baseline_issues")

    (cfg.output_root / "catalog" / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )

    readme = f"""# Step 3 Baseline Reconstruction

This folder contains the Step 3 baseline reconstruction outputs built from:

- `{cfg.step2_root.as_posix()}`

Baseline method:
- great-circle interpolation between the last cleaned ADS-B before point and the first cleaned ADS-B after point
- predicted timestamps aligned to the cleaned ADS-C timestamps
- evaluation against the cleaned ADS-C segment as ground truth

Primary outputs:
- `catalog/baseline_metrics.parquet`
- `catalog/baseline_metrics.csv`
- `catalog/baseline_summary.json`
- `catalog/baseline_issues.parquet`
- `catalog/baseline_issues.csv`

Optional per-flight outputs:
- `flights/<segment_id>/baseline_prediction.parquet`
- `flights/<segment_id>/baseline_prediction.csv`
- `flights/<segment_id>/baseline_metadata.json`

Run configuration:
- write_per_flight_outputs: {cfg.write_per_flight_outputs}
- preview_mode: {cfg.max_flights_to_process is not None}
- max_flights_to_process: {cfg.max_flights_to_process}

Counts:
- Step 2 kept flights loaded: {total_kept}
- Flights evaluated this run: {summary.get("flights_evaluated", 0)}
"""
    (cfg.output_root / "README.md").write_text(readme, encoding="utf-8")

    _log("Step 3 baseline build finished.", cfg.verbose)
    return summary

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Step 3 baseline reconstruction outputs from Step 2 clean data.")
    parser.add_argument("--repo-root", type=str, default=None, help="Optional repo root override.")
    parser.add_argument("--max-flights", type=int, default=None, help="Preview mode: process only the first N kept Step 2 flights.")
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Write only catalog metrics/issues, not per-flight prediction files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N flights.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete the existing step3_baseline output root before writing.",
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
    base_cfg = default_step3_config(repo_root)

    cfg = Step3Config(
        step2_root=base_cfg.step2_root,
        output_root=base_cfg.output_root,
        write_per_flight_outputs=not args.catalog_only,
        max_flights_to_process=args.max_flights,
        progress_every=args.progress_every,
        clean_existing_output=not args.keep_output,
        verbose=not args.quiet,
    )

    summary = run_step3_baseline(cfg)
    print(json.dumps(summary, indent=2, default=_json_default))

if __name__ == "__main__":
    main()
