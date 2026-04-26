"""Build Step 4 training datasets from cleaned flights."""

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

BEFORE_STEPS      = 64    # steps of ADS-B before to keep (at 60 s → ~64 min)
AFTER_STEPS       = 32    # steps of ADS-B after  to keep (at 60 s → ~32 min)
RESAMPLE_SECONDS  = 60    # resample interval for before/after sequences
MAX_ADSC_WP       = 32    # maximum ADS-C waypoints per flight (pad to this)

N_SEQ_FEATURES = 6

LAT_MEAN,  LAT_STD  = 53.0,   8.0
LON_MEAN,  LON_STD  = -30.0,  25.0
VEL_MEAN,  VEL_STD  = 240.0,  30.0    # m/s
ALT_MEAN,  ALT_STD  = 10500.0, 1000.0 # m

@dataclass(frozen=True)
class Step4Config:
    step2_root:              Path           = Path("artifacts/step2_clean")
    output_root:             Path           = Path("artifacts/step4_ml_dataset")
    min_adsc_points_for_ml:  int            = 3
    max_flights_to_process:  int | None     = None
    train_fraction:          float          = 0.70
    val_fraction:            float          = 0.15
    test_fraction:           float          = 0.15
    random_seed:             int            = 42
    clean_existing_output:   bool           = True
    write_per_flight_outputs: bool          = False
    verbose:                 bool           = True
    progress_every:          int            = 25

def _step2_catalog_path(step2_root: Path) -> Path:
    validated = step2_root / "catalog" / "clean_flights_validated.parquet"
    if validated.exists():
        return validated
    fallback = step2_root / "catalog" / "clean_flights_catalog.parquet"
    return fallback

def _load_step2_catalog(step2_root: Path) -> pd.DataFrame:
    p = _step2_catalog_path(step2_root)
    if not p.exists():
        raise FileNotFoundError(f"Missing Step 2 catalog: {p}")
    df = pd.read_parquet(p)
    if "step2_keep" not in df.columns:
        raise ValueError("Step 2 catalog missing 'step2_keep' column.")
    return df

def _resolve_section_path(row: pd.Series, step2_root: Path,
                           filename: str, candidates: list[str]) -> Path:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            p = Path(str(row[col]))
            if p.exists():
                return p
    return step2_root / "flights" / str(row["segment_id"]) / filename

def _before_path(row, r): return _resolve_section_path(row, r, "adsb_before_clean.parquet", ["clean_adsb_before_path"])
def _adsc_path(row, r):   return _resolve_section_path(row, r, "adsc_clean.parquet",         ["clean_adsc_path"])
def _after_path(row, r):  return _resolve_section_path(row, r, "adsb_after_clean.parquet",   ["clean_adsb_after_path"])

def _load_section(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)

def _to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)

EARTH_RADIUS_M = 6_371_000.0

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return EARTH_RADIUS_M * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))

def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y  = math.sin(dl) * math.cos(phi2)
    x  = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def _latlon_to_xyz(lat, lon):
    lat, lon = math.radians(lat), math.radians(lon)
    return np.array([math.cos(lat)*math.cos(lon),
                     math.cos(lat)*math.sin(lon),
                     math.sin(lat)], dtype=float)

def _xyz_to_latlon(xyz):
    xyz = xyz / np.linalg.norm(xyz)
    return math.degrees(math.asin(np.clip(xyz[2], -1, 1))), \
           math.degrees(math.atan2(xyz[1], xyz[0]))

def great_circle_interpolate(lat0, lon0, lat1, lon1, tau) -> tuple[float, float]:
    tau  = float(np.clip(tau, 0, 1))
    p0, p1 = _latlon_to_xyz(lat0, lon0), _latlon_to_xyz(lat1, lon1)
    omega = math.acos(float(np.clip(np.dot(p0, p1), -1, 1)))
    if abs(omega) < 1e-12:
        return lat0, lon0
    s = math.sin(omega)
    return _xyz_to_latlon(math.sin((1-tau)*omega)/s * p0 + math.sin(tau*omega)/s * p1)

def local_residual_m(blat, blon, tlat, tlon) -> tuple[float, float]:
    north = (tlat - blat) * 111_320.0
    east  = (tlon - blon) * 111_320.0 * math.cos(math.radians(blat))
    return east, north

def _assign_splits_by_icao24(
    flights: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Split at the aircraft level (icao24) to prevent data leakage across routes."""
    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0,
                        rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/val/test fractions must sum to 1.0")

    if "icao24" not in flights.columns:
        raise ValueError("Step 2 catalog missing 'icao24' column needed for leak-free split.")

    rng = np.random.default_rng(random_seed)
    unique_aircraft = sorted(flights["icao24"].dropna().astype(str).unique())
    unique_aircraft = list(rng.permutation(unique_aircraft))

    n = len(unique_aircraft)
    n_train = int(round(n * train_fraction))
    n_val   = int(round(n * val_fraction))
    n_train = min(n_train, n)
    n_val   = min(n_val, n - n_train)

    aircraft_split = {}
    for i, ac in enumerate(unique_aircraft):
        if i < n_train:
            aircraft_split[ac] = "train"
        elif i < n_train + n_val:
            aircraft_split[ac] = "val"
        else:
            aircraft_split[ac] = "test"

    result = flights.copy()
    result["split"] = result["icao24"].astype(str).map(aircraft_split).fillna("train")
    return result

def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["timestamp"] = _to_timestamp(w["timestamp"])
    w["latitude"]  = pd.to_numeric(w["latitude"],  errors="coerce")
    w["longitude"] = pd.to_numeric(w["longitude"], errors="coerce")
    for col in ["velocity_mps", "heading_deg", "geoaltitude_m", "baroaltitude_m"]:
        if col in w.columns:
            w[col] = pd.to_numeric(w[col], errors="coerce")
    return w.dropna(subset=["timestamp","latitude","longitude"]).sort_values("timestamp").reset_index(drop=True)

def _scalar(row: pd.Series, col: str) -> float:
    if col not in row.index or pd.isna(row[col]):
        return float("nan")
    return float(row[col])

def _build_point_rows(catalog_row, before, adsc, after):
    """Build one row per ADS-C waypoint (tabular / pointwise format)."""
    b_anc  = before.iloc[-1]
    a_anc  = after.iloc[0]
    t0, t1 = pd.Timestamp(b_anc["timestamp"]), pd.Timestamp(a_anc["timestamp"])
    dur    = float((t1 - t0).total_seconds())
    if dur <= 0:
        raise ValueError("non-positive gap duration")

    dist_m  = haversine_m(float(b_anc["latitude"]), float(b_anc["longitude"]),
                          float(a_anc["latitude"]), float(a_anc["longitude"]))
    bear    = bearing_deg(float(b_anc["latitude"]), float(b_anc["longitude"]),
                          float(a_anc["latitude"]), float(a_anc["longitude"]))

    rows = []
    for idx, row in adsc.iterrows():
        elapsed = float((row["timestamp"] - t0).total_seconds())
        tau_raw = elapsed / dur
        tau     = float(np.clip(tau_raw, 0, 1))
        blat, blon = great_circle_interpolate(
            float(b_anc["latitude"]), float(b_anc["longitude"]),
            float(a_anc["latitude"]), float(a_anc["longitude"]), tau)
        tlat, tlon = float(row["latitude"]), float(row["longitude"])
        err_m = haversine_m(blat, blon, tlat, tlon)
        res_e, res_n = local_residual_m(blat, blon, tlat, tlon)
        rows.append({
            "segment_id":                   str(catalog_row["segment_id"]),
            "source_run":                   catalog_row.get("source_run"),
            "processing_day":               catalog_row.get("processing_day"),
            "icao24":                       catalog_row.get("icao24"),
            "adsc_point_index":             int(idx),
            "adsc_point_count_clean":       int(len(adsc)),
            "timestamp":                    row["timestamp"],
            "split":                        catalog_row["split"],
            "tau":                          tau,
            "tau_raw":                      tau_raw,
            "elapsed_sec_from_before_anchor": elapsed,
            "gap_duration_sec":             dur,
            "gap_duration_minutes":         dur / 60.0,
            "before_anchor_lat":            float(b_anc["latitude"]),
            "before_anchor_lon":            float(b_anc["longitude"]),
            "after_anchor_lat":             float(a_anc["latitude"]),
            "after_anchor_lon":             float(a_anc["longitude"]),
            "before_anchor_velocity_mps":   _scalar(b_anc, "velocity_mps"),
            "after_anchor_velocity_mps":    _scalar(a_anc, "velocity_mps"),
            "before_anchor_heading_deg":    _scalar(b_anc, "heading_deg"),
            "after_anchor_heading_deg":     _scalar(a_anc, "heading_deg"),
            "before_anchor_geoaltitude_m":  _scalar(b_anc, "geoaltitude_m"),
            "after_anchor_geoaltitude_m":   _scalar(a_anc, "geoaltitude_m"),
            "anchor_distance_m":            dist_m,
            "anchor_bearing_deg":           bear,
            "baseline_lat":                 blat,
            "baseline_lon":                 blon,
            "target_lat":                   tlat,
            "target_lon":                   tlon,
            "baseline_error_m":             err_m,
            "target_residual_east_m":       res_e,
            "target_residual_north_m":      res_n,
        })
    meta = {
        "segment_id":           str(catalog_row["segment_id"]),
        "source_run":           catalog_row.get("source_run"),
        "icao24":               catalog_row.get("icao24"),
        "split":                catalog_row["split"],
        "gap_duration_sec":     dur,
        "anchor_distance_m":    dist_m,
        "adsc_point_count_clean": int(len(adsc)),
    }
    return pd.DataFrame(rows), meta

def _normalize_feature(val: float, mean: float, std: float) -> float:
    if not math.isfinite(val):
        return 0.0
    return (val - mean) / std

def _resample_track_to_grid(df: pd.DataFrame, resample_sec: int) -> pd.DataFrame:
    """Resample a track to a fixed time grid using time interpolation."""
    if len(df) < 2:
        return df
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    start = ordered["timestamp"].min().floor(f"{resample_sec}s")
    end   = ordered["timestamp"].max().ceil(f"{resample_sec}s")
    if start == end:
        return ordered
    grid = pd.date_range(start=start, end=end, freq=f"{resample_sec}s")
    numeric_cols = ["latitude", "longitude", "velocity_mps", "heading_deg", "geoaltitude_m"]
    present = [c for c in numeric_cols if c in ordered.columns]
    resampled = (
        ordered.set_index("timestamp")[present]
        .sort_index()
        .reindex(grid)
        .interpolate(method="time", limit_direction="forward", limit_area="inside")
    )
    return resampled.reset_index().rename(columns={"index": "timestamp"})

def _track_to_feature_array(df: pd.DataFrame, n_steps: int,
                             from_end: bool = True) -> np.ndarray:
    """Convert a track DataFrame to a fixed-length normalized feature array with mask."""
    out = np.zeros((n_steps, N_SEQ_FEATURES), dtype=np.float32)
    mask = np.zeros(n_steps, dtype=np.float32)

    if df.empty:
        return out, mask

    df = df.sort_values("timestamp").reset_index(drop=True)

    if from_end:
        df = df.tail(n_steps).reset_index(drop=True)
        offset = n_steps - len(df)   # left-pad
    else:
        df = df.head(n_steps).reset_index(drop=True)
        offset = 0                    # right-pad

    for i, row in df.iterrows():
        dest = offset + i if from_end else i
        if dest >= n_steps:
            break

        lat = _safe_float(row.get("latitude",  np.nan))
        lon = _safe_float(row.get("longitude", np.nan))
        vel = float(row["velocity_mps"])  if "velocity_mps"  in row.index and pd.notna(row["velocity_mps"])  else np.nan
        hdg = float(row["heading_deg"])   if "heading_deg"   in row.index and pd.notna(row["heading_deg"])   else np.nan
        alt = float(row["geoaltitude_m"]) if "geoaltitude_m" in row.index and pd.notna(row["geoaltitude_m"]) else np.nan

        out[dest, 0] = _normalize_feature(lat, LAT_MEAN, LAT_STD)
        out[dest, 1] = _normalize_feature(lon, LON_MEAN, LON_STD)
        out[dest, 2] = _normalize_feature(vel, VEL_MEAN, VEL_STD)
        out[dest, 3] = math.sin(math.radians(hdg)) if math.isfinite(hdg) else 0.0
        out[dest, 4] = math.cos(math.radians(hdg)) if math.isfinite(hdg) else 0.0
        out[dest, 5] = _normalize_feature(alt, ALT_MEAN, ALT_STD)
        mask[dest] = 1.0

    return out, mask

def _safe_float(val) -> float:
    """Convert to float safely - returns nan for NaT, None, or non-numeric."""
    if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'NaTType'):
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")

def _safe_timedelta_sec(ts, t0) -> float:
    """Compute (ts - t0).total_seconds() safely, returning nan if either is NaT."""
    if pd.isna(ts) or pd.isna(t0):
        return float("nan")
    try:
        return float((pd.Timestamp(ts) - pd.Timestamp(t0)).total_seconds())
    except Exception:
        return float("nan")

def _build_sequence_sample(catalog_row, before, adsc, after):
    """Build one sequence sample dict for GRU training."""
    b_anc = before.sort_values("timestamp").iloc[-1]
    a_anc = after.sort_values("timestamp").iloc[0]

    t0_raw = b_anc["timestamp"]
    t1_raw = a_anc["timestamp"]
    if pd.isna(t0_raw) or pd.isna(t1_raw):
        raise ValueError("NaT anchor timestamp in before/after track")

    t0  = pd.Timestamp(t0_raw)
    t1  = pd.Timestamp(t1_raw)
    dur = _safe_timedelta_sec(t1, t0)
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError(f"non-positive gap duration: {dur}")

    before_resampled = _resample_track_to_grid(before, RESAMPLE_SECONDS)
    after_resampled  = _resample_track_to_grid(after,  RESAMPLE_SECONDS)

    before_seq,  before_mask  = _track_to_feature_array(before_resampled, BEFORE_STEPS, from_end=True)
    after_seq,   after_mask   = _track_to_feature_array(after_resampled,  AFTER_STEPS,  from_end=False)

    adsc_sorted = adsc.sort_values("timestamp").reset_index(drop=True)
    n_wp = min(len(adsc_sorted), MAX_ADSC_WP)

    adsc_targets = np.zeros((MAX_ADSC_WP, 2),  dtype=np.float32)
    adsc_tau     = np.zeros(MAX_ADSC_WP,        dtype=np.float32)
    adsc_mask    = np.zeros(MAX_ADSC_WP,        dtype=np.float32)

    for i in range(n_wp):
        wp = adsc_sorted.iloc[i]
        lat = _safe_float(wp["latitude"])
        lon = _safe_float(wp["longitude"])
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        adsc_targets[i, 0] = np.float32(lat)
        adsc_targets[i, 1] = np.float32(lon)
        elapsed = _safe_timedelta_sec(wp["timestamp"], t0)
        if math.isfinite(elapsed):
            adsc_tau[i] = float(np.clip(elapsed / dur, 0, 1))
        adsc_mask[i] = 1.0

    return {
        "segment_id":    str(catalog_row["segment_id"]),
        "icao24":        str(catalog_row.get("icao24", "")),
        "split":         str(catalog_row["split"]),
        "before_seq":    before_seq,
        "before_mask":   before_mask,
        "after_seq":     after_seq,
        "after_mask":    after_mask,
        "adsc_targets":  adsc_targets,
        "adsc_tau":      adsc_tau,
        "adsc_mask":     adsc_mask,
        "gap_dur_sec":   np.float32(dur),
        "n_adsc_wp":     np.int32(n_wp),
        "before_anchor_lat": np.float32(b_anc["latitude"]),
        "before_anchor_lon": np.float32(b_anc["longitude"]),
        "after_anchor_lat":  np.float32(a_anc["latitude"]),
        "after_anchor_lon":  np.float32(a_anc["longitude"]),
    }

def _save_sequence_split(samples: list[dict], path: Path) -> None:
    """Stack samples into arrays and save as compressed NPZ."""
    if not samples:
        return
    np.savez_compressed(
        path,
        segment_ids   = np.array([s["segment_id"]   for s in samples], dtype=object),
        icao24        = np.array([s["icao24"]        for s in samples], dtype=object),
        before_seq    = np.stack([s["before_seq"]    for s in samples]),  # (N, T, D)
        before_mask   = np.stack([s["before_mask"]   for s in samples]),  # (N, T)
        after_seq     = np.stack([s["after_seq"]     for s in samples]),  # (N, T, D)
        after_mask    = np.stack([s["after_mask"]    for s in samples]),  # (N, T)
        adsc_targets  = np.stack([s["adsc_targets"]  for s in samples]),  # (N, K, 2)
        adsc_tau      = np.stack([s["adsc_tau"]      for s in samples]),  # (N, K)
        adsc_mask     = np.stack([s["adsc_mask"]     for s in samples]),  # (N, K)
        gap_dur_sec   = np.array([s["gap_dur_sec"]   for s in samples], dtype=np.float32),
        n_adsc_wp     = np.array([s["n_adsc_wp"]     for s in samples], dtype=np.int32),
        before_anchor_lat = np.array([s["before_anchor_lat"] for s in samples], dtype=np.float32),
        before_anchor_lon = np.array([s["before_anchor_lon"] for s in samples], dtype=np.float32),
        after_anchor_lat  = np.array([s["after_anchor_lat"]  for s in samples], dtype=np.float32),
        after_anchor_lon  = np.array([s["after_anchor_lon"]  for s in samples], dtype=np.float32),
    )

def run_step4_build_ml_dataset(cfg: Step4Config) -> dict[str, Any]:
    def _log(message: str) -> None:
        if cfg.verbose:
            print(message, flush=True)

    step2_catalog = _load_step2_catalog(cfg.step2_root)

    catalog_used = str(_step2_catalog_path(cfg.step2_root).name)
    _log(f"Catalog loaded: {catalog_used}")

    kept = step2_catalog[step2_catalog["step2_keep"] == True].copy().reset_index(drop=True)

    if cfg.max_flights_to_process is not None:
        kept = kept.head(int(cfg.max_flights_to_process)).reset_index(drop=True)

    adsc_col = next((c for c in ["adsc_rows_clean","adsc_point_count_clean","adsc_point_count"]
                     if c in kept.columns), None)
    if adsc_col:
        kept = kept[pd.to_numeric(kept[adsc_col], errors="coerce").fillna(0)
                    >= cfg.min_adsc_points_for_ml].copy()

    if kept.empty:
        raise ValueError("No eligible Step 2 flights for Step 4.")

    kept = _assign_splits_by_icao24(
        kept,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        random_seed=cfg.random_seed,
    )

    counts = kept["split"].value_counts()
    _log(f"Split (by ICAO24): train={counts.get('train',0)}  "
         f"val={counts.get('val',0)}  test={counts.get('test',0)}")

    if cfg.clean_existing_output and cfg.output_root.exists():
        shutil.rmtree(cfg.output_root)
    catalog_root = cfg.output_root / "catalog"
    dataset_root = cfg.output_root / "dataset"
    catalog_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    if cfg.write_per_flight_outputs:
        (cfg.output_root / "flights").mkdir(parents=True, exist_ok=True)

    point_frames:   list[pd.DataFrame] = []
    flight_metas:   list[dict]         = []
    issue_rows:     list[dict]         = []
    seq_samples:    dict[str, list]    = {"train": [], "val": [], "test": []}

    total = len(kept)
    _log(f"Processing {total} flights...")

    for idx, row in kept.iterrows():
        if cfg.verbose and ((idx == 0) or ((idx+1) % cfg.progress_every == 0) or (idx+1 == total)):
            print(f"[Step4] {idx+1}/{total}", flush=True)

        seg_id = str(row["segment_id"])
        try:
            before = _clean_frame(_load_section(_before_path(row, cfg.step2_root)))
            adsc   = _clean_frame(_load_section(_adsc_path(row,   cfg.step2_root)))
            after  = _clean_frame(_load_section(_after_path(row,  cfg.step2_root)))

            pts, meta = _build_point_rows(row, before, adsc, after)
            if not pts.empty:
                point_frames.append(pts)
                flight_metas.append(meta)

            if cfg.write_per_flight_outputs:
                seg_dir = cfg.output_root / "flights" / seg_id
                seg_dir.mkdir(parents=True, exist_ok=True)
                pts.to_parquet(seg_dir / "ml_points.parquet", index=False)
                (seg_dir / "ml_metadata.json").write_text(
                    json.dumps(meta, indent=2, default=str))

        except Exception as exc:
            issue_rows.append({
                "segment_id": seg_id,
                "source_run": row.get("source_run"),
                "issue_type": "pointwise_build_failure",
                "issue_detail": str(exc),
            })
            continue   # skip sequence step if pointwise failed

        try:
            seq = _build_sequence_sample(row, before, adsc, after)
            seq_samples[row["split"]].append(seq)
        except Exception as exc:
            issue_rows.append({
                "segment_id": seg_id,
                "source_run": row.get("source_run"),
                "issue_type": "sequence_build_failure",
                "issue_detail": str(exc),
            })

    if not point_frames:
        raise ValueError("Step 4 produced no point dataset rows.")

    _log("Finished per-flight processing. Concatenating point dataset...")
    point_dataset = pd.concat(point_frames, ignore_index=True)

    _log("Writing point_dataset.parquet...")
    point_dataset.to_parquet(dataset_root / "point_dataset.parquet", index=False)
    _log("Writing point_dataset.csv...")
    point_dataset.to_csv(dataset_root / "point_dataset.csv", index=False)

    for split in ["train", "val", "test"]:
        split_df = point_dataset[point_dataset["split"] == split]
        _log(f"Writing {split}_points.parquet...")
        split_df.to_parquet(dataset_root / f"{split}_points.parquet", index=False)

    feature_columns = [
        "tau","elapsed_sec_from_before_anchor","gap_duration_sec","gap_duration_minutes",
        "before_anchor_lat","before_anchor_lon","after_anchor_lat","after_anchor_lon",
        "before_anchor_velocity_mps","after_anchor_velocity_mps",
        "before_anchor_heading_deg","after_anchor_heading_deg",
        "before_anchor_geoaltitude_m","after_anchor_geoaltitude_m",
        "anchor_distance_m","anchor_bearing_deg",
        "baseline_lat","baseline_lon","adsc_point_count_clean",
    ]
    target_columns = ["target_lat","target_lon","target_residual_east_m","target_residual_north_m"]

    numeric = point_dataset[feature_columns + target_columns + ["split"]].copy()
    split_map = {"train": 0, "val": 1, "test": 2}
    _log("Writing ml_dataset_arrays.npz...")
    np.savez_compressed(
        dataset_root / "ml_dataset_arrays.npz",
        X=numeric[feature_columns].to_numpy(dtype=np.float32),
        y=numeric[target_columns].to_numpy(dtype=np.float32),
        split_ids=numeric["split"].map(split_map).to_numpy(dtype=np.int64),
        feature_columns=np.array(feature_columns, dtype=object),
        target_columns=np.array(target_columns, dtype=object),
    )

    for split in ["train", "val", "test"]:
        samples = seq_samples[split]
        if samples:
            _log(f"Writing sequences_{split}.npz...")
            _save_sequence_split(samples, dataset_root / f"sequences_{split}.npz")
            _log(f"Sequences saved: {split} → {len(samples)} flights")

    norm_stats = {
        "lat":  {"mean": LAT_MEAN,  "std": LAT_STD},
        "lon":  {"mean": LON_MEAN,  "std": LON_STD},
        "vel":  {"mean": VEL_MEAN,  "std": VEL_STD},
        "alt":  {"mean": ALT_MEAN,  "std": ALT_STD},
        "before_steps":   BEFORE_STEPS,
        "after_steps":    AFTER_STEPS,
        "max_adsc_wp":    MAX_ADSC_WP,
        "n_seq_features": N_SEQ_FEATURES,
        "resample_sec":   RESAMPLE_SECONDS,
    }
    _log("Writing normalization_stats.json...")
    (dataset_root / "normalization_stats.json").write_text(
        json.dumps(norm_stats, indent=2))

    _log("Writing Step 4 catalogs...")
    kept.to_parquet(catalog_root / "flight_splits.parquet", index=False)
    kept.to_csv(catalog_root / "flight_splits.csv", index=False)
    pd.DataFrame(flight_metas).to_parquet(catalog_root / "flight_ml_metadata.parquet", index=False)
    issues_df = pd.DataFrame(issue_rows) if issue_rows else \
        pd.DataFrame(columns=["segment_id","source_run","issue_type","issue_detail"])
    issues_df.to_parquet(catalog_root / "step4_issues.parquet", index=False)
    issues_df.to_csv(catalog_root / "step4_issues.csv", index=False)

    summary = {
        "step2_kept_flights_input":  int(len(step2_catalog[step2_catalog["step2_keep"]==True])),
        "step4_flights_eligible":    int(total),
        "step4_flights_failed":      int(len(issue_rows)),
        "point_rows_total":          int(len(point_dataset)),
        "split_method":              "by_icao24",
        "catalog_used":              catalog_used,
        "split_point_rows":   {s: int((point_dataset["split"]==s).sum()) for s in ["train","val","test"]},
        "split_flights":      {s: int((kept["split"]==s).sum())          for s in ["train","val","test"]},
        "sequence_flights":   {s: len(seq_samples[s])                    for s in ["train","val","test"]},
        "sequence_shape": {
            "before_seq":   f"(N, {BEFORE_STEPS}, {N_SEQ_FEATURES})",
            "after_seq":    f"(N, {AFTER_STEPS}, {N_SEQ_FEATURES})",
            "adsc_targets": f"(N, {MAX_ADSC_WP}, 2)",
            "adsc_mask":    f"(N, {MAX_ADSC_WP})",
        },
        "feature_columns":           feature_columns,
        "target_columns":            target_columns,
        "min_adsc_points_for_ml":    cfg.min_adsc_points_for_ml,
        "random_seed":               cfg.random_seed,
        "train_fraction":            cfg.train_fraction,
        "val_fraction":              cfg.val_fraction,
        "test_fraction":             cfg.test_fraction,
    }

    (catalog_root / "step4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log("Step 4 ML dataset build finished.")
    if cfg.verbose:
        print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Step 4 ML dataset from Step 2 cleaned flights.")
    p.add_argument("--step2-root",           type=str,   default="artifacts/step2_clean")
    p.add_argument("--output-root",          type=str,   default="artifacts/step4_ml_dataset")
    p.add_argument("--max-flights",          type=int,   default=None)
    p.add_argument("--min-adsc-points-ml",   type=int,   default=3)
    p.add_argument("--train-fraction",       type=float, default=0.70)
    p.add_argument("--val-fraction",         type=float, default=0.15)
    p.add_argument("--test-fraction",        type=float, default=0.15)
    p.add_argument("--random-seed",          type=int,   default=42)
    p.add_argument("--progress-every",       type=int,   default=25)
    p.add_argument("--keep-output",          action="store_true")
    p.add_argument("--write-per-flight-outputs", action="store_true")
    p.add_argument("--quiet",                action="store_true")
    return p

def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg  = Step4Config(
        step2_root=Path(args.step2_root),
        output_root=Path(args.output_root),
        min_adsc_points_for_ml=args.min_adsc_points_ml,
        max_flights_to_process=args.max_flights,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
        clean_existing_output=not args.keep_output,
        write_per_flight_outputs=args.write_per_flight_outputs,
        verbose=not args.quiet,
        progress_every=args.progress_every,
    )
    run_step4_build_ml_dataset(cfg)

if __name__ == "__main__":
    main()
