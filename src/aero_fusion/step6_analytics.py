"""Compute downstream analytics for reconstructed trajectories."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from aero_fusion.emissions_calculator import compute_emissions_kg_co2

STEP2_ROOT   = Path("artifacts/step2_clean")
STEP4_ROOT   = Path("artifacts/step4_ml_dataset")
STEP5_ROOT   = Path("artifacts/step5_gru")
OUTPUT_ROOT  = Path("artifacts/step6_analytics")

INTERP_STEP_SEC = 60

EARTH_R = 6_371_000.0

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a)))) / 1000.0

def track_length_km(lats: np.ndarray, lons: np.ndarray) -> float:
    """Total geodesic length of a sequence of lat/lon points."""
    if len(lats) < 2:
        return 0.0
    phi1 = np.radians(lats[:-1]);  phi2 = np.radians(lats[1:])
    lam1 = np.radians(lons[:-1]);  lam2 = np.radians(lons[1:])
    dphi = phi2 - phi1;            dlam = lam2 - lam1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return float((EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))).sum()) / 1000.0

def gc_point(lat0, lon0, lat1, lon1, tau: float):
    """Single great-circle interpolation at fraction tau."""
    lat0r,lon0r = math.radians(lat0), math.radians(lon0)
    lat1r,lon1r = math.radians(lat1), math.radians(lon1)
    x0=math.cos(lat0r)*math.cos(lon0r); y0=math.cos(lat0r)*math.sin(lon0r); z0=math.sin(lat0r)
    x1=math.cos(lat1r)*math.cos(lon1r); y1=math.cos(lat1r)*math.sin(lon1r); z1=math.sin(lat1r)
    dot   = max(-1.0, min(1.0, x0*x1+y0*y1+z0*z1))
    omega = math.acos(dot)
    if abs(omega) < 1e-10:
        return lat0, lon0
    s = math.sin(omega)
    w0 = math.sin((1-tau)*omega)/s;  w1 = math.sin(tau*omega)/s
    xp,yp,zp = w0*x0+w1*x1, w0*y0+w1*y1, w0*z0+w1*z1
    n  = math.sqrt(xp**2+yp**2+zp**2)
    xp,yp,zp = xp/n, yp/n, zp/n
    return math.degrees(math.asin(max(-1,min(1,zp)))), math.degrees(math.atan2(yp,xp))

def dense_gc_track(lat0, lon0, lat1, lon1,
                   t_start: pd.Timestamp, t_end: pd.Timestamp,
                   step_sec: int = INTERP_STEP_SEC) -> pd.DataFrame:
    """Build a dense great-circle track between two anchor points, with one row per step_sec seconds."""
    dur = float((t_end - t_start).total_seconds())
    if dur <= 0:
        return pd.DataFrame(columns=["timestamp","latitude","longitude","source"])
    n_steps = max(2, int(dur / step_sec) + 1)
    taus    = np.linspace(0, 1, n_steps)
    rows    = []
    for tau in taus:
        lat, lon = gc_point(lat0, lon0, lat1, lon1, float(tau))
        ts = t_start + pd.Timedelta(seconds=float(tau * dur))
        rows.append({"timestamp": ts, "latitude": lat, "longitude": lon, "source": "gc_interp"})
    return pd.DataFrame(rows)

def cross_track_distance_km(
    point_lat, point_lon,
    path_lat0, path_lon0,
    path_lat1, path_lon1,
) -> float:
    """Cross-track distance of a point from the great-circle path between two reference points."""
    d13 = haversine_km(path_lat0, path_lon0, point_lat, point_lon) / (EARTH_R/1000)
    bear12 = math.radians(_bearing(path_lat0, path_lon0, path_lat1, path_lon1))
    bear13 = math.radians(_bearing(path_lat0, path_lon0, point_lat, point_lon))
    xt = math.asin(math.sin(d13) * math.sin(bear13 - bear12))
    return xt * (EARTH_R / 1000)

def _bearing(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y  = math.sin(dl) * math.cos(phi2)
    x  = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def interpolate_gap_from_waypoints(
    waypoints: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    step_sec: int = INTERP_STEP_SEC,
) -> pd.DataFrame:
    """Build a dense track inside the gap by interpolating between sparse waypoints along great-circle arcs."""
    if waypoints.empty:
        return dense_gc_track(float("nan"), float("nan"), float("nan"), float("nan"),
                              t_start, t_end, step_sec)

    wp = waypoints.sort_values("timestamp").reset_index(drop=True)

    first_wp_time = pd.Timestamp(wp["timestamp"].iloc[0])
    last_wp_time  = pd.Timestamp(wp["timestamp"].iloc[-1])

    rows = []
    for i in range(len(wp) - 1):
        r0, r1 = wp.iloc[i], wp.iloc[i+1]
        seg = dense_gc_track(
            float(r0["latitude"]), float(r0["longitude"]),
            float(r1["latitude"]), float(r1["longitude"]),
            pd.Timestamp(r0["timestamp"]),
            pd.Timestamp(r1["timestamp"]),
            step_sec,
        )
        if i > 0 and not seg.empty:
            seg = seg.iloc[1:]   # avoid duplicating waypoint
        rows.append(seg)

    if not rows:
        return pd.DataFrame(columns=["timestamp","latitude","longitude","source"])

    dense = pd.concat(rows, ignore_index=True)
    dense["source"] = "gru_interp"
    return dense

def build_full_track(
    before_df: pd.DataFrame,
    gap_fill: pd.DataFrame,
    after_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Stitch before + gap fill + after into one continuous track."""
    parts = []

    if not before_df.empty:
        b = before_df[["timestamp","latitude","longitude"]].copy()
        b["source"] = "adsb_before"
        parts.append(b)

    if not gap_fill.empty:
        g = gap_fill[["timestamp","latitude","longitude","source"]].copy()
        parts.append(g)

    if not after_df.empty:
        a = after_df[["timestamp","latitude","longitude"]].copy()
        a["source"] = "adsb_after"
        parts.append(a)

    if not parts:
        return pd.DataFrame()

    track = pd.concat(parts, ignore_index=True)
    track["timestamp"] = pd.to_datetime(track["timestamp"], errors="coerce")
    track = track.dropna(subset=["timestamp","latitude","longitude"])
    track = track.sort_values("timestamp").reset_index(drop=True)
    track["method"] = label
    return track

def compute_flight_analytics(
    segment_id: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    adsc_df: pd.DataFrame,
    gru_waypoints_lat: np.ndarray,
    gru_waypoints_lon: np.ndarray,
    baseline_waypoints_lat: np.ndarray,
    baseline_waypoints_lon: np.ndarray,
    adsc_tau: np.ndarray,
    adsc_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute route distance, emissions, and cross-track deviation for one test flight across three methods."""

    before = before_df.sort_values("timestamp").reset_index(drop=True)
    after  = after_df.sort_values("timestamp").reset_index(drop=True)

    if before.empty or after.empty:
        return {}

    b_anc = before.iloc[-1]
    a_anc = after.iloc[0]
    t0    = pd.Timestamp(b_anc["timestamp"])
    t1    = pd.Timestamp(a_anc["timestamp"])
    dur   = float((t1 - t0).total_seconds())
    if dur <= 0:
        return {}

    valid = adsc_mask > 0
    n_wp  = int(valid.sum())
    if n_wp == 0:
        return {}

    gru_wp_df = pd.DataFrame({
        "timestamp": [t0 + pd.Timedelta(seconds=float(tau*dur))
                      for tau in adsc_tau[valid]],
        "latitude":  gru_waypoints_lat[valid].tolist(),
        "longitude": gru_waypoints_lon[valid].tolist(),
    })
    anchor_start = pd.DataFrame([{"timestamp": t0, "latitude": float(b_anc["latitude"]), "longitude": float(b_anc["longitude"])}])
    anchor_end   = pd.DataFrame([{"timestamp": t1, "latitude": float(a_anc["latitude"]), "longitude": float(a_anc["longitude"])}])
    gru_wp_df = pd.concat([anchor_start, gru_wp_df, anchor_end], ignore_index=True)

    baseline_wp_df = pd.DataFrame({
        "timestamp": [t0 + pd.Timedelta(seconds=float(tau*dur))
                      for tau in adsc_tau[valid]],
        "latitude":  baseline_waypoints_lat[valid].tolist(),
        "longitude": baseline_waypoints_lon[valid].tolist(),
    })

    adsc_truth_df = pd.DataFrame({
        "timestamp": [t0 + pd.Timedelta(seconds=float(tau*dur))
                      for tau in adsc_tau[valid]],
        "latitude":  adsc_df["latitude"].values[:n_wp] if not adsc_df.empty else [],
        "longitude": adsc_df["longitude"].values[:n_wp] if not adsc_df.empty else [],
    }) if not adsc_df.empty else pd.DataFrame()

    gru_fill      = interpolate_gap_from_waypoints(gru_wp_df,      t0, t1)
    baseline_fill = dense_gc_track(
        float(b_anc["latitude"]), float(b_anc["longitude"]),
        float(a_anc["latitude"]), float(a_anc["longitude"]),
        t0, t1,
    )
    baseline_fill["source"] = "gc_interp"

    gru_track      = build_full_track(before, gru_fill,      after, "gru")
    baseline_track = build_full_track(before, baseline_fill, after, "baseline")

    def track_dist(df):
        if df.empty or len(df) < 2:
            return float("nan")
        return track_length_km(df["latitude"].values, df["longitude"].values)

    gru_dist_km      = track_dist(gru_track)
    baseline_dist_km = track_dist(baseline_track)

    adsc_gap_dist_km = float("nan")
    if not adsc_truth_df.empty and len(adsc_truth_df) >= 2:
        adsc_gap_dist_km = track_length_km(
            adsc_truth_df["latitude"].values,
            adsc_truth_df["longitude"].values,
        )

    anchor_dist_km = haversine_km(
        float(b_anc["latitude"]), float(b_anc["longitude"]),
        float(a_anc["latitude"]), float(a_anc["longitude"]),
    )

    gru_em          = compute_emissions_kg_co2(gru_dist_km)
    baseline_em     = compute_emissions_kg_co2(baseline_dist_km)
    gru_co2_kg      = gru_em["co2_kg"]
    baseline_co2_kg = baseline_em["co2_kg"]
    co2_diff_kg     = gru_co2_kg - baseline_co2_kg

    xt_devs = []
    for lat, lon in zip(gru_waypoints_lat[valid], gru_waypoints_lon[valid]):
        try:
            xt = cross_track_distance_km(
                float(lat), float(lon),
                float(b_anc["latitude"]), float(b_anc["longitude"]),
                float(a_anc["latitude"]), float(a_anc["longitude"]),
            )
            xt_devs.append(abs(xt))
        except Exception:
            pass

    mean_xt_km = float(np.mean(xt_devs)) if xt_devs else float("nan")
    max_xt_km  = float(np.max(xt_devs))  if xt_devs else float("nan")

    return {
        "segment_id":           segment_id,
        "gap_duration_minutes": dur / 60.0,
        "n_adsc_waypoints":     int(n_wp),
        "anchor_dist_km":       anchor_dist_km,

        "gru_total_dist_km":       gru_dist_km,
        "baseline_total_dist_km":  baseline_dist_km,
        "adsc_gap_dist_km":        adsc_gap_dist_km,
        "dist_diff_km":            gru_dist_km - baseline_dist_km,

        "gru_co2_kg":           gru_co2_kg,
        "baseline_co2_kg":      baseline_co2_kg,
        "co2_diff_kg":          co2_diff_kg,
        "fuel_burn_coeff":      gru_em["co2_per_km"],

        "gru_mean_xt_km":       mean_xt_km,
        "gru_max_xt_km":        max_xt_km,

        "_gru_track":      gru_track,
        "_baseline_track": baseline_track,
    }

def run_step6_analytics(
    step2_root:  Path = STEP2_ROOT,
    step4_root:  Path = STEP4_ROOT,
    step5_root:  Path = STEP5_ROOT,
    output_root: Path = OUTPUT_ROOT,
    write_per_flight: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:

    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "catalog").mkdir(parents=True, exist_ok=True)
    if write_per_flight:
        (output_root / "reconstructions").mkdir(parents=True, exist_ok=True)

    preds_path = step5_root / "test_predictions.npz"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing: {preds_path}\nRun step 5 training first.")

    preds = np.load(preds_path, allow_pickle=True)
    pred_lat      = preds["pred_lat"]       # (N, K)
    pred_lon      = preds["pred_lon"]
    baseline_lat  = preds["baseline_lat"]
    baseline_lon  = preds["baseline_lon"]
    true_lat      = preds["true_lat"]
    true_lon      = preds["true_lon"]
    mask          = preds["mask"]
    N             = pred_lat.shape[0]

    splits_path = step4_root / "catalog" / "flight_splits.parquet"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing: {splits_path}")
    splits    = pd.read_parquet(splits_path)
    test_rows = splits[splits["split"] == "test"].reset_index(drop=True)

    if len(test_rows) != N:
        print(f"[WARN] test_rows={len(test_rows)} vs predictions={N} - using min({N})")
        test_rows = test_rows.head(N)

    if verbose:
        print(f"Computing analytics for {N} test flights...")

    analytics_rows = []
    issues = []

    for i, row in test_rows.iterrows():
        seg_id = str(row["segment_id"])
        idx    = test_rows.index.get_loc(i)

        try:
            flight_dir = step2_root / "flights" / seg_id
            before_df  = pd.read_parquet(flight_dir / "adsb_before_clean.parquet")
            after_df   = pd.read_parquet(flight_dir / "adsb_after_clean.parquet")
            adsc_df    = pd.read_parquet(flight_dir / "adsc_clean.parquet")

            for df in [before_df, after_df, adsc_df]:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            before_sorted = before_df.sort_values("timestamp")
            after_sorted  = after_df.sort_values("timestamp")
            t0 = pd.Timestamp(before_sorted["timestamp"].iloc[-1])
            t1 = pd.Timestamp(after_sorted["timestamp"].iloc[0])
            dur = float((t1 - t0).total_seconds())

            adsc_tau_i  = preds["adsc_tau"][idx]   if "adsc_tau"  in preds else mask[idx]
            adsc_mask_i = mask[idx]

            if not adsc_df.empty and dur > 0:
                adsc_sorted = adsc_df.sort_values("timestamp").reset_index(drop=True)
                n_adsc = min(len(adsc_sorted), len(adsc_mask_i))
                adsc_tau_computed = np.zeros(len(adsc_mask_i), dtype=np.float32)
                for j in range(n_adsc):
                    elapsed = float((pd.Timestamp(adsc_sorted["timestamp"].iloc[j]) - t0).total_seconds())
                    adsc_tau_computed[j] = float(np.clip(elapsed / dur, 0, 1))
                adsc_tau_i = adsc_tau_computed

            result = compute_flight_analytics(
                segment_id           = seg_id,
                before_df            = before_df,
                after_df             = after_df,
                adsc_df              = adsc_df,
                gru_waypoints_lat    = pred_lat[idx],
                gru_waypoints_lon    = pred_lon[idx],
                baseline_waypoints_lat = baseline_lat[idx],
                baseline_waypoints_lon = baseline_lon[idx],
                adsc_tau             = adsc_tau_i,
                adsc_mask            = adsc_mask_i,
            )

            if not result:
                raise ValueError("Empty analytics result")

            if write_per_flight:
                rec_dir = output_root / "reconstructions" / seg_id
                rec_dir.mkdir(parents=True, exist_ok=True)
                result["_gru_track"].to_parquet(
                    rec_dir / "full_track_gru.parquet", index=False)
                result["_baseline_track"].to_parquet(
                    rec_dir / "full_track_baseline.parquet", index=False)

            row_out = {k: v for k, v in result.items() if not k.startswith("_")}
            analytics_rows.append(row_out)

        except Exception as exc:
            issues.append({"segment_id": seg_id, "error": str(exc)})
            if verbose:
                print(f"  [SKIP] {seg_id}: {exc}")

        if verbose and (idx+1) % 50 == 0:
            print(f"  Processed {idx+1}/{N}")

    if not analytics_rows:
        raise ValueError("No analytics rows produced.")

    analytics_df = pd.DataFrame(analytics_rows)
    analytics_df.to_parquet(output_root / "catalog" / "flight_analytics.parquet", index=False)
    analytics_df.to_csv(output_root / "catalog" / "flight_analytics.csv", index=False)

    def safe_mean(col):
        s = analytics_df[col].dropna()
        return float(s.mean()) if len(s) > 0 else float("nan")

    summary = {
        "flights_analyzed":              len(analytics_df),
        "flights_skipped":               len(issues),

        "route_distance": {
            "gru_mean_km":               safe_mean("gru_total_dist_km"),
            "baseline_mean_km":          safe_mean("baseline_total_dist_km"),
            "mean_diff_km":              safe_mean("dist_diff_km"),
            "gru_longer_pct":            float(
                (analytics_df["dist_diff_km"] > 0).sum() / len(analytics_df) * 100
            ),
        },
        "emissions_proxy": {
            "fuel_burn_coeff_kg_co2_per_km": safe_mean("fuel_burn_coeff"),
            "gru_mean_co2_kg":           safe_mean("gru_co2_kg"),
            "baseline_mean_co2_kg":      safe_mean("baseline_co2_kg"),
            "mean_co2_diff_kg":          safe_mean("co2_diff_kg"),
        },
        "cross_track_deviation": {
            "gru_mean_xt_km":            safe_mean("gru_mean_xt_km"),
            "gru_max_xt_km":             safe_mean("gru_max_xt_km"),
        },
    }

    with open(output_root / "catalog" / "analytics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nStep 6 analytics complete - {len(analytics_df)} flights")
        print(json.dumps(summary, indent=2))

    return summary

def main():
    run_step6_analytics()

if __name__ == "__main__":
    main()
