"""Track coverage and reconstruction drift over time."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

STEP2_ROOT   = Path("artifacts/step2_clean")
STEP4_ROOT   = Path("artifacts/step4_ml_dataset")
STEP5_ROOT   = Path("artifacts/step5_gru")
STEP3B_ROOT  = Path("artifacts/step3b_kalman")
STEP5K_ROOT  = Path("artifacts/step5_kalman")
OUTPUT_ROOT  = Path("artifacts/step8_monitoring")

MIN_IMPROVEMENT_PCT     = 5.0   # alert if GRU improvement drops below this
MAX_GAP_RATE_DRIFT_PCT  = 20.0   # alert if monthly gap rate deviates >20% from mean
EARTH_R = 6371.0

def _parse_month(seg_id: str) -> str | None:
    """Extract YYYY-MM from segment_id like 20241002_..."""
    try:
        return pd.Timestamp(str(seg_id)[:8], ).strftime("%Y-%m")
    except Exception:
        return None

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return EARTH_R * 2 * math.asin(min(1.0, max(0.0, a)**0.5))

def compute_dataset_stats(catalog: pd.DataFrame) -> pd.DataFrame:
    """Per-month statistics from the validated flight catalog."""
    catalog = catalog.copy()
    catalog["month"] = catalog["segment_id"].apply(_parse_month)
    catalog = catalog.dropna(subset=["month"])

    agg_dict = {
        "n_segments":        ("segment_id",          "count"),
        "n_unique_aircraft": ("icao24",               "nunique"),
    }

    if "gap_duration_minutes" in catalog.columns:
        agg_dict["mean_gap_min"]    = ("gap_duration_minutes", "mean")
        agg_dict["median_gap_min"]  = ("gap_duration_minutes", "median")

    adsc_col = next((c for c in ["adsc_point_count","adsc_point_count_clean",
                                  "n_adsc_waypoints","adsc_rows_clean"]
                     if c in catalog.columns), None)
    if adsc_col:
        agg_dict["mean_adsc_points"] = (adsc_col, "mean")

    stats = (
        catalog.groupby("month")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("month")
    )

    stats["gap_rate_proxy"] = stats["n_segments"] / stats["n_segments"].max()
    catalog["_is_old_code"] = catalog["segment_id"].apply(
        lambda s: str(s)[:6] in ["202307","202308"]
    )
    return stats

def compute_gru_monthly_errors(splits: pd.DataFrame) -> pd.DataFrame:
    """Load step5 test predictions and compute per-month error stats."""
    preds_path = STEP5_ROOT / "test_predictions.npz"
    if not preds_path.exists():
        return pd.DataFrame()

    preds = np.load(preds_path, allow_pickle=True)
    ge    = preds["gru_errors_m"]           # (N, K)
    be    = preds["baseline_errors_m"]      # (N, K)
    mask  = preds["mask"]                   # (N, K)

    test_splits = splits[splits["split"] == "test"].reset_index(drop=True)
    N = min(len(test_splits), len(ge))

    rows = []
    for i in range(N):
        seg_id = str(test_splits["segment_id"].iloc[i])
        m      = mask[i] > 0
        if m.sum() == 0:
            continue
        rows.append({
            "segment_id":    seg_id,
            "month":         _parse_month(seg_id),
            "gru_error_km":  float(ge[i][m].mean()) / 1000.0,
            "bl_error_km":   float(be[i][m].mean()) / 1000.0,
            "gap_dur_min":   float(test_splits.get("gap_duration_minutes", pd.Series([0]*N)).iloc[i]),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["month"]).sort_values("month")
    df["improvement_pct"] = (1 - df["gru_error_km"] / df["bl_error_km"].clip(lower=0.01)) * 100

    monthly = (
        df.groupby("month")
        .agg(
            n_flights           = ("segment_id",      "count"),
            gru_mean_km         = ("gru_error_km",    "mean"),
            gru_median_km       = ("gru_error_km",    "median"),
            bl_mean_km          = ("bl_error_km",     "mean"),
            bl_median_km        = ("bl_error_km",     "median"),
            improvement_mean    = ("improvement_pct", "mean"),
            improvement_median  = ("improvement_pct", "median"),
        )
        .reset_index()
    )
    return monthly

def compute_kf_monthly_errors(splits: pd.DataFrame) -> pd.DataFrame:
    """Load step5_kalman per-flight metrics and compute per-month stats."""
    kf_path = STEP5K_ROOT / "per_flight_metrics_test.parquet"
    if not kf_path.exists():
        kf_path = STEP3B_ROOT / "catalog" / "kf_flights.parquet"
    if not kf_path.exists():
        return pd.DataFrame()

    kf = pd.read_parquet(kf_path)
    if "segment_id" not in kf.columns:
        return pd.DataFrame()

    kf["month"] = kf["segment_id"].apply(_parse_month)
    kf = kf.dropna(subset=["month"])

    err_col = next((c for c in ["kalman_mean_error_km","kf_mean_error_km"]
                    if c in kf.columns), None)
    if err_col is None:
        return pd.DataFrame()

    monthly = (
        kf.groupby("month")
        .agg(kf_mean_km=(err_col,"mean"), kf_median_km=(err_col,"median"))
        .reset_index()
    )
    return monthly

def detect_alerts(dataset_stats: pd.DataFrame,
                  gru_monthly: pd.DataFrame) -> list[dict]:
    alerts = []

    if not dataset_stats.empty:
        mean_n = dataset_stats["n_segments"].mean()
        for _, row in dataset_stats.iterrows():
            if row["n_segments"] < mean_n * 0.3:
                alerts.append({
                    "type":    "LOW_COVERAGE",
                    "month":   row["month"],
                    "message": f"Only {int(row['n_segments'])} segments (mean={mean_n:.0f}). "
                               f"Possible ADS-C coverage gap.",
                    "severity": "WARNING",
                })

    if not gru_monthly.empty:
        for _, row in gru_monthly.iterrows():
            if row.get("n_flights", 0) >= 5 and row["improvement_mean"] < MIN_IMPROVEMENT_PCT:
                alerts.append({
                    "type":    "MODEL_DRIFT",
                    "month":   row["month"],
                    "message": f"GRU improvement only {row['improvement_mean']:.1f}% "
                               f"(threshold={MIN_IMPROVEMENT_PCT}%). Possible model drift.",
                    "severity": "WARNING",
                })

    if not dataset_stats.empty and len(dataset_stats) >= 4:
        first_half = dataset_stats.head(len(dataset_stats)//2)["mean_gap_min"].mean()
        last_half  = dataset_stats.tail(len(dataset_stats)//2)["mean_gap_min"].mean()
        if last_half > first_half * 1.3:
            alerts.append({
                "type":    "GAP_DURATION_TREND",
                "month":   "overall",
                "message": f"Average gap duration increased from {first_half:.0f}min "
                           f"to {last_half:.0f}min. Possible ADS-C reporting change.",
                "severity": "INFO",
            })

    return alerts

def run_step8_monitoring(
    step4_root:  Path = STEP4_ROOT,
    output_root: Path = OUTPUT_ROOT,
    verbose:     bool = True,
) -> dict:

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits_path = step4_root / "catalog" / "flight_splits.parquet"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing: {splits_path}")
    splits = pd.read_parquet(splits_path)

    dataset_stats  = compute_dataset_stats(splits)
    gru_monthly    = compute_gru_monthly_errors(splits)
    kf_monthly     = compute_kf_monthly_errors(splits)
    alerts         = detect_alerts(dataset_stats, gru_monthly)

    dataset_stats.to_parquet(output_root / "dataset_stats_monthly.parquet", index=False)
    dataset_stats.to_csv(output_root / "dataset_stats_monthly.csv", index=False)

    if not gru_monthly.empty:
        gru_monthly.to_parquet(output_root / "gru_errors_monthly.parquet", index=False)
        gru_monthly.to_csv(output_root / "gru_errors_monthly.csv", index=False)

    if not kf_monthly.empty:
        kf_monthly.to_parquet(output_root / "kf_errors_monthly.parquet", index=False)

    summary = {
        "months_covered":     int(len(dataset_stats)),
        "total_flights":      int(splits["segment_id"].count()),
        "date_range":         {
            "first": str(dataset_stats["month"].iloc[0])  if not dataset_stats.empty else "-",
            "last":  str(dataset_stats["month"].iloc[-1]) if not dataset_stats.empty else "-",
        },
        "overall_gru": {
            "mean_improvement_pct":   float(gru_monthly["improvement_mean"].mean())   if not gru_monthly.empty else None,
            "median_improvement_pct": float(gru_monthly["improvement_median"].median()) if not gru_monthly.empty else None,
        },
        "alerts": alerts,
        "n_alerts": len(alerts),
    }

    (output_root / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    if verbose:
        print("=" * 60)
        print("STEP 8 - MONITORING SUMMARY")
        print("=" * 60)
        print(f"  Months covered     : {summary['months_covered']}")
        print(f"  Total flights      : {summary['total_flights']:,}")
        print(f"  Date range         : {summary['date_range']['first']} to {summary['date_range']['last']}")
        if summary["overall_gru"]["mean_improvement_pct"]:
            print(f"  GRU improvement    : {summary['overall_gru']['mean_improvement_pct']:.1f}% mean")
        print(f"\n  Alerts ({len(alerts)}):")
        if alerts:
            for a in alerts:
                print(f"    [{a['severity']}] {a['month']}: {a['message']}")
        else:
            print("    No alerts - data and model quality look stable.")
        print("=" * 60)
        print(f"\nOutputs saved to: {output_root}")

    return summary

def main():
    run_step8_monitoring()

if __name__ == "__main__":
    main()
