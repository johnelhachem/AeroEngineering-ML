from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_RESAMPLE_SEC = 60
MAX_ADSC_WP = 32
BEFORE_CONTEXT_STEPS = 64
AFTER_CONTEXT_STEPS = 32

@dataclass(frozen=True)
class KalmanConfig:
    step2_root: Path
    step4_dataset_root: Path
    output_root: Path
    context_mode: str = "resampled_local"
    resample_seconds: int = DEFAULT_RESAMPLE_SEC
    before_context_limit: int | None = None
    after_context_limit: int | None = None
    tuning_grid: str = "compact"
    max_flights_per_split: int | None = None
    tune_on_split: str = "val"
    verbose: bool = True
    clean_existing_output: bool = True

@dataclass(frozen=True)
class KalmanParams:
    measurement_std_m: float
    accel_std_along_mps2: float
    accel_std_cross_mps2: float

@dataclass
class PreparedFlight:
    split: str
    segment_id: str
    t0: pd.Timestamp
    t1: pd.Timestamp
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    measurement_times: list[pd.Timestamp]
    measurement_along_cross_m: np.ndarray
    target_times: list[pd.Timestamp]
    valid_indices: np.ndarray
    adsc_tau_full: np.ndarray
    adsc_mask_full: np.ndarray
    true_lat_full: np.ndarray
    true_lon_full: np.ndarray
    baseline_lat_full: np.ndarray
    baseline_lon_full: np.ndarray

def default_kalman_config(repo_root: Path | None = None) -> KalmanConfig:
    project_root = repo_root or Path(__file__).resolve().parents[2]
    return KalmanConfig(
        step2_root=project_root / "artifacts" / "step2_clean",
        step4_dataset_root=project_root / "artifacts" / "step4_ml_dataset" / "dataset",
        output_root=project_root / "artifacts" / "step5_kalman",
    )

def _log(message: str, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)

def _wrap_lon_deg(lon_deg: float) -> float:
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0

def _coerce_track(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ["latitude", "longitude"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return (
        work.dropna(subset=["timestamp", "latitude", "longitude"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

def _resample_track(df: pd.DataFrame, resample_seconds: int) -> pd.DataFrame:
    work = _coerce_track(df)
    if len(work) < 2:
        return work
    grid = pd.date_range(
        start=pd.Timestamp(work["timestamp"].iloc[0]),
        end=pd.Timestamp(work["timestamp"].iloc[-1]),
        freq=f"{int(resample_seconds)}s",
    )
    if len(grid) == 0:
        return work
    indexed = work.set_index("timestamp")[["latitude", "longitude"]]
    expanded = indexed.reindex(indexed.index.union(grid)).sort_index()
    expanded[["latitude", "longitude"]] = expanded[["latitude", "longitude"]].interpolate(method="time")
    resampled = expanded.loc[grid].reset_index().rename(columns={"index": "timestamp"})
    return _coerce_track(resampled)

def _merge_context_measurements(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    context_mode: str,
    resample_seconds: int,
    before_context_limit: int | None,
    after_context_limit: int | None,
) -> pd.DataFrame:
    before = _coerce_track(before_df)
    after = _coerce_track(after_df)

    if context_mode == "resampled_local":
        before_ctx = _resample_track(before, resample_seconds).tail(BEFORE_CONTEXT_STEPS)
        after_ctx = _resample_track(after, resample_seconds).head(AFTER_CONTEXT_STEPS)
    elif context_mode == "native_clean":
        before_ctx = before.copy()
        after_ctx = after.copy()
    else:
        raise ValueError(f"Unsupported context_mode: {context_mode}")

    if before_context_limit is not None and before_context_limit > 0:
        before_ctx = before_ctx.tail(int(before_context_limit))
    if after_context_limit is not None and after_context_limit > 0:
        after_ctx = after_ctx.head(int(after_context_limit))

    keep_parts = []
    if not before.empty:
        keep_parts.append(before.tail(1))
    if not after.empty:
        keep_parts.append(after.head(1))

    parts = [frame for frame in [before_ctx, after_ctx, *keep_parts] if not frame.empty]
    if not parts:
        return pd.DataFrame(columns=["timestamp", "latitude", "longitude"])
    return _coerce_track(pd.concat(parts, ignore_index=True)[["timestamp", "latitude", "longitude"]])

def _angular_distance_rad(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    phi1 = math.radians(float(lat1_deg))
    phi2 = math.radians(float(lat2_deg))
    dphi = phi2 - phi1
    dlam = math.radians(float(lon2_deg) - float(lon1_deg))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, a))))

def haversine_m(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    return EARTH_RADIUS_M * _angular_distance_rad(lat1_deg, lon1_deg, lat2_deg, lon2_deg)

def bearing_deg(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    phi1 = math.radians(float(lat1_deg))
    phi2 = math.radians(float(lat2_deg))
    dlam = math.radians(float(lon2_deg) - float(lon1_deg))
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def _latlon_to_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat_rad = math.radians(float(lat_deg))
    lon_rad = math.radians(float(lon_deg))
    return np.array(
        [
            math.cos(lat_rad) * math.cos(lon_rad),
            math.cos(lat_rad) * math.sin(lon_rad),
            math.sin(lat_rad),
        ],
        dtype=float,
    )

def _xyz_to_latlon(xyz: np.ndarray) -> tuple[float, float]:
    norm = float(np.linalg.norm(xyz))
    if norm <= 0.0:
        raise ValueError("Cannot convert zero vector to lat/lon.")
    unit = xyz / norm
    lat = math.degrees(math.asin(float(np.clip(unit[2], -1.0, 1.0))))
    lon = math.degrees(math.atan2(float(unit[1]), float(unit[0])))
    return lat, _wrap_lon_deg(lon)

def great_circle_interpolate(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    tau: float,
) -> tuple[float, float]:
    tau = float(np.clip(tau, 0.0, 1.0))
    p0 = _latlon_to_xyz(start_lat, start_lon)
    p1 = _latlon_to_xyz(end_lat, end_lon)
    omega = math.acos(float(np.clip(np.dot(p0, p1), -1.0, 1.0)))
    if abs(omega) < 1e-12:
        return float(start_lat), _wrap_lon_deg(float(start_lon))
    sin_omega = math.sin(omega)
    xyz = (math.sin((1.0 - tau) * omega) / sin_omega) * p0 + (math.sin(tau * omega) / sin_omega) * p1
    return _xyz_to_latlon(xyz)

def destination_point(lat_deg: float, lon_deg: float, bearing_deg_value: float, distance_m: float) -> tuple[float, float]:
    angular_distance = float(distance_m) / EARTH_RADIUS_M
    theta = math.radians(float(bearing_deg_value))
    phi1 = math.radians(float(lat_deg))
    lam1 = math.radians(float(lon_deg))
    sin_phi2 = math.sin(phi1) * math.cos(angular_distance) + math.cos(phi1) * math.sin(angular_distance) * math.cos(theta)
    phi2 = math.asin(float(np.clip(sin_phi2, -1.0, 1.0)))
    y = math.sin(theta) * math.sin(angular_distance) * math.cos(phi1)
    x = math.cos(angular_distance) - math.sin(phi1) * math.sin(phi2)
    lam2 = lam1 + math.atan2(y, x)
    return math.degrees(phi2), _wrap_lon_deg(math.degrees(lam2))

def along_cross_track_m(
    point_lat: float,
    point_lon: float,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> tuple[float, float]:
    delta13 = _angular_distance_rad(start_lat, start_lon, point_lat, point_lon)
    theta13 = math.radians(bearing_deg(start_lat, start_lon, point_lat, point_lon))
    theta12 = math.radians(bearing_deg(start_lat, start_lon, end_lat, end_lon))
    cross = math.asin(np.clip(math.sin(delta13) * math.sin(theta13 - theta12), -1.0, 1.0)) * EARTH_RADIUS_M
    along = math.atan2(math.sin(delta13) * math.cos(theta13 - theta12), math.cos(delta13)) * EARTH_RADIUS_M
    return float(along), float(cross)

def _course_along_path_deg(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    tau: float,
) -> float:
    total_dist = max(haversine_m(start_lat, start_lon, end_lat, end_lon), 1.0)
    eps = min(0.01, 10_000.0 / total_dist)
    tau0 = float(np.clip(tau - eps, 0.0, 1.0))
    tau1 = float(np.clip(tau + eps, 0.0, 1.0))
    if tau1 <= tau0:
        tau0 = max(0.0, tau - 1e-6)
        tau1 = min(1.0, tau + 1e-6)
    lat0, lon0 = great_circle_interpolate(start_lat, start_lon, end_lat, end_lon, tau0)
    lat1, lon1 = great_circle_interpolate(start_lat, start_lon, end_lat, end_lon, tau1)
    return bearing_deg(lat0, lon0, lat1, lon1)

def path_state_to_latlon(
    along_track_m_value: float,
    cross_track_m_value: float,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> tuple[float, float]:
    total_dist = max(haversine_m(start_lat, start_lon, end_lat, end_lon), 1.0)
    tau = float(np.clip(along_track_m_value / total_dist, 0.0, 1.0))
    base_lat, base_lon = great_circle_interpolate(start_lat, start_lon, end_lat, end_lon, tau)
    course = _course_along_path_deg(start_lat, start_lon, end_lat, end_lon, tau)
    offset_bearing = course + 90.0 if cross_track_m_value >= 0.0 else course - 90.0
    return destination_point(base_lat, base_lon, offset_bearing, abs(float(cross_track_m_value)))

def _baseline_waypoints(
    before_anchor_lat: float,
    before_anchor_lon: float,
    after_anchor_lat: float,
    after_anchor_lon: float,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_lat = np.zeros_like(tau, dtype=np.float32)
    pred_lon = np.zeros_like(tau, dtype=np.float32)
    for idx, frac in enumerate(tau):
        lat, lon = great_circle_interpolate(
            before_anchor_lat,
            before_anchor_lon,
            after_anchor_lat,
            after_anchor_lon,
            float(frac),
        )
        pred_lat[idx] = np.float32(lat)
        pred_lon[idx] = np.float32(lon)
    return pred_lat, pred_lon

def _state_transition(dt_sec: float) -> np.ndarray:
    dt = float(max(dt_sec, 0.0))
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

def _process_noise(dt_sec: float, accel_std_along: float, accel_std_cross: float) -> np.ndarray:
    dt = float(max(dt_sec, 0.0))
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    q_along = float(accel_std_along) ** 2
    q_cross = float(accel_std_cross) ** 2
    return np.array(
        [
            [q_along * dt4 / 4.0, 0.0, q_along * dt3 / 2.0, 0.0],
            [0.0, q_cross * dt4 / 4.0, 0.0, q_cross * dt3 / 2.0],
            [q_along * dt3 / 2.0, 0.0, q_along * dt2, 0.0],
            [0.0, q_cross * dt3 / 2.0, 0.0, q_cross * dt2],
        ],
        dtype=float,
    )

def _make_sorted_timeline(measurement_times: list[pd.Timestamp], target_times: list[pd.Timestamp]) -> list[pd.Timestamp]:
    all_values = {pd.Timestamp(ts).value: pd.Timestamp(ts) for ts in [*measurement_times, *target_times]}
    return [all_values[key] for key in sorted(all_values)]

def kalman_smooth_gap(flight: PreparedFlight, params: KalmanParams) -> tuple[np.ndarray, np.ndarray]:
    timeline = _make_sorted_timeline(flight.measurement_times, flight.target_times)
    if not timeline:
        raise ValueError(f"{flight.segment_id}: empty timeline")

    measurements_by_time = {
        pd.Timestamp(ts).value: np.asarray(z, dtype=float)
        for ts, z in zip(flight.measurement_times, flight.measurement_along_cross_m, strict=True)
    }
    target_index = {pd.Timestamp(ts).value: idx for idx, ts in enumerate(flight.target_times)}

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    R = np.diag([params.measurement_std_m**2, params.measurement_std_m**2]).astype(float)
    I = np.eye(4, dtype=float)

    meas = np.asarray(flight.measurement_along_cross_m, dtype=float)
    meas_times = [pd.Timestamp(ts) for ts in flight.measurement_times]
    obs_order = np.argsort([ts.value for ts in meas_times])
    meas = meas[obs_order]
    meas_times = [meas_times[i] for i in obs_order]

    if len(meas) >= 2:
        dt0 = max(float((meas_times[1] - meas_times[0]).total_seconds()), 1.0)
        v0 = (meas[1] - meas[0]) / dt0
    else:
        v0 = np.zeros(2, dtype=float)

    x0 = np.array([meas[0, 0], meas[0, 1], v0[0], v0[1]], dtype=float)
    pos_var = params.measurement_std_m**2
    P0 = np.diag([pos_var * 4.0, pos_var * 4.0, 250.0**2, 50.0**2]).astype(float)

    n_steps = len(timeline)
    x_pred = np.zeros((n_steps, 4), dtype=float)
    P_pred = np.zeros((n_steps, 4, 4), dtype=float)
    x_filt = np.zeros((n_steps, 4), dtype=float)
    P_filt = np.zeros((n_steps, 4, 4), dtype=float)

    x_pred[0] = x0
    P_pred[0] = P0
    z0 = measurements_by_time.get(pd.Timestamp(timeline[0]).value)
    if z0 is not None:
        innovation = z0 - (H @ x_pred[0])
        S = H @ P_pred[0] @ H.T + R
        K = P_pred[0] @ H.T @ np.linalg.inv(S)
        x_filt[0] = x_pred[0] + K @ innovation
        P_filt[0] = (I - K @ H) @ P_pred[0]
    else:
        x_filt[0] = x_pred[0]
        P_filt[0] = P_pred[0]

    for idx in range(1, n_steps):
        dt_sec = float((pd.Timestamp(timeline[idx]) - pd.Timestamp(timeline[idx - 1])).total_seconds())
        F = _state_transition(dt_sec)
        Q = _process_noise(dt_sec, params.accel_std_along_mps2, params.accel_std_cross_mps2)
        x_pred[idx] = F @ x_filt[idx - 1]
        P_pred[idx] = F @ P_filt[idx - 1] @ F.T + Q
        z = measurements_by_time.get(pd.Timestamp(timeline[idx]).value)
        if z is not None:
            innovation = z - (H @ x_pred[idx])
            S = H @ P_pred[idx] @ H.T + R
            K = P_pred[idx] @ H.T @ np.linalg.inv(S)
            x_filt[idx] = x_pred[idx] + K @ innovation
            P_filt[idx] = (I - K @ H) @ P_pred[idx]
        else:
            x_filt[idx] = x_pred[idx]
            P_filt[idx] = P_pred[idx]

    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for idx in range(n_steps - 2, -1, -1):
        dt_sec = float((pd.Timestamp(timeline[idx + 1]) - pd.Timestamp(timeline[idx])).total_seconds())
        F = _state_transition(dt_sec)
        gain = P_filt[idx] @ F.T @ np.linalg.inv(P_pred[idx + 1])
        x_smooth[idx] = x_filt[idx] + gain @ (x_smooth[idx + 1] - x_pred[idx + 1])
        P_smooth[idx] = P_filt[idx] + gain @ (P_smooth[idx + 1] - P_pred[idx + 1]) @ gain.T

    pred_lat = np.zeros(len(flight.target_times), dtype=np.float32)
    pred_lon = np.zeros(len(flight.target_times), dtype=np.float32)
    for ts in flight.target_times:
        smooth_idx = timeline.index(pd.Timestamp(ts))
        lat, lon = path_state_to_latlon(
            along_track_m_value=float(x_smooth[smooth_idx, 0]),
            cross_track_m_value=float(x_smooth[smooth_idx, 1]),
            start_lat=flight.start_lat,
            start_lon=flight.start_lon,
            end_lat=flight.end_lat,
            end_lon=flight.end_lon,
        )
        out_idx = target_index[pd.Timestamp(ts).value]
        pred_lat[out_idx] = np.float32(lat)
        pred_lon[out_idx] = np.float32(lon)

    return pred_lat, pred_lon

def _load_split_dataset(dataset_root: Path, split: str) -> dict[str, np.ndarray]:
    path = dataset_root / f"sequences_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 4 split dataset: {path}")
    return dict(np.load(path, allow_pickle=True))

def _flight_paths(step2_root: Path, segment_id: str) -> tuple[Path, Path]:
    flight_dir = step2_root / "flights" / segment_id
    return flight_dir / "adsb_before_clean.parquet", flight_dir / "adsb_after_clean.parquet"

def prepare_split_flights(config: KalmanConfig, split: str) -> list[PreparedFlight]:
    dataset = _load_split_dataset(config.step4_dataset_root, split)
    n_samples = len(dataset["segment_ids"])
    if config.max_flights_per_split is not None:
        n_samples = min(n_samples, int(config.max_flights_per_split))

    flights: list[PreparedFlight] = []
    _log(f"Preparing {split} flights: {n_samples}", config.verbose)

    for idx in range(n_samples):
        segment_id = str(dataset["segment_ids"][idx])
        before_path, after_path = _flight_paths(config.step2_root, segment_id)
        before_df = _coerce_track(pd.read_parquet(before_path))
        after_df = _coerce_track(pd.read_parquet(after_path))
        if before_df.empty or after_df.empty:
            continue

        b_anchor = before_df.iloc[-1]
        a_anchor = after_df.iloc[0]
        t0 = pd.Timestamp(b_anchor["timestamp"])
        t1 = pd.Timestamp(a_anchor["timestamp"])
        if t1 <= t0:
            continue

        merged = _merge_context_measurements(
            before_df,
            after_df,
            context_mode=config.context_mode,
            resample_seconds=config.resample_seconds,
            before_context_limit=config.before_context_limit,
            after_context_limit=config.after_context_limit,
        )
        if merged.empty or len(merged) < 2:
            continue

        start_lat = float(dataset["before_anchor_lat"][idx])
        start_lon = float(dataset["before_anchor_lon"][idx])
        end_lat = float(dataset["after_anchor_lat"][idx])
        end_lon = float(dataset["after_anchor_lon"][idx])

        meas_pos = merged.apply(
            lambda row: along_cross_track_m(
                point_lat=float(row["latitude"]),
                point_lon=float(row["longitude"]),
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
            ),
            axis=1,
            result_type="expand",
        ).to_numpy(dtype=float)

        tau_full = dataset["adsc_tau"][idx].astype(np.float32)
        mask_full = dataset["adsc_mask"][idx].astype(np.float32)
        valid_indices = np.where(mask_full > 0)[0]
        target_times = [
            t0 + pd.Timedelta(seconds=float(tau_full[j] * dataset["gap_dur_sec"][idx]))
            for j in valid_indices
        ]

        baseline_lat_full, baseline_lon_full = _baseline_waypoints(
            before_anchor_lat=start_lat,
            before_anchor_lon=start_lon,
            after_anchor_lat=end_lat,
            after_anchor_lon=end_lon,
            tau=tau_full,
        )

        flights.append(
            PreparedFlight(
                split=split,
                segment_id=segment_id,
                t0=t0,
                t1=t1,
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
                measurement_times=[pd.Timestamp(ts) for ts in merged["timestamp"].tolist()],
                measurement_along_cross_m=meas_pos,
                target_times=target_times,
                valid_indices=valid_indices.astype(int),
                adsc_tau_full=tau_full,
                adsc_mask_full=mask_full,
                true_lat_full=dataset["adsc_targets"][idx, :, 0].astype(np.float32),
                true_lon_full=dataset["adsc_targets"][idx, :, 1].astype(np.float32),
                baseline_lat_full=baseline_lat_full,
                baseline_lon_full=baseline_lon_full,
            )
        )

        if config.verbose and (idx + 1) % 50 == 0:
            _log(f"  prepared {idx + 1}/{n_samples} raw {split} flights", config.verbose)

    _log(f"Prepared usable {split} flights: {len(flights)}", config.verbose)
    return flights

def evaluate_split(
    flights: list[PreparedFlight],
    params: KalmanParams,
    produce_predictions: bool = False,
    verbose: bool = False,
    label: str = "split",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    per_flight_rows: list[dict[str, Any]] = []
    kalman_all: list[np.ndarray] = []
    baseline_all: list[np.ndarray] = []

    pred_lat_all = []
    pred_lon_all = []
    true_lat_all = []
    true_lon_all = []
    baseline_lat_all = []
    baseline_lon_all = []
    mask_all = []
    tau_all = []
    seg_all = []

    for idx, flight in enumerate(flights, start=1):
        pred_lat_valid, pred_lon_valid = kalman_smooth_gap(flight, params)
        pred_lat_full = np.zeros(MAX_ADSC_WP, dtype=np.float32)
        pred_lon_full = np.zeros(MAX_ADSC_WP, dtype=np.float32)
        pred_lat_full[flight.valid_indices] = pred_lat_valid
        pred_lon_full[flight.valid_indices] = pred_lon_valid

        valid = flight.adsc_mask_full > 0
        kalman_err = np.full(MAX_ADSC_WP, np.nan, dtype=np.float32)
        baseline_err = np.full(MAX_ADSC_WP, np.nan, dtype=np.float32)
        for j in np.where(valid)[0]:
            kalman_err[j] = np.float32(
                haversine_m(
                    pred_lat_full[j],
                    pred_lon_full[j],
                    flight.true_lat_full[j],
                    flight.true_lon_full[j],
                )
            )
            baseline_err[j] = np.float32(
                haversine_m(
                    flight.baseline_lat_full[j],
                    flight.baseline_lon_full[j],
                    flight.true_lat_full[j],
                    flight.true_lon_full[j],
                )
            )

        k = kalman_err[valid]
        b = baseline_err[valid]
        kalman_all.append(k.astype(np.float64))
        baseline_all.append(b.astype(np.float64))

        per_flight_rows.append(
            {
                "split": flight.split,
                "segment_id": flight.segment_id,
                "n_waypoints": int(valid.sum()),
                "kalman_mean_error_km": float(np.mean(k) / 1000.0),
                "kalman_median_error_km": float(np.median(k) / 1000.0),
                "baseline_mean_error_km": float(np.mean(b) / 1000.0),
                "baseline_median_error_km": float(np.median(b) / 1000.0),
                "improvement_mean_pct": float((1.0 - np.mean(k) / max(np.mean(b), 1e-9)) * 100.0),
                "improvement_median_pct": float((1.0 - np.median(k) / max(np.median(b), 1e-9)) * 100.0),
            }
        )

        if produce_predictions:
            pred_lat_all.append(pred_lat_full)
            pred_lon_all.append(pred_lon_full)
            true_lat_all.append(flight.true_lat_full)
            true_lon_all.append(flight.true_lon_full)
            baseline_lat_all.append(flight.baseline_lat_full)
            baseline_lon_all.append(flight.baseline_lon_full)
            mask_all.append(flight.adsc_mask_full)
            tau_all.append(flight.adsc_tau_full)
            seg_all.append(flight.segment_id)

        if verbose and (idx % 25 == 0 or idx == len(flights)):
            _log(f"  {label}: evaluated {idx}/{len(flights)} flights", verbose)

    kalman_flat = np.concatenate(kalman_all) if kalman_all else np.array([], dtype=np.float64)
    baseline_flat = np.concatenate(baseline_all) if baseline_all else np.array([], dtype=np.float64)
    kalman_per_flight_means = (
        np.array([arr.mean() for arr in kalman_all], dtype=np.float64)
        if kalman_all
        else np.array([], dtype=np.float64)
    )
    baseline_per_flight_means = (
        np.array([arr.mean() for arr in baseline_all], dtype=np.float64)
        if baseline_all
        else np.array([], dtype=np.float64)
    )

    summary = {
        "split": flights[0].split if flights else "unknown",
        "n_flights": int(len(flights)),
        "kalman_mean_error_km": float(np.mean(kalman_per_flight_means) / 1000.0),
        "kalman_median_error_km": float(np.median(kalman_per_flight_means) / 1000.0),
        "kalman_p90_error_km": float(np.percentile(kalman_flat, 90) / 1000.0),
        "baseline_mean_error_km": float(np.mean(baseline_per_flight_means) / 1000.0),
        "baseline_median_error_km": float(np.median(baseline_per_flight_means) / 1000.0),
        "baseline_p90_error_km": float(np.percentile(baseline_flat, 90) / 1000.0),
        "improvement_mean_pct": float(
            (1.0 - np.mean(kalman_per_flight_means) / max(np.mean(baseline_per_flight_means), 1e-9)) * 100.0
        ),
        "improvement_median_pct": float(
            (1.0 - np.median(kalman_per_flight_means) / max(np.median(baseline_per_flight_means), 1e-9)) * 100.0
        ),
    }

    extras = None
    if produce_predictions:
        extras = {
            "segment_ids": np.array(seg_all, dtype=object),
            "pred_lat": np.stack(pred_lat_all).astype(np.float32),
            "pred_lon": np.stack(pred_lon_all).astype(np.float32),
            "true_lat": np.stack(true_lat_all).astype(np.float32),
            "true_lon": np.stack(true_lon_all).astype(np.float32),
            "baseline_lat": np.stack(baseline_lat_all).astype(np.float32),
            "baseline_lon": np.stack(baseline_lon_all).astype(np.float32),
            "mask": np.stack(mask_all).astype(np.float32),
            "adsc_tau": np.stack(tau_all).astype(np.float32),
            "per_flight_df": pd.DataFrame(per_flight_rows),
        }

    return summary, extras

def _candidate_grid(grid_mode: str) -> list[KalmanParams]:
    if grid_mode == "compact":
        measurement_std_vals = [250.0, 500.0, 1000.0]
        accel_along_vals = [0.005, 0.01, 0.02, 0.05]
        accel_cross_vals = [0.001, 0.003, 0.006, 0.01]
    elif grid_mode == "deep":
        measurement_std_vals = [100.0, 250.0, 500.0, 1000.0, 2000.0]
        accel_along_vals = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
        accel_cross_vals = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.02]
    else:
        raise ValueError(f"Unsupported tuning_grid: {grid_mode}")
    return [
        KalmanParams(m, qa, qc)
        for m, qa, qc in itertools.product(measurement_std_vals, accel_along_vals, accel_cross_vals)
    ]

def tune_kalman_params(
    val_flights: list[PreparedFlight],
    grid_mode: str,
    verbose: bool,
) -> tuple[KalmanParams, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_params: KalmanParams | None = None
    best_score = float("inf")

    grid = _candidate_grid(grid_mode)
    _log(f"Tuning Kalman hyperparameters on validation: {len(grid)} candidates", verbose)

    for idx, params in enumerate(grid, start=1):
        _log(
            (
                f"  candidate {idx}/{len(grid)} "
                f"(measurement_std_m={params.measurement_std_m}, "
                f"accel_std_along_mps2={params.accel_std_along_mps2}, "
                f"accel_std_cross_mps2={params.accel_std_cross_mps2})"
            ),
            verbose,
        )
        summary, _ = evaluate_split(
            val_flights,
            params,
            produce_predictions=False,
            verbose=verbose,
            label=f"val candidate {idx}/{len(grid)}",
        )
        row = {
            "candidate_rank": idx,
            "measurement_std_m": params.measurement_std_m,
            "accel_std_along_mps2": params.accel_std_along_mps2,
            "accel_std_cross_mps2": params.accel_std_cross_mps2,
            **summary,
        }
        rows.append(row)
        score = float(summary["kalman_mean_error_km"])
        if score < best_score:
            best_score = score
            best_params = params
            _log(
                f"    new best: mean_error={best_score:.3f} km with current candidate",
                verbose,
            )

    results_df = pd.DataFrame(rows).sort_values("kalman_mean_error_km").reset_index(drop=True)
    if best_params is None:
        raise ValueError("Validation tuning produced no candidate result.")
    return best_params, results_df

def run_step5_kalman(config: KalmanConfig) -> dict[str, Any]:
    if config.clean_existing_output and config.output_root.exists():
        shutil.rmtree(config.output_root)
    config.output_root.mkdir(parents=True, exist_ok=True)

    train_flights = prepare_split_flights(config, "train")
    val_flights = prepare_split_flights(config, "val")
    test_flights = prepare_split_flights(config, "test")

    if not val_flights:
        raise ValueError("No usable validation flights for Kalman tuning.")
    if not test_flights:
        raise ValueError("No usable test flights for Kalman evaluation.")

    best_params, tuning_df = tune_kalman_params(val_flights, config.tuning_grid, config.verbose)
    tuning_df.to_parquet(config.output_root / "val_tuning_results.parquet", index=False)
    tuning_df.to_csv(config.output_root / "val_tuning_results.csv", index=False)

    _log("Running final validation summary with best Kalman parameters...", config.verbose)
    val_summary, _ = evaluate_split(
        val_flights,
        best_params,
        produce_predictions=False,
        verbose=config.verbose,
        label="validation final",
    )
    _log("Running final test evaluation with best Kalman parameters...", config.verbose)
    test_summary, test_outputs = evaluate_split(
        test_flights,
        best_params,
        produce_predictions=True,
        verbose=config.verbose,
        label="test final",
    )
    assert test_outputs is not None

    per_flight_df = test_outputs.pop("per_flight_df")
    per_flight_df.to_parquet(config.output_root / "per_flight_metrics_test.parquet", index=False)
    per_flight_df.to_csv(config.output_root / "per_flight_metrics_test.csv", index=False)

    np.savez_compressed(config.output_root / "test_predictions.npz", **test_outputs)

    summary = {
        "config": {
            **asdict(config),
            "step2_root": str(config.step2_root),
            "step4_dataset_root": str(config.step4_dataset_root),
            "output_root": str(config.output_root),
        },
        "prepared_counts": {
            "train": len(train_flights),
            "val": len(val_flights),
            "test": len(test_flights),
        },
        "selected_params": asdict(best_params),
        "validation_summary": val_summary,
        "test_summary": test_summary,
    }

    with open(config.output_root / "test_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _log(json.dumps(summary, indent=2), config.verbose)
    return summary

def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = default_kalman_config()
    parser = argparse.ArgumentParser(description="Validation-tuned Kalman smoother baseline for oceanic gap reconstruction.")
    parser.add_argument("--step2-root", type=Path, default=defaults.step2_root)
    parser.add_argument("--step4-dataset-root", type=Path, default=defaults.step4_dataset_root)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--context-mode", choices=["native_clean", "resampled_local"], default=defaults.context_mode)
    parser.add_argument("--resample-seconds", type=int, default=defaults.resample_seconds)
    parser.add_argument("--before-context-limit", type=int, default=defaults.before_context_limit)
    parser.add_argument("--after-context-limit", type=int, default=defaults.after_context_limit)
    parser.add_argument("--tuning-grid", choices=["compact", "deep"], default=defaults.tuning_grid)
    parser.add_argument("--max-flights-per-split", type=int, default=defaults.max_flights_per_split)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser

def main() -> None:
    args = _build_arg_parser().parse_args()
    config = KalmanConfig(
        step2_root=args.step2_root,
        step4_dataset_root=args.step4_dataset_root,
        output_root=args.output_root,
        context_mode=args.context_mode,
        resample_seconds=args.resample_seconds,
        before_context_limit=args.before_context_limit,
        after_context_limit=args.after_context_limit,
        tuning_grid=args.tuning_grid,
        max_flights_per_split=args.max_flights_per_split,
        verbose=not args.quiet,
        clean_existing_output=not args.keep_output,
    )
    run_step5_kalman(config)

if __name__ == "__main__":
    main()
