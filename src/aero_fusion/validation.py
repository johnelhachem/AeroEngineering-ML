from __future__ import annotations

from dataclasses import asdict, dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Any

import pandas as pd

EARTH_RADIUS_KM = 6371.0088
KM_TO_NAUTICAL_MILES = 0.539956803

@dataclass(frozen=True)
class ValidationThresholds:
    """Physical plausibility thresholds for Step 1 candidate validation."""

    max_boundary_speed_kts: float = 750.0
    max_internal_speed_kts: float = 700.0

@dataclass
class ValidationResult:
    """Validation outcome and compact metrics for one candidate segment."""

    is_valid: bool
    reasons: list[str]
    before_count: int
    during_count: int
    after_count: int
    adsc_point_count: int
    gap_duration_minutes: float
    before_boundary_speed_kts: float | None
    after_boundary_speed_kts: float | None
    max_internal_speed_kts: float | None
    median_internal_speed_kts: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    a = sin(d_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))

def implied_speed_kts(point_a: pd.Series, point_b: pd.Series, time_col: str = "timestamp") -> float | None:
    """Compute implied speed in knots between two trajectory points."""

    delta_seconds = (point_b[time_col] - point_a[time_col]).total_seconds()
    if delta_seconds <= 0:
        return None

    distance_km = haversine_km(
        point_a["latitude"],
        point_a["longitude"],
        point_b["latitude"],
        point_b["longitude"],
    )
    distance_nm = distance_km * KM_TO_NAUTICAL_MILES
    hours = delta_seconds / 3600.0
    if hours <= 0:
        return None
    return distance_nm / hours

def internal_speed_profile(adsc_points: pd.DataFrame) -> pd.Series:
    """Compute point-to-point ADS-C implied speeds in knots."""

    ordered = adsc_points.sort_values("timestamp").reset_index(drop=True)
    if len(ordered) < 2:
        return pd.Series(dtype="float64")

    speeds: list[float] = []
    for index in range(len(ordered) - 1):
        speed = implied_speed_kts(ordered.iloc[index], ordered.iloc[index + 1])
        if speed is not None:
            speeds.append(speed)
    return pd.Series(speeds, dtype="float64")

def validate_fusion_candidate(
    *,
    segment_row: pd.Series,
    matched_flights: pd.DataFrame,
    adsc_points: pd.DataFrame,
    adsb_before: pd.DataFrame,
    adsb_during: pd.DataFrame,
    adsb_after: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> ValidationResult:
    """Apply the Step 1 rules for validated fusion-ready flights."""

    reasons: list[str] = []

    if matched_flights.empty:
        reasons.append("no_overlapping_flight")
    elif len(matched_flights) != 1:
        reasons.append("multiple_overlapping_flights")

    if not matched_flights.empty:
        flight_icao24 = str(matched_flights.iloc[0]["icao24"]).lower()
        segment_icao24 = str(segment_row["icao24"]).lower()
        if flight_icao24 != segment_icao24:
            reasons.append("icao24_mismatch")

    if adsb_before.empty:
        reasons.append("missing_adsb_before")
    if len(adsb_during) != 0:
        reasons.append("adsb_present_during_gap")
    if adsb_after.empty:
        reasons.append("missing_adsb_after")

    before_boundary_speed = None
    after_boundary_speed = None
    if not adsb_before.empty and not adsc_points.empty:
        before_boundary_speed = implied_speed_kts(
            adsb_before.sort_values("timestamp").iloc[-1],
            adsc_points.sort_values("timestamp").iloc[0],
        )
        if before_boundary_speed is None or before_boundary_speed > thresholds.max_boundary_speed_kts:
            reasons.append("boundary_speed_before_unreasonable")

    if not adsc_points.empty and not adsb_after.empty:
        after_boundary_speed = implied_speed_kts(
            adsc_points.sort_values("timestamp").iloc[-1],
            adsb_after.sort_values("timestamp").iloc[0],
        )
        if after_boundary_speed is None or after_boundary_speed > thresholds.max_boundary_speed_kts:
            reasons.append("boundary_speed_after_unreasonable")

    internal_speeds = internal_speed_profile(adsc_points)
    max_internal_speed = float(internal_speeds.max()) if not internal_speeds.empty else None
    median_internal_speed = float(internal_speeds.median()) if not internal_speeds.empty else None
    if max_internal_speed is None:
        reasons.append("insufficient_adsc_points")
    elif max_internal_speed > thresholds.max_internal_speed_kts:
        reasons.append("internal_adsc_speed_unreasonable")

    return ValidationResult(
        is_valid=len(reasons) == 0,
        reasons=sorted(set(reasons)),
        before_count=int(len(adsb_before)),
        during_count=int(len(adsb_during)),
        after_count=int(len(adsb_after)),
        adsc_point_count=int(len(adsc_points)),
        gap_duration_minutes=float(segment_row["gap_duration_minutes"]),
        before_boundary_speed_kts=before_boundary_speed,
        after_boundary_speed_kts=after_boundary_speed,
        max_internal_speed_kts=max_internal_speed,
        median_internal_speed_kts=median_internal_speed,
    )
