"""Estimate route emissions for reconstructed trajectories."""

from __future__ import annotations
import math
import numpy as np

NAT_FLEET_MIX = [
    (5.98,  3.16, 0.28),   # A330-200/300
    (7.20,  3.16, 0.22),   # B777-200ER/300ER
    (5.50,  3.16, 0.20),   # B787-9
    (5.30,  3.16, 0.15),   # A350-900
    (6.80,  3.16, 0.10),   # B767-300ER
    (4.90,  3.16, 0.05),   # A321XLR (emerging)
]

_FLEET_AVG_CO2_KG_PER_KM = sum(
    fuel * co2 * share for fuel, co2, share in NAT_FLEET_MIX
) / sum(share for _, _, share in NAT_FLEET_MIX)

def altitude_efficiency_factor(altitude_ft: float) -> float:
    """Fuel-burn correction factor relative to optimal cruise altitude (FL370); 1.0 = nominal."""
    optimal_ft = 37000.0

    if altitude_ft <= 0:
        return 1.0   # unknown altitude - use nominal

    delta = altitude_ft - optimal_ft

    if delta < 0:
        factor = 1.0 + (delta / optimal_ft) ** 2 * 8.0
    else:
        factor = 1.0 + (delta / 40000.0) * 3.0

    return max(0.85, min(1.20, factor))

def compute_emissions_kg_co2(
    distance_km: float,
    altitude_ft_profile: list[float] | None = None,
    aircraft_type: str | None = None,
) -> dict:
    """Estimate CO2 emissions (kg) for a flight segment with optional altitude correction."""
    if distance_km <= 0:
        return {"co2_kg": 0.0, "fuel_kg": 0.0, "co2_per_km": 0.0,
                "mean_altitude_ft": 35000.0, "efficiency_factor": 1.0,
                "aircraft_type": "fleet_average"}

    aircraft_map = {
        "A330":  (5.98, 3.16),
        "B777":  (7.20, 3.16),
        "B787":  (5.50, 3.16),
        "A350":  (5.30, 3.16),
        "B767":  (6.80, 3.16),
    }

    if aircraft_type and aircraft_type.upper() in aircraft_map:
        fuel_per_km, co2_factor = aircraft_map[aircraft_type.upper()]
        ac_label = aircraft_type.upper()
    else:
        fuel_per_km = sum(f*s for f,_,s in NAT_FLEET_MIX) / sum(s for _,_,s in NAT_FLEET_MIX)
        co2_factor  = 3.16
        ac_label = "fleet_average"

    if altitude_ft_profile and len(altitude_ft_profile) > 0:
        valid_alts = [a for a in altitude_ft_profile
                      if a is not None and math.isfinite(float(a)) and float(a) > 1000]
        mean_alt = float(np.mean(valid_alts)) if valid_alts else 35000.0
    else:
        mean_alt = 35000.0   # assume FL350 if unknown

    eff_factor  = altitude_efficiency_factor(mean_alt)
    fuel_kg     = fuel_per_km * distance_km * eff_factor
    co2_kg      = fuel_kg * co2_factor
    co2_per_km  = co2_kg / distance_km

    return {
        "co2_kg":           round(co2_kg, 2),
        "fuel_kg":          round(fuel_kg, 2),
        "co2_per_km":       round(co2_per_km, 3),
        "mean_altitude_ft": round(mean_alt, 0),
        "efficiency_factor": round(eff_factor, 4),
        "aircraft_type":    ac_label,
        "distance_km":      round(distance_km, 2),
    }

def compare_reconstructions_emissions(
    gru_dist_km:      float,
    baseline_dist_km: float,
    gru_alt_profile:  list[float] | None = None,
    baseline_alt_profile: list[float] | None = None,
) -> dict:
    """Return CO2 emissions for GRU and baseline reconstructions and their difference."""
    gru_em  = compute_emissions_kg_co2(gru_dist_km,      gru_alt_profile)
    base_em = compute_emissions_kg_co2(baseline_dist_km, baseline_alt_profile)

    co2_diff = gru_em["co2_kg"] - base_em["co2_kg"]
    pct_diff = (co2_diff / max(base_em["co2_kg"], 0.01)) * 100

    return {
        "gru":         gru_em,
        "baseline":    base_em,
        "co2_diff_kg": round(co2_diff, 2),
        "pct_diff":    round(pct_diff, 2),
        "interpretation": (
            f"GRU reconstruction implies {abs(co2_diff):.0f} kg CO2 "
            f"{'more' if co2_diff > 0 else 'less'} than baseline "
            f"({abs(pct_diff):.1f}% {'higher' if co2_diff > 0 else 'lower'})."
        ),
    }

if __name__ == "__main__":
    print("Fleet average CO2 per km:", round(_FLEET_AVG_CO2_KG_PER_KM, 2), "kg/km")
    print()

    result = compute_emissions_kg_co2(
        distance_km=5500,
        altitude_ft_profile=[35000] * 50,
    )
    print("Typical NAT flight (5500km, FL350):")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print()
    comparison = compare_reconstructions_emissions(
        gru_dist_km=5520, baseline_dist_km=5500,
        gru_alt_profile=[36000]*50, baseline_alt_profile=[35000]*50,
    )
    print("GRU vs Baseline comparison:")
    print(f"  {comparison['interpretation']}")
    print(f"  CO2 diff: {comparison['co2_diff_kg']} kg")
