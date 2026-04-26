from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


COLOR_MAP = {
    "ADS-B before": "#1f77b4",
    "ADS-C gap": "#d62728",
    "ADS-B after": "#2ca02c",
}


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_validated_flights_by_day(summary_df: pd.DataFrame):
    apply_plot_style()
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("No validated flights found")
        ax.axis("off")
        return fig, ax

    daily_counts = (
        summary_df.assign(processing_day=pd.to_datetime(summary_df["processing_day"]))
        .groupby("processing_day")
        .size()
        .reset_index(name="validated_flights")
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=daily_counts, x="processing_day", y="validated_flights", color="#4c78a8", ax=ax)
    ax.set_title("Validated Fusion-Ready Flights by Day")
    ax.set_xlabel("Processing day")
    ax.set_ylabel("Validated segments")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax


def plot_gap_duration_hist(summary_df: pd.DataFrame):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(summary_df, x="gap_duration_minutes", bins=25, color="#f58518", ax=ax)
    ax.set_title("ADS-C Gap Duration Distribution")
    ax.set_xlabel("Gap duration (minutes)")
    ax.set_ylabel("Segment count")
    fig.tight_layout()
    return fig, ax


def plot_adsc_point_count_hist(summary_df: pd.DataFrame):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(summary_df, x="adsc_point_count", bins=25, color="#54a24b", ax=ax)
    ax.set_title("ADS-C Point Count per Validated Segment")
    ax.set_xlabel("ADS-C points")
    ax.set_ylabel("Segment count")
    fig.tight_layout()
    return fig, ax


def plot_boundary_speed_hist(summary_df: pd.DataFrame):
    apply_plot_style()
    speed_columns = ["before_boundary_speed_kts", "after_boundary_speed_kts"]
    speed_data = (
        summary_df[speed_columns]
        .rename(
            columns={
                "before_boundary_speed_kts": "Before boundary",
                "after_boundary_speed_kts": "After boundary",
            }
        )
        .melt(var_name="boundary", value_name="speed_kts")
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        speed_data,
        x="speed_kts",
        hue="boundary",
        bins=25,
        stat="count",
        common_norm=False,
        alpha=0.55,
        ax=ax,
    )
    ax.set_title("Implied Boundary Speed Distribution")
    ax.set_xlabel("Speed (knots)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig, ax


def plot_stitched_route(stitched_df: pd.DataFrame, title: str):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    for phase, phase_df in stitched_df.groupby("phase", sort=False):
        ax.plot(
            phase_df["longitude"],
            phase_df["latitude"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=phase,
            color=COLOR_MAP.get(phase, "#555555"),
        )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_time_progress(
    adsb_before: pd.DataFrame,
    adsc_gap: pd.DataFrame,
    adsb_after: pd.DataFrame,
    title: str,
):
    apply_plot_style()
    combined = pd.concat(
        [
            adsb_before.assign(phase="ADS-B before"),
            adsc_gap.assign(phase="ADS-C gap"),
            adsb_after.assign(phase="ADS-B after"),
        ],
        ignore_index=True,
    ).sort_values("timestamp")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for phase, phase_df in combined.groupby("phase", sort=False):
        axes[0].plot(
            phase_df["timestamp"],
            phase_df["latitude"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=phase,
            color=COLOR_MAP.get(phase, "#555555"),
        )
        axes[1].plot(
            phase_df["timestamp"],
            phase_df["longitude"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=phase,
            color=COLOR_MAP.get(phase, "#555555"),
        )

    axes[0].set_ylabel("Latitude")
    axes[1].set_ylabel("Longitude")
    axes[1].set_xlabel("UTC timestamp")
    axes[0].set_title(title)
    axes[0].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, axes
