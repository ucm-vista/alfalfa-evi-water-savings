"""
ET Daily Scatter Plot

Displays daily ET (OpenET) as scatter points on a secondary axis.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from es_analysis.data_providers.et_provider import (
    compute_daily_and_monthly_for_uid,
    _norm_county_name,
    water_year_bounds,
)
from es_analysis.utils.plotting import save_figure


def et_daily_scatter_plot(
    county: str,
    water_year: int,
    uid: str,
    method: str = "A",
    use_left_axis: bool = True,
    left_axis_offset: float = -0.10,
    output_dir: Path = Path("es_analysis/output/figures"),
) -> Dict[str, str]:
    county_norm = _norm_county_name(county)
    start, end = water_year_bounds(water_year)

    daily_df, monthly_df, harvest_dates, passes_df = compute_daily_and_monthly_for_uid(
        county=county_norm,
        wy=water_year,
        uid=str(uid),
        chosen_method=method.upper(),
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(right=0.78)

    y_max_month = float(np.nanmax(monthly_df[["ET_open", "ET_corr"]].to_numpy())) * 1.25
    ax.set_ylim(0, max(y_max_month, 1.0))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    ax_daily: Optional[plt.Axes] = None
    if use_left_axis:
        ax_daily = ax.twinx()
        ax_daily.spines["right"].set_visible(False)
        ax_daily.spines["left"].set_visible(True)
        ax_daily.spines["left"].set_position(("axes", left_axis_offset))
        ax_daily.yaxis.set_label_position("left")
        ax_daily.yaxis.tick_left()
        ax_daily.patch.set_visible(False)

        ax_daily.scatter(
            daily_df.index,
            daily_df["ET_open"].to_numpy(float),
            s=16,
            marker="o",
            facecolors="red",
            edgecolors="black",
            linewidths=0.2,
            alpha=0.45,
            label="Daily ET (OpenET)",
            zorder=4,
        )

        y_max_daily = max(float(np.nanmax(daily_df["ET_open"].to_numpy())), 0.1) * 1.25
        ax_daily.set_ylim(0, y_max_daily)
        ax_daily.set_ylabel("Daily ET (mm/day)")
    else:
        ax.plot(
            daily_df.index,
            daily_df["ET_open"].to_numpy(float),
            color="red",
            lw=1.1,
            label="Daily ET (OpenET)",
            zorder=4,
        )

    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Water year timeline")
    ax.set_ylabel("Monthly ET (mm/month)")
    ax.set_title(f"{county_norm} | WY{water_year} | UID {uid} – Daily ET")

    lines1, labels1 = ax.get_legend_handles_labels()
    if ax_daily is not None:
        linesd, labelsd = ax_daily.get_legend_handles_labels()
    else:
        linesd, labelsd = [], []
    ax.legend(lines1 + linesd, labels1 + labelsd, loc="lower right")

    output_path = output_dir / "et_corrections" / f"{county_norm}_WY{water_year}_UID{uid}_daily_et.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(output_path),
        "county": county_norm,
        "water_year": water_year,
        "uid": uid,
        "method": method.upper(),
        "use_left_axis": use_left_axis,
        "n_daily_points": len(daily_df),
        "figure_type": "daily_et_scatter",
    }


if __name__ == "__main__":
    result = et_daily_scatter_plot("Fresno", 2020, "1003334", "A", True)
    print("Created:", result["figure_path"])