"""
ET Harvest Dates Scatter Plot

Displays harvest/cut dates at the top of the plot with X markers.
"""

from pathlib import Path
from typing import Dict
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


def et_harvest_dates_scatter_plot(
    county: str,
    water_year: int,
    uid: str,
    method: str = "A",
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

    y_max = float(np.nanmax(monthly_df[["ET_open", "ET_corr"]].to_numpy())) * 1.25

    if harvest_dates:
        ax.scatter(
            harvest_dates,
            [y_max * 0.93] * len(harvest_dates),
            marker="x",
            s=90,
            color="black",
            linewidths=2.0,
            label="Harvest dates (cuts)",
            zorder=5,
        )

    ax.set_xlim(start, end)
    ax.set_ylim(0, max(y_max, 1.0))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Water year timeline")
    ax.set_ylabel("Height (relative)")
    ax.set_title(f"{county_norm} | WY{water_year} | UID {uid} – Harvest Dates")
    ax.legend()

    output_path = output_dir / "et_corrections" / f"{county_norm}_WY{water_year}_UID{uid}_harvest_dates.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(output_path),
        "county": county_norm,
        "water_year": water_year,
        "uid": uid,
        "method": method.upper(),
        "n_harvest_dates": len(harvest_dates),
        "figure_type": "harvest_dates_scatter",
    }


if __name__ == "__main__":
    result = et_harvest_dates_scatter_plot("Fresno", 2020, "1003334", "A")
    print("Created:", result["figure_path"])