"""
ETof Daily Line Plot

Displays ETof (fraction of ETo) as a dashed line on a right secondary axis.
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


def etof_daily_line_plot(
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
    fig.subplots_adjust(right=0.78)

    y_max_month = float(np.nanmax(monthly_df[["ET_open", "ET_corr"]].to_numpy())) * 1.25
    ax.set_ylim(0, max(y_max_month, 1.0))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(
        daily_df.index,
        daily_df["ETof"].to_numpy(float),
        color="black",
        lw=1.4,
        linestyle="--",
        label="ETof (fraction of ETo)",
    )
    ax2.set_ylabel("Fraction of ETo (–)")

    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Water year timeline")
    ax.set_ylabel("Monthly ET (mm/month)")
    ax.set_title(f"{county_norm} | WY{water_year} | UID {uid} – ETof Daily")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, labels2, loc="upper right")

    output_path = output_dir / "et_corrections" / f"{county_norm}_WY{water_year}_UID{uid}_etof_daily.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(output_path),
        "county": county_norm,
        "water_year": water_year,
        "uid": uid,
        "method": method.upper(),
        "figure_type": "etof_daily_line",
    }


if __name__ == "__main__":
    result = etof_daily_line_plot("Fresno", 2020, "1003334", "A")
    print("Created:", result["figure_path"])