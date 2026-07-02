"""
ET Monthly Corrected Bar Plot

Generates a bar chart showing monthly corrected ET (method A or B).
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


def et_monthly_corrected_bar_plot(
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

    m_starts = pd.DatetimeIndex(monthly_df.index)
    m_centers = m_starts + ((m_starts.to_period("M").end_time - m_starts) / 2)

    bar_w = pd.Timedelta(days=14)
    x_corr = m_centers

    ax.bar(
        x_corr,
        monthly_df["ET_corr"].to_numpy(float),
        width=bar_w,
        align="center",
        color="#6BCB77",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.95,
        label=f"Monthly ET (corrected, method {str(method).upper()})",
        zorder=2,
    )

    ax.set_xlim(start, end)
    y_max = float(np.nanmax(monthly_df["ET_corr"].to_numpy())) * 1.25
    ax.set_ylim(0, max(y_max, 1.0))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Water year timeline")
    ax.set_ylabel("Monthly ET (mm/month)")
    ax.set_title(f"{county_norm} | WY{water_year} | UID {uid} – Monthly Corrected ET (method {str(method).upper()})")
    ax.legend()

    output_path = output_dir / "et_corrections" / f"{county_norm}_WY{water_year}_UID{uid}_monthly_corrected.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(output_path),
        "county": county_norm,
        "water_year": water_year,
        "uid": uid,
        "method": method.upper(),
        "figure_type": "monthly_corrected_bar",
    }


if __name__ == "__main__":
    result = et_monthly_corrected_bar_plot("Fresno", 2020, "1003334", "A")
    print("Created:", result["figure_path"])