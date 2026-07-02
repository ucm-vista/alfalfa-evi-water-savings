"""
EVI Raw Plot - Original EVI with NaN gaps and observation points
Source reference: alfalfa_evi_jovyan.py lines 222-259
"""

import sys
from pathlib import Path
from typing import List, Optional
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from es_analysis.data_providers import EviDataProvider, SG_WINDOW, SG_POLY, INTERP_WINDOW_DAYS
from es_analysis.utils import water_year_bounds_multi


def plot_evi_raw(county: str, parcel_id: str, years: List[int],
                  output_dir: Optional[Path] = None,
                  interp_window: int = INTERP_WINDOW_DAYS,
                  sg_window: int = SG_WINDOW,
                  sg_poly: int = SG_POLY) -> Path:
    """
    Plot original (raw) EVI with NaN gaps and blue markers for observed days
    
    Source: alfalfa_evi_jovyan.py lines 222-259
    """
    provider = EviDataProvider()
    daily, filled, smoothed = provider.process_parcel(county, parcel_id, years, interp_window, sg_window, sg_poly)

    if daily.empty:
        raise ValueError("No data available for the selected filters.")

    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()

    ax.plot(daily["date"], daily["mean_evi"], label="Original (raw)", linewidth=1.2)
    raw_mask = ~daily["mean_evi"].isna()
    ax.scatter(daily.loc[raw_mask, "date"], daily.loc[raw_mask, "mean_evi"],
               s=12, alpha=0.9, label="Raw obs (points)")

    start, end = water_year_bounds_multi(years)
    ax.set_xlim(start, end)
    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    title = f"{county} | parcel {parcel_id} | WY {min(years)}–{max(years)} | win={interp_window}d, SG={sg_window}/{sg_poly}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)

    if output_dir is None:
        output_dir = Path("es_analysis/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / (title.replace(" ", "_").replace("/", "_").replace("|", "_") + "_raw.png")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


if __name__ == "__main__":
    plot_evi_raw(
        county="Fresno",
        parcel_id="1005187",
        years=[2021],
    )