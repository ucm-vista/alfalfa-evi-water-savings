"""
EVI Smoothed Plot - Smoothed EVI using Savitzky-Golay
Source reference: alfalfa_evi_jovyan.py line 244
"""

import sys
from pathlib import Path
from typing import List, Optional
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt

from es_analysis.data_providers import EviDataProvider, SG_WINDOW, SG_POLY, INTERP_WINDOW_DAYS
from es_analysis.utils import water_year_bounds_multi


def plot_evi_smoothed(county: str, parcel_id: str, years: List[int],
                      output_dir: Optional[Path] = None,
                      interp_window: int = INTERP_WINDOW_DAYS,
                      sg_window: int = SG_WINDOW,
                      sg_poly: int = SG_POLY) -> Path:
    """
    Plot smoothed EVI time series using Savitzky-Golay
    
    Source: alfalfa_evi_jovyan.py line 244
    """
    provider = EviDataProvider()
    daily, filled, smoothed = provider.process_parcel(county, parcel_id, years, interp_window, sg_window, sg_poly)

    if daily.empty:
        raise ValueError("No data available for the selected filters.")

    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()

    ax.plot(smoothed.index, smoothed, label="Smoothed (SG)", linewidth=1.6, alpha=0.95)

    start, end = water_year_bounds_multi(years)
    ax.set_xlim(start, end)
    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    title = f"{county} | parcel {parcel_id} | WY {min(years)}–{max(years)} | smoothed"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)

    if output_dir is None:
        output_dir = Path("es_analysis/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / (title.replace(" ", "_").replace("/", "_").replace("|", "_") + "_smoothed.png")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


if __name__ == "__main__":
    plot_evi_smoothed(
        county="Fresno",
        parcel_id="1005187",
        years=[2021],
    )