"""
EVI County Year Smoothed Plot - Smoothed EVI from county/year CSV
Source reference: alfalfa_evi_jovyan.py line 748
"""

import sys
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from es_analysis.data_providers import COUNTY_YEAR_EXPORT_ROOT, INTERP_WINDOW_DAYS, SG_WINDOW, SG_POLY
from es_analysis.utils import water_year_bounds


def plot_evi_county_year_smoothed(county: str, wy: int, parcel_id: Optional[str] = None,
                                   output_dir: Optional[Path] = None) -> Path:
    """
    Plot smoothed (SG) EVI from county/year CSV
    
    Source: alfalfa_evi_jovyan.py line 748
    """
    csv_path = COUNTY_YEAR_EXPORT_ROOT / county / f"WY{wy}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")
    dfc = pd.read_csv(csv_path, parse_dates=["date"])
    if dfc.empty:
        raise ValueError("Empty CSV for that county/year.")

    parcels = sorted(dfc["parcel_id"].astype(str).unique().tolist())
    to_plot = parcels if (parcel_id is None or str(parcel_id).strip() == "") else [str(parcel_id)]

    start, end = water_year_bounds(wy)
    out_dir = output_dir if output_dir is not None else Path("es_analysis/output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in to_plot:
        s = dfc[dfc["parcel_id"].astype(str) == pid].copy()
        if s.empty:
            continue

        fig = plt.figure(figsize=(11, 5.5))
        ax = plt.gca()

        ax.plot(s["date"], s["smoothed_mean_evi"], label="Smoothed (SG)", linewidth=1.6, alpha=0.95)

        ax.set_xlim(start, end)
        ax.set_xlabel("Date")
        ax.set_ylabel("EVI")
        ax.set_title(f"{county} | parcel {pid} | WY {wy} | smoothed")
        ax.legend()
        ax.grid(True, alpha=0.2)

        outfile = out_dir / f"{county}_WY{wy}_parcel_{pid}_smoothed.png"
        fig.tight_layout()
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        return outfile

    raise ValueError(f"No valid parcel found for {county} WY{wy}")


if __name__ == "__main__":
    plot_evi_county_year_smoothed(
        county="Fresno",
        wy=2021,
    )