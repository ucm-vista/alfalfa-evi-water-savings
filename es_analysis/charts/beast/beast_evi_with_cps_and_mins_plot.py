"""
BEAST EVI with change points and minima plot
Line plot with vertical lines showing BEAST CPs (red) and detected minima (green).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from es_analysis.data_providers import BeastDataProvider


def plot_beast_evi_with_cps_and_mins(
    county: str,
    wy: int,
    parcel: str,
    cp_dates: Optional[List[pd.Timestamp]] = None,
    minima_all: Optional[List[pd.Timestamp]] = None,
    minima_matched: Optional[List[pd.Timestamp]] = None,
    output_dir: Optional[Path] = None,
    label_suffix: str = "",
    figsize: tuple = (12, 5),
    dpi: int = 150
) -> dict:
    """
    Plot EVI with BEAST change points (red) and detected minima (hollow=qualified, filled=cut).
    
    Args:
        county: County name (e.g., 'Fresno')
        wy: Water year (e.g., 2021)
        parcel: Parcel ID/identifier
        cp_dates: List of change point dates
        minima_all: List of all qualified minima (hollow markers)
        minima_matched: List of minima matched to CPs (filled markers)
        output_dir: Directory to save plot
        label_suffix: Suffix to append to title
        figsize: Figure size (width, height)
        dpi: Figure resolution for output
        
    Returns:
        dict: Summary containing file path and metadata
    """
    provider = BeastDataProvider()
    
    if output_dir is None:
        output_dir = provider.output_dir / provider.normalize_county_name(county)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_parcel = provider.load_county_year_csv(county, wy)
    
    if df_parcel is None or df_parcel.empty:
        return {
            "status": "error",
            "message": f"No data found for county={county}, wy={wy}",
            "file_path": None
        }
    
    parcel_data = df_parcel[df_parcel["parcel_id"] == parcel] if "parcel_id" in df_parcel.columns else df_parcel.iloc[[0]]
    
    if parcel_data.empty:
        return {
            "status": "error",
            "message": f"Parcel {parcel} not found in county={county}, wy={wy}",
            "file_path": None
        }
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    ax.plot(parcel_data["date"], parcel_data["original_mean_evi"], 
            label="Original (raw)", linewidth=1.0)
    
    mask = ~parcel_data["original_mean_evi"].isna()
    ax.scatter(parcel_data.loc[mask, "date"], parcel_data.loc[mask, "original_mean_evi"], 
              s=10, alpha=0.9, label="Raw obs")
    
    ax.plot(parcel_data["date"], parcel_data["gapfilled_mean_evi"], 
            label="Gap-filled (quartic)", linewidth=1.2, alpha=0.9)
    
    ax.plot(parcel_data["date"], parcel_data["smoothed_mean_evi"], 
            label="Smoothed (SG)", linewidth=1.5, alpha=0.95)
    
    for d in cp_dates:
        ax.axvline(d, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8,
                  label="BEAST CP" if d == (cp_dates[0] if cp_dates else None) else None)
    
    df_indexed = parcel_data.set_index("date")
    
    if minima_all:
        min_y_values = [df_indexed.loc[t, "smoothed_mean_evi"] if t in df_indexed.index else np.nan 
                       for t in minima_all]
        ax.scatter(minima_all, min_y_values,
                  marker="v", s=45, facecolors="none", edgecolors="black", 
                  zorder=5, label="All qualified minima")
    
    if minima_matched:
        min_y_values = [df_indexed.loc[t, "smoothed_mean_evi"] if t in df_indexed.index else np.nan 
                       for t in minima_matched]
        ax.scatter(minima_matched, min_y_values,
                  marker="v", s=55, color="black", zorder=6, 
                  label="Minima matched to CPs (cuts)")
    
    start, end = provider.water_year_bounds(wy)
    ax.set_xlim(start, end)
    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    ax.set_title(f"{county} | Parcel {parcel} | WY {wy} (Cuts={len(minima_matched) if minima_matched else 0}{label_suffix})")
    ax.legend()
    ax.grid(True, alpha=0.25)
    
    outfile = output_dir / f"WY{wy}_parcel_{parcel}_EVI_CP_MIN.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.show()
    plt.close()
    
    return {
        "status": "success",
        "file_path": str(outfile),
        "county": county,
        "water_year": wy,
        "parcel": parcel,
        "n_cps": len(cp_dates) if cp_dates else 0,
        "n_minima_all": len(minima_all) if minima_all else 0,
        "n_minima_matched": len(minima_matched) if minima_matched else 0
    }