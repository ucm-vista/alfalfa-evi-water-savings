"""
BEAST EVI with change points - smoothed plot
Plots the smoothed EVI time series using Savitzky-Golay filter.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from es_analysis.data_providers import BeastDataProvider


def plot_beast_evi_with_cps_smoothed(
    county: str,
    wy: int,
    parcel: str,
    cp_dates: Optional[List[pd.Timestamp]] = None,
    output_dir: Optional[Path] = None,
    linewidth: float = 1.5,
    alpha: float = 0.95,
    dpi: int = 150
) -> dict:
    """
    Plot smoothed EVI time series (Savitzky-Golay) with change points marked.
    
    Args:
        county: County name (e.g., 'Fresno')
        wy: Water year (e.g., 2021)
        parcel: Parcel ID/identifier
        cp_dates: List of change point dates (if None, extracts from BEAST output)
        output_dir: Directory to save plot
        linewidth: Line width for the plot
        alpha: Transparency of line
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
    
    if "smoothed_mean_evi" not in parcel_data.columns:
        return {
            "status": "error",
            "message": f"Smoothed EVI column not found for parcel {parcel}",
            "file_path": None
        }
    
    if cp_dates is None:
        cp_result = provider.load_seasonal_cuts_csv(county, wy)
        if cp_result is not None and not cp_result.empty:
            parcel_cps = cp_result[cp_result["parcel_id"] == parcel] if "parcel_id" in cp_result.columns else cp_result.iloc[[0]]
            cp_dates = [pd.Timestamp(d) for d in parcel_cps["cut_dates_iso"].values[0].split(";") if d] if "cut_dates_iso" in parcel_cps.columns else []
    
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    
    ax.plot(parcel_data["date"], parcel_data["smoothed_mean_evi"], 
            label="Smoothed (SG)", linewidth=linewidth, alpha=alpha)
    
    for d in cp_dates:
        ax.axvline(d, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8)
    
    start, end = provider.water_year_bounds(wy)
    ax.set_xlim(start, end)
    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    ax.set_title(f"{county} | Parcel {parcel} | WY {wy} (Smoothed EVI with CPs)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    outfile = output_dir / f"WY{wy}_parcel_{parcel}_evi_smoothed_cps.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.show()
    plt.close()
    
    n_smoothed = parcel_data["smoothed_mean_evi"].notna().sum()
    
    return {
        "status": "success",
        "file_path": str(outfile),
        "county": county,
        "water_year": wy,
        "parcel": parcel,
        "n_points": n_smoothed,
        "n_cps": len(cp_dates) if cp_dates else 0
    }