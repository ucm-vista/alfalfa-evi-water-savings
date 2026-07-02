"""
BEAST EVI with change points - original/raw EVI plot
Plots the original raw EVI time series.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from es_analysis.data_providers import BeastDataProvider


def plot_beast_evi_with_cps_original(
    county: str,
    wy: int,
    parcel: str,
    cp_dates: Optional[List[pd.Timestamp]] = None,
    output_dir: Optional[Path] = None,
    show_observations: bool = True,
    linewidth: float = 1.0,
    dpi: int = 150
) -> dict:
    """
    Plot original (raw) EVI time series with change points marked as vertical lines.
    
    Args:
        county: County name (e.g., 'Fresno')
        wy: Water year (e.g., 2021)
        parcel: Parcel ID/identifier
        cp_dates: List of change point dates (if None, extracts from BEAST output)
        output_dir: Directory to save plot
        show_observations: Whether to show raw observation points
        linewidth: Line width for the plot
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
    
    if cp_dates is None:
        cp_result = provider.load_seasonal_cuts_csv(county, wy)
        if cp_result is not None and not cp_result.empty:
            parcel_cps = cp_result[cp_result["parcel_id"] == parcel] if "parcel_id" in cp_result.columns else cp_result.iloc[[0]]
            cp_dates = [pd.Timestamp(d) for d in parcel_cps["cut_dates_iso"].values[0].split(";") if d] if "cut_dates_iso" in parcel_cps.columns else []
    
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    
    ax.plot(parcel_data["date"], parcel_data["original_mean_evi"], 
            label="Original (raw)", linewidth=linewidth)
    
    if show_observations:
        mask = ~parcel_data["original_mean_evi"].isna()
        ax.scatter(parcel_data.loc[mask, "date"], parcel_data.loc[mask, "original_mean_evi"], 
                  s=10, alpha=0.9, label="Raw obs")
    
    for d in cp_dates:
        ax.axvline(d, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8)
    
    start, end = provider.water_year_bounds(wy)
    ax.set_xlim(start, end)
    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    ax.set_title(f"{county} | Parcel {parcel} | WY {wy} (Original EVI with CPs)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    outfile = output_dir / f"WY{wy}_parcel_{parcel}_evi_original_cps.png"
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
        "n_cps": len(cp_dates) if cp_dates else 0
    }