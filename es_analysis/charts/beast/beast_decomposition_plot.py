"""
BEAST decomposition plot
Generates large BEAST seasonal/trend decomposition with change points.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import Rbeast as rb
except ImportError:
    rb = None

from es_analysis.data_providers import BeastDataProvider


def plot_beast_decomposition(
    county: str,
    wy: int,
    parcel: str,
    result: Optional[object] = None,
    output_dir: Optional[Path] = None,
    display_result: bool = True,
    dpi: int = 150
) -> dict:
    """
    Plot BEAST decomposition for a single parcel's EVI time series.
    
    Args:
        county: County name (e.g., 'Fresno')
        wy: Water year (e.g., 2021)
        parcel: Parcel ID/identifier
        result: Pre-computed BEAST result object (if None, loads from provider)
        output_dir: Directory to save plot (if None, uses provider default)
        display_result: Whether to display result text/table
        dpi: Figure resolution for output
        
    Returns:
        dict: Summary containing file path and metadata
    """
    if rb is None:
        raise ImportError("Rbeast package not installed. Run: pip install Rbeast")
    
    provider = BeastDataProvider()
    
    if output_dir is None:
        output_dir = provider.output_dir / provider.normalize_county_name(county)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result is None:
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
        
        series = parcel_data["gapfilled_mean_evi"].values if "gapfilled_mean_evi" in parcel_data.columns else \
                 parcel_data["original_mean_evi"].values
        
        if pd.isna(series).all():
            return {
                "status": "error",
                "message": f"All EVI values are NaN for parcel {parcel}",
                "file_path": None
            }
        
        result = provider.run_beast(series, cp_component="season")
    
    plt.figure(figsize=(12, 10))
    rb.plot(result, ncpStat='median')
    plt.suptitle(f"BEAST EVI Decomposition\nCounty={county}, WY={wy}, Parcel={parcel}",
                 y=0.98, fontsize=14)
    plt.subplots_adjust(top=0.90, hspace=0.0, wspace=0.0)
    plt.show()
    
    if display_result:
        try:
            from IPython.display import display
            display(result)
        except Exception:
            print(result)
    
    outfile = output_dir / f"WY{wy}_parcel_{parcel}_decomposition.png"
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return {
        "status": "success",
        "file_path": str(outfile),
        "county": county,
        "water_year": wy,
        "parcel": parcel,
        "n_cps": len(result.trend.cp) if hasattr(result, 'trend') else 0
    }