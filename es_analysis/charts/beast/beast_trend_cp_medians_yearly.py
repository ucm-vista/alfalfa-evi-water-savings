"""
BEAST trend change point medians yearly plot
Line plot of median trend change point dates plotted across years with IQR band.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from es_analysis.data_providers import BeastDataProvider


def plot_beast_trend_cp_medians_yearly(
    county: str,
    span: Optional[str] = None,
    output_dir: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
    dpi: int = 150
) -> dict:
    """
    Plot yearly median DOY of trend CPs (all CPs together) with IQR band.
    
    Args:
        county: County name (e.g., 'Fresno')
        span: Optional span string (e.g., '2019_2023') for specific file
        output_dir: Directory to save plot (if None, uses CSV parent dir)
        show: Whether to display the plot
        title: Custom title
        figsize: Figure size (width, height)
        dpi: Figure resolution for output
        
    Returns:
        dict: Summary containing file path and statistics
    """
    provider = BeastDataProvider()
    county_norm = provider.normalize_county_name(county)
    
    csv = provider.find_trend_csv(county, span)
    
    if csv is None or not csv.exists():
        return {
            "status": "error",
            "message": f"No trend CSV found for county={county}",
            "file_path": None
        }
    
    df = pd.read_csv(csv)
    long = provider.explode_trend_dates(df)
    
    if long.empty:
        return {
            "status": "error",
            "message": "No trend CP dates found",
            "file_path": None
        }
    
    tab = provider.tab_yearly_medians(long)
    
    if output_dir is None:
        output_dir = csv.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out = output_dir / f"{csv.stem}_yearly_median.png"
    
    if title is None:
        title = f"{county_norm}: Yearly median trend change-point timing\n(all CPs; DOY median with IQR)"
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(tab["year"], tab["median_doy"], marker="o")
    
    ax.fill_between(tab["year"].to_numpy(dtype=float),
                    tab["q25"].to_numpy(dtype=float),
                    tab["q75"].to_numpy(dtype=float), alpha=0.2)
    
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Day of Year (median)")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return {
        "status": "success",
        "file_path": str(out),
        "county": county,
        "n_years": len(tab),
        "median_doy_mean": float(tab["median_doy"].mean()),
        "median_doy_std": float(tab["median_doy"].std()),
        "table": tab.to_dict("records")
    }