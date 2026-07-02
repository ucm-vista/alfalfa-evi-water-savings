"""
BEAST trend change points all points scatter plot
Scatter plot of all trend CP dates with county colors.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from es_analysis.data_providers import BeastDataProvider


def plot_beast_trend_cps_allpoints(
    county: str,
    csv_path: Optional[Path] = None,
    span: Optional[str] = None,
    year_min: int = 2019,
    year_max: int = 2023,
    output_dir: Optional[Path] = None,
    seed: int = 0,
    show: bool = True,
    title: Optional[str] = None,
    figsize: tuple = (11, 6),
    dpi: int = 150
) -> dict:
    """
    Plot all trend CP dates as DOY scatter with x-jitter.
    
    Args:
        county: County name (e.g., 'Fresno')
        csv_path: Optional explicit path to trend CSV
        span: Optional span string (e.g., '2019_2023') for specific file
        year_min: Minimum year to include
        year_max: Maximum year to include
        output_dir: Directory to save plot
        seed: Random seed for jitter
        show: Whether to display the plot
        title: Custom title
        figsize: Figure size (width, height)
        dpi: Figure resolution for output
        
    Returns:
        dict: Summary containing file path and statistics
    """
    provider = BeastDataProvider()
    county_norm = provider.normalize_county_name(county)
    
    if csv_path is None:
        csv_path = provider.find_trend_csv(county, span)
    
    if csv_path is None or (isinstance(csv_path, Path) and not csv_path.exists()):
        return {
            "status": "error",
            "message": f"No trend CSV found for county={county}",
            "file_path": None
        }
    
    df = provider.explode_trend_cps(csv_path, year_min=year_min, year_max=year_max)
    
    if df.empty:
        return {
            "status": "error",
            "message": f"No CPs in the selected year window ({year_min}-{year_max})",
            "file_path": None
        }
    
    if output_dir is None:
        output_dir = Path(csv_path).parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    years = sorted(df["year"].unique())
    xs = []
    ys = []
    
    for y in years:
        s = df.loc[df["year"] == y, "doy"].to_numpy()
        if s.size == 0:
            continue
        jitter = rng.uniform(-0.20, 0.20, size=s.size)
        xs.extend(y + jitter)
        ys.extend(s)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys, s=10, alpha=0.35)
    ax.set_ylim(1, 366)
    ax.set_ylabel("Day of Year (DOY)")
    ax.set_xlabel("Year")
    
    xticks = sorted(df["year"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(y) for y in xticks])
    
    if title is None:
        title = f"{county_norm}: All trend CP dates (DOY), {year_min}-{year_max}"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    
    fig.tight_layout()
    
    out_name = f"{county_norm.replace(' ', '_').lower()}_trend_cps_allpoints_{year_min}_{year_max}.png"
    out = output_dir / out_name
    fig.savefig(out, dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    stats_by_year = {
        int(y): {
            "n_cps": int((df["year"] == y).sum()),
            "mean_doy": float(df.loc[df["year"] == y, "doy"].mean()),
            "median_doy": float(df.loc[df["year"] == y, "doy"].median())
        }
        for y in years
    }
    
    return {
        "status": "success",
        "file_path": str(out),
        "county": county,
        "year_range": (year_min, year_max),
        "n_years": len(years),
        "total_cps": len(df),
        "stats_by_year": stats_by_year
    }