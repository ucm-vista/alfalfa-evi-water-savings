"""
BEAST trend change point medians by order with errors
Bar plot with error bars showing median trend CP dates with variability.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from es_analysis.data_providers import BeastDataProvider


def plot_beast_trend_cp_medians_by_order_with_errors(
    county: str,
    csv_path: Optional[Path] = None,
    span: Optional[str] = None,
    year_min: int = 2019,
    year_max: int = 2023,
    max_orders: int = 10,
    min_group_n: int = 3,
    output_dir: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
    figsize: tuple = (12, 7),
    dpi: int = 150
) -> dict:
    """
    Plot median DOY of trend CPs by order with IQR error bars.
    
    Args:
        county: County name (e.g., 'Fresno')
        csv_path: Optional explicit path to trend CSV
        span: Optional span string (e.g., '2019_2023') for specific file
        year_min: Minimum year to include
        year_max: Maximum year to include
        max_orders: Maximum CP order to plot
        min_group_n: Minimum number of points per year/order to include
        output_dir: Directory to save plot
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
    
    fig, ax = plt.subplots(figsize=figsize)
    palette = plt.cm.tab10.colors
    
    legend_entries = []
    stats_results = {}
    
    for k in range(1, max_orders + 1):
        d = df[df["cp_order"] == k]
        if d.empty:
            continue
        
        g = (d.groupby("year")["doy"]
               .agg(median="median", q25=lambda s: s.quantile(0.25), 
                    q75=lambda s: s.quantile(0.75), n="size")
               .reset_index())
        
        g = g[g["n"] >= int(min_group_n)]
        if g.empty:
            continue
        
        years = g["year"].values
        medians = g["median"].values
        q25 = g["q25"].values
        q75 = g["q75"].values
        
        ax.errorbar(years, medians, 
                   yerr=[medians - q25, q75 - medians],
                   fmt='-o', markersize=6, capsize=4,
                   color=palette[(k-1) % len(palette)],
                   label=f"Cut {k}")
        
        stats_results[f"cut_{k}"] = {
            "n_years": len(years),
            "mean_doy": float(np.mean(medians)),
            "mean_n_per_year": float(g["n"].mean())
        }
        legend_entries.append(k)
    
    if title is None:
        title = f"{county_norm}: Trend CP Medians by Order with IQR ({year_min}-{year_max})"
    
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Day of Year (median)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    out_name = f"{csv_path.stem}_by_order_errors_{year_min}_{year_max}.png"
    out = output_dir / out_name
    fig.savefig(out, dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return {
        "status": "success",
        "file_path": str(out),
        "county": county,
        "year_range": (year_min, year_max),
        "n_cuts_plotted": len(legend_entries),
        "cuts": legend_entries,
        "stats": stats_results
    }