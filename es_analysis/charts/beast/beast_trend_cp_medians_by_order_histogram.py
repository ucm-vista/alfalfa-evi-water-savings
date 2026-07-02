"""
BEAST trend change point medians by order histogram
Frequency bar plot of trend CP order (1st, 2nd, 3rd cuts).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from es_analysis.data_providers import BeastDataProvider


def plot_beast_trend_cp_medians_by_order_histogram(
    county: str,
    span: Optional[str] = None,
    output_dir: Optional[Path] = None,
    show: bool = True,
    min_parcels: int = 3,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150
) -> dict:
    """
    Plot yearly median DOY of trend CPs grouped by CP order (1st, 2nd, 3rd cuts).
    
    Args:
        county: County name (e.g., 'Fresno')
        span: Optional span string (e.g., '2019_2023') for specific file
        output_dir: Directory to save plot
        show: Whether to display the plot
        min_parcels: Minimum number of parcels required to include in plot
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
    
    if output_dir is None:
        output_dir = csv.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out = output_dir / f"{csv.stem}_by_order_histogram.png"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10.colors
    
    unique_orders = sorted(long["cp_order"].unique())
    freq_by_order = long.groupby("cp_order")["parcel_id"].nunique()
    
    x_pos = np.arange(len(unique_orders))
    bars = ax.bar(x_pos, [freq_by_order.get(o, 0) for o in unique_orders], 
                  color=[colors[i % len(colors)] for i in range(len(unique_orders))],
                  alpha=0.7)
    
    ax.set_xlabel("Change Point Order")
    ax.set_ylabel("Number of Parcels")
    
    if title is None:
        title = f"{county_norm}: Frequency of Trend CP Order (1st, 2nd, 3rd cuts)"
    ax.set_title(title)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(o)}th cut" for o in unique_orders])
    ax.grid(True, axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    stats = {
        f"order_{int(o)}": {
            "n_parcels": int(freq_by_order.get(o, 0)),
            "n_cps": int((long["cp_order"] == o).sum())
        }
        for o in unique_orders
    }
    
    return {
        "status": "success",
        "file_path": str(out),
        "county": county,
        "unique_orders": unique_orders,
        "stats": stats
    }