"""
BEAST county/year boxplots
Boxplot showing distribution of cuttings by county and year.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from es_analysis.data_providers import BeastDataProvider


def plot_beast_county_year_boxplots(
    county: str,
    years: Optional[List[int]] = None,
    column: str = "n_cuttings",
    output_dir: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150
) -> dict:
    """
    For a county, draw a horizontal boxplot per year.
    
    Args:
        county: County name (e.g., 'Fresno')
        years: List of water years (if None, defaults to 2019-2024)
        column: Column to plot values from
        output_dir: Directory to save plot
        title: Custom title
        x_label: Custom x-axis label
        figsize: Figure size (width, height)
        dpi: Figure resolution for output
        
    Returns:
        dict: Summary containing file path and statistics
    """
    provider = BeastDataProvider()
    county_norm = provider.normalize_county_name(county)
    
    if output_dir is None:
        output_dir = provider.output_dir / county_norm
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if years is None:
        years = list(range(2019, 2025))
    
    cdir = provider.output_dir / county_norm
    files = [cdir / f"beast_changepoints_WY{y}.csv" for y in years]
    
    data = []
    valid_years = []
    
    for y, f in zip(years, files):
        if f.exists():
            d = pd.read_csv(f)
            if column in d.columns:
                vals = d[column].dropna().astype(float).values
                if vals.size:
                    data.append(vals)
                    valid_years.append(y)
    
    if not data:
        return {
            "status": "error",
            "message": f"No data for {county_norm} (column: {column})",
            "file_path": None
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        bp = ax.boxplot(data, vert=False, labels=[str(y) for y in valid_years], 
                       patch_artist=True, showfliers=False, whis=(0, 100))
    except TypeError:
        bp = ax.boxplot(data, labels=[str(y) for y in valid_years], 
                       patch_artist=True, showfliers=False, whis=(0, 100), vert=False)
    
    for patch in bp["boxes"]:
        patch.set(facecolor="#dddddd", alpha=0.85, edgecolor="#aaaaaa")
    
    means = [np.mean(d) for d in data]
    ax.scatter(means, range(1, len(data) + 1), color="black", marker="o", s=50, zorder=5)
    
    ax.set_xlabel(x_label if x_label else column.replace("_", " ").title())
    ax.set_ylabel("Water Year")
    
    if title is None:
        title = f"{county_norm}: {column.replace('_', ' ').title()} by Year (grey=min..max; \u2022=mean)"
    ax.set_title(title)
    
    fig.tight_layout()
    
    out = output_dir / f"boxplot_{column}_by_year.png"
    fig.savefig(out, dpi=dpi)
    plt.show()
    plt.close(fig)
    
    stats = {
        f"y{y}": {
            "n_parcels": len(d),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "min": float(np.min(d)),
            "max": float(np.max(d)),
            "std": float(np.std(d))
        }
        for y, d in zip(valid_years, data)
    }
    
    return {
        "status": "success",
        "file_path": str(out),
        "county": county,
        "years": valid_years,
        "n_years": len(valid_years),
        "column": column,
        "stats": stats
    }