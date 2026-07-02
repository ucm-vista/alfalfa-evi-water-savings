"""
BEAST all counties boxplot
Boxplot showing distribution across all counties for comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from es_analysis.data_providers import BeastDataProvider


def plot_beast_all_counties_boxplot(
    counties: Optional[List[str]] = None,
    column: str = "n_cuttings",
    year: Optional[int] = None,
    output_dir: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    figsize: tuple = (12, 8),
    dpi: int = 150
) -> dict:
    """
    Draw a horizontal boxplot comparing all counties for a given year or all years combined.
    
    Args:
        counties: List of county names (if None, uses all available)
        column: Column to plot values from
        year: Specific water year (if None, aggregates all years)
        output_dir: Directory to save plot
        title: Custom title
        x_label: Custom x-axis label
        figsize: Figure size (width, height)
        dpi: Figure resolution for output
        
    Returns:
        dict: Summary containing file path and statistics
    """
    provider = BeastDataProvider()
    
    if counties is None:
        counties = provider.get_selected_counties()
    
    if output_dir is None:
        output_dir = provider.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    county_labels = []
    
    for county in counties:
        county_norm = provider.normalize_county_name(county)
        cdir = provider.output_dir / county_norm
        
        if year is not None:
            files = [cdir / f"beast_changepoints_WY{year}.csv"]
        else:
            files = sorted(cdir.glob("beast_changepoints_WY*.csv"))
        
        all_vals = []
        for f in files:
            if f.exists():
                d = pd.read_csv(f)
                if column in d.columns:
                    vals = d[column].dropna().astype(float).values
                    if vals.size:
                        all_vals.extend(vals)
        
        if all_vals:
            data.append(np.array(all_vals))
            county_labels.append(county_norm)
    
    if not data:
        return {
            "status": "error",
            "message": f"No data found for any counties (column: {column})",
            "file_path": None
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        bp = ax.boxplot(data, vert=False, labels=county_labels, 
                       patch_artist=True, showfliers=False, whis=(0, 100))
    except TypeError:
        bp = ax.boxplot(data, labels=county_labels, 
                       patch_artist=True, showfliers=False, whis=(0, 100), vert=False)
    
    colors = plt.cm.tab20.colors
    for i, patch in enumerate(bp["boxes"]):
        patch.set(facecolor=colors[i % len(colors)], alpha=0.7, edgecolor="#555555")
    
    means = [np.mean(d) for d in data]
    ax.scatter(means, range(1, len(data) + 1), color="black", marker="o", s=50, zorder=5)
    
    ax.set_xlabel(x_label if x_label else column.replace("_", " ").title())
    ax.set_ylabel("County")
    
    year_text = f"WY{year}" if year else "All Years"
    if title is None:
        title = f"All Counties: {column.replace('_', ' ').title()} - {year_text}"
    ax.set_title(title)
    
    fig.tight_layout()
    
    year_suffix = f"_WY{year}" if year else "_all_years"
    out = output_dir / f"boxplot_{column}_all_counties{year_suffix}.png"
    fig.savefig(out, dpi=dpi)
    plt.show()
    plt.close(fig)
    
    stats = {
        county_labels[i]: {
            "n_observations": len(d),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "min": float(np.min(d)),
            "max": float(np.max(d)),
            "std": float(np.std(d))
        }
        for i, d in enumerate(data)
    }
    
    return {
        "status": "success",
        "file_path": str(out),
        "n_counties": len(county_labels),
        "counties": county_labels,
        "column": column,
        "year": year,
        "stats": stats
    }