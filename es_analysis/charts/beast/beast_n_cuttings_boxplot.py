"""
BEAST number of cuttings boxplot
Horizontal boxplot showing distribution of cuttings per parcel across years with mean markers.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from es_analysis.data_providers import BeastDataProvider


def plot_beast_n_cuttings_boxplot(
    county: str,
    years: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: str = "Number of Cuttings",
    figsize: tuple = (10, 6),
    dpi: int = 150
) -> dict:
    """
    Draw per-year horizontal boxplots for n_cuttings.
    
    Args:
        county: County name (e.g., 'Fresno')
        years: List of water years to include (if None, detects from disk)
        output_dir: Directory to save plot
        title: Custom title (if None, uses default)
        x_label: Label for x-axis
        figsize: Figure size (width, height)
        dpi: Figure resolution for output
        
    Returns:
        dict: Summary containing file path and metadata
    """
    provider = BeastDataProvider()
    county_norm = provider.normalize_county_name(county)
    
    if output_dir is None:
        output_dir = provider.output_dir / county_norm
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if years is None:
        years = provider.detect_years_on_disk(county_norm)
    
    if not years:
        return {
            "status": "error",
            "message": f"No years found for county={county}",
            "file_path": None
        }
    
    data, valid_years = provider.gather_year_values(county_norm, years, column="n_cuttings")
    
    if not data:
        return {
            "status": "error",
            "message": f"No seasonal CSVs for {county_norm} in {years}",
            "file_path": None
        }
    
    labels = [str(y) for y in valid_years]
    out_path = output_dir / "boxplot_n_cuttings.png"
    
    if title is None:
        title = f"{county_norm}: Yearly Cutting Distribution (grey=min..max; \u2022=mean)"
    
    fig_h = max(5, 0.4 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    
    kw = dict(vert=False, patch_artist=True, showfliers=False, whis=(0, 100))
    try:
        bp = ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, **kw)
    
    for patch in bp["boxes"]:
        patch.set(facecolor="#dddddd", alpha=0.85, edgecolor="#aaaaaa")
    
    means = [np.mean(d) for d in data]
    ax.scatter(means, range(1, len(data) + 1), color="black", marker="o", s=50, zorder=5)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel("Water Year")
    ax.set_title(title)
    fig.tight_layout()
    
    fig.savefig(out_path, dpi=dpi)
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
        "file_path": str(out_path),
        "county": county,
        "years": valid_years,
        "n_years": len(valid_years),
        "stats": stats
    }