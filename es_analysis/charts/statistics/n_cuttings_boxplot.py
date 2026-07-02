"""Boxplot of number of cuttings per year for a county.

Source: alfalfa_evi_jovyan.py line 6421
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from es_analysis.utils.helper import norm_county_name
from es_analysis.data_providers.config import BEAST_OUT_ROOT, config


def _norm_county_name(name: str) -> str:
    """Normalize county name."""
    s = str(name).replace("_", " ").strip()
    return " ".join(s.split()).title()


def _detect_years_on_disk(county: str) -> List[int]:
    """Find years available for a county."""
    years = []
    for f in (BEAST_OUT_ROOT / _norm_county_name(county)).glob("beast_seasonal_cuts_WY*.csv"):
        tail = f.stem.replace("beast_seasonal_cuts_WY", "")
        yr = "".join(ch for ch in tail if ch.isdigit())
        if len(yr) >= 4:
            years.append(int(yr[:4]))
    return sorted(set(years))


def _gather_year_values(county: str, years: List[int], column: str = "n_cuttings") -> Tuple[List[np.ndarray], List[int]]:
    """Gather values for given county, years, and column."""
    data = []
    filtered_years = []
    for y in years:
        f = BEAST_OUT_ROOT / _norm_county_name(county) / f"beast_seasonal_cuts_WY{y}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if column not in df.columns:
            continue
        vals = pd.to_numeric(df[column], errors="coerce").dropna()
        if column == "n_cuttings" and config.min_cuttings > 0:
            vals = vals[vals >= config.min_cuttings]
        vals = vals.to_numpy()
        if vals.size > 0:
            data.append(vals)
            filtered_years.append(y)
    return data, filtered_years


def _boxplot_horizontal(data: List[np.ndarray], labels: List[str], title: str, ylabel: str, outfile: Path) -> Tuple[plt.Figure, plt.Axes]:
    """Create horizontal boxplot.

    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.7)))

    positions = list(range(len(data), 0, -1))
    bp = ax.boxplot(data, vert=False, positions=positions, patch_artist=True, widths=0.6)

    cmap = plt.cm.tab20.colors
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap[i % len(cmap)])
        patch.set_alpha(0.60)

    for patch in bp["boxes"]:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    means = [float(np.mean(arr)) for arr in data]
    ax.scatter(means, positions, s=30, c="black", marker="o", zorder=5, label="mean")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylabel("Year")
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.15)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
    legend.get_texts()[0].set_text(f"grey=min..max; •=mean (n={sum(len(arr) for arr in data)})")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.show()
    plt.close(fig)

    return fig, ax


def n_cuttings_boxplot(
    county: str,
    years: Optional[List[int]] = None,
    column: str = "n_cuttings",
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create boxplot of number of cuttings per year for a county.

    Args:
        county: County name.
        years: List of years to include. If None, auto-detects from disk.
        column: Column name to plot (default: "n_cuttings").
        outfile: Output path for saving figure. If None, auto-generates.

    Returns:
        Tuple of (figure, axes, summary_dict).

    Summary dict contains:
        - county: County name.
        - years: List of years included.
        - column: Column plotted.
        - num_parcels: Total number of parcels across all years.
        - num_years: Number of years with data.
        - mean_value: Overall mean value across all parcels.
        - outfile: Path where figure was saved.
    """
    county_norm = _norm_county_name(county)
    if years is None:
        years = _detect_years_on_disk(county_norm)
    data, filtered_years = _gather_year_values(county_norm, years, column=column)

    if not data:
        print(f"No seasonal CSVs for {county_norm} in {years}.")
        return None, None, {}

    labels = [str(y) for y in filtered_years]

    if outfile is None:
        outfile = BEAST_OUT_ROOT / county_norm / f"boxplot_{column}.png"

    ylabel = "Cuttings" if column == "n_cuttings" else "Seasonal change points"
    title = f"{county_norm}: Yearly {ylabel} Distribution (grey=min..max; •=mean)"

    fig, ax = _boxplot_horizontal(data, labels, title, ylabel, outfile)

    total_parcels = sum(len(arr) for arr in data)
    mean_value = float(np.mean(np.concatenate(data))) if data else np.nan

    summary = {
        "county": county_norm,
        "years": filtered_years,
        "column": column,
        "num_parcels": total_parcels,
        "num_years": len(filtered_years),
        "mean_value": mean_value,
        "outfile": str(outfile),
    }
    print(f"N cuttings boxplot:county={county_norm}, years={filtered_years}, parcels={total_parcels}")
    return fig, ax, summary