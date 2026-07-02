"""Bar plot by county.

Source: alfalfa_evi_jovyan.py line 17357
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from es_analysis.utils.helper import norm_county_name

COUNTY_ORDER = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]


def _norm_county_name(name: str) -> str:
    """Normalize county name."""
    s = str(name).replace("_", " ").strip()
    return " ".join(s.split()).title()


def _style_axes_black_border(ax: plt.Axes):
    """Style axes with black borders."""
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1.0)
    ax.tick_params(colors="black")


def bar_by_county_plot(
    df: pd.DataFrame,
    wy_start: int,
    wy_end: int,
    value_col: str,
    ylabel: str,
    title: str,
    counties: Optional[Iterable[str]] = None,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create bar plot by county.

    Args:
        df: Input DataFrame with columns "WY", "county", and value_col.
        wy_start: Start water year.
        wy_end: End water year.
        value_col: Column name for values to plot.
        ylabel: Label for Y-axis.
        title: Plot title.
        counties: List of counties to include. If None, uses COUNTY_ORDER.
        outfile: Output path for saving figure. If None, figure is not saved.

    Returns:
        Tuple of (figure, axes, summary_dict).

    Summary dict contains:
        - wy_start: Start water year.
        - wy_end: End water year.
        - value_col: Column plotted.
        - counties: List of counties included.
        - num_counties: Number of counties with data.
        - num_missing: Number of records with missing values.
        - mean_value: Mean value across all records.
        - outfile: Path where figure was saved (None if not saved).
    """
    d = df[df["WY"].between(wy_start, wy_end)].copy()
    
    num_total = len(d)
    num_missing = d[value_col].isna().sum()
    d = d.dropna(subset=[value_col])
    
    num_valid = len(d)
    mean_value = float(d[value_col].mean()) if num_valid > 0 else np.nan

    if counties is None:
        counties_use = COUNTY_ORDER
    else:
        counties_use = [_norm_county_name(c) for c in counties]

    d = d[d["county"].isin(counties_use)].copy()

    if d.empty:
        print("No data to plot after filtering.")
        return None, None, {}

    county_year = (
        d.groupby(["county", "WY"])[value_col].mean().reset_index()
    )
    out = (
        county_year.groupby("county")[value_col].mean()
        .reindex([c for c in counties_use if c in county_year["county"].unique()])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(out))
    ax.bar(x, out[value_col].to_numpy(dtype=float))

    ax.set_xticks(x)
    ax.set_xticklabels(out["county"].astype(str).tolist(), rotation=35, ha="right")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _style_axes_black_border(ax)
    fig.tight_layout()

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
        plt.show()
        plt.close(fig)

    summary = {
        "wy_start": wy_start,
        "wy_end": wy_end,
        "value_col": value_col,
        "counties": counties_use,
        "num_counties": len(out),
        "num_total": num_total,
        "num_missing": num_missing,
        "num_valid": num_valid,
        "mean_value": mean_value,
        "outfile": str(outfile) if outfile else None,
    }
    print(f"Bar by county plot: {len(out)} counties, mean={mean_value:.3g}")
    return fig, ax, summary