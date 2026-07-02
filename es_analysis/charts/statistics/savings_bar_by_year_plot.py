"""Late-water savings bar plot by water year.

Produces a bar chart of mean savings intensity (e.g. ac-ft/acre or
mm depth) grouped by water year for a given cap scenario.

Source: alfalfa_evi_jovyan.py lines 19894-19925
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_axes_black_border(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1.0)
    ax.tick_params(colors="black")


# ---------------------------------------------------------------------------
# Main chart function
# ---------------------------------------------------------------------------

def savings_bar_by_year_plot(
    df: pd.DataFrame,
    wy_start: int,
    wy_end: int,
    value_col: str = "water_saved",
    ylabel: str = "Water saved (ac-ft/acre)",
    title: str = "Water saving intensity by WY",
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create a bar chart of savings by water year.

    Args:
        df: Summary DataFrame from make_savings_summary() with
            columns "WY" and value_col.
        wy_start: Start water year.
        wy_end: End water year.
        value_col: Column with savings values to plot.
        ylabel: Y-axis label.
        title: Plot title.
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    d = df[df["WY"].between(wy_start, wy_end)].copy()
    num_total = len(d)
    num_missing = int(d[value_col].isna().sum())
    d = d.dropna(subset=[value_col])

    out = (
        d.groupby("WY")[value_col]
        .mean()
        .reset_index()
        .sort_values("WY")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(out))
    ax.bar(x, out[value_col].to_numpy(dtype=float))

    ax.set_xticks(x)
    ax.set_xticklabels(
        out["WY"].astype(int).astype(str).tolist(), rotation=0
    )

    ax.set_xlabel("Water year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _style_axes_black_border(ax)
    fig.tight_layout()

    mean_value = float(d[value_col].mean()) if len(d) > 0 else np.nan

    summary = {
        "wy_start": wy_start,
        "wy_end": wy_end,
        "value_col": value_col,
        "num_years": len(out),
        "num_total": num_total,
        "num_missing": num_missing,
        "mean_value": mean_value,
    }

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        summary["outfile"] = str(outfile)
        print(f"Savings bar by year saved: {outfile}")

    print(
        f"Savings bar by year: {len(out)} years, "
        f"mean={mean_value:.3g}"
    )
    return fig, ax, summary
