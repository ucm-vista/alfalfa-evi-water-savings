"""Marginal ET per late cutting — grouped bar chart.

Shows that each late cutting costs substantial water (ac-ft/acre),
with southern counties paying more per cut than northern ones.
Bars grouped by county (N→S) and colored by cutting order (1st/2nd/3rd)
using a sequential blue palette to reinforce diminishing returns.
"""

import ast
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data_providers.spatial_provider import COUNTY_ORDER
from ...utils.units import mm_to_acft_per_acre
from .multicounty_parcel_scatter_plot import _style_axes_full_border


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _explode_late_cycle_et(df: pd.DataFrame) -> pd.DataFrame:
    """Parse late_cycle_et_mm_list and explode to long format.

    Returns a DataFrame with columns:
        UniqueID, county, WY, cutting_order (1-based), et_mm
    """
    rows = []
    for _, r in df.iterrows():
        raw = r.get("late_cycle_et_mm_list", "[]")
        if isinstance(raw, str):
            try:
                vals = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
        elif isinstance(raw, list):
            vals = raw
        else:
            continue
        for i, v in enumerate(vals, start=1):
            if pd.notna(v):
                rows.append({
                    "UniqueID": r["UniqueID"],
                    "county": r["county"],
                    "WY": r["WY"],
                    "cutting_order": i,
                    "et_mm": float(v),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main chart function
# ---------------------------------------------------------------------------

def marginal_et_per_cutting_plot(
    df: pd.DataFrame,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Grouped bar chart of mean ET per late cutting by county and order.

    Args:
        df: DataFrame from late_cut_base_parcel_year.csv with columns
            UniqueID, county, WY, late_cycle_et_mm_list.
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    # --- Explode to long format ---
    long = _explode_late_cycle_et(df)
    long = long[(long["et_mm"] > 0) & (long["cutting_order"] <= 3)]
    long["et_acft_per_acre"] = mm_to_acft_per_acre(long["et_mm"])

    # --- Aggregate: mean and SEM per county × cutting order ---
    agg = (
        long.groupby(["county", "cutting_order"])["et_acft_per_acre"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Order counties N→S
    present = agg["county"].unique().tolist()
    counties = [c for c in COUNTY_ORDER if c in present]
    others = sorted([c for c in present if c not in counties])
    counties = counties + others

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 5.5))

    orders = [1, 2, 3]
    labels = ["1st late cut", "2nd late cut", "3rd late cut"]
    blues = ["#08519c", "#3182bd", "#9ecae1"]  # dark → light blue
    n_groups = len(counties)
    bar_width = 0.25
    x = np.arange(n_groups)

    for j, (order, label, color) in enumerate(zip(orders, labels, blues)):
        sub = agg[agg["cutting_order"] == order].set_index("county")
        means = [sub.loc[c, "mean"] if c in sub.index else 0 for c in counties]
        sds = [sub.loc[c, "std"] if c in sub.index else 0 for c in counties]
        counts = [int(sub.loc[c, "count"]) if c in sub.index else 0 for c in counties]
        positions = x + (j - 1) * bar_width

        bars = ax.bar(
            positions, means, bar_width,
            yerr=sds, capsize=3,
            label=label, color=color, edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 1.0},
        )

        # n= annotations above each bar cluster (only for 1st cut to avoid clutter)
        if order == 1:
            for pos, cnt in zip(positions, counts):
                if cnt > 0:
                    ax.text(
                        pos + bar_width, ax.get_ylim()[1] * 0.01,
                        f"n={cnt}", ha="center", va="bottom",
                        fontsize=6, color="gray", rotation=90,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(counties, rotation=35, ha="right")
    ax.set_ylabel("Mean ET per cutting (ac-ft/acre)")
    ax.set_title(
        "Mean ET per late cutting by county and cutting order\n"
        "(WYs 2019\u20132024)"
    )
    ax.legend(frameon=False, loc="upper left")
    _style_axes_full_border(ax)
    fig.tight_layout()

    # --- Summary ---
    summary = {
        "n_counties": len(counties),
        "n_observations": len(long),
        "counties": counties,
        "overall_means_acft": {
            f"order_{o}": float(
                long.loc[long["cutting_order"] == o, "et_acft_per_acre"].mean()
            )
            for o in orders
            if (long["cutting_order"] == o).any()
        },
    }

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        summary["outfile"] = str(outfile)
        print(f"Marginal ET per cutting plot saved: {outfile}")

    return fig, ax, summary
