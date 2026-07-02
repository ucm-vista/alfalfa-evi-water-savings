"""County late-season water budget — two-panel bar chart.

Left panel:  Mean late-season ET intensity (ac-ft/acre) per county, N→S.
Right panel: Total late-season ET volume (ac-ft) per county, N→S.

Shows both per-acre gradient and absolute magnitude of "sellable" water.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data_providers.spatial_provider import COUNTY_ORDER
from ...utils.units import mm_to_acft_per_acre
from .multicounty_parcel_scatter_plot import _county_color_map, _style_axes_full_border


def county_late_water_bar_plot(
    df: pd.DataFrame,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Two-panel bar chart of late-season water by county.

    Args:
        df: DataFrame from late_cut_base_parcel_year.csv with columns
            county, late_et_mm, late_et_acft.
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes_array, summary_dict).
    """
    work = df[df["late_et_mm"] > 0].copy()
    work["late_et_acft_per_acre"] = mm_to_acft_per_acre(work["late_et_mm"])

    # --- Aggregate per county ---
    agg = (
        work.groupby("county")
        .agg(
            mean_acft_per_acre=("late_et_acft_per_acre", "mean"),
            total_acft=("late_et_acft", "sum"),
            n_parcel_years=("late_et_mm", "count"),
        )
        .reset_index()
    )

    # Order counties N→S
    present = agg["county"].unique().tolist()
    counties = [c for c in COUNTY_ORDER if c in present]
    others = sorted([c for c in present if c not in counties])
    counties = counties + others

    agg = agg.set_index("county").loc[counties].reset_index()
    ccolors = _county_color_map(counties)
    bar_colors = [ccolors[c] for c in counties]

    grand_total = float(agg["total_acft"].sum())

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(counties))

    # Left: intensity
    ax1.bar(x, agg["mean_acft_per_acre"], color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(counties, rotation=35, ha="right")
    ax1.set_ylabel("Mean late-season ET (ac-ft/acre)")
    ax1.set_title("(a) Late-season ET intensity by county")
    _style_axes_full_border(ax1)

    # Right: total volume
    ax2.bar(x, agg["total_acft"], color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(counties, rotation=35, ha="right")
    ax2.set_ylabel("Total late-season ET (ac-ft)")
    ax2.set_title("(b) Total late-season ET volume by county")
    ax2.annotate(
        f"Total: {grand_total:,.0f} ac-ft",
        xy=(0.95, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, edgecolor="gray"),
    )
    _style_axes_full_border(ax2)

    fig.tight_layout()

    # --- Summary ---
    summary = {
        "n_counties": len(counties),
        "n_parcel_years": int(agg["n_parcel_years"].sum()),
        "grand_total_acft": grand_total,
        "counties": counties,
        "per_county": agg.set_index("county")[
            ["mean_acft_per_acre", "total_acft", "n_parcel_years"]
        ].to_dict(orient="index"),
    }

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        summary["outfile"] = str(outfile)
        print(f"County late water bar plot saved: {outfile}")

    return fig, np.array([ax1, ax2]), summary
