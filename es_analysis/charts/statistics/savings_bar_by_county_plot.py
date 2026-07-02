"""Late-water savings bar plot by county.

Produces a bar chart of mean savings intensity (e.g. ac-ft/acre or
mm depth) grouped by county for a given cap scenario.

Source: alfalfa_evi_jovyan.py lines 19928-19939
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data_providers.evi_provider import normalize_county_name
from ...data_providers.spatial_provider import COUNTY_ORDER


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

def savings_bar_by_county_plot(
    df: pd.DataFrame,
    wy_start: int,
    wy_end: int,
    value_col: str = "water_saved",
    ylabel: str = "Water saved (ac-ft/acre)",
    title: str = "Mean water saving intensity by county",
    counties: Optional[Iterable[str]] = None,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create a bar chart of savings by county.

    The chart shows the mean of county-year values across all WYs
    for each county.

    Args:
        df: Summary DataFrame from make_savings_summary() with
            columns "WY", "county", and value_col.
        wy_start: Start water year.
        wy_end: End water year.
        value_col: Column with savings values to plot.
        ylabel: Y-axis label.
        title: Plot title.
        counties: Optional list of counties to include.
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    d = df[df["WY"].between(wy_start, wy_end)].copy()
    d = d.dropna(subset=[value_col])

    if counties is None:
        counties_use = list(COUNTY_ORDER)
    else:
        counties_use = [normalize_county_name(c) for c in counties]

    d = d[d["county"].isin(counties_use)].copy()

    if d.empty:
        print("No data to plot after filtering.")
        return None, None, {}

    # County-year means -> then mean across years per county
    county_year = (
        d.groupby(["county", "WY"])[value_col].mean().reset_index()
    )
    out = (
        county_year.groupby("county")[value_col]
        .mean()
        .reindex([
            c for c in counties_use
            if c in county_year["county"].unique()
        ])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(out))
    ax.bar(x, out[value_col].to_numpy(dtype=float))

    ax.set_xticks(x)
    ax.set_xticklabels(
        out["county"].astype(str).tolist(), rotation=35, ha="right"
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _style_axes_black_border(ax)
    fig.tight_layout()

    mean_value = float(out[value_col].mean()) if len(out) > 0 else np.nan

    summary = {
        "wy_start": wy_start,
        "wy_end": wy_end,
        "value_col": value_col,
        "num_counties": len(out),
        "mean_value": mean_value,
    }

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        summary["outfile"] = str(outfile)
        print(f"Savings bar by county saved: {outfile}")

    print(
        f"Savings bar by county: {len(out)} counties, "
        f"mean={mean_value:.3g}"
    )
    return fig, ax, summary
