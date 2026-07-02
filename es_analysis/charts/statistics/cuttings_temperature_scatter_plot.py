"""Simple per-parcel scatter: cuttings vs. temperature, colored by county.

Each point is one parcel-year (all water years pooled). Two panels:
    (a) cuttings vs. mean air temperature
    (b) cuttings vs. max  air temperature
Points are colored by county. Companion to ``cuttings_temperature_quadrant_plot.py``;
it reuses the same cached ``cuttings_temperature`` parcel-year data (mean Daymet
temperature over each parcel's cut-cycle segments).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.utils.publication_style import (
    DOUBLE_COL_WIDTH,
    apply_style,
    save_pub_figure,
    add_panel_label,
)
from es_analysis.charts.statistics.cuttings_temperature_quadrant_plot import (
    _build_or_load_temp_df,
    _style_axes_black_border,
)

# (parcel-year column, x-axis label) for the two panels
_COLUMNS = [
    ("tmean_mean", "Mean air temperature (°C)"),
    ("tmax_mean", "Max air temperature (°C)"),
]


def cuttings_temperature_scatter(
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
    wy_start: int = 2019,
    wy_end: int = 2024,
    jitter: float = 0.15,
) -> Tuple[plt.Figure, Dict]:
    """Per-parcel-year scatter of cuttings vs. temperature, colored by county.

    Args:
        df: Per-parcel-year frame with ``county``, ``WY``, ``n_cuttings``,
            ``tmax_mean``, ``tmean_mean`` (from
            :func:`cuttings_temperature_quadrant_plot._build_or_load_temp_df`).
        out_dir: Output directory (``png/`` + ``pdf/`` subdirs created).
        wy_start: First water year (inclusive).
        wy_end: Last water year (inclusive).
        jitter: Uniform +/- vertical jitter on the integer cutting count to reduce
            overplotting (0 disables). Purely visual; stats use the raw values.

    Returns:
        Tuple of (figure, summary_dict).
    """
    apply_style()
    rng = np.random.default_rng(0)

    d = df[df["WY"].between(wy_start, wy_end)].copy()
    d = d.dropna(subset=["county", "n_cuttings", "tmax_mean", "tmean_mean"])

    counties = [c for c in COUNTY_ORDER if c in d["county"].unique()]
    cmap = mpl.colormaps.get_cmap("tab10")
    county_color = {c: cmap(i % 10) for i, c in enumerate(counties)}

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH * 1.5, DOUBLE_COL_WIDTH * 0.62), sharey=True,
    )

    summary: Dict = {"counties": counties, "n_points": int(len(d)), "corr": {}}
    letters = ["a", "b"]

    for col, (xkey, xlabel) in enumerate(_COLUMNS):
        ax = axes[col]

        # one scatter call per county so the color legend stays clean
        for c in counties:
            sub = d[d["county"] == c]
            if sub.empty:
                continue
            yv = sub["n_cuttings"].to_numpy(float)
            if jitter:
                yv = yv + rng.uniform(-jitter, jitter, size=yv.shape)
            ax.scatter(
                sub[xkey].to_numpy(float), yv,
                s=12, facecolor=county_color[c], edgecolor="none",
                alpha=0.40, zorder=3,
            )

        xall = d[xkey].to_numpy(float)
        yall = d["n_cuttings"].to_numpy(float)
        m = np.isfinite(xall) & np.isfinite(yall)

        ax.text(0.025, 0.97, f"n = {int(m.sum())}", transform=ax.transAxes,
                va="top", ha="left", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"), zorder=5)

        ax.set_xlabel(xlabel, fontsize=9)
        if col == 0:
            ax.set_ylabel("Number of cuttings per parcel-year", fontsize=9)
        _style_axes_black_border(ax)
        add_panel_label(ax, letters[col])

        corr = float(np.corrcoef(xall[m], yall[m])[0, 1]) if m.sum() > 1 else float("nan")
        summary["corr"][letters[col]] = corr

    fig.suptitle(
        f"Per-parcel cuttings vs. temperature, all counties (WY{wy_start}–{wy_end})",
        fontsize=11, fontweight="bold", y=1.02,
    )

    county_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=county_color[c], markeredgecolor="none", label=c)
        for c in counties
    ]
    fig.legend(handles=county_handles, title="County", frameon=False,
               loc="center left", bbox_to_anchor=(0.88, 0.5), fontsize=8,
               title_fontsize=9, labelspacing=0.3, handletextpad=0.4)

    fig.tight_layout(rect=(0, 0, 0.87, 0.96))

    if out_dir is not None:
        save_pub_figure(fig, "cuttings_temperature_scatter", out_dir)
        summary["out_dir"] = str(out_dir)

    print(f"  cuttings_temperature_scatter: {len(d)} parcel-year pts; "
          f"corr={summary['corr']}")
    return fig, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    df = _build_or_load_temp_df("alfalfa_run_6")
    out = (
        Path(__file__).parent.parent.parent
        / "output" / "figures" / "alfalfa_run_6"
        / "cuttings_analysis" / "all_counties"
    )
    fig, summary = cuttings_temperature_scatter(df, out_dir=out)
    print(f"Summary: {summary}")
