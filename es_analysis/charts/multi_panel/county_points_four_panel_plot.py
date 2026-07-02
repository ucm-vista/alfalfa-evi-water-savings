"""
County Points Four Panel Plot

Four-panel grid comprehensive county-level display.
- Top-left: GDD5 cumulative vs cuttings
- Top-right: GDD5 mean per cut vs cuttings
- Bottom-left: ET cumulative vs cuttings
- Bottom-right: ET mean per cut vs cuttings

Source: alfalfa_evi_jovyan.py line 13936
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from es_analysis.data_providers.statistics_provider import (
    aggregate_to_county_year_points_four,
    _compute_regression_stats,
    _style_axes_full_border,
    _add_stats_text_top_left,
    _county_color_map,
    _marker_for_wy,
    COUNTY_ORDER,
)


def plot_county_points_four_panel(
    df_points_four: pd.DataFrame,
    *,
    cut_metric: str,
    title_suffix: str = "",
    county_order: Optional[List[str]] = None,
    marker_size: float = 70.0,
    alpha: float = 0.85,
    edge_color: str = "0.25",
    edge_lw: float = 0.9,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Create four-panel scatter plot of county-level points.

    Four panels show different relationships between environmental metrics
    and cuttings:
    1. GDD5 cumulative vs cuttings
    2. GDD5 mean per cut vs cuttings
    3. ET cumulative vs cuttings
    4. ET mean per cut vs cuttings

    Parameters
    ----------
    df_points_four : pd.DataFrame
        County-year aggregated data. Must contain columns: county, WY,
        gdd5_cum, gdd5_mean_per_cut, et_cum_mm, et_mean_per_cut_mm,
        {cut_metric}
    cut_metric : str
        Cutting metric name (e.g., "n_cp_season", "n_cuttings")
    title_suffix : str, optional
        Additional text to append to titles
    county_order : list of str, optional
        Order of counties for display
    marker_size : float, default 70.0
        Size of scatter markers
    alpha : float, default 0.85
        Marker transparency
    edge_color : str, default "0.25"
        Edge color of markers
    edge_lw : float, default 0.9
        Edge line width

    Returns
    -------
    df_points_filtered : pd.DataFrame
        Filtered dataframe used for plotting
    fig : plt.Figure
        Matplotlib figure object
    """
    cols = ["gdd5_cum", "gdd5_mean_per_cut", "et_cum_mm", "et_mean_per_cut_mm"]
    dfp = df_points_four.dropna(subset=["county", "WY", cut_metric] + cols).copy()
    if dfp.empty:
        raise ValueError("No data to plot after dropping NaNs.")

    if county_order is None:
        county_order = list(COUNTY_ORDER)

    keep = [c for c in county_order if c in dfp["county"].unique()]
    if keep:
        dfp["county"] = pd.Categorical(dfp["county"], categories=keep, ordered=True)
        dfp = dfp.sort_values(["county", "WY"])

    counties = dfp["county"].astype(str).unique().tolist()
    county_colors = _county_color_map(counties)

    y_scatter = dfp[cut_metric].astype(int).to_numpy()
    y_reg = y_scatter.astype(float)

    panels = [
        ("gdd5_cum", "GDD5 cumulative (°C-day)\n(sum over union of cut segments)"),
        ("gdd5_mean_per_cut", "GDD5 mean per cut (°C-day/cut)\n(mean of per-cut segment sums)"),
        ("et_cum_mm", "ET cumulative (mm)\n(sum over union of cut segments)"),
        ("et_mean_per_cut_mm", "ET mean per cut (mm/cut)\n(mean of per-cut segment sums)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0), sharey=True)
    axes = axes.ravel()

    for ax, (xcol, xlabel) in zip(axes, panels):
        xvals = dfp[xcol].to_numpy(float)
        stats = _compute_regression_stats(xvals, y_reg)

        for c in counties:
            sub = dfp[dfp["county"].astype(str) == c]
            for wy in sorted(sub["WY"].unique()):
                s2 = sub[sub["WY"] == wy]
                ax.scatter(
                    s2[xcol].to_numpy(float),
                    s2[cut_metric].astype(int).to_numpy(),
                    s=marker_size,
                    marker=_marker_for_wy(int(wy)),
                    facecolors=[county_colors[c]],
                    edgecolors=edge_color,
                    linewidths=edge_lw,
                    alpha=alpha,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of cuttings (integer)")

        if np.isfinite(stats["slope"]) and np.isfinite(stats["intercept"]) and np.isfinite(xvals).any():
            xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
            ax.plot(xs, stats["slope"] * xs + stats["intercept"],
                    linestyle="--", linewidth=1.2, color="black", alpha=0.7)

        _add_stats_text_top_left(ax, stats)
        _style_axes_full_border(ax)

    axes[0].set_title(f"GDD5 cumulative vs cuttings{title_suffix}")
    axes[1].set_title(f"GDD5 mean-per-cut vs cuttings{title_suffix}")
    axes[2].set_title(f"ET cumulative vs cuttings{title_suffix}")
    axes[3].set_title(f"ET mean-per-cut vs cuttings{title_suffix}")

    axL = axes[3]
    county_handles = [
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=county_colors[c],
            markeredgecolor=edge_color,
            markeredgewidth=edge_lw,
            label=c
        )
        for c in counties
    ]
    leg1 = axL.legend(
        handles=county_handles, title="County", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.70)
    )
    axL.add_artist(leg1)

    years = sorted(dfp["WY"].unique().tolist())
    year_handles = [
        mpl.lines.Line2D(
            [0], [0], marker=_marker_for_wy(int(wy)), linestyle="",
            markerfacecolor="white",
            markeredgecolor=edge_color,
            markeredgewidth=edge_lw,
            label=f"WY {int(wy)}"
        )
        for wy in years
    ]
    axL.legend(
        handles=year_handles, title="Water year (marker)", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.20)
    )

    fig.tight_layout()
    plt.show()
    return dfp, fig


def run_four_panel_scatter(
    *,
    df_parcel_year: pd.DataFrame,
    cut_metric: str = "n_cp_season",
    wy_start: int = 2019,
    wy_end: int = 2024,
    counties: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Run four-panel scatter plot.

    Parameters
    ----------
    df_parcel_year : pd.DataFrame
        Parcel-year level data. Must contain columns: county, WY,
        gdd5_cum, gdd5_mean_per_cut, et_cum_mm, et_mean_per_cut_mm, {cut_metric}
    cut_metric : str, default "n_cp_season"
        Cutting metric name
    wy_start : int, default 2019
        Starting water year
    wy_end : int, default 2024
        Ending water year
    counties : list of str, optional
        Specific counties to include

    Returns
    -------
    df_points : pd.DataFrame
        County-year aggregated data
    fig : plt.Figure
        Matplotlib figure object
    """
    if counties is None:
        counties = list(COUNTY_ORDER)

    df_points = aggregate_to_county_year_points_four(
        df_parcel_year,
        counties=counties,
        wy_start=wy_start,
        wy_end=wy_end,
    )
    title_suffix = f"\nCounty-year means (WY {wy_start}–{wy_end})"
    return plot_county_points_four_panel(
        df_points,
        cut_metric=cut_metric,
        title_suffix=title_suffix,
        county_order=list(COUNTY_ORDER),
    )