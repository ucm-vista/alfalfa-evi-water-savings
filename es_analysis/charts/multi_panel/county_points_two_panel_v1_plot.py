"""
County Points Two Panel Plot (v1)

Two-panel scatter plot showing county-year relationships.
Left panel: Monthly trends (Daymet/GDD vs cuttings).
Right panel: Annual summaries (ET vs cuttings).

Source: alfalfa_evi_jovyan.py line 12324
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from es_analysis.data_providers.statistics_provider import (
    aggregate_to_county_year_points,
    aggregate_to_county_overall_points,
    _compute_regression_stats,
    _style_axes_full_border,
    _add_stats_text_top_left,
    _county_color_map,
    _marker_for_wy,
    COUNTY_ORDER,
)


def plot_county_points_two_panel(
    df_points: pd.DataFrame,
    *,
    daymet_var: str,
    cut_metric: str,
    mode: str,
    title_suffix: str = "",
    county_order: Optional[List[str]] = None,
    marker_size: float = 70.0,
    alpha: float = 0.85,
    edge_color: str = "0.25",
    edge_lw: float = 0.9,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Create two-panel scatter plot of county-level points.

    Left panel: Daymet variable (GDD or temperature) vs cuttings
    Right panel: Cumulative ET vs cuttings

    Parameters
    ----------
    df_points : pd.DataFrame
        County-level aggregated data. Must contain columns: county,
        {cut_metric}, {daymet_var}_mean, et_cum_minET_to_last_cut毫米
        If mode="county_year", also include WY column.
    daymet_var : str
        Daymet variable name (e.g., "gdd5", "tmax", "tmin")
    cut_metric : str
        Cutting metric name (e.g., "n_cp_season", "n_cuttings")
    mode : str
        Either "county_year" (with water year markers) or "county_overall"
    title_suffix : str, optional
        Additional text to append to title
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
    dm_col = f"{daymet_var}_mean"
    dfp = df_points.dropna(subset=["county", cut_metric, dm_col, "et_cum_minET_to_last_cut_mm"]).copy()
    if dfp.empty:
        raise ValueError("No data to plot after dropping NaNs.")

    if county_order is None:
        county_order = list(COUNTY_ORDER)

    keep = [c for c in county_order if c in dfp["county"].unique()]
    if keep:
        dfp["county"] = pd.Categorical(dfp["county"], categories=keep, ordered=True)
        dfp = dfp.sort_values("county")

    counties = dfp["county"].astype(str).unique().tolist()
    county_colors = _county_color_map(counties)

    x_dm = dfp[dm_col].to_numpy(float)
    x_et = dfp["et_cum_minET_to_last_cut_mm"].to_numpy(float)
    y = dfp[cut_metric].to_numpy(float)
    stats_dm = _compute_regression_stats(x_dm, y)
    stats_et = _compute_regression_stats(x_et, y)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)

    ax = axes[0]
    if mode == "county_year":
        for c in counties:
            sub = dfp[dfp["county"].astype(str) == c]
            for wy in sorted(sub["WY"].unique()):
                s2 = sub[sub["WY"] == wy]
                ax.scatter(
                    s2[dm_col], s2[cut_metric],
                    s=marker_size,
                    marker=_marker_for_wy(int(wy)),
                    facecolors=[county_colors[c]],
                    edgecolors=edge_color,
                    linewidths=edge_lw,
                    alpha=alpha,
                )
    else:
        for c in counties:
            sub = dfp[dfp["county"].astype(str) == c]
            ax.scatter(
                sub[dm_col], sub[cut_metric],
                s=marker_size,
                marker="o",
                facecolors=[county_colors[c]],
                edgecolors=edge_color,
                linewidths=edge_lw,
                alpha=alpha,
            )

    if daymet_var.lower() == "gdd5":
        ax.set_xlabel("Cumulative GDD (base 5°C) from tmin/tmax\n(parcel ET/GDD window)")
        ax.set_title(f"Number of cuttings vs GDD5{title_suffix}")
    else:
        ax.set_xlabel(f"Mean Daymet {daymet_var}")
        ax.set_title(f"Number of cuttings vs {daymet_var}{title_suffix}")

    ax.set_ylabel("Number of cuttings")

    if np.isfinite(stats_dm["slope"]) and np.isfinite(stats_dm["intercept"]):
        xs = np.linspace(np.nanmin(x_dm), np.nanmax(x_dm), 100)
        ax.plot(xs, stats_dm["slope"] * xs + stats_dm["intercept"], linestyle="--", linewidth=1.2, color="black", alpha=0.7)

    _add_stats_text_top_left(ax, stats_dm)
    _style_axes_full_border(ax)

    ax2 = axes[1]
    if mode == "county_year":
        for c in counties:
            sub = dfp[dfp["county"].astype(str) == c]
            for wy in sorted(sub["WY"].unique()):
                s2 = sub[sub["WY"] == wy]
                ax2.scatter(
                    s2["et_cum_minET_to_last_cut_mm"], s2[cut_metric],
                    s=marker_size,
                    marker=_marker_for_wy(int(wy)),
                    facecolors=[county_colors[c]],
                    edgecolors=edge_color,
                    linewidths=edge_lw,
                    alpha=alpha,
                )
    else:
        for c in counties:
            sub = dfp[dfp["county"].astype(str) == c]
            ax2.scatter(
                sub["et_cum_minET_to_last_cut_mm"], sub[cut_metric],
                s=marker_size,
                marker="o",
                facecolors=[county_colors[c]],
                edgecolors=edge_color,
                linewidths=edge_lw,
                alpha=alpha,
            )

    ax2.set_xlabel("Cumulative OpenET ET (mm)\n(mean over included parcels/years)")
    ax2.set_title(f"Number of cuttings vs ET{title_suffix}")

    if np.isfinite(stats_et["slope"]) and np.isfinite(stats_et["intercept"]):
        xs2 = np.linspace(np.nanmin(x_et), np.nanmax(x_et), 100)
        ax2.plot(xs2, stats_et["slope"] * xs2 + stats_et["intercept"], linestyle="--", linewidth=1.2, color="black", alpha=0.7)

    _add_stats_text_top_left(ax2, stats_et)
    _style_axes_full_border(ax2)

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
    leg1 = ax2.legend(
        handles=county_handles, title="County", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.65)
    )
    ax2.add_artist(leg1)

    if mode == "county_year":
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
        ax2.legend(
            handles=year_handles, title="Water year (marker)", frameon=False,
            loc="center left", bbox_to_anchor=(1.02, 0.15)
        )

    fig.tight_layout()
    plt.show()
    return dfp, fig


def run_all_counties_county_year_scatter(
    *,
    df_parcel_year: pd.DataFrame,
    daymet_var: str,
    cut_metric: str = "n_cp_season",
    wy_start: int = 2019,
    wy_end: int = 2024,
    counties: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Run two-panel scatter plot for all counties, county-year mode.

    Parameters
    ----------
    df_parcel_year : pd.DataFrame
        Parcel-year level data
    daymet_var : str
        Daymet variable name
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

    df_points = aggregate_to_county_year_points(
        df_parcel_year,
        daymet_var=daymet_var,
        cut_metric=cut_metric,
        counties=counties,
        wy_start=wy_start,
        wy_end=wy_end,
    )
    title_suffix = f"\nCounty-year means (WY {wy_start}–{wy_end})"
    return plot_county_points_two_panel(
        df_points,
        daymet_var=daymet_var,
        cut_metric=cut_metric,
        mode="county_year",
        title_suffix=title_suffix,
        county_order=list(COUNTY_ORDER),
    )


def run_all_counties_county_overall_scatter(
    *,
    df_parcel_year: pd.DataFrame,
    daymet_var: str,
    cut_metric: str = "n_cp_season",
    wy_start: int = 2019,
    wy_end: int = 2024,
    counties: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Run two-panel scatter plot for all counties, overall mode.

    Parameters
    ----------
    df_parcel_year : pd.DataFrame
        Parcel-year level data
    daymet_var : str
        Daymet variable name
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
        County-level aggregated data
    fig : plt.Figure
        Matplotlib figure object
    """
    if counties is None:
        counties = list(COUNTY_ORDER)

    df_points = aggregate_to_county_overall_points(
        df_parcel_year,
        daymet_var=daymet_var,
        cut_metric=cut_metric,
        counties=counties,
        wy_start=wy_start,
        wy_end=wy_end,
    )
    title_suffix = f"\nCounty overall mean (WY {wy_start}–{wy_end})"
    return plot_county_points_two_panel(
        df_points,
        daymet_var=daymet_var,
        cut_metric=cut_metric,
        mode="county_overall",
        title_suffix=title_suffix,
        county_order=list(COUNTY_ORDER),
    )