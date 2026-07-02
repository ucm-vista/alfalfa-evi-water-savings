"""
County Map and Ridgelines Plot

Spatial distribution of parcels with temporal ridgeline overlay.
- Left panel: County map with inset bar charts showing mean cuttings per water year
- Right panel: Ridgeline plots showing distribution of cuttings by county

Source: alfalfa_evi_jovyan.py line 15927
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

try:
    from scipy.stats import gaussian_kde
    HAS_GAUSSIAN_KDE = True
except ImportError:
    HAS_GAUSSIAN_KDE = False
    gaussian_kde = None

from es_analysis.data_providers.statistics_provider import COUNTY_ORDER
from es_analysis.data_providers.spatial_provider import load_county_boundaries


COUNTY_ORDER = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]

INSET_W = 0.060
INSET_H = 0.060
INSET_BAR_WIDTH = 0.85
INSET_XROT = 90
INSET_YTICKS_FIXED = [0, 5, 8, 10]

INSET_OFFSETS: Dict[str, Tuple[float, float]] = {
    "San Joaquin": (-0.030, 0.070),
    "Stanislaus":  (0.05, 0.075),
    "Merced":      (-0.07, 0.010),
    "Madera":      (0.025, 0.080),
    "Fresno":      (-0.065, 0.010),
    "Tulare":      (0.01, 0.010),
    "Kings":       (-0.075, 0.00),
    "Kern":        (0.00, 0.010),
    "Riverside":   (-0.065, 0.025),
    "Imperial":    (0.00, 0.025),
}

RIDGE_ALPHA = 0.05
RIDGE_EDGE_LW = 0.90


def _year_tick_labels(years: List[int]) -> List[str]:
    """
    Create year tick labels.

    Parameters
    ----------
    years : list of int
        List of years

    Returns
    -------
    list of str
        Label strings (first year full, others last 2 digits)
    """
    if not years:
        return []
    labels = [str(years[0])]
    labels += [str(y)[-2:] for y in years[1:]]
    return labels


def _build_year_color_map(years: List[int]) -> Dict[int, tuple]:
    """
    Create a color map for years.

    Parameters
    ----------
    years : list of int
        List of years

    Returns
    -------
    dict
        Dictionary mapping years to RGB tuples
    """
    years = sorted(list(years))
    cmap = mpl.colormaps.get_cmap("tab10")
    return {wy: cmap(i % 10) for i, wy in enumerate(years)}


def plot_county_map_and_ridgelines(
    gdf,
    wy_map_start: int,
    wy_map_end: Optional[int] = None,
    wy_ridge_start: Optional[int] = None,
    wy_ridge_end: Optional[int] = None,
    cut_metric: str = "n_cp_season",
    cut_min: Optional[float] = None,
    cut_max: Optional[float] = None,
    merge_years: bool = False,
):
    """
    Create map with inset bars + ridgelines plot.

    Left panel: County map with inset bar charts showing mean cuttings per water year.
    Right panel: Ridgeline plots showing distribution of cuttings by county.

    Parameters
    ----------
    gdf : GeoDataFrame or pd.DataFrame
        Spatial data or regular dataframe with columns: county, WY, {cut_metric}, geometry (if GeoDataFrame)
        If spatial data unavailable, only ridgeline panel will be shown.
    wy_map_start : int
        Starting water year for map aggregation
    wy_map_end : int, optional
        Ending water year for map aggregation (defaults to wy_map_start)
    wy_ridge_start : int, optional
        Starting water year for ridgeline data (defaults to wy_map_start)
    wy_ridge_end : int, optional
        Ending water year for ridgeline data (defaults to wy_map_end)
    cut_metric : str, default "n_cp_season"
        Cutting metric name (must be "n_cp_season" or "n_cuttings")
    cut_min : float, optional
        Minimum cut metric value filter for ridgelines
    cut_max : float, optional
        Maximum cut metric value filter for ridgelines
    merge_years : bool, default False
        If True, combine all years into single ridge per county

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object, or None if no data available

    Notes
    -----
    If geopandas is not available or spatial data is missing,
    only the ridgeline panel will be displayed.
    """
    if cut_metric not in {"n_cp_season", "n_cuttings"}:
        raise ValueError("cut_metric must be 'n_cp_season' or 'n_cuttings'.")

    wy_map_end = wy_map_end if wy_map_end is not None else wy_map_start
    if wy_ridge_start is None:
        wy_ridge_start = wy_map_start
    if wy_ridge_end is None:
        wy_ridge_end = wy_map_end

    if not HAS_GEOPANDAS or not isinstance(gdf, (gpd.GeoDataFrame, pd.DataFrame)):
        print("[warn] GeoPandas not available or invalid input data. Only ridgeline will be shown.")
        return _plot_ridgeline_only(
            gdf if isinstance(gdf, pd.DataFrame) else None,
            wy_ridge_start, wy_ridge_end, cut_metric, cut_min, cut_max, merge_years
        )

    data_map = gdf[gdf["WY"].between(wy_map_start, wy_map_end)].copy()
    data_map = data_map.dropna(subset=[cut_metric])
    if data_map.empty:
        raise ValueError(f"No map data for WYs {wy_map_start}–{wy_map_end} metric={cut_metric}.")

    tmp = (
        data_map.groupby(["county", "WY"])[cut_metric]
        .mean()
        .reset_index()
        .rename(columns={cut_metric: "mean_per_wy"})
    )

    counties_in_data = [c for c in COUNTY_ORDER if c in tmp["county"].unique()]
    if not counties_in_data:
        raise ValueError("No overlapping counties between data and COUNTY_ORDER.")

    tmp = tmp[tmp["county"].isin(counties_in_data)].copy()
    map_years = sorted(tmp["WY"].unique().tolist())
    if not map_years:
        raise ValueError("No map years found after filtering.")

    tmp["mean_int"] = tmp["mean_per_wy"].round().astype(int)
    grand = (
        tmp.groupby("county")["mean_per_wy"].mean().round().astype(int)
        .rename("grand_mean_int").reset_index().set_index("county")
        .reindex(counties_in_data)
    )
    pivot_int = (
        tmp.pivot(index="county", columns="WY", values="mean_int")
        .reindex(index=counties_in_data, columns=map_years)
    )

    all_vals = pivot_int.to_numpy(dtype=float)
    vmin = np.nanmin(all_vals) if np.isfinite(np.nanmin(all_vals)) else 0.0
    vmax = np.nanmax(all_vals) if np.isfinite(np.nanmax(all_vals)) else 1.0
    inset_y0 = 0
    inset_y1 = max(10, int(np.ceil(vmax)) + 1)

    inset_yticks = [t for t in INSET_YTICKS_FIXED if inset_y0 <= t <= inset_y1]
    year_colors = _build_year_color_map(map_years)

    data_ridge = gdf[gdf["WY"].between(wy_ridge_start, wy_ridge_end)].copy()
    data_ridge = data_ridge.dropna(subset=[cut_metric])
    if cut_min is not None:
        data_ridge = data_ridge[data_ridge[cut_metric] >= cut_min]
    if cut_max is not None:
        data_ridge = data_ridge[data_ridge[cut_metric] <= cut_max]
    if data_ridge.empty:
        raise ValueError("No ridge data after filtering.")

    data_ridge = data_ridge[data_ridge["county"].isin(counties_in_data)]
    ridge_years = sorted(data_ridge["WY"].unique().tolist())

    gdf_counties = None
    if HAS_GEOPANDAS and isinstance(gdf, gpd.GeoDataFrame):
        gdf_counties = load_county_boundaries()
        if gdf_counties is not None:
            gdf_plot = gdf_counties[gdf_counties["COUNTY_norm"].isin(counties_in_data)].copy()
        else:
            gdf_plot = None
    else:
        gdf_plot = None

    if gdf_plot is None:
        print("[warn] No county boundaries available. Only ridgeline will be shown.")
        return _plot_ridgeline_only(
            data_ridge,
            wy_ridge_start, wy_ridge_end, cut_metric, cut_min, cut_max, merge_years,
            counties_in_data, year_colors
        )

    n_cnty = len(counties_in_data)
    height = max(13.0, 0.95 * n_cnty + 4.5)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(26, height))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2.8, 1.9], wspace=0.10)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_ridge = fig.add_subplot(gs[0, 1])

    gdf_plot.plot(
        ax=ax_map,
        facecolor="0.90",
        edgecolor="white",
        linewidth=0.9,
        zorder=1,
    )
    ax_map.grid(False)
    for sp in ax_map.spines.values():
        sp.set_visible(False)

    ax_map.set_xlabel("Longitude", fontsize=17)
    ax_map.set_ylabel("Latitude", fontsize=17)
    ax_map.set_aspect("equal", adjustable="box")

    xmin, ymin, xmax, ymax = gdf_plot.total_bounds
    y_pad = (ymax - ymin) * 0.10
    x_pad = (xmax - xmin) * 0.06
    ax_map.set_ylim(ymin - y_pad, ymax + y_pad)
    ax_map.set_xlim(xmin - x_pad, xmax + x_pad)

    years_lbl = _year_tick_labels(map_years)
    x_pos = np.arange(len(map_years), dtype=float)

    for _, row in gdf_plot.iterrows():
        cname = row["COUNTY_norm"]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if cname not in pivot_int.index:
            continue

        rp = geom.representative_point()
        cx, cy = float(rp.x), float(rp.y)

        x_disp, y_disp = ax_map.transData.transform((cx, cy))
        x_af, y_af = ax_map.transAxes.inverted().transform((x_disp, y_disp))

        dx, dy = INSET_OFFSETS.get(cname, (0.0, 0.0))
        x_af += dx
        y_af += dy

        left = np.clip(x_af - INSET_W / 2, 0.0, 1.0 - INSET_W)
        bottom = np.clip(y_af - INSET_H / 2, 0.0, 1.0 - INSET_H)

        ax_in = ax_map.inset_axes([left, bottom, INSET_W, INSET_H], transform=ax_map.transAxes, zorder=20)
        ax_in.set_facecolor((1, 1, 1, 0.92))

        vals = pivot_int.loc[cname].to_numpy(dtype=float)

        for i, wy in enumerate(map_years):
            v = vals[i]
            if np.isfinite(v):
                ax_in.bar(
                    x_pos[i], int(v),
                    width=INSET_BAR_WIDTH,
                    color=year_colors.get(wy, "0.7"),
                    edgecolor="black",
                    linewidth=0.35,
                    alpha=0.80,
                    zorder=2,
                )

        gm = grand.loc[cname, "grand_mean_int"] if cname in grand.index else np.nan
        if pd.notna(gm):
            ax_in.axhline(int(gm), color="red", linestyle=":", linewidth=1.8, zorder=3)

        ax_in.set_ylim(inset_y0, inset_y1)
        ax_in.set_yticks(inset_yticks)
        ax_in.tick_params(axis="y", labelsize=10, length=2.5, colors="black")

        ax_in.set_xticks(x_pos)
        ax_in.set_xticklabels(years_lbl, fontsize=10, rotation=INSET_XROT, ha="center", va="top", color="black")
        ax_in.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_in.set_xlabel("")
        ax_in.set_ylabel("")
        ax_in.grid(False)

        for s in ["top", "right"]:
            ax_in.spines[s].set_visible(False)
        ax_in.spines["left"].set_linewidth(0.7)
        ax_in.spines["bottom"].set_linewidth(0.7)

        ax_in.text(
            0.5, -0.10,
            cname,
            transform=ax_in.transAxes,
            ha="center", va="top",
            fontsize=16,
            color="black",
            alpha=1.0,
            clip_on=False,
            zorder=30,
        )

    x_all = data_ridge[cut_metric].values
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    x_pad_r = max(0.2, 0.05 * (x_max - x_min or 1.0))
    x_grid = np.linspace(x_min - x_pad_r, x_max + x_pad_r, 400)

    for idx, county in enumerate(counties_in_data):
        y0 = n_cnty - idx
        sub_c = data_ridge[data_ridge["county"] == county]
        if sub_c.empty:
            continue

        ax_ridge.hlines(y0, x_min - x_pad_r, x_max + x_pad_r, color="black", linewidth=0.35)

        if merge_years:
            vals = sub_c[cut_metric].dropna().values
            if vals.size < 1:
                continue
            if HAS_GAUSSIAN_KDE and gaussian_kde is not None and vals.size > 1:
                pdf = gaussian_kde(vals)(x_grid)
            else:
                counts, bin_edges = np.histogram(vals, bins=min(12, max(3, vals.size)))
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                pdf = np.interp(x_grid, centers, counts, left=0.0, right=0.0)
            if np.all(pdf == 0):
                continue

            pdf = pdf / pdf.max()
            ridge = pdf * 0.55
            baseline = np.full_like(x_grid, y0, dtype=float)

            ax_ridge.fill_between(
                x_grid, baseline, baseline + ridge,
                color="0.6",
                alpha=0.25,
                linewidth=RIDGE_EDGE_LW,
                edgecolor="black",
            )
        else:
            years_c = sorted(sub_c["WY"].unique().tolist())
            for wy in years_c:
                vals = sub_c.loc[sub_c["WY"] == wy, cut_metric].dropna().values
                if vals.size < 1:
                    continue

                if HAS_GAUSSIAN_KDE and gaussian_kde is not None and vals.size > 1:
                    pdf = gaussian_kde(vals)(x_grid)
                else:
                    counts, bin_edges = np.histogram(vals, bins=min(12, max(3, vals.size)))
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    pdf = np.interp(x_grid, centers, counts, left=0.0, right=0.0)

                if np.all(pdf == 0):
                    continue

                pdf = pdf / pdf.max()
                ridge = pdf * 0.55
                baseline = np.full_like(x_grid, y0, dtype=float)

                border_alpha = 0.30

                ax_ridge.plot(
                    x_grid, baseline + ridge,
                    color=year_colors.get(wy, "0.7"),
                    linewidth=1.5,
                    alpha=1.0,
                )

    ax_ridge.set_xlim(x_min - x_pad_r, x_max + x_pad_r)
    ax_ridge.set_ylim(1.0, n_cnty + 0.75)

    yticks = [n_cnty - i for i in range(n_cnty)]
    ax_ridge.set_yticks(yticks)
    ax_ridge.set_yticklabels(counties_in_data, fontsize=15, color="black")

    ax_ridge.set_xlabel("Alfalfa number of cuttings", fontsize=16, color="black")
    ax_ridge.set_ylabel("")
    ax_ridge.grid(False)
    for side in ["top", "right"]:
        sp = ax_ridge.spines.get(side)
        if sp is not None:
            sp.set_visible(False)

    if (not merge_years) and ridge_years:
        handles = [
            mpl.patches.Patch(
                facecolor=year_colors.get(wy, "0.7"),
                edgecolor="black",
                alpha=0.70,
                label=f"WY {wy}",
            )
            for wy in ridge_years
        ]
        ax_ridge.legend(
            handles=handles,
            title="Water years",
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(-0.25, 0.95),
            borderaxespad=0.0,
            fontsize=14,
            title_fontsize=16,
        )

    fig.suptitle(
        f"County {cut_metric}: inset WY means on map (WYs {wy_map_start}–{wy_map_end}) "
        f"+ parcel densities (WYs {wy_ridge_start}–{wy_ridge_end})",
        y=0.95,
        fontsize=20,
    )

    plt.subplots_adjust(left=0.05, right=0.93, bottom=0.09, top=0.90, wspace=0.10)
    plt.show()
    return fig


def _plot_ridgeline_only(
    data,
    wy_start: int,
    wy_end: int,
    cut_metric: str,
    cut_min: Optional[float],
    cut_max: Optional[float],
    merge_years: bool,
    counties_in_data: List[str] = None,
    year_colors: Dict[int, tuple] = None,
):
    """
    Plot only the ridgeline panel when spatial data unavailable.

    Parameters
    ----------
    data : pd.DataFrame or None
        Plotting data with columns: county, WY, {cut_metric}
    wy_start : int
        Starting water year
    wy_end : int
        Ending water year
    cut_metric : str
        Cutting metric name
    cut_min : float, optional
        Minimum cut metric filter
    cut_max : float, optional
        Maximum cut metric filter
    merge_years : bool
        Whether to merge years
    counties_in_data : list of str, optional
        List of counties to include
    year_colors : dict, optional
        Year color mapping

    Returns
    -------
    fig : plt.Figure or None
        Matplotlib figure, or None if no data
    """
    if data is None:
        print("[warn] No data available for ridgeline plot.")
        return None

    data_ridge = data[data["WY"].between(wy_start, wy_end)].copy()
    data_ridge = data_ridge.dropna(subset=[cut_metric])
    if cut_min is not None:
        data_ridge = data_ridge[data_ridge[cut_metric] >= cut_min]
    if cut_max is not None:
        data_ridge = data_ridge[data_ridge[cut_metric] <= cut_max]
    if data_ridge.empty:
        print("[warn] No data after filtering for ridgeline plot.")
        return None

    counties = data_ridge["county"].unique().tolist()
    if counties_in_data is None:
        counties_in_data = [c for c in COUNTY_ORDER if c in counties]
    else:
        counties_in_data = [c for c in counties_in_data if c in counties]

    if not counties_in_data:
        print("[warn] No counties with data for ridgeline plot.")
        return None

    data_ridge = data_ridge[data_ridge["county"].isin(counties_in_data)]
    ridge_years = sorted(data_ridge["WY"].unique().tolist())

    if year_colors is None:
        year_colors = _build_year_color_map(ridge_years)

    n_cnty = len(counties_in_data)
    height = max(10.0, 0.95 * n_cnty + 2.0)

    fig, ax_ridge = plt.subplots(figsize=(14, height))

    x_all = data_ridge[cut_metric].values
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    x_pad_r = max(0.2, 0.05 * (x_max - x_min or 1.0))
    x_grid = np.linspace(x_min - x_pad_r, x_max + x_pad_r, 400)

    for idx, county in enumerate(counties_in_data):
        y0 = n_cnty - idx
        sub_c = data_ridge[data_ridge["county"] == county]
        if sub_c.empty:
            continue

        ax_ridge.hlines(y0, x_min - x_pad_r, x_max + x_pad_r, color="black", linewidth=0.35)

        if merge_years:
            vals = sub_c[cut_metric].dropna().values
            if vals.size < 1:
                continue
            if HAS_GAUSSIAN_KDE and gaussian_kde is not None and vals.size > 1:
                pdf = gaussian_kde(vals)(x_grid)
            else:
                counts, bin_edges = np.histogram(vals, bins=min(12, max(3, vals.size)))
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                pdf = np.interp(x_grid, centers, counts, left=0.0, right=0.0)
            if np.all(pdf == 0):
                continue

            pdf = pdf / pdf.max()
            ridge = pdf * 0.55
            baseline = np.full_like(x_grid, y0, dtype=float)

            ax_ridge.fill_between(
                x_grid, baseline, baseline + ridge,
                color="0.6",
                alpha=0.25,
                linewidth=RIDGE_EDGE_LW,
                edgecolor="black",
            )
        else:
            years_c = sorted(sub_c["WY"].unique().tolist())
            for wy in years_c:
                vals = sub_c.loc[sub_c["WY"] == wy, cut_metric].dropna().values
                if vals.size < 1:
                    continue

                if HAS_GAUSSIAN_KDE and gaussian_kde is not None and vals.size > 1:
                    pdf = gaussian_kde(vals)(x_grid)
                else:
                    counts, bin_edges = np.histogram(vals, bins=min(12, max(3, vals.size)))
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    pdf = np.interp(x_grid, centers, counts, left=0.0, right=0.0)

                if np.all(pdf == 0):
                    continue

                pdf = pdf / pdf.max()
                ridge = pdf * 0.55
                baseline = np.full_like(x_grid, y0, dtype=float)

                border_alpha = 0.30

                ax_ridge.plot(
                    x_grid, baseline + ridge,
                    color=year_colors.get(wy, "0.7"),
                    linewidth=1.5,
                    alpha=1.0,
                )

    ax_ridge.set_xlim(x_min - x_pad_r, x_max + x_pad_r)
    ax_ridge.set_ylim(1.0, n_cnty + 0.75)

    yticks = [n_cnty - i for i in range(n_cnty)]
    ax_ridge.set_yticks(yticks)
    ax_ridge.set_yticklabels(counties_in_data, fontsize=15, color="black")

    ax_ridge.set_xlabel("Alfalfa number of cuttings", fontsize=16, color="black")
    ax_ridge.set_ylabel("")
    ax_ridge.grid(False)
    for side in ["top", "right"]:
        sp = ax_ridge.spines.get(side)
        if sp is not None:
            sp.set_visible(False)

    if (not merge_years) and ridge_years:
        handles = [
            mpl.patches.Patch(
                facecolor=year_colors.get(wy, "0.7"),
                edgecolor="black",
                alpha=0.70,
                label=f"WY {wy}",
            )
            for wy in ridge_years
        ]
        ax_ridge.legend(
            handles=handles,
            title="Water years",
            frameon=False,
            loc="upper right",
            fontsize=14,
            title_fontsize=16,
        )

    fig.suptitle(
        f"County {cut_metric}: parcel densities (WYs {wy_start}–{wy_end})",
        fontsize=20,
    )

    plt.tight_layout()
    plt.show()
    return fig