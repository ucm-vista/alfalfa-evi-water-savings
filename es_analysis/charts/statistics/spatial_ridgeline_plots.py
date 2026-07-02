"""Spatial ridgeline plots: county map with inset WY bars + density ridges.

Two figures:
  1. spatial_ridgeline_cuttings  - Number of cuttings
  2. spatial_ridgeline_seg_et    - Cumulative segment ET (mm)

Each figure has a left panel (county map with inset bar charts per WY)
and a right panel (ridgeline density per county, overlaid by WY).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from es_analysis.data_providers.spatial_provider import (
    COUNTY_ORDER,
    load_county_boundaries,
)
from es_analysis.utils.publication_style import save_pub_figure


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
INSET_W = 0.054
INSET_H = 0.054
INSET_BAR_WIDTH = 0.74
INSET_XROT = 90
INSET_YTICKS_CUTTINGS = [0, 5, 8, 10]

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


def _year_tick_labels(years: List[int]) -> List[str]:
    if not years:
        return []
    return [str(years[0])] + [str(y)[-2:] for y in years[1:]]


def _build_year_color_map(years: List[int]) -> Dict[int, tuple]:
    cmap = mpl.colormaps.get_cmap("tab10")
    return {wy: cmap(i % 10) for i, wy in enumerate(sorted(years))}


# ---------------------------------------------------------------------------
# Core plot: map with inset bars + ridgeline panel
# ---------------------------------------------------------------------------
def _plot_map_ridgeline(
    df: pd.DataFrame,
    gdf_counties,
    metric_col: str,
    metric_label: str,
    inset_yticks: List[int],
    ridge_min: Optional[float] = None,
    ridge_max: Optional[float] = None,
    out_dir: Optional[Path] = None,
    out_name: str = "spatial_ridgeline",
    is_integer_metric: bool = False,
    inset_ylim_top: Optional[float] = None,
) -> Tuple[plt.Figure, Dict]:
    """Build two-panel figure: county map with inset bars + ridgelines.

    Args:
        df: DataFrame with columns county, WY, metric_col.
        gdf_counties: GeoDataFrame of county boundaries with COUNTY_norm column.
        metric_col: Column to aggregate and plot.
        metric_label: X-axis label for ridgeline panel.
        inset_yticks: Fixed y-tick positions for inset bar charts.
        ridge_min: Min filter for ridgeline data (None = no filter).
        ridge_max: Max filter for ridgeline data (None = no filter).
        out_dir: Output directory (None = don't save).
        out_name: Filename stem.

    Returns:
        (fig, summary_dict)
    """
    df = df.dropna(subset=[metric_col]).copy()
    counties_in_data = [c for c in COUNTY_ORDER if c in df["county"].unique()]

    # --- MAP aggregation: mean per county-WY, grand mean ---
    tmp = (
        df.groupby(["county", "WY"])[metric_col]
        .mean()
        .reset_index()
        .rename(columns={metric_col: "mean_per_wy"})
    )
    tmp = tmp[tmp["county"].isin(counties_in_data)].copy()
    map_years = sorted(tmp["WY"].unique().tolist())

    tmp["mean_display"] = tmp["mean_per_wy"].round(1)
    grand = (
        tmp.groupby("county")["mean_per_wy"]
        .mean()
        .rename("grand_mean")
        .reset_index()
        .set_index("county")
        .reindex(counties_in_data)
    )
    pivot = (
        tmp.pivot(index="county", columns="WY", values="mean_per_wy")
        .reindex(index=counties_in_data, columns=map_years)
    )

    # Inset y-range
    all_vals = pivot.to_numpy(dtype=float)
    vmax_inset = np.nanmax(all_vals) if np.any(np.isfinite(all_vals)) else 1.0
    if inset_ylim_top is not None:
        inset_y1 = inset_ylim_top
    else:
        inset_y1 = max(int(np.ceil(vmax_inset)) + 1, 10)
    inset_yticks_use = [t for t in inset_yticks if t <= inset_y1]

    year_colors = _build_year_color_map(map_years)

    # --- RIDGE data ---
    data_ridge = df[df["county"].isin(counties_in_data)].copy()
    if ridge_min is not None:
        data_ridge = data_ridge[data_ridge[metric_col] >= ridge_min]
    if ridge_max is not None:
        data_ridge = data_ridge[data_ridge[metric_col] <= ridge_max]

    # --- Layout ---
    n_cnty = len(counties_in_data)
    height = max(12.0, 0.85 * n_cnty + 4.0)
    fig = plt.figure(figsize=(24, height))

    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2.65, 1.7], wspace=0.10)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_ridge = fig.add_subplot(gs[0, 1])

    # ---- MAP ----
    if gdf_counties is not None:
        gdf_plot = gdf_counties[gdf_counties["COUNTY_norm"].isin(counties_in_data)].copy()
        gdf_plot.plot(
            ax=ax_map, facecolor="0.90", edgecolor="white",
            linewidth=0.9, zorder=1,
        )

        xmin, ymin, xmax, ymax = gdf_plot.total_bounds
        y_pad = (ymax - ymin) * 0.14
        x_pad = (xmax - xmin) * 0.09
        ax_map.set_ylim(ymin - y_pad, ymax + y_pad)
        ax_map.set_xlim(xmin - x_pad, xmax + x_pad)
    else:
        gdf_plot = None

    ax_map.grid(False)
    for sp in ax_map.spines.values():
        sp.set_visible(False)
    ax_map.set_xlabel("Longitude", fontsize=22)
    ax_map.set_ylabel("Latitude", fontsize=22)
    ax_map.tick_params(axis="both", labelsize=18)
    ax_map.set_aspect("equal", adjustable="box")

    # Inset bar charts
    years_lbl = _year_tick_labels(map_years)
    x_pos = np.arange(len(map_years), dtype=float)

    if gdf_plot is not None:
        for _, row in gdf_plot.iterrows():
            cname = row["COUNTY_norm"]
            geom = row.geometry
            if geom is None or geom.is_empty or cname not in pivot.index:
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

            ax_in = ax_map.inset_axes(
                [left, bottom, INSET_W, INSET_H],
                transform=ax_map.transAxes, zorder=20,
            )
            # Fully transparent background
            ax_in.set_facecolor("none")
            ax_in.patch.set_alpha(0.0)

            vals = pivot.loc[cname].to_numpy(dtype=float)
            for i, wy in enumerate(map_years):
                v = vals[i]
                if np.isfinite(v):
                    ax_in.bar(
                        x_pos[i], v, width=INSET_BAR_WIDTH,
                        color=year_colors.get(wy, "0.7"),
                        edgecolor="black", linewidth=0.35, alpha=0.80, zorder=2,
                    )

            gm = grand.loc[cname, "grand_mean"] if cname in grand.index else np.nan
            if pd.notna(gm):
                # For integer metrics, round to nearest int and draw line there
                if is_integer_metric:
                    gm_draw = round(gm)
                    gm_label = str(int(gm_draw))
                else:
                    gm_draw = gm
                    gm_label = f"{gm:.0f}"
                ax_in.axhline(gm_draw, color="red", linestyle=":", linewidth=1.8, zorder=3)
                ax_in.text(
                    len(map_years) - 0.3, gm_draw + (inset_y1 * 0.02),
                    gm_label, fontsize=10, color="red",
                    ha="right", va="bottom", fontweight="bold", zorder=4,
                )

            ax_in.set_ylim(0, inset_y1)
            ax_in.set_yticks(inset_yticks_use)
            ytick_size = 10 if is_integer_metric else 7
            ax_in.tick_params(axis="y", labelsize=ytick_size, length=2.5, colors="black")
            ax_in.set_xticks(x_pos)
            ax_in.set_xticklabels(
                years_lbl, fontsize=11, rotation=INSET_XROT,
                ha="center", va="top", color="black",
            )
            ax_in.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax_in.set_xlabel("")
            ax_in.set_ylabel("")
            ax_in.grid(False)
            for s in ["top", "right"]:
                ax_in.spines[s].set_visible(False)
            ax_in.spines["left"].set_linewidth(0.7)
            ax_in.spines["bottom"].set_linewidth(0.7)

            ax_in.text(
                0.5, -0.10, cname, transform=ax_in.transAxes,
                ha="center", va="top", fontsize=17, color="black",
                clip_on=False, zorder=30,
            )

    # ---- RIDGELINES ----
    if not data_ridge.empty:
        x_all = data_ridge[metric_col].values
        x_min_r = float(np.min(x_all))
        x_max_r = float(np.max(x_all))
        x_pad_r = max(0.2, 0.05 * (x_max_r - x_min_r or 1.0))
        x_grid = np.linspace(x_min_r - x_pad_r, x_max_r + x_pad_r, 400)
        ridge_years = sorted(data_ridge["WY"].unique().tolist())

        for idx_c, county in enumerate(counties_in_data):
            y0 = n_cnty - idx_c
            sub_c = data_ridge[data_ridge["county"] == county]
            if sub_c.empty:
                continue

            ax_ridge.hlines(y0, x_min_r - x_pad_r, x_max_r + x_pad_r,
                            color="black", linewidth=0.35)

            for wy in sorted(sub_c["WY"].unique()):
                vals = sub_c.loc[sub_c["WY"] == wy, metric_col].dropna().values
                if vals.size < 2:
                    continue
                if gaussian_kde is not None:
                    pdf = gaussian_kde(vals)(x_grid)
                else:
                    counts, edges = np.histogram(vals, bins=min(12, max(3, vals.size)))
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    pdf = np.interp(x_grid, centers, counts, left=0.0, right=0.0)
                if np.all(pdf == 0):
                    continue
                pdf = pdf / pdf.max()
                ridge = pdf * 0.48
                baseline = np.full_like(x_grid, y0, dtype=float)

                ax_ridge.plot(
                    x_grid, baseline + ridge,
                    color=year_colors.get(wy, "0.7"),
                    linewidth=1.35, alpha=1.0,
                )

        ax_ridge.set_xlim(x_min_r - x_pad_r, x_max_r + x_pad_r)
        ax_ridge.set_ylim(1.0, n_cnty + 0.75)

        yticks_r = [n_cnty - i for i in range(n_cnty)]
        ax_ridge.set_yticks(yticks_r)
        ax_ridge.set_yticklabels(counties_in_data, fontsize=16, color="black")
        ax_ridge.set_xlabel(metric_label, fontsize=18, color="black")
        ax_ridge.tick_params(axis="x", labelsize=13)
        ax_ridge.set_ylabel("")
        ax_ridge.grid(False)
        for side in ["top", "right"]:
            ax_ridge.spines[side].set_visible(False)

        # Legend
        handles = [
            mpl.patches.Patch(
                facecolor=year_colors.get(wy, "0.7"), edgecolor="black",
                alpha=0.70, label=f"WY {wy}",
            )
            for wy in ridge_years
        ]
        ax_ridge.legend(
            handles=handles, title="Water years", frameon=False,
            loc="upper right", bbox_to_anchor=(-0.25, 0.95),
            fontsize=15, title_fontsize=17,
        )

    wys = sorted(df["WY"].unique())
    fig.suptitle(
        f"County {metric_label}: inset WY means on map + parcel densities "
        f"(WY{min(wys)}\u2013{max(wys)})",
        y=0.95, fontsize=22,
    )
    plt.subplots_adjust(left=0.05, right=0.93, bottom=0.09, top=0.90, wspace=0.10)

    summary = {
        "chart": out_name,
        "metric": metric_col,
        "n_counties": len(counties_in_data),
        "n_wys": len(map_years),
        "n_parcel_years": len(df),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in ("png", "pdf"):
            fmt_dir = out_dir / fmt
            fmt_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                fmt_dir / f"{out_name}.{fmt}",
                dpi=200, bbox_inches="tight",
            )
            sz = (fmt_dir / f"{out_name}.{fmt}").stat().st_size // 1024
            print(f"  Saved: {fmt_dir / f'{out_name}.{fmt}'} ({sz} KB)")
        summary["out_dir"] = str(out_dir)

    return fig, summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def spatial_ridgeline_cuttings(
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
    cut_min: float = 2,
    cut_max: float = 12,
) -> Tuple[plt.Figure, Dict]:
    """Map + ridgeline for number of cuttings."""
    gdf_counties = load_county_boundaries()
    return _plot_map_ridgeline(
        df=df,
        gdf_counties=gdf_counties,
        metric_col="n_cuttings",
        metric_label="Number of cuttings",
        inset_yticks=[0, 5, 8, 10],
        ridge_min=cut_min,
        ridge_max=cut_max,
        out_dir=out_dir,
        out_name="spatial_ridgeline_cuttings",
        is_integer_metric=True,
    )


def spatial_ridgeline_seg_et(
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
    et_min: Optional[float] = None,
    et_max: Optional[float] = None,
) -> Tuple[plt.Figure, Dict]:
    """Map + ridgeline for cumulative segment ET (mm)."""
    gdf_counties = load_county_boundaries()

    return _plot_map_ridgeline(
        df=df,
        gdf_counties=gdf_counties,
        metric_col="et_cum_minET_to_last_cut_mm",
        metric_label="Cumulative segment ET (mm)",
        inset_yticks=[0, 800, 1000, 1200, 1400],
        ridge_min=et_min,
        ridge_max=et_max,
        out_dir=out_dir,
        out_name="spatial_ridgeline_seg_et",
        is_integer_metric=False,
        inset_ylim_top=1400,
    )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.utils.run_output import load_dataframe

    df = load_dataframe("multicounty_matched", "alfalfa_run_6")
    if df is None:
        from es_analysis.data_providers.parcel_summary_provider import build_multicounty_matched
        df = build_multicounty_matched()

    out = Path(__file__).parent.parent.parent / "output" / "figures" / "alfalfa_run_6" / "cuttings_analysis" / "all_counties"

    print("--- Cuttings ridgeline ---")
    fig1, s1 = spatial_ridgeline_cuttings(df, out_dir=out)
    print(f"  {s1}")
    plt.close(fig1)

    print("--- Segment ET ridgeline ---")
    fig2, s2 = spatial_ridgeline_seg_et(df, out_dir=out)
    print(f"  {s2}")
    plt.close(fig2)
