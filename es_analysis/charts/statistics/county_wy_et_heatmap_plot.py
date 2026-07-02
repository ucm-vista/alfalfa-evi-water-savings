"""Triple heatmap: County x WY ET summary.

Three-panel figure:
  (a) Mean annual OpenET actual ET by county x WY
  (b) Mean cumulative segment ET (cut-cycle) by county x WY
  (c) Difference (a - b): ET not captured by cut segments

Data sources:
  - Panel (a): ET stats CSV (ET_open_annual_mm) or parcel_summary matched data
  - Panel (b): parcel_summary matched (et_cum_minET_to_last_cut_mm)
  - Panel (c): computed difference
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.data_providers.wy_type_provider import get_wy_type
from es_analysis.utils.publication_style import (
    DOUBLE_COL_WIDTH,
    apply_style,
    save_pub_figure,
    add_panel_label,
)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)


def _draw_panel(ax, pivot, cmap, fmt, title, vmin=None, vmax=None):
    """Draw annotated heatmap on ax, return imshow handle."""
    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1.0

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isfinite(v):
                text_color = "white" if v > 0.6 * vmax else "black"
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=5, color=text_color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=5, color="grey", fontstyle="italic")

    ax.set_xticks(np.arange(n_cols))
    wys = list(pivot.columns)
    ax.set_xticklabels([f"{int(wy)}\n({get_wy_type(int(wy))[0]})" for wy in wys],
                        fontsize=5)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(pivot.index), fontsize=6)
    ax.set_title(title, fontsize=7)
    _style_ax(ax)

    return im


def county_wy_et_triple_heatmap(
    df: pd.DataFrame,
    et_annual_col: str = "ET_open_annual_mm",
    et_segment_col: str = "et_cum_minET_to_last_cut_mm",
    counties: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Three-panel heatmap: annual ET, segment ET, and difference.

    Args:
        df: DataFrame with county, WY, et_annual_col, et_segment_col.
            If et_annual_col is missing, panel (a) uses total_et_mm.
        et_annual_col: Column for annual OpenET.
        et_segment_col: Column for cumulative segment ET.
        counties: Ordered county list (default COUNTY_ORDER).
        out_dir: Output directory.

    Returns:
        (fig, axes, summary)
    """
    apply_style()

    # Find appropriate annual ET column
    if et_annual_col not in df.columns:
        fallbacks = ["total_et_mm", "ET_open_annual_mm", "et_annual_mm"]
        for fb in fallbacks:
            if fb in df.columns:
                et_annual_col = fb
                break

    # If still missing, try loading from ET stats parcel-year CSV
    if et_annual_col not in df.columns:
        import glob
        _pkg_root = Path(__file__).parent.parent.parent
        pattern = str(_pkg_root / "output" / "et_correction" / "offphase_parcel_year_*.csv")
        csvs = sorted(glob.glob(pattern))
        if csvs:
            df_et = pd.read_csv(csvs[-1], usecols=["UniqueID", "WY", "ET_open_annual_mm"])
            df_et["UniqueID"] = df_et["UniqueID"].astype(str)
            df["UniqueID"] = df["UniqueID"].astype(str)
            df = df.merge(df_et, on=["UniqueID", "WY"], how="left")
            et_annual_col = "ET_open_annual_mm"

    if et_annual_col not in df.columns or et_segment_col not in df.columns:
        missing = []
        if et_annual_col not in df.columns:
            missing.append(et_annual_col)
        if et_segment_col not in df.columns:
            missing.append(et_segment_col)
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)[:20]}")

    if counties is None:
        counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    wys = sorted(df["WY"].unique())

    # Build pivot tables
    pivot_annual = df.groupby(["county", "WY"])[et_annual_col].mean().unstack()
    pivot_annual = pivot_annual.reindex(index=counties, columns=wys)

    pivot_segment = df.groupby(["county", "WY"])[et_segment_col].mean().unstack()
    pivot_segment = pivot_segment.reindex(index=counties, columns=wys)

    pivot_diff = pivot_annual - pivot_segment

    # Figure
    n_rows = len(counties)
    fig_h = max(3.5, n_rows * 0.35 + 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH * 2.0, fig_h))

    # Shared color range for panels a and b
    vmax_ab = max(
        np.nanmax(pivot_annual.values) if np.any(np.isfinite(pivot_annual.values)) else 1,
        np.nanmax(pivot_segment.values) if np.any(np.isfinite(pivot_segment.values)) else 1,
    )

    im_a = _draw_panel(axes[0], pivot_annual, "YlOrRd", ".0f",
                        f"Annual OpenET (actual, mm)", vmax=vmax_ab)
    add_panel_label(axes[0], "a")

    im_b = _draw_panel(axes[1], pivot_segment, "YlOrRd", ".0f",
                        f"Cumulative segment ET (mm)", vmax=vmax_ab)
    add_panel_label(axes[1], "b")
    axes[1].set_yticklabels([])

    # Panel c: difference (diverging colormap)
    diff_abs_max = np.nanmax(np.abs(pivot_diff.values)) if np.any(np.isfinite(pivot_diff.values)) else 100
    im_c = _draw_panel(axes[2], pivot_diff, "RdBu_r", ".0f",
                        f"Difference (annual \u2212 segment, mm)",
                        vmin=-diff_abs_max, vmax=diff_abs_max)
    add_panel_label(axes[2], "c")
    axes[2].set_yticklabels([])

    # Colorbars
    cbar_ab = fig.colorbar(im_a, ax=axes[:2].tolist(), shrink=0.7, pad=0.02, location="bottom")
    cbar_ab.set_label("ET (mm)", fontsize=6)
    cbar_ab.ax.tick_params(labelsize=5)

    cbar_c = fig.colorbar(im_c, ax=axes[2], shrink=0.7, pad=0.02, location="bottom")
    cbar_c.set_label("Difference (mm)", fontsize=6)
    cbar_c.ax.tick_params(labelsize=5)

    fig.suptitle(
        f"County \u00d7 Water Year ET Summary \u2014 {len(counties)} counties, "
        f"WY{min(wys)}\u2013{max(wys)}",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()

    # Summary
    summary = {
        "chart": "county_wy_et_triple_heatmap",
        "n_counties": len(counties),
        "n_wys": len(wys),
        "mean_annual_et_mm": float(np.nanmean(pivot_annual.values)),
        "mean_segment_et_mm": float(np.nanmean(pivot_segment.values)),
        "mean_diff_mm": float(np.nanmean(pivot_diff.values)),
    }

    if out_dir is not None:
        save_pub_figure(fig, "county_wy_et_triple_heatmap", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


def _draw_panel_with_pct(ax, pivot, ref_pivot, cmap, title, vmin, vmax,
                         pivot_min=None, pivot_max=None):
    """Draw heatmap with value + range/% annotations.

    Each cell shows the mean value.  If pivot_min/pivot_max are provided,
    a second line shows the [min\u2013max] range.  If ref_pivot is provided,
    a line shows the % change vs the reference panel.
    """
    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isfinite(v):
                # Contrast: dark text on light cells, white on dark
                norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = "white" if norm_v > 0.55 else "black"

                lines = [f"{v:.0f}"]
                if pivot_min is not None and pivot_max is not None:
                    lo = float(pivot_min.values[i, j])
                    hi = float(pivot_max.values[i, j])
                    if np.isfinite(lo) and np.isfinite(hi):
                        lines.append(f"[{lo:.0f}\u2013{hi:.0f}]")
                if ref_pivot is not None:
                    ref_v = float(ref_pivot.values[i, j])
                    if np.isfinite(ref_v) and ref_v != 0:
                        pct = (v - ref_v) / ref_v * 100
                        lines.append(f"({pct:+.0f}%)")
                label = "\n".join(lines)
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=7, color=text_color, linespacing=1.1)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=7, color="grey", fontstyle="italic")

    ax.set_xticks(np.arange(n_cols))
    wys = list(pivot.columns)
    ax.set_xticklabels([f"{int(wy)}\n({get_wy_type(int(wy))[0]})" for wy in wys],
                        fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(pivot.index), fontsize=9)
    ax.set_title(title, fontsize=10)
    _style_ax(ax)

    return im


def county_wy_et_dual_heatmap(
    df: pd.DataFrame,
    et_annual_col: str = "ET_open_annual_mm",
    et_segment_col: str = "et_cum_minET_to_last_cut_mm",
    counties: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Two-panel heatmap with independent colorbars and % annotations.

    Panel (a): Annual OpenET actual ET (mm).
    Panel (b): Cumulative segment ET (mm) with % change vs panel (a).

    Each panel has its own colorbar spanning its own data range.

    Args:
        df: DataFrame with county, WY, ET columns.
        et_annual_col: Column for annual OpenET.
        et_segment_col: Column for cumulative segment ET.
        counties: Ordered county list.
        out_dir: Output directory.

    Returns:
        (fig, axes, summary) where summary includes pivot tables.
    """
    apply_style()

    # Find appropriate annual ET column
    if et_annual_col not in df.columns:
        fallbacks = ["total_et_mm", "ET_open_annual_mm", "et_annual_mm"]
        for fb in fallbacks:
            if fb in df.columns:
                et_annual_col = fb
                break

    # If still missing, try loading from ET stats parcel-year CSV
    if et_annual_col not in df.columns:
        import glob as _glob
        _pkg_root = Path(__file__).parent.parent.parent
        # Search run-specific dirs first, then global
        csvs = []
        for search_dir in sorted(_pkg_root.glob("output/figures/*/et_correction"), reverse=True):
            csvs = sorted(_glob.glob(str(search_dir / "offphase_parcel_year_*.csv")))
            if csvs:
                break
        if not csvs:
            pattern = str(_pkg_root / "output" / "et_correction" / "offphase_parcel_year_*.csv")
            csvs = sorted(_glob.glob(pattern))
        if csvs:
            df_et = pd.read_csv(csvs[-1], usecols=["UniqueID", "WY", "ET_open_annual_mm"])
            df_et["UniqueID"] = df_et["UniqueID"].astype(str)
            df["UniqueID"] = df["UniqueID"].astype(str)
            df = df.merge(df_et, on=["UniqueID", "WY"], how="left")
            et_annual_col = "ET_open_annual_mm"

    if et_annual_col not in df.columns or et_segment_col not in df.columns:
        missing = []
        if et_annual_col not in df.columns:
            missing.append(et_annual_col)
        if et_segment_col not in df.columns:
            missing.append(et_segment_col)
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)[:20]}")

    if counties is None:
        counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    wys = sorted(df["WY"].unique())

    # Build pivot tables — mean, min, max
    pivot_annual = df.groupby(["county", "WY"])[et_annual_col].mean().unstack()
    pivot_annual = pivot_annual.reindex(index=counties, columns=wys)
    pivot_annual_min = df.groupby(["county", "WY"])[et_annual_col].min().unstack()
    pivot_annual_min = pivot_annual_min.reindex(index=counties, columns=wys)
    pivot_annual_max = df.groupby(["county", "WY"])[et_annual_col].max().unstack()
    pivot_annual_max = pivot_annual_max.reindex(index=counties, columns=wys)

    pivot_segment = df.groupby(["county", "WY"])[et_segment_col].mean().unstack()
    pivot_segment = pivot_segment.reindex(index=counties, columns=wys)
    pivot_segment_min = df.groupby(["county", "WY"])[et_segment_col].min().unstack()
    pivot_segment_min = pivot_segment_min.reindex(index=counties, columns=wys)
    pivot_segment_max = df.groupby(["county", "WY"])[et_segment_col].max().unstack()
    pivot_segment_max = pivot_segment_max.reindex(index=counties, columns=wys)

    # Figure — 2 panels, taller to fit three-line annotations
    n_rows = len(counties)
    fig_h = max(6.0, n_rows * 0.60 + 2.5)
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH * 1.7, fig_h),
        constrained_layout=True,
    )

    # Shared color range — data-driven min/max (not from zero)
    all_vals = np.concatenate([
        pivot_annual.values[np.isfinite(pivot_annual.values)],
        pivot_segment.values[np.isfinite(pivot_segment.values)],
    ])
    vmin_shared = float(np.percentile(all_vals, 1)) * 0.95 if len(all_vals) else 500
    vmax_shared = float(np.percentile(all_vals, 99)) * 1.02 if len(all_vals) else 1400

    # Green-blue colormap
    cmap = "GnBu"

    # Panel (a): annual ET with [min-max] range
    im_a = _draw_panel_with_pct(axes[0], pivot_annual, None, cmap,
                                 "Annual OpenET (actual, mm)", vmin_shared, vmax_shared,
                                 pivot_min=pivot_annual_min, pivot_max=pivot_annual_max)
    add_panel_label(axes[0], "a")

    # Panel (b): segment ET with [min-max] range + % vs annual
    im_b = _draw_panel_with_pct(axes[1], pivot_segment, pivot_annual, cmap,
                                 "Cumulative segment ET (mm)", vmin_shared, vmax_shared,
                                 pivot_min=pivot_segment_min, pivot_max=pivot_segment_max)
    add_panel_label(axes[1], "b")
    axes[1].set_yticklabels([])

    # Single colorbar on the right side spanning both panels
    cbar = fig.colorbar(im_b, ax=axes.tolist(), shrink=0.75, pad=0.03,
                        location="right", aspect=30)
    cbar.set_label("ET (mm)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"County \u00d7 Water Year ET Summary \u2014 {len(counties)} counties, "
        f"WY{min(wys)}\u2013{max(wys)}",
        fontsize=11, y=1.02,
    )

    summary = {
        "chart": "county_wy_et_dual_heatmap",
        "n_counties": len(counties),
        "n_wys": len(wys),
        "mean_annual_et_mm": float(np.nanmean(pivot_annual.values)),
        "mean_segment_et_mm": float(np.nanmean(pivot_segment.values)),
        "pivot_annual": pivot_annual,
        "pivot_segment": pivot_segment,
    }

    if out_dir is not None:
        save_pub_figure(fig, "county_wy_et_dual_heatmap", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.parcel_summary_provider import build_multicounty_matched

    df = build_multicounty_matched()
    out = Path(__file__).parent.parent.parent / "output" / "figures" / "statistics"
    fig, axes, summary = county_wy_et_triple_heatmap(df, out_dir=out)
    print(f"Summary: {summary}")
