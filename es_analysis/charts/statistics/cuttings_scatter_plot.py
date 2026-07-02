"""Scatter plots of n_cuttings vs OpenET ET and vs GDD5.

Year-by-year two-panel figures and an aggregate figure with IQR whiskers.
No regression lines (explicit design choice).

Uses publication style (Wong/Okabe-Ito colorblind-safe palette, journal sizing).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.evi_provider import normalize_county_name
from es_analysis.utils.publication_style import (
    WONG_PALETTE,
    DOUBLE_COL_WIDTH,
    MAX_HEIGHT,
    apply_style,
    save_pub_figure,
    add_panel_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_axes_full_border(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Year-by-year scatter: one figure per WY with 2 panels
# ---------------------------------------------------------------------------

def cuttings_scatter_by_year(
    df: pd.DataFrame,
    wy: int,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    gdd_col: str = "gdd5_mean",
    cut_col: str = "n_cuttings",
    outfile: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    et_mode: str = "actual",
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Two-panel scatter for one water year: n_cuttings vs ET (left), vs GDD5 (right).

    Args:
        df: Parcel-year DataFrame with cut_col, et_col, gdd_col columns.
        wy: Water year for title.
        et_col: ET column name.
        gdd_col: GDD5 column name.
        cut_col: Cutting count column name.
        outfile: Optional output path (PNG). If None and out_dir given, auto-names.
        out_dir: Optional output directory for save_pub_figure (PNG + PDF).
        et_mode: "actual", "corrected", or "both".

    Returns:
        Tuple of (figure, axes_array, summary_dict).
    """
    apply_style()

    corr_col = "et_cum_corrected_mm"
    has_corr = corr_col in df.columns and et_mode in ("corrected", "both")

    sub = df[df["WY"] == wy].copy() if "WY" in df.columns else df.copy()

    has_et = et_col in sub.columns
    has_gdd = gdd_col in sub.columns

    if has_et:
        sub = sub.dropna(subset=[et_col])
    if has_gdd:
        sub = sub.dropna(subset=[gdd_col])

    n_panels = int(has_et) + int(has_gdd)
    if n_panels == 0:
        raise ValueError(f"Neither {et_col} nor {gdd_col} found in DataFrame.")

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45),
        sharey=True,
    )
    if n_panels == 1:
        axes = np.array([axes])

    panel_idx = 0
    summary = {"wy": wy, "n_parcels": len(sub), "et_mode": et_mode}

    if has_et:
        ax = axes[panel_idx]
        y = sub[cut_col].to_numpy(float)

        if et_mode == "both" and has_corr:
            x_act = sub[et_col].to_numpy(float)
            x_corr = sub[corr_col].to_numpy(float)
            ax.scatter(x_act, y, s=12, alpha=0.4, edgecolor="none",
                       color="lightgrey", label="Actual ET")
            ax.scatter(x_corr, y, s=12, alpha=0.5, edgecolor="none",
                       color="#6BCB77", label="Corrected ET")
            ax.legend(fontsize=6, loc="lower right", framealpha=0.7)
        elif et_mode == "corrected" and has_corr:
            x = sub[corr_col].to_numpy(float)
            ax.scatter(x, y, s=12, alpha=0.5, edgecolor="none", color="#6BCB77")
        else:
            x = sub[et_col].to_numpy(float)
            ax.scatter(x, y, s=12, alpha=0.5, edgecolor="none", color=WONG_PALETTE[5])

        et_label = {"actual": "Cumulative OpenET ET (mm)",
                    "corrected": "Corrected ET (mm)",
                    "both": "ET (mm)"}[et_mode]
        ax.set_xlabel(et_label)
        ax.set_ylabel("Number of cuttings")
        ax.set_title(f"WY {wy}")
        _style_axes_full_border(ax)
        add_panel_label(ax, "a")
        summary["et_range"] = (float(np.nanmin(sub[et_col])), float(np.nanmax(sub[et_col])))
        panel_idx += 1

    if has_gdd:
        ax = axes[panel_idx]
        x = sub[gdd_col].to_numpy(float)
        y = sub[cut_col].to_numpy(float)
        ax.scatter(x, y, s=12, alpha=0.5, edgecolor="none", color=WONG_PALETTE[3])
        ax.set_xlabel("Cumulative GDD5 (\u00b0C\u00b7day)")
        if not has_et:
            ax.set_ylabel("Number of cuttings")
        ax.set_title(f"WY {wy}")
        _style_axes_full_border(ax)
        add_panel_label(ax, "b" if has_et else "a")
        summary["gdd_range"] = (float(np.nanmin(x)), float(np.nanmax(x)))
        panel_idx += 1

    fig.tight_layout()

    suffix = f"_et_{et_mode}" if et_mode != "actual" else ""
    if out_dir is not None:
        save_pub_figure(fig, f"cuttings_scatter_WY{wy}{suffix}", out_dir)
        summary["outdir"] = str(out_dir)
    elif outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Cuttings scatter saved: {outfile}")
        summary["outfile"] = str(outfile)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# Aggregate scatter: mean/median across all years with IQR whiskers
# ---------------------------------------------------------------------------

def cuttings_scatter_aggregate(
    df: pd.DataFrame,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    gdd_col: str = "gdd5_mean",
    cut_col: str = "n_cuttings",
    outfile: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    et_mode: str = "actual",
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Aggregate scatter: median ET and GDD5 per n_cuttings with IQR whiskers.

    Groups all parcel-years by n_cuttings, computes median and IQR for
    ET and GDD5, plots as errorbar scatter (no regression lines).

    Args:
        df: Multi-year parcel-year DataFrame.
        et_col: ET column name.
        gdd_col: GDD5 column name.
        cut_col: Cutting count column name.
        outfile: Optional output path (PNG).
        out_dir: Optional output directory for save_pub_figure (PNG + PDF).
        et_mode: "actual", "corrected", or "both".

    Returns:
        Tuple of (figure, axes_array, summary_dict).
    """
    apply_style()

    corr_col = "et_cum_corrected_mm"
    has_corr = corr_col in df.columns and et_mode in ("corrected", "both")

    has_et = et_col in df.columns
    has_gdd = gdd_col in df.columns

    n_panels = int(has_et) + int(has_gdd)
    if n_panels == 0:
        raise ValueError(f"Neither {et_col} nor {gdd_col} found in DataFrame.")

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45),
        sharey=True,
    )
    if n_panels == 1:
        axes = np.array([axes])

    panel_idx = 0
    summary = {"n_parcels_total": len(df), "et_mode": et_mode}

    if has_et:
        ax = axes[panel_idx]

        def _plot_errorbar(ax, data_col, color, label_prefix=""):
            sub = df.dropna(subset=[data_col, cut_col])
            grouped = sub.groupby(cut_col)[data_col]
            stats = grouped.agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "count"])
            stats.columns = ["median", "q25", "q75", "count"]
            stats = stats[stats["count"] >= 5]
            if stats.empty:
                return 0
            x_vals = stats["median"].values
            y_vals = stats.index.values
            xerr_lo = stats["median"].values - stats["q25"].values
            xerr_hi = stats["q75"].values - stats["median"].values
            lbl = f"{label_prefix}Median ET" if label_prefix else None
            ax.errorbar(
                x_vals, y_vals, xerr=[xerr_lo, xerr_hi],
                fmt="o", markersize=5, color=color,
                ecolor=color, elinewidth=0.8, capsize=3, capthick=0.8,
                label=lbl,
            )
            return len(stats)

        if et_mode == "both" and has_corr:
            _plot_errorbar(ax, et_col, "lightgrey", "Actual ")
            n_groups = _plot_errorbar(ax, corr_col, "#6BCB77", "Corrected ")
            ax.legend(fontsize=6, loc="lower right", framealpha=0.7)
        elif et_mode == "corrected" and has_corr:
            n_groups = _plot_errorbar(ax, corr_col, "#6BCB77")
        else:
            n_groups = _plot_errorbar(ax, et_col, WONG_PALETTE[5])

        et_label = {"actual": "Median cumulative ET (mm)",
                    "corrected": "Median corrected ET (mm)",
                    "both": "Median ET (mm)"}[et_mode]
        ax.set_xlabel(et_label)
        ax.set_ylabel("Number of cuttings")
        ax.set_title("All WYs aggregate")
        _style_axes_full_border(ax)
        add_panel_label(ax, "a")
        summary["et_groups"] = n_groups
        panel_idx += 1

    if has_gdd:
        ax = axes[panel_idx]
        sub = df.dropna(subset=[gdd_col, cut_col])
        grouped = sub.groupby(cut_col)[gdd_col]
        stats = grouped.agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "count"])
        stats.columns = ["median", "q25", "q75", "count"]
        stats = stats[stats["count"] >= 5]

        x_vals = stats["median"].values
        y_vals = stats.index.values
        xerr_lo = stats["median"].values - stats["q25"].values
        xerr_hi = stats["q75"].values - stats["median"].values

        ax.errorbar(
            x_vals, y_vals,
            xerr=[xerr_lo, xerr_hi],
            fmt="o", markersize=5, color=WONG_PALETTE[3],
            ecolor=WONG_PALETTE[3], elinewidth=0.8, capsize=3, capthick=0.8,
        )
        ax.set_xlabel("Median cumulative GDD5 (\u00b0C\u00b7day)")
        if not has_et:
            ax.set_ylabel("Number of cuttings")
        ax.set_title("All WYs aggregate")
        _style_axes_full_border(ax)
        add_panel_label(ax, "b" if has_et else "a")
        summary["gdd_groups"] = len(stats)
        panel_idx += 1

    fig.tight_layout()

    suffix = f"_et_{et_mode}" if et_mode != "actual" else ""
    if out_dir is not None:
        save_pub_figure(fig, f"cuttings_scatter_aggregate{suffix}", out_dir)
        summary["outdir"] = str(out_dir)
    elif outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Cuttings scatter aggregate saved: {outfile}")
        summary["outfile"] = str(outfile)

    return fig, axes, summary
