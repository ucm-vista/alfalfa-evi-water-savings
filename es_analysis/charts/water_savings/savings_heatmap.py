"""Heatmap of water savings intensity by county and water year.

County (rows, N to S) x WY (columns) matrix colored by savings
intensity (ac-ft/acre), with cell annotations showing values.

Supports et_mode: "actual", "corrected", or "both" (side-by-side panels).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
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


def _draw_heatmap(ax, pivot, cmap, fmt, title=None, show_ylabel=True):
    """Draw a single heatmap on ax, return data array."""
    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape
    vmin = 0
    vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1.0

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isfinite(v):
                text_color = "white" if v > 0.6 * vmax else "black"
                ax.text(
                    j, i, f"{v:{fmt}}",
                    ha="center", va="center",
                    fontsize=6, color=text_color,
                )
            else:
                ax.text(
                    j, i, "N/A",
                    ha="center", va="center",
                    fontsize=5, color="grey", fontstyle="italic",
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"WY{int(c)}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    if show_ylabel:
        ax.set_yticklabels(list(pivot.index))
    else:
        ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=8)
    _style_ax(ax)

    return im, data


def savings_heatmap(
    df_summary: pd.DataFrame,
    value_col: str = "water_saved_acft_per_acre",
    cap_k: Optional[int] = None,
    cmap: str = "YlOrRd",
    fmt: str = ".3f",
    et_mode: str = "actual",
    corrected_value_col: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, object, Dict]:
    """Heatmap of savings intensity: county rows x WY columns.

    Args:
        df_summary: County-WY savings summary with columns
            "county", "WY", and value_col.
        value_col: Column to visualize for actual ET.
        cap_k: If set, filter df_summary to cut_cap_k == cap_k.
        cmap: Matplotlib colormap name.
        fmt: Number format for cell annotations.
        et_mode: "actual", "corrected", or "both".
        corrected_value_col: Column for corrected ET. If None and
            et_mode involves corrected, tries value_col with
            "_corrected" inserted.
        out_dir: Output directory for save_pub_figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    apply_style()

    work = df_summary.copy()
    if cap_k is not None and "cut_cap_k" in work.columns:
        work = work[work["cut_cap_k"] == cap_k]

    if value_col not in work.columns:
        raise ValueError(f"Column '{value_col}' not found. Available: {list(work.columns)}")

    # Determine corrected column
    if corrected_value_col is None:
        # Try common naming: water_saved_acft_per_acre -> water_saved_corrected_acft_per_acre
        corrected_value_col = value_col.replace("water_saved_", "water_saved_corrected_")
        if corrected_value_col == value_col:
            corrected_value_col = f"{value_col}_corrected"

    def _make_pivot(col):
        if col not in work.columns:
            return None
        return work.pivot_table(
            index="county", columns="WY", values=col, aggfunc="mean",
        )

    pivot_actual = _make_pivot(value_col)
    pivot_corr = _make_pivot(corrected_value_col) if et_mode in ("corrected", "both") else None

    # Order counties N->S
    def _order(pv):
        if pv is None:
            return pv
        ordered = [c for c in COUNTY_ORDER if c in pv.index]
        pv = pv.reindex(ordered)
        pv = pv[sorted(pv.columns)]
        return pv

    pivot_actual = _order(pivot_actual)
    if pivot_corr is not None:
        pivot_corr = _order(pivot_corr)

    unit_label = "ac-ft/acre" if "acft" in value_col else "mm"
    cap_label = f" (cap={cap_k})" if cap_k is not None else ""

    if et_mode == "both" and pivot_corr is not None:
        n_rows = pivot_actual.shape[0]
        fig_h = max(DOUBLE_COL_WIDTH * 0.4, n_rows * 0.35 + 1.0)
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH * 1.6, fig_h))

        im1, data1 = _draw_heatmap(axes[0], pivot_actual, cmap, fmt,
                                    title=f"Actual ET savings{cap_label}, mean per parcel-year")
        add_panel_label(axes[0], "a")

        im2, data2 = _draw_heatmap(axes[1], pivot_corr, cmap, fmt,
                                    title=f"Corrected ET savings{cap_label}, mean per parcel-year",
                                    show_ylabel=False)
        add_panel_label(axes[1], "b")

        # Shared colorbar
        cbar = fig.colorbar(im1, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Savings ({unit_label})", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        data = data1  # use actual for summary
    else:
        pivot_use = pivot_corr if (et_mode == "corrected" and pivot_corr is not None) else pivot_actual
        n_rows = pivot_use.shape[0]
        fig_h = max(DOUBLE_COL_WIDTH * 0.4, n_rows * 0.35 + 1.0)
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, fig_h))

        mode_label = "Corrected" if et_mode == "corrected" else "Actual"
        im, data = _draw_heatmap(ax, pivot_use, cmap, fmt,
                                  title=f"{mode_label} ET savings{cap_label}, mean per parcel-year")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f"Savings ({unit_label})", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        axes = ax

    # Imperial County coverage note
    if "Imperial" in list(pivot_actual.index):
        fig.text(
            0.01, 0.01,
            "Note: Imperial County has severe OpenET data gaps; treat with caution.",
            fontsize=5, fontstyle="italic", color="grey",
            transform=fig.transFigure,
        )

    fig.tight_layout()

    ordered_counties = list(pivot_actual.index)
    summary = {
        "n_counties": len(ordered_counties),
        "n_years": pivot_actual.shape[1],
        "value_col": value_col,
        "cap_k": cap_k,
        "et_mode": et_mode,
        "overall_mean": float(np.nanmean(data)),
        "max_county_wy": None,
    }

    if np.any(np.isfinite(data)):
        idx_flat = np.nanargmax(data)
        i_max, j_max = np.unravel_index(idx_flat, data.shape)
        summary["max_county_wy"] = {
            "county": ordered_counties[i_max],
            "wy": int(pivot_actual.columns[j_max]),
            "value": float(data[i_max, j_max]),
        }

    if out_dir is not None:
        suffix = f"_cap{cap_k}" if cap_k is not None else ""
        mode_suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"savings_heatmap{suffix}{mode_suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.config import config

    parser = argparse.ArgumentParser(description="Savings heatmap (standalone)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory to read input CSV from (overrides config default)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures")
    args = parser.parse_args()

    data_base = Path(args.data_dir) if args.data_dir else config.water_saving_out_dir
    csv_path = data_base / "water_saving_summary_county_WY_by_cap.csv"
    df = pd.read_csv(csv_path)
    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent.parent / "output" / "figures" / "water_savings"
    )
    for cap in [0, 1]:
        fig, ax, summary = savings_heatmap(df, cap_k=cap, out_dir=out)
        print(f"Cap={cap}: mean={summary['overall_mean']:.4f}, max={summary['max_county_wy']}")
