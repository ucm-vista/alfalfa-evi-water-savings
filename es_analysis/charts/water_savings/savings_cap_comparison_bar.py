"""Grouped bar chart comparing water savings across cap scenarios by county.

Shows side-by-side bars for cap=0/1/2/3 for each county (N to S),
answering: "How much water would we save if we capped late cuttings at K?"

Supports et_mode: "actual" (raw OpenET), "corrected" (ETof-adjusted), or
"both" (two-panel: actual on top, corrected on bottom).
"""

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.utils.units import mm_to_acft_per_acre
from es_analysis.utils.publication_style import (
    WONG_PALETTE,
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


# Cap scenario colors: darkest (remove all) -> lightest (keep 3)
CAP_COLORS = [WONG_PALETTE[6], WONG_PALETTE[1], WONG_PALETTE[2], WONG_PALETTE[3]]


def _col_prefix(et_mode: str) -> str:
    """Return column prefix for saved mm columns."""
    if et_mode == "corrected":
        return "saved_corrected_mm_cap"
    return "saved_mm_cap"


def _plot_one_panel(
    ax, df_savings, counties_use, valid_caps, col_prefix, unit, title=None,
):
    """Draw grouped bars for one et_mode on ax."""
    val_col_name = "saved_acft_per_acre" if unit == "acft_per_acre" else "saved_mm"

    rows = []
    for county in counties_use:
        sub = df_savings[df_savings["county"] == county]
        for k in valid_caps:
            col = f"{col_prefix}{k}"
            mean_mm = float(sub[col].mean()) if col in sub.columns else 0.0
            rows.append({
                "county": county,
                "cap": k,
                "saved_mm": mean_mm,
                "saved_acft_per_acre": float(mm_to_acft_per_acre(mean_mm)),
            })
    agg = pd.DataFrame(rows)

    n_counties = len(counties_use)
    n_caps = len(valid_caps)
    x = np.arange(n_counties)
    width = 0.8 / n_caps

    for j, k in enumerate(valid_caps):
        sub = agg[agg["cap"] == k]
        vals = []
        for c in counties_use:
            row = sub[sub["county"] == c]
            vals.append(float(row[val_col_name].iloc[0]) if len(row) > 0 else 0.0)
        offset = (j - n_caps / 2 + 0.5) * width
        label = "Remove all late cuts" if k == 0 else f"Cap at {k} late cut{'s' if k > 1 else ''}"
        color = CAP_COLORS[j % len(CAP_COLORS)]
        ax.bar(
            x + offset, vals, width=width * 0.88,
            label=label, color=color,
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(counties_use, rotation=35, ha="right")
    ylabel = "Mean water savings (ac-ft/acre)" if unit == "acft_per_acre" else "Mean water savings (mm)"
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=8)
    _style_ax(ax)

    return agg


def savings_cap_comparison_bar(
    df_savings: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1, 2, 3),
    counties: Optional[Iterable[str]] = None,
    unit: str = "acft_per_acre",
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, object, Dict]:
    """Grouped bar chart of mean water savings by county for each cap scenario.

    Args:
        df_savings: Parcel-year DataFrame with saved_mm_cap{K} columns
            and optionally saved_corrected_mm_cap{K} columns.
        cap_values: Cap values to compare (default 0,1,2,3).
        counties: County list (default: all present, in COUNTY_ORDER).
        unit: "acft_per_acre" or "mm" for y-axis.
        et_mode: "actual", "corrected", or "both".
        out_dir: Output directory for save_pub_figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    apply_style()

    present = df_savings["county"].unique().tolist()
    if counties is None:
        counties_use = [c for c in COUNTY_ORDER if c in present]
    else:
        counties_use = [c for c in counties if c in present]

    # Verify cap columns exist for each mode
    def _valid_caps(prefix):
        return [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]

    if et_mode == "both":
        valid_actual = _valid_caps("saved_mm_cap")
        valid_corr = _valid_caps("saved_corrected_mm_cap")
        if not valid_actual:
            raise ValueError("No saved_mm_cap{K} columns found.")
        if not valid_corr:
            raise ValueError("No saved_corrected_mm_cap{K} columns. Run with et_mode='corrected'.")
        valid_caps_use = valid_actual

        fig, axes = plt.subplots(
            2, 1,
            figsize=(DOUBLE_COL_WIDTH * 1.3, DOUBLE_COL_WIDTH * 0.9),
            sharex=True,
        )
        _plot_one_panel(axes[0], df_savings, counties_use, valid_actual, "saved_mm_cap", unit, title="Actual ET")
        axes[0].legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        add_panel_label(axes[0], "a")

        _plot_one_panel(axes[1], df_savings, counties_use, valid_corr, "saved_corrected_mm_cap", unit, title="Corrected ET")
        add_panel_label(axes[1], "b")

        fig.suptitle(
            f"Water Savings by Cap Scenario \u2014 Actual & Corrected ET, "
            f"{len(counties_use)} counties, WY2019\u20132024, mean per parcel-year",
            fontsize=8, y=1.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        prefix = _col_prefix(et_mode)
        valid_caps_use = _valid_caps(prefix)
        if not valid_caps_use:
            raise ValueError(f"No {prefix}{{K}} columns found.")

        mode_label = "Corrected" if et_mode == "corrected" else "Actual"
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.3, DOUBLE_COL_WIDTH * 0.5))
        _plot_one_panel(ax, df_savings, counties_use, valid_caps_use, prefix, unit,
                        title=f"Water Savings by Cap Scenario ({mode_label} ET)")
        ax.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.suptitle(
            f"{len(counties_use)} counties, WY2019\u20132024, mean per parcel-year",
            fontsize=7, y=1.0,
        )
        axes = ax
        fig.tight_layout(rect=[0, 0, 1, 0.92])

    # Summary stats
    overall_mean = {}
    for k in valid_caps_use:
        col = f"saved_mm_cap{k}"
        if col in df_savings.columns:
            m = float(df_savings[col].mean())
            overall_mean[f"cap{k}_mm"] = m
            overall_mean[f"cap{k}_acft_per_acre"] = float(mm_to_acft_per_acre(m))

    summary = {
        "n_counties": len(counties_use),
        "cap_values": list(valid_caps_use),
        "n_parcel_years": len(df_savings),
        "et_mode": et_mode,
        "overall_means": overall_mean,
    }

    if out_dir is not None:
        suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"savings_cap_comparison_bar{suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.config import config

    csv_path = config.water_saving_out_dir / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = Path(__file__).parent.parent.parent / "output" / "figures" / "water_savings"
    fig, ax, summary = savings_cap_comparison_bar(df, out_dir=out)
    print(f"Summary: {summary}")
