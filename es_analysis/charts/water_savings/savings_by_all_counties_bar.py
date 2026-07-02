"""All-counties savings bar chart (cap=0 and cap=1).

Shows per-county mean savings (ac-ft/acre) with side-by-side cap=0 / cap=1
bars. When et_mode="both", plots actual and corrected as paired bars.

Answers: "How much water can each county save by capping late cuttings?"
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def savings_by_all_counties_bar(
    df_savings: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    unit: str = "acft_per_acre",
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, object, Dict]:
    """All-counties savings bar chart for cap=0 and cap=1.

    Args:
        df_savings: Parcel-year DataFrame with saved_mm_cap{K} columns.
        cap_values: Cap scenarios to show (default: 0 and 1).
        unit: "acft_per_acre" or "mm".
        et_mode: "actual", "corrected", or "both".
        out_dir: Output directory.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    apply_style()

    present = df_savings["county"].unique().tolist()
    counties = [c for c in COUNTY_ORDER if c in present]
    n_counties = len(counties)

    def _convert(mm_val):
        return float(mm_to_acft_per_acre(mm_val)) if unit == "acft_per_acre" else mm_val

    ylabel = "Mean water savings (ac-ft/acre)" if unit == "acft_per_acre" else "Mean water savings (mm)"

    if et_mode == "both":
        # Two-panel: (a) actual, (b) corrected
        fig, axes = plt.subplots(
            2, 1,
            figsize=(DOUBLE_COL_WIDTH * 1.4, DOUBLE_COL_WIDTH * 0.9),
            sharex=True,
        )

        for panel_idx, (prefix, title_label) in enumerate([
            ("saved_mm_cap", "Actual ET"),
            ("saved_corrected_mm_cap", "Corrected ET"),
        ]):
            ax = axes[panel_idx]
            x = np.arange(n_counties)
            valid_caps = [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]
            n_caps = len(valid_caps)
            bar_width = 0.7 / max(n_caps, 1)

            cap_colors = ["#1f77b4", "#7fbfdf", "#b0d4e8", "#d4e8f0"]

            for j, k in enumerate(valid_caps):
                col = f"{prefix}{k}"
                vals = []
                sds = []
                for county in counties:
                    sub = df_savings[df_savings["county"] == county]
                    mm_vals = sub[col].dropna()
                    vals.append(_convert(float(mm_vals.mean())) if len(mm_vals) > 0 else 0.0)
                    sd = float(mm_vals.std()) if len(mm_vals) > 1 else 0.0
                    sds.append(_convert(sd))

                offset = (j - n_caps / 2 + 0.5) * bar_width
                label = "Remove all late cuts" if k == 0 else f"Cap at {k}"
                ax.bar(
                    x + offset, vals, width=bar_width * 0.88,
                    yerr=sds, capsize=2,
                    color=cap_colors[j % len(cap_colors)],
                    edgecolor="black", linewidth=0.5,
                    label=label,
                    error_kw={"elinewidth": 0.7, "capthick": 0.7},
                )

            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(title_label, fontsize=8)
            if panel_idx == 0:
                ax.legend(fontsize=8, ncol=n_caps, loc="lower center", bbox_to_anchor=(0.5, 1.002), frameon=False, borderaxespad=0.0, columnspacing=1.0, handlelength=1.8)
            _style_ax(ax)
            add_panel_label(ax, chr(ord("a") + panel_idx))

        axes[-1].set_xticks(np.arange(n_counties))
        axes[-1].set_xticklabels(counties, rotation=35, ha="right")
        fig.suptitle(
            f"Water Savings by County \u2014 Actual & Corrected ET, "
            f"WY2019\u20132024, mean per parcel-year",
            fontsize=8, y=1.01,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    else:
        # Single panel
        prefix = "saved_corrected_mm_cap" if et_mode == "corrected" else "saved_mm_cap"
        valid_caps = [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]
        n_caps = len(valid_caps)

        if not valid_caps:
            raise ValueError(f"No {prefix}{{K}} columns found for caps {cap_values}.")

        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.4, DOUBLE_COL_WIDTH * 0.5))
        x = np.arange(n_counties)
        bar_width = 0.7 / max(n_caps, 1)
        cap_colors = ["#1f77b4", "#7fbfdf", "#b0d4e8", "#d4e8f0"]

        for j, k in enumerate(valid_caps):
            col = f"{prefix}{k}"
            vals = []
            sds = []
            for county in counties:
                sub = df_savings[df_savings["county"] == county]
                mm_vals = sub[col].dropna()
                vals.append(_convert(float(mm_vals.mean())) if len(mm_vals) > 0 else 0.0)
                sd = float(mm_vals.std()) if len(mm_vals) > 1 else 0.0
                sds.append(_convert(sd))

            offset = (j - n_caps / 2 + 0.5) * bar_width
            label = "Remove all late cuts" if k == 0 else f"Cap at {k}"
            ax.bar(
                x + offset, vals, width=bar_width * 0.88,
                yerr=sds, capsize=2,
                color=cap_colors[j % len(cap_colors)],
                edgecolor="black", linewidth=0.5,
                label=label,
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
            )

        mode_label = "Corrected" if et_mode == "corrected" else "Actual"
        ax.set_xticks(x)
        ax.set_xticklabels(counties, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(
            f"Water Savings by County ({mode_label} ET, WY2019\u20132024, mean per parcel-year)",
            fontsize=8,
            pad=18,
        )
        ax.legend(fontsize=8, ncol=n_caps, loc="lower center", bbox_to_anchor=(0.5, 1.002), frameon=False, borderaxespad=0.0, columnspacing=1.0, handlelength=1.8)
        _style_ax(ax)
        axes = ax
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    # Summary
    summary = {
        "n_counties": n_counties,
        "counties": counties,
        "cap_values": list(cap_values),
        "et_mode": et_mode,
        "n_parcel_years": len(df_savings),
    }
    for k in cap_values:
        col = f"saved_mm_cap{k}"
        if col in df_savings.columns:
            by_county = {}
            for county in counties:
                sub = df_savings[df_savings["county"] == county]
                mm = float(sub[col].mean()) if len(sub) > 0 else 0.0
                by_county[county] = round(_convert(mm), 4)
            summary[f"cap{k}_by_county"] = by_county

    if out_dir is not None:
        mode_suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"savings_by_all_counties_bar_combined{mode_suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.config import config

    csv_path = config.water_saving_out_dir / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = Path(__file__).parent.parent.parent / "output" / "figures" / "water_savings"
    fig, ax, summary = savings_by_all_counties_bar(df, et_mode="both", out_dir=out)
    for k, v in summary.items():
        if k.startswith("cap"):
            print(f"  {k}: {v}")
