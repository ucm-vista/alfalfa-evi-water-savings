"""Water savings by water year bar chart (combined cap0 + cap1).

Mean savings (ac-ft/acre) per water year across all counties, with
cap=0 and cap=1 side by side.

Answers: "Which water years had the greatest savings potential?"

Supports et_mode: "actual", "corrected", or "both" (side-by-side bars).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.data_providers.wy_type_provider import get_wy_type
from es_analysis.utils.units import mm_to_acft_per_acre
from es_analysis.utils.publication_style import (
    WONG_PALETTE,
    DOUBLE_COL_WIDTH,
    apply_style,
    save_pub_figure,
)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)


def savings_by_wy_bar_combined(
    df_savings: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    unit: str = "acft_per_acre",
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Combined bar chart with cap0 and cap1 side by side per water year.

    Cap 0 = blue, Cap 1 = light blue.
    """
    apply_style()

    prefix = "saved_corrected_mm_cap" if et_mode == "corrected" else "saved_mm_cap"
    valid_caps = [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]
    if not valid_caps:
        raise ValueError(f"No {prefix}{{K}} columns found for caps {cap_values}.")

    wys = sorted(df_savings["WY"].unique())

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.1, DOUBLE_COL_WIDTH * 0.45))

    def _convert(mm_val):
        return float(mm_to_acft_per_acre(mm_val)) if unit == "acft_per_acre" else mm_val

    n_caps = len(valid_caps)
    bar_width = 0.7 / max(n_caps, 1)
    x = np.arange(len(wys))

    cap_colors = {0: "#1f77b4", 1: "#7fbfdf", 2: "#b0d4e8", 3: "#d4e8f0"}
    cap_labels = {0: "Remove all late cuts (cap 0)", 1: "Cap at 1 late cut"}

    for j, k in enumerate(valid_caps):
        col = f"{prefix}{k}"
        vals = []
        sds = []
        for wy in wys:
            sub = df_savings[df_savings["WY"] == wy]
            mm_vals = sub[col].dropna()
            vals.append(_convert(float(mm_vals.mean())) if len(mm_vals) > 0 else 0.0)
            sd = float(mm_vals.std()) if len(mm_vals) > 1 else 0.0
            sds.append(_convert(sd))
        offset = (j - n_caps / 2 + 0.5) * bar_width
        label = cap_labels.get(k, f"Cap at {k}")
        ax.bar(
            x + offset, vals, width=bar_width * 0.88,
            yerr=sds, capsize=3,
            color=cap_colors.get(k, "#aaaaaa"),
            edgecolor="black", linewidth=0.5,
            label=label,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"WY{wy}\n({get_wy_type(wy)})" for wy in wys], rotation=35, ha="right", fontsize=8,
    )
    ylabel = "Mean water savings (ac-ft/acre)" if unit == "acft_per_acre" else "Mean water savings (mm)"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    mode_label = {"actual": "Actual ET", "corrected": "Corrected ET", "both": "Actual & Corrected ET"}[et_mode]
    n_counties = df_savings["county"].nunique()
    ax.set_title(
        f"Water savings by water year \u2014 {mode_label}, "
        f"{n_counties} counties pooled, mean per parcel-year",
        fontsize=9,
        pad=18,
    )
    ax.legend(fontsize=8, ncol=n_caps, loc="lower center", bbox_to_anchor=(0.5, 1.002), frameon=False, borderaxespad=0.0, columnspacing=1.0, handlelength=1.8)
    ax.set_ylim(bottom=0)
    _style_ax(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    summary = {
        "cap_values": list(valid_caps),
        "et_mode": et_mode,
        "n_water_years": len(wys),
        "n_parcel_years": len(df_savings),
    }

    if out_dir is not None:
        save_pub_figure(fig, "savings_by_wy_bar_combined", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, ax, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.config import config

    csv_path = config.water_saving_out_dir / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = Path(__file__).parent.parent.parent / "output" / "figures" / "water_savings"
    fig, ax, summary = savings_by_wy_bar_combined(df, cap_values=(0, 1), out_dir=out)
    print(f"Cap values: {summary['cap_values']}")
