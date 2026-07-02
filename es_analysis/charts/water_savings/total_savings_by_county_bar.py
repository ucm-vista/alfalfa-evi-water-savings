"""Total water saved by county (annual mean, ac-ft/yr).

Companion to the per-acre savings chart. Where that chart shows efficiency
(rate per acre, mostly uniform N→S), this chart shows magnitude — total
volume saved per year — which is dominated by the area gradient.

For each parcel-year, total saved (ac-ft) = saved_mm × area_acres / 304.8.
Bars show the annual mean (sum across all WYs / number of WYs in the dataset).
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
)


CAP_COLORS = {0: "#1f77b4", 1: "#7fbfdf"}
CAP_LABELS = {0: "Remove all late cuts (cap 0)", 1: "Cap at 1 late cut"}
MM_PER_FOOT = 304.8


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)


def total_savings_by_county_bar(
    df_savings: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    apply_style()

    prefix = "saved_corrected_mm_cap" if et_mode == "corrected" else "saved_mm_cap"
    valid_caps = [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]
    if not valid_caps:
        raise ValueError(f"No {prefix}{{K}} columns found.")

    present = df_savings["county"].unique().tolist()
    counties = [c for c in COUNTY_ORDER if c in present]
    n_counties = len(counties)
    n_wy = df_savings["WY"].nunique()

    # Per-parcel-year total ac-ft saved
    df = df_savings.copy()
    for k in valid_caps:
        df[f"_acft_cap{k}"] = df[f"{prefix}{k}"].fillna(0) * df["area_acres"] / MM_PER_FOOT

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.4, DOUBLE_COL_WIDTH * 0.55))
    x = np.arange(n_counties)
    bar_width = 0.7 / max(len(valid_caps), 1)

    annual_by_cap = {}
    for j, k in enumerate(valid_caps):
        annual_vals = []
        for c in counties:
            sub = df[df["county"] == c]
            total_acft = float(sub[f"_acft_cap{k}"].sum())
            annual_vals.append(total_acft / n_wy)   # mean per WY
        offset = (j - len(valid_caps) / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, annual_vals, width=bar_width * 0.88,
            color=CAP_COLORS.get(k, "#888"),
            edgecolor="black", linewidth=0.5,
            label=CAP_LABELS.get(k, f"Cap at {k}"),
        )
        annual_by_cap[k] = dict(zip(counties, annual_vals))

    ax.set_xticks(x)
    ax.set_xticklabels(counties, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Total water saved (ac-ft / year)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    mode_label = "Corrected" if et_mode == "corrected" else "Actual"
    ax.set_title(
        f"Total water saved by county ({mode_label} ET, "
        f"WY2019–2024 annual mean)",
        fontsize=9, pad=18,
    )
    ax.legend(
        fontsize=8, ncol=len(valid_caps),
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        frameon=False, borderaxespad=0.0,
        columnspacing=1.2, handlelength=1.8,
    )
    _style_ax(ax)

    fig.text(
        0.5, 0.005,
        "Annual mean = total ac-ft saved across WY2019–2024 ÷ 6. "
        "Per-parcel-year saved (ac-ft) = saved_mm × area_acres / 304.8.",
        ha="center", fontsize=7, style="italic", color="#444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    summary = {
        "et_mode": et_mode,
        "n_water_years": n_wy,
        "annual_acft_by_county": {f"cap{k}": annual_by_cap[k] for k in valid_caps},
    }

    if out_dir is not None:
        suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"total_savings_by_county_bar{suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, ax, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    csv_path = Path(__file__).parent.parent.parent / "output" / "figures" / "alfalfa_run_6" / "water_savings" / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = csv_path.parent
    fig, ax, summary = total_savings_by_county_bar(df, out_dir=out)
    print("Wrote to:", out)
    for c in COUNTY_ORDER:
        c0 = summary["annual_acft_by_county"]["cap0"].get(c, 0)
        c1 = summary["annual_acft_by_county"]["cap1"].get(c, 0)
        print(f"  {c:12s} cap0={c0:>9,.0f}  cap1={c1:>9,.0f}  ac-ft/yr")
