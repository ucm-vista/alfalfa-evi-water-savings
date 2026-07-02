"""Total water saved by water year (ac-ft, 10 counties pooled).

Companion to the per-acre WY savings chart. Shows annual volume saved
across all 10 study counties per WY.

For each parcel-year, total saved (ac-ft) = saved_mm × area_acres / 304.8.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.wy_type_provider import get_wy_type
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


def total_savings_by_wy_bar(
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

    wys = sorted(df_savings["WY"].unique())

    df = df_savings.copy()
    for k in valid_caps:
        df[f"_acft_cap{k}"] = df[f"{prefix}{k}"].fillna(0) * df["area_acres"] / MM_PER_FOOT

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.15, DOUBLE_COL_WIDTH * 0.55))
    x = np.arange(len(wys))
    bar_width = 0.7 / max(len(valid_caps), 1)

    by_cap = {}
    for j, k in enumerate(valid_caps):
        vals = []
        for wy in wys:
            sub = df[df["WY"] == wy]
            vals.append(float(sub[f"_acft_cap{k}"].sum()))
        offset = (j - len(valid_caps) / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, vals, width=bar_width * 0.88,
            color=CAP_COLORS.get(k, "#888"),
            edgecolor="black", linewidth=0.5,
            label=CAP_LABELS.get(k, f"Cap at {k}"),
        )
        by_cap[k] = dict(zip([int(w) for w in wys], vals))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"WY{wy}\n({get_wy_type(wy)})" for wy in wys],
        rotation=0, ha="center", fontsize=8,
    )
    ax.set_ylabel("Total water saved (ac-ft / year)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    mode_label = "Corrected" if et_mode == "corrected" else "Actual"
    n_counties = df_savings["county"].nunique()
    ax.set_title(
        f"Total water saved by water year ({mode_label} ET, "
        f"{n_counties} counties pooled)",
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
        "Per-parcel-year saved (ac-ft) = saved_mm × area_acres / 304.8, "
        "summed over all parcel-years that WY.",
        ha="center", fontsize=7, style="italic", color="#444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    summary = {
        "et_mode": et_mode,
        "n_water_years": len(wys),
        "annual_acft_by_wy": {f"cap{k}": by_cap[k] for k in valid_caps},
    }

    if out_dir is not None:
        suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"total_savings_by_wy_bar{suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, ax, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    csv_path = Path(__file__).parent.parent.parent / "output" / "figures" / "alfalfa_run_6" / "water_savings" / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = csv_path.parent
    fig, ax, summary = total_savings_by_wy_bar(df, out_dir=out)
    print("Wrote to:", out)
    for wy in sorted(summary["annual_acft_by_wy"]["cap0"]):
        c0 = summary["annual_acft_by_wy"]["cap0"][wy]
        c1 = summary["annual_acft_by_wy"]["cap1"][wy]
        print(f"  WY{wy}  cap0={c0:>9,.0f}  cap1={c1:>9,.0f}  ac-ft")
