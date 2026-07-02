"""WY savings bar chart with acreage lines on a secondary axis.

Bars (left axis): mean water savings per parcel-year (ac-ft/acre) for cap 0
and cap 1, pooled across all 10 counties.

Lines (right axis): alfalfa parcel-year acreage per WY.
  - Total: sum of area_acres over all parcel-years that WY (the denominator
    behind the bar mean).
  - Contributing under cap 0: parcels with at least one late cut after the
    cutoff (n_late_cuts >= 1) — drives the "remove all late cuts" bar.
  - Contributing under cap 1: parcels with two or more late cuts
    (n_late_cuts >= 2) — drives the "cap at 1" bar.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.wy_type_provider import get_wy_type
from es_analysis.utils.units import mm_to_acft_per_acre
from es_analysis.utils.publication_style import (
    DOUBLE_COL_WIDTH,
    apply_style,
    save_pub_figure,
)


CAP_LINE_COLORS = {0: "#d62728", 1: "#ff7f0e"}     # red, orange
TOTAL_LINE_COLOR = "#000000"


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)


def _acres_by_wy(df: pd.DataFrame, wys) -> Dict[str, Dict[int, float]]:
    out = {"total": {}, 0: {}, 1: {}}
    for wy in wys:
        sub = df[df["WY"] == wy]
        out["total"][wy] = float(sub["area_acres"].sum())
        out[0][wy] = float(sub.loc[sub["n_late_cuts"].fillna(0) >= 1, "area_acres"].sum())
        out[1][wy] = float(sub.loc[sub["n_late_cuts"].fillna(0) >= 2, "area_acres"].sum())
    return out


def savings_by_wy_bar_combined_with_area(
    df_savings: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    unit: str = "acft_per_acre",
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    apply_style()

    prefix = "saved_corrected_mm_cap" if et_mode == "corrected" else "saved_mm_cap"
    valid_caps = [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]
    if not valid_caps:
        raise ValueError(f"No {prefix}{{K}} columns found for caps {cap_values}.")

    wys = sorted(df_savings["WY"].unique())
    acres = _acres_by_wy(df_savings, wys)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.15, DOUBLE_COL_WIDTH * 0.55))

    def _convert(mm_val):
        return float(mm_to_acft_per_acre(mm_val)) if unit == "acft_per_acre" else mm_val

    n_caps = len(valid_caps)
    bar_width = 0.7 / max(n_caps, 1)
    x = np.arange(len(wys))

    cap_colors = {0: "#1f77b4", 1: "#7fbfdf", 2: "#b0d4e8", 3: "#d4e8f0"}
    cap_labels = {0: "Remove all late cuts (cap 0)", 1: "Cap at 1 late cut"}

    bar_handles = []
    for j, k in enumerate(valid_caps):
        col = f"{prefix}{k}"
        vals, sds = [], []
        for wy in wys:
            sub = df_savings[df_savings["WY"] == wy]
            mm_vals = sub[col].dropna()
            vals.append(_convert(float(mm_vals.mean())) if len(mm_vals) > 0 else 0.0)
            sd = float(mm_vals.std()) if len(mm_vals) > 1 else 0.0
            sds.append(_convert(sd))
        offset = (j - n_caps / 2 + 0.5) * bar_width
        label = cap_labels.get(k, f"Cap at {k}")
        bh = ax.bar(
            x + offset, vals, width=bar_width * 0.88,
            yerr=sds, capsize=3,
            color=cap_colors.get(k, "#aaaaaa"),
            edgecolor="black", linewidth=0.5,
            label=label,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        bar_handles.append(bh)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"WY{wy}\n({get_wy_type(wy)})" for wy in wys],
        rotation=0, ha="center", fontsize=8,
    )
    ylabel = "Mean water savings (ac-ft/acre)" if unit == "acft_per_acre" else "Mean water savings (mm)"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    mode_label = {"actual": "Actual ET", "corrected": "Corrected ET", "both": "Actual & Corrected ET"}[et_mode]
    n_counties = df_savings["county"].nunique()
    ax.set_title(
        f"Water savings by water year — {mode_label}, "
        f"{n_counties} counties pooled, mean per parcel-year",
        fontsize=9, pad=24,
    )
    ax.set_ylim(bottom=0)
    _style_ax(ax)

    # Acreage lines on secondary axis
    ax2 = ax.twinx()
    total = [acres["total"][wy] for wy in wys]
    line_handles = []
    h, = ax2.plot(
        x, total, linestyle="-", color=TOTAL_LINE_COLOR,
        linewidth=1.6, label="Total alfalfa acreage", zorder=5,
    )
    line_handles.append(h)
    for k in valid_caps:
        if k not in acres:
            continue
        vals = [acres[k][wy] for wy in wys]
        ls = (0, (4, 2)) if k == 0 else (0, (2, 2))
        lab = (
            "Contributing acreage (≥1 late cut, cap 0)"
            if k == 0
            else "Contributing acreage (≥2 late cuts, cap 1)"
        )
        h, = ax2.plot(
            x, vals, linestyle=ls,
            color=CAP_LINE_COLORS.get(k, "#888"),
            linewidth=1.6, label=lab, zorder=5,
        )
        line_handles.append(h)

    ax2.set_ylabel("Alfalfa acreage (acres)", fontsize=9, color="#333")
    ax2.tick_params(axis="y", labelsize=8, colors="#333")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.set_ylim(0, max(total) * 1.15)

    handles = list(bar_handles) + line_handles
    labels = [h.get_label() for h in handles]
    ax.legend(
        handles, labels,
        fontsize=7, ncol=min(len(handles), 5),
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        frameon=False, borderaxespad=0.0,
        columnspacing=1.2, handlelength=2.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    summary = {
        "cap_values": list(valid_caps),
        "et_mode": et_mode,
        "n_water_years": len(wys),
        "acres_total_by_wy": {int(k): v for k, v in acres["total"].items()},
        "acres_cap0_contrib_by_wy": {int(k): v for k, v in acres[0].items()},
        "acres_cap1_contrib_by_wy": {int(k): v for k, v in acres[1].items()},
    }

    if out_dir is not None:
        save_pub_figure(fig, "savings_by_wy_bar_combined_with_area", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, ax, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    csv_path = Path(__file__).parent.parent.parent / "output" / "figures" / "alfalfa_run_6" / "water_savings" / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = csv_path.parent
    fig, ax, summary = savings_by_wy_bar_combined_with_area(df, cap_values=(0, 1), out_dir=out)
    print("Wrote to:", out)
    for wy in sorted(summary["acres_total_by_wy"]):
        t = summary["acres_total_by_wy"][wy]
        c0 = summary["acres_cap0_contrib_by_wy"][wy]
        c1 = summary["acres_cap1_contrib_by_wy"][wy]
        print(f"  WY{wy}  total={t:>8,.0f}  cap0={c0:>8,.0f}  cap1={c1:>8,.0f}")
