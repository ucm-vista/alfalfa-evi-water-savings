"""All-counties savings bar chart with acreage lines on a secondary axis.

Bars (left axis): mean water savings per parcel-year (ac-ft/acre) for cap 0
and cap 1 — identical to savings_by_all_counties_bar.

Lines (right axis): alfalfa parcel-year acreage per county.
  - Total: sum of area_acres over all parcel-years (the denominator behind
    each bar's mean).
  - Contributing under cap 0: parcels with at least one late cut after the
    cutoff (n_late_cuts >= 1) — the acreage that actually drives the
    "remove all late cuts" bar.
  - Contributing under cap 1: parcels with two or more late cuts
    (n_late_cuts >= 2) — the acreage that drives the "cap at 1" bar.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.utils.units import mm_to_acft_per_acre
from es_analysis.utils.publication_style import (
    DOUBLE_COL_WIDTH,
    apply_style,
    save_pub_figure,
    add_panel_label,
)


CAP_LINE_COLORS = {0: "#d62728", 1: "#ff7f0e"}     # red, orange
TOTAL_LINE_COLOR = "#000000"


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)


def _acres_by_county(df: pd.DataFrame, counties) -> Dict[str, Dict[str, float]]:
    """Total + cap-conditioned parcel-year acreage per county."""
    out = {"total": {}, 0: {}, 1: {}}
    for c in counties:
        sub = df[df["county"] == c]
        out["total"][c] = float(sub["area_acres"].sum())
        out[0][c] = float(sub.loc[sub["n_late_cuts"].fillna(0) >= 1, "area_acres"].sum())
        out[1][c] = float(sub.loc[sub["n_late_cuts"].fillna(0) >= 2, "area_acres"].sum())
    return out


def _add_acreage_lines(ax_left, x, counties, acres, valid_caps):
    """Overlay total + per-cap contributing acreage lines on a twin axis."""
    ax2 = ax_left.twinx()

    total = [acres["total"][c] for c in counties]
    line_handles = []
    h, = ax2.plot(
        x, total, linestyle="-", color=TOTAL_LINE_COLOR,
        linewidth=1.6, label="Total alfalfa acreage",
        zorder=5,
    )
    line_handles.append(h)

    for k in valid_caps:
        if k not in acres:
            continue
        vals = [acres[k][c] for c in counties]
        ls = (0, (4, 2)) if k == 0 else (0, (2, 2))
        lab = (
            "Contributing acreage (≥1 late cut, cap 0)"
            if k == 0
            else "Contributing acreage (≥2 late cuts, cap 1)"
        )
        h, = ax2.plot(
            x, vals, linestyle=ls,
            color=CAP_LINE_COLORS.get(k, "#888"),
            linewidth=1.6, label=lab,
            zorder=5,
        )
        line_handles.append(h)

    ax2.set_ylabel("Alfalfa acreage (acres)", fontsize=9, color="#333")
    ax2.tick_params(axis="y", labelsize=8, colors="#333")
    # comma thousands formatting
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    # leave headroom above the highest line
    ymax = max(total) * 1.15
    ax2.set_ylim(0, ymax)
    return ax2, line_handles


def savings_by_all_counties_bar_with_area(
    df_savings: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    unit: str = "acft_per_acre",
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, object, Dict]:
    apply_style()

    present = df_savings["county"].unique().tolist()
    counties = [c for c in COUNTY_ORDER if c in present]
    n_counties = len(counties)
    acres = _acres_by_county(df_savings, counties)

    def _convert(mm_val):
        return float(mm_to_acft_per_acre(mm_val)) if unit == "acft_per_acre" else mm_val

    ylabel = (
        "Mean water savings (ac-ft/acre)" if unit == "acft_per_acre"
        else "Mean water savings (mm)"
    )

    if et_mode == "both":
        fig, axes = plt.subplots(
            2, 1,
            figsize=(DOUBLE_COL_WIDTH * 1.4, DOUBLE_COL_WIDTH * 1.0),
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

            bar_handles = []
            for j, k in enumerate(valid_caps):
                col = f"{prefix}{k}"
                vals, sds = [], []
                for county in counties:
                    sub = df_savings[df_savings["county"] == county]
                    mm_vals = sub[col].dropna()
                    vals.append(_convert(float(mm_vals.mean())) if len(mm_vals) > 0 else 0.0)
                    sd = float(mm_vals.std()) if len(mm_vals) > 1 else 0.0
                    sds.append(_convert(sd))

                offset = (j - n_caps / 2 + 0.5) * bar_width
                label = "Remove all late cuts" if k == 0 else f"Cap at {k}"
                bh = ax.bar(
                    x + offset, vals, width=bar_width * 0.88,
                    yerr=sds, capsize=2,
                    color=cap_colors[j % len(cap_colors)],
                    edgecolor="black", linewidth=0.5,
                    label=label,
                    error_kw={"elinewidth": 0.7, "capthick": 0.7},
                )
                bar_handles.append(bh)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(title_label, fontsize=8)
            _style_ax(ax)
            add_panel_label(ax, chr(ord("a") + panel_idx))

            ax2, line_handles = _add_acreage_lines(ax, x, counties, acres, valid_caps)

            if panel_idx == 0:
                handles = list(bar_handles) + line_handles
                labels = [h.get_label() for h in handles]
                ax.legend(
                    handles, labels,
                    fontsize=7, ncol=min(len(handles), 5),
                    loc="lower center", bbox_to_anchor=(0.5, 1.04),
                    frameon=False, borderaxespad=0.0,
                    columnspacing=1.2, handlelength=2.0,
                )

        axes[-1].set_xticks(np.arange(n_counties))
        axes[-1].set_xticklabels(counties, rotation=35, ha="right")
        fig.suptitle(
            "Water Savings by County — Actual & Corrected ET, "
            "WY2019–2024, mean per parcel-year",
            fontsize=8, y=1.01,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    else:
        prefix = "saved_corrected_mm_cap" if et_mode == "corrected" else "saved_mm_cap"
        valid_caps = [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]
        n_caps = len(valid_caps)
        if not valid_caps:
            raise ValueError(f"No {prefix}{{K}} columns found for caps {cap_values}.")

        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.4, DOUBLE_COL_WIDTH * 0.55))
        x = np.arange(n_counties)
        bar_width = 0.7 / max(n_caps, 1)
        cap_colors = ["#1f77b4", "#7fbfdf", "#b0d4e8", "#d4e8f0"]

        bar_handles = []
        for j, k in enumerate(valid_caps):
            col = f"{prefix}{k}"
            vals, sds = [], []
            for county in counties:
                sub = df_savings[df_savings["county"] == county]
                mm_vals = sub[col].dropna()
                vals.append(_convert(float(mm_vals.mean())) if len(mm_vals) > 0 else 0.0)
                sd = float(mm_vals.std()) if len(mm_vals) > 1 else 0.0
                sds.append(_convert(sd))

            offset = (j - n_caps / 2 + 0.5) * bar_width
            label = "Remove all late cuts" if k == 0 else f"Cap at {k}"
            bh = ax.bar(
                x + offset, vals, width=bar_width * 0.88,
                yerr=sds, capsize=2,
                color=cap_colors[j % len(cap_colors)],
                edgecolor="black", linewidth=0.5,
                label=label,
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
            )
            bar_handles.append(bh)

        mode_label = "Corrected" if et_mode == "corrected" else "Actual"
        ax.set_xticks(x)
        ax.set_xticklabels(counties, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(
            f"Water Savings by County ({mode_label} ET, WY2019–2024, mean per parcel-year)",
            fontsize=8, pad=24,
        )
        _style_ax(ax)

        ax2, line_handles = _add_acreage_lines(ax, x, counties, acres, valid_caps)

        handles = list(bar_handles) + line_handles
        labels = [h.get_label() for h in handles]
        ax.legend(
            handles, labels,
            fontsize=7, ncol=min(len(handles), 5),
            loc="lower center", bbox_to_anchor=(0.5, 1.02),
            frameon=False, borderaxespad=0.0,
            columnspacing=1.2, handlelength=2.0,
        )
        axes = ax
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    summary = {
        "n_counties": n_counties,
        "counties": counties,
        "et_mode": et_mode,
        "acres_total_by_county": acres["total"],
        "acres_cap0_contrib_by_county": acres[0],
        "acres_cap1_contrib_by_county": acres[1],
    }

    if out_dir is not None:
        suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"savings_by_all_counties_bar_combined_with_area{suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    csv_path = Path(__file__).parent.parent.parent / "output" / "figures" / "alfalfa_run_6" / "water_savings" / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = csv_path.parent
    fig, ax, summary = savings_by_all_counties_bar_with_area(df, et_mode="actual", out_dir=out)
    print("Wrote to:", out)
    for c in summary["counties"]:
        t = summary["acres_total_by_county"][c]
        c0 = summary["acres_cap0_contrib_by_county"][c]
        c1 = summary["acres_cap1_contrib_by_county"][c]
        print(f"  {c:12s}  total={t:>9,.0f}  cap0={c0:>9,.0f}  cap1={c1:>9,.0f}")
