"""County water-savings figure — two stacked panels.

Panel (a): mean water savings per parcel-year (ac-ft/acre), cap 0 & cap 1.
Panel (b): total water saved per year (ac-ft/yr, annual mean over WYs).

Each mean bar sits in front of a faded "maximum" bar in the same color
(panel a: max parcel-year value; panel b: max annual total across WYs).
Each bar pair shows two stacked value labels above the faded max bar:
the max value on top (small, grey) and the mean value below it
(black, bold, with a thin white stroke for legibility).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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


CAP_COLORS = {0: "#1f77b4", 1: "#7fbfdf"}
CAP_LABELS = {0: "Remove all late cuts (cap 0)", 1: "Cap at 1 late cut"}
MAX_ALPHA = 0.18
MM_PER_FOOT = 304.8

# Honest in-figure ET label + filename suffix per ET mode. The underlying data
# column is chosen by ``et_mode`` (saved_mm_cap* = uncorrected OpenET ETa;
# saved_corrected_mm_cap* = pass-timing-corrected).
ET_TITLE = {
    "actual": "Uncorrected OpenET ETa",
    "corrected": "Corrected ET",
    "both": "Actual vs Corrected ET",
}
ET_SUFFIX = {"actual": "_uncorrected", "corrected": "_corrected", "both": "_both"}


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)


def _draw_grouped_with_max(
    ax, x, mean_by_cap, max_by_cap, caps, bar_width,
    mean_fmt: str, max_fmt: str, label_fs_mean: int = 7, label_fs_max: int = 6,
):
    """Draw faded max bars behind mean bars; add two stacked labels above each pair."""
    n_caps = len(caps)
    mean_handles = []
    bar_positions = []   # list of (x_pos, mean_val, max_val)
    for j, k in enumerate(caps):
        offset = (j - n_caps / 2 + 0.5) * bar_width
        color = CAP_COLORS.get(k, "#888")
        ax.bar(
            x + offset, max_by_cap[k], width=bar_width * 0.88,
            color=color, alpha=MAX_ALPHA, edgecolor="none", linewidth=0,
            zorder=1,
        )
        bars = ax.bar(
            x + offset, mean_by_cap[k], width=bar_width * 0.88,
            color=color, edgecolor="black", linewidth=0.5,
            label=CAP_LABELS.get(k, f"Cap at {k}"),
            zorder=3,
        )
        mean_handles.append(bars)
        for xp, mv, xv in zip(x + offset, mean_by_cap[k], max_by_cap[k]):
            bar_positions.append((float(xp), float(mv), float(xv)))

    # Headroom so the stacked labels don't clip
    all_max = [xv for _, _, xv in bar_positions] or [0]
    ymax = max(all_max) * 1.22 if max(all_max) > 0 else 1.0
    ax.set_ylim(0, ymax)

    pad_mean = ymax * 0.012
    pad_max = ymax * 0.075
    pe_white = [pe.withStroke(linewidth=2, foreground="white")]
    for xp, mv, xv in bar_positions:
        ax.text(
            xp, xv + pad_mean, mean_fmt % mv,
            ha="center", va="bottom", fontsize=label_fs_mean,
            color="black", fontweight="bold",
            path_effects=pe_white, zorder=10,
        )
        ax.text(
            xp, xv + pad_max, max_fmt % xv,
            ha="center", va="bottom", fontsize=label_fs_max,
            color="#555", zorder=10,
        )

    max_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#777777", alpha=MAX_ALPHA, edgecolor="none",
        label="Maximum (faded background)",
    )
    return mean_handles, max_handle


def county_savings_combined_stacked(
    df_savings: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, object, Dict]:
    apply_style()

    prefix = "saved_corrected_mm_cap" if et_mode == "corrected" else "saved_mm_cap"
    caps = [k for k in cap_values if f"{prefix}{k}" in df_savings.columns]
    if not caps:
        raise ValueError(f"No {prefix}{{K}} columns found.")

    present = df_savings["county"].unique().tolist()
    counties = [c for c in COUNTY_ORDER if c in present]
    n = len(counties)
    n_wy = df_savings["WY"].nunique()
    x = np.arange(n)
    bar_width = 0.7 / max(len(caps), 1)

    # Panel (a): per-acre mean and max parcel-year value per county
    perac_mean = {k: [] for k in caps}
    perac_max = {k: [] for k in caps}
    for c in counties:
        sub = df_savings[df_savings["county"] == c]
        for k in caps:
            vals = sub[f"{prefix}{k}"].dropna()
            perac_mean[k].append(float(mm_to_acft_per_acre(vals.mean())) if len(vals) else 0.0)
            perac_max[k].append(float(mm_to_acft_per_acre(vals.max())) if len(vals) else 0.0)

    # Panel (b): annual mean total ac-ft and max annual total across WYs (per county)
    df = df_savings.copy()
    for k in caps:
        df[f"_acft_cap{k}"] = df[f"{prefix}{k}"].fillna(0) * df["area_acres"] / MM_PER_FOOT

    total_mean = {k: [] for k in caps}
    total_max = {k: [] for k in caps}
    for c in counties:
        sub = df[df["county"] == c]
        for k in caps:
            wy_totals = sub.groupby("WY")[f"_acft_cap{k}"].sum()
            total_mean[k].append(float(wy_totals.sum() / n_wy))
            total_max[k].append(float(wy_totals.max()) if len(wy_totals) else 0.0)

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1,
        figsize=(DOUBLE_COL_WIDTH * 1.4, DOUBLE_COL_WIDTH * 1.15),
        sharex=True,
    )

    mean_handles_a, max_handle = _draw_grouped_with_max(
        ax_a, x, perac_mean, perac_max, caps, bar_width,
        mean_fmt="%.2f", max_fmt="%.2f",
    )
    ax_a.set_ylabel("Mean water savings\n(ac-ft/acre per parcel-year)", fontsize=9)
    ax_a.tick_params(axis="y", labelsize=8)
    _style_ax(ax_a)
    add_panel_label(ax_a, "a")

    handles = list(mean_handles_a) + [max_handle]
    labels = [h.get_label() for h in handles]
    ax_a.legend(
        handles, labels,
        fontsize=8, ncol=len(handles),
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
        frameon=False, borderaxespad=0.0,
        columnspacing=1.2, handlelength=1.8,
    )

    _draw_grouped_with_max(
        ax_b, x, total_mean, total_max, caps, bar_width,
        mean_fmt="%.0f", max_fmt="%.0f",
    )
    ax_b.set_ylabel("Total water saved\n(ac-ft / year)", fontsize=9)
    ax_b.tick_params(axis="y", labelsize=8)
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _style_ax(ax_b)
    add_panel_label(ax_b, "b")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(counties, rotation=35, ha="right", fontsize=8)

    fig.suptitle(
        f"Water savings by county ({ET_TITLE.get(et_mode, et_mode)}, WY2019–2024)",
        fontsize=10, y=0.985,
    )
    fig.text(
        0.5, 0.005,
        "Bar labels: top = max, bottom = mean. "
        "Faded bar = within-group max "
        "(panel a: max parcel-year; panel b: max annual total across WYs).",
        ha="center", fontsize=7, style="italic", color="#444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    summary = {
        "et_mode": et_mode,
        "n_counties": n,
        "perac_mean": {f"cap{k}": dict(zip(counties, perac_mean[k])) for k in caps},
        "perac_max": {f"cap{k}": dict(zip(counties, perac_max[k])) for k in caps},
        "total_mean_acft_yr": {f"cap{k}": dict(zip(counties, total_mean[k])) for k in caps},
        "total_max_acft_yr": {f"cap{k}": dict(zip(counties, total_max[k])) for k in caps},
    }

    if out_dir is not None:
        suffix = ET_SUFFIX.get(et_mode, f"_{et_mode}")
        save_pub_figure(fig, f"county_savings_combined_stacked{suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, (ax_a, ax_b), summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    csv_path = Path(__file__).parent.parent.parent / "output" / "figures" / "alfalfa_run_6" / "water_savings" / "late_cut_savings_parcel_year_mm.csv"
    df = pd.read_csv(csv_path)
    out = csv_path.parent
    fig, axes, summary = county_savings_combined_stacked(df, out_dir=out)
    print("Wrote to:", out)
