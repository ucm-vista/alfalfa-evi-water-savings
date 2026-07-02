"""Late-cut frequency distribution bar chart.

Shows how many parcel-years have 0, 1, 2, 3, 4+ late cuttings,
with mean late-season ET by bin count.

Supports et_mode: "actual", "corrected", or "both" (overlay on panel b).
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


def late_cut_frequency_bar(
    df: pd.DataFrame,
    max_display: int = 5,
    by_county: bool = False,
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Two-panel figure: (a) count histogram, (b) mean late ET by n_late_cuts.

    Args:
        df: Parcel-year DataFrame with n_late_cuts and late_et_mm columns.
            When et_mode includes "corrected", also needs late_et_corrected_mm.
        max_display: Group all values >= max_display into "N+" bin.
        by_county: If True, panel (a) uses stacked bars colored by county.
        et_mode: "actual", "corrected", or "both".
        out_dir: Output directory for save_pub_figure.

    Returns:
        Tuple of (figure, axes_array, summary_dict).
    """
    apply_style()

    work = df.copy()
    work["n_late_bin"] = work["n_late_cuts"].clip(upper=max_display)

    bin_labels = {i: str(i) for i in range(max_display)}
    bin_labels[max_display] = f"{max_display}+"

    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_WIDTH * 1.2, DOUBLE_COL_WIDTH * 0.45),
    )

    # --- Panel (a): Count distribution ---
    ax_count = axes[0]

    if by_county:
        present = [c for c in COUNTY_ORDER if c in work["county"].unique()]
        colors = {c: WONG_PALETTE[(i + 1) % len(WONG_PALETTE)] for i, c in enumerate(present)}
        bins = sorted(work["n_late_bin"].unique())
        bottom = np.zeros(len(bins))
        for county in present:
            sub = work[work["county"] == county]
            counts = sub.groupby("n_late_bin").size().reindex(bins, fill_value=0)
            ax_count.bar(
                np.arange(len(bins)), counts.values, bottom=bottom,
                color=colors[county], edgecolor="white", linewidth=0.3,
                label=county,
            )
            bottom += counts.values
        ax_count.legend(fontsize=5, ncol=2, loc="upper right", framealpha=0.7)
        ax_count.set_xticks(np.arange(len(bins)))
        ax_count.set_xticklabels([bin_labels.get(b, str(b)) for b in bins])
    else:
        freq = work.groupby("n_late_bin").size().sort_index()
        bins = freq.index.tolist()
        ax_count.bar(
            np.arange(len(bins)), freq.values,
            color=WONG_PALETTE[5], edgecolor="black", linewidth=0.5,
        )
        total = freq.sum()
        for i, (b, cnt) in enumerate(zip(bins, freq.values)):
            pct = 100 * cnt / total
            ax_count.text(
                i, cnt + total * 0.01, f"{cnt:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=6,
            )
        ax_count.set_xticks(np.arange(len(bins)))
        ax_count.set_xticklabels([bin_labels.get(b, str(b)) for b in bins])

    ax_count.set_xlabel("Number of late cuttings (after Jul 1)")
    ax_count.set_ylabel("Parcel-years (count)")
    ax_count.set_title("Late-cut frequency distribution", fontsize=8)
    _style_ax(ax_count)
    add_panel_label(ax_count, "a")

    # --- Panel (b): Mean late ET by n_late_cuts ---
    ax_et = axes[1]

    # Determine which ET columns to show
    et_cols = []
    if et_mode in ("actual", "both"):
        et_cols.append(("late_et_mm", "Actual ET", WONG_PALETTE[1], 1.0))
    if et_mode in ("corrected", "both"):
        if "late_et_corrected_mm" in work.columns:
            et_cols.append(("late_et_corrected_mm", "Corrected ET", WONG_PALETTE[3], 0.85 if et_mode == "both" else 1.0))

    n_et = len(et_cols)
    bar_width = 0.7 / max(n_et, 1)

    for idx, (col, label, color, alpha) in enumerate(et_cols):
        if col not in work.columns:
            continue
        stats = (
            work.groupby("n_late_bin")[col]
            .agg(["mean", "median", "count", "std"])
            .sort_index()
        )
        bins_et = stats.index.tolist()
        x_et = np.arange(len(bins_et))
        sd = stats["std"]

        offset = (idx - n_et / 2 + 0.5) * bar_width if n_et > 1 else 0
        ax_et.bar(
            x_et + offset, stats["mean"].values,
            width=bar_width * 0.88, yerr=sd.values, capsize=3,
            color=color, edgecolor="black", linewidth=0.5,
            alpha=alpha, label=label,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        # Overlay median as markers
        ax_et.scatter(
            x_et + offset, stats["median"].values,
            marker="D", s=20, color="black", zorder=5,
            label="Median" if idx == 0 else None,
        )

    # Use bins from the first available ET column
    primary_col = et_cols[0][0] if et_cols else "late_et_mm"
    if primary_col in work.columns:
        bins_final = work.groupby("n_late_bin")[primary_col].mean().sort_index().index.tolist()
    else:
        bins_final = sorted(work["n_late_bin"].unique())
    ax_et.set_xticks(np.arange(len(bins_final)))
    ax_et.set_xticklabels([bin_labels.get(b, str(b)) for b in bins_final])
    mode_label = {"actual": "actual", "corrected": "corrected", "both": "actual & corrected"}[et_mode]
    ax_et.set_xlabel("Number of late cuttings (after Jul 1)")
    ax_et.set_ylabel(f"Mean late-season ET ({mode_label}, mm)")
    ax_et.set_title("Mean late ET by late-cut count", fontsize=8)
    ax_et.legend(fontsize=6, loc="upper left", framealpha=0.7)
    _style_ax(ax_et)
    add_panel_label(ax_et, "b")

    n_py = len(work)
    n_counties = work["county"].nunique() if "county" in work.columns else 0
    fig.suptitle(
        f"Late-Cutting Frequency \u2014 {n_py:,} parcel-years, "
        f"{n_counties} counties, WY2019\u20132024",
        fontsize=8, y=1.02,
    )
    fig.tight_layout()

    # Summary
    primary_stats = work.groupby("n_late_bin")[primary_col].mean().sort_index() if primary_col in work.columns else pd.Series()
    summary = {
        "n_parcel_years": len(work),
        "et_mode": et_mode,
        "distribution": {
            bin_labels.get(b, str(b)): int(work[work["n_late_bin"] == b].shape[0])
            for b in sorted(work["n_late_bin"].unique())
        },
        "mean_late_et_by_bin": {
            bin_labels.get(b, str(b)): float(primary_stats.loc[b])
            for b in primary_stats.index
        },
        "pct_with_late_cuts": float(
            100 * (work["n_late_cuts"] > 0).sum() / len(work)
        ),
    }

    if out_dir is not None:
        suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"late_cut_frequency_bar{suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.config import config

    csv_path = config.water_saving_out_dir / "late_cut_base_parcel_year.csv"
    df = pd.read_csv(csv_path)
    out = Path(__file__).parent.parent.parent / "output" / "figures" / "water_savings"
    fig, axes, summary = late_cut_frequency_bar(df, out_dir=out)
    print(f"Distribution: {summary['distribution']}")
    print(f"Parcels with late cuts: {summary['pct_with_late_cuts']:.1f}%")
