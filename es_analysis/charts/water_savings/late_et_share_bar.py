"""Late-season ET as share of total seasonal ET by county.

Stacked/grouped bar chart showing the proportion of total cycle ET
consumed by late-season (post-cutoff) cuttings. Directly supports
the paper's argument that late-season cuttings are not worth the
water cost.

Supports et_mode: "actual", "corrected", or "both" (side-by-side).
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


def late_et_share_bar(
    df: pd.DataFrame,
    group_by: str = "county",
    et_mode: str = "actual",
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Bar chart of late-season ET as percentage of total cycle ET.

    Args:
        df: Parcel-year DataFrame with late_et_mm and total_et_mm columns.
            When et_mode includes "corrected", also needs
            late_et_corrected_mm and total_et_corrected_mm.
        group_by: "county" or "WY" — x-axis grouping.
        et_mode: "actual", "corrected", or "both".
        out_dir: Output directory.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    apply_style()

    if "total_et_mm" not in df.columns:
        raise ValueError(
            "Column 'total_et_mm' not found. Re-run late-water workflow "
            "with latest provider code to include total_et_mm."
        )

    work = df[df["total_et_mm"] > 0].copy()

    # Compute share
    work["late_share_pct"] = 100.0 * work["late_et_mm"] / work["total_et_mm"]
    if "late_et_corrected_mm" in work.columns and "total_et_corrected_mm" in work.columns:
        mask = work["total_et_corrected_mm"] > 0
        work.loc[mask, "late_share_corrected_pct"] = (
            100.0 * work.loc[mask, "late_et_corrected_mm"]
            / work.loc[mask, "total_et_corrected_mm"]
        )

    if group_by == "county":
        groups = [c for c in COUNTY_ORDER if c in work["county"].unique()]
        group_col = "county"
        x_labels = groups
    elif group_by == "WY":
        groups = sorted(work["WY"].unique())
        group_col = "WY"
        x_labels = [f"WY{int(g)}" for g in groups]
    else:
        raise ValueError(f"group_by must be 'county' or 'WY', got '{group_by}'")

    # Bars to plot
    bars = []
    if et_mode in ("actual", "both"):
        bars.append(("late_share_pct", "Actual ET", WONG_PALETTE[1]))
    if et_mode in ("corrected", "both") and "late_share_corrected_pct" in work.columns:
        bars.append(("late_share_corrected_pct", "Corrected ET", WONG_PALETTE[3]))

    n_bars = len(bars)
    bar_width = 0.7 / max(n_bars, 1)
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.1, DOUBLE_COL_WIDTH * 0.45))

    for idx, (col, label, color) in enumerate(bars):
        means = []
        sds = []
        for g in groups:
            sub = work[work[group_col] == g]
            vals = sub[col].dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            sds.append(float(vals.std()) if len(vals) > 1 else 0.0)

        offset = (idx - n_bars / 2 + 0.5) * bar_width if n_bars > 1 else 0
        rects = ax.bar(
            x + offset, means, width=bar_width * 0.88,
            yerr=sds, capsize=3,
            color=color, edgecolor="black", linewidth=0.5,
            label=label,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        # Annotate values
        for rect, m in zip(rects, means):
            if m > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35 if group_by == "county" else 0, ha="right")
    mode_label = {"actual": "Actual ET", "corrected": "Corrected ET", "both": "Actual & Corrected ET"}[et_mode]
    scope = "WY2019\u20132024 pooled" if group_by == "county" else "all counties pooled"
    ax.set_ylabel("Late-season ET share (%)")
    ax.set_title(
        f"Late-season ET share of total cut-cycle ET ({mode_label}, {scope})",
        fontsize=7,
    )
    ax.legend(fontsize=6, loc="upper right", frameon=False)
    _style_ax(ax)
    fig.tight_layout()

    # Summary
    overall = float(work["late_share_pct"].mean()) if "late_share_pct" in work.columns else 0.0
    summary = {
        "group_by": group_by,
        "et_mode": et_mode,
        "n_parcel_years": len(work),
        "overall_mean_pct": overall,
        "mean_by_group": {
            str(g): float(work[work[group_col] == g]["late_share_pct"].mean())
            for g in groups
        },
    }

    if out_dir is not None:
        mode_suffix = f"_{et_mode}" if et_mode != "actual" else ""
        save_pub_figure(fig, f"late_et_share_bar_{group_by}{mode_suffix}", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, ax, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.config import config

    csv_path = config.water_saving_out_dir / "late_cut_base_parcel_year.csv"
    df = pd.read_csv(csv_path)
    out = Path(__file__).parent.parent.parent / "output" / "figures" / "water_savings"

    # By county
    fig, ax, summary = late_et_share_bar(df, group_by="county", out_dir=out)
    print(f"Overall mean late ET share: {summary['overall_mean_pct']:.1f}%")
    print(f"By county: {summary['mean_by_group']}")

    # By WY
    fig, ax, summary = late_et_share_bar(df, group_by="WY", out_dir=out)
    print(f"By WY: {summary['mean_by_group']}")
