"""Separate daily + monthly ET correction two-panel plot.

Top panel: Daily OpenET scatter (red), harvest dates (X), Landsat passes
           (diamonds), ETof dashed on right y-axis.
Bottom panel: Monthly ET bars (actual lightgrey, corrected green), CI errorbars.

Supports ``et_mode``:
  - "actual"    : monthly grey bars only (no corrected)
  - "corrected" : monthly green bars only
  - "both"      : both overlaid (default)

The daily panel always shows raw OpenET ET — correction is applied only at
the monthly level via ETof-based off-phase adjustment.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.et_provider import (
    compute_daily_and_monthly_for_uid,
    _norm_county_name,
    water_year_bounds,
)
from es_analysis.utils.publication_style import (
    DOUBLE_COL_WIDTH,
    apply_style,
)

COLOR_ACTUAL = "red"
COLOR_CORRECTED = "#6BCB77"
COLOR_ACTUAL_BAR = "lightgrey"


def _month_centers(month_starts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Compute mid-month timestamps for bar centering."""
    centers = []
    for m0 in month_starts:
        m0 = pd.to_datetime(m0).normalize()
        mend = (m0 + pd.offsets.MonthEnd(0)).normalize()
        n_days = (mend - m0).days + 1
        centers.append(m0 + pd.Timedelta(days=n_days / 2.0))
    return pd.DatetimeIndex(centers)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def et_separate_daily_monthly(
    county: str,
    water_year: int,
    uid: str,
    method: str = "A",
    et_mode: str = "both",
    ci_alpha: float = 0.10,
    n_boot: int = 400,
    cloud_cover_max: float = 20.0,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Generate a two-panel daily+monthly ET correction plot.

    Args:
        county: County name.
        water_year: Water year.
        uid: Parcel UniqueID.
        method: Correction method "A" or "B".
        et_mode: "actual", "corrected", or "both".
        ci_alpha: CI alpha for bootstrap (default 0.10 = 90% CI).
        n_boot: Number of bootstrap replicates.
        cloud_cover_max: Max cloud cover for Landsat passes.
        output_dir: Where to save. Defaults to es_analysis/output/figures/et_corrections/.

    Returns:
        Dict with figure metadata.
    """
    apply_style()

    county_norm = _norm_county_name(county)
    start, end = water_year_bounds(water_year)

    daily_df, monthly_df, harvest_dates, passes_df = compute_daily_and_monthly_for_uid(
        county=county_norm,
        wy=water_year,
        uid=str(uid),
        chosen_method=method.upper(),
        ci_alpha=ci_alpha,
        n_boot=n_boot,
        cloud_cover_max=cloud_cover_max,
    )

    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL_WIDTH * 1.6, DOUBLE_COL_WIDTH * 0.9),
                             sharex=True)

    # Title at top of figure
    fig.suptitle(
        f"{county_norm} | WY{water_year} | UID {uid} | Method {method.upper()}",
        fontsize=10, y=0.98,
    )

    # ---- Top panel: Daily OpenET + satellite pass + harvest + ETof ----
    ax_daily = axes[0]

    ax_daily.scatter(
        daily_df.index, daily_df["ET_open"].to_numpy(float),
        s=14, marker="o", facecolors=COLOR_ACTUAL, edgecolors="black",
        linewidths=0.2, alpha=0.45, label="Daily ET (OpenET)", zorder=3,
    )

    y_max_daily = max(float(np.nanmax(daily_df["ET_open"].to_numpy())), 0.1) * 1.25

    if harvest_dates:
        ax_daily.scatter(
            harvest_dates, [y_max_daily * 0.93] * len(harvest_dates),
            marker="x", s=80, color="black", linewidths=1.8,
            label="Harvest dates", zorder=5,
        )

    if not passes_df.empty:
        pass_dates = pd.to_datetime(passes_df["date_only"].unique())
        ax_daily.scatter(
            pass_dates, [y_max_daily * 0.85] * len(pass_dates),
            marker="D", s=50, facecolors="none", edgecolors="black",
            linewidths=1.2, label="Landsat passes", zorder=5,
        )

    # ETof on right y-axis
    ax_etof = ax_daily.twinx()
    ax_etof.plot(
        daily_df.index, daily_df["ETof"].to_numpy(float),
        color="black", lw=1.2, linestyle="--", label="ETof", zorder=2,
    )
    ax_etof.set_ylabel("ETof (fraction of ETo)")

    ax_daily.set_ylim(0, y_max_daily)
    ax_daily.set_ylabel("Daily ET (mm/day)")
    _style_ax(ax_daily)

    # Legend above the top panel
    h1, l1 = ax_daily.get_legend_handles_labels()
    h2, l2 = ax_etof.get_legend_handles_labels()
    ax_daily.legend(
        h1 + h2, l1 + l2, fontsize=6, ncol=4,
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )

    # ---- Bottom panel: Monthly ET bars ----
    ax_month = axes[1]
    m_starts = pd.DatetimeIndex(monthly_df.index)
    m_centers = _month_centers(m_starts)
    bar_w = pd.Timedelta(days=14)
    off = pd.Timedelta(days=7)

    if et_mode == "both":
        x_actual = m_centers - off
        x_corr = m_centers + off
    else:
        x_actual = m_centers
        x_corr = m_centers

    if et_mode in ("actual", "both"):
        ax_month.bar(
            x_actual, monthly_df["ET_open"].to_numpy(float),
            width=bar_w, align="center", color=COLOR_ACTUAL_BAR,
            edgecolor="black", linewidth=0.6, alpha=0.85,
            label="Monthly ET (OpenET)", zorder=1,
        )

    if et_mode in ("corrected", "both"):
        ax_month.bar(
            x_corr, monthly_df["ET_corr"].to_numpy(float),
            width=bar_w, align="center", color=COLOR_CORRECTED,
            edgecolor="black", linewidth=0.8, alpha=0.95,
            label=f"Monthly ET (corrected, {method.upper()})", zorder=2,
        )

        if monthly_df["ET_corr_ci_low"].notna().any():
            y_mid = monthly_df["ET_corr"].to_numpy(float)
            y_lo = monthly_df["ET_corr_ci_low"].to_numpy(float)
            y_hi = monthly_df["ET_corr_ci_high"].to_numpy(float)
            err_lower = np.clip(y_mid - y_lo, 0.0, np.inf)
            err_upper = np.clip(y_hi - y_mid, 0.0, np.inf)
            ax_month.errorbar(
                x_corr, y_mid, yerr=[err_lower, err_upper],
                fmt="none", ecolor="black", elinewidth=1.5,
                capsize=4, capthick=1.5,
                label=f"CI ({int((1 - ci_alpha) * 100)}%)", zorder=3,
            )

    ax_month.set_xlim(start, end)
    ax_month.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_month.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_month.set_xlabel("Water year timeline")
    ax_month.set_ylabel("Monthly ET (mm/month)")
    _style_ax(ax_month)

    # Legend above the bottom panel
    ax_month.legend(
        fontsize=6, ncol=3,
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save — organized as {base}/et_corrections/{County}/WY{year}/
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "output" / "figures" / "et_corrections"
    county_dir = output_dir / county_norm.replace(" ", "_") / f"WY{water_year}"
    county_dir.mkdir(parents=True, exist_ok=True)
    fname = f"UID{uid}_separate_{et_mode}_{method.upper()}"
    for ext in ("png", "pdf"):
        fig.savefig(county_dir / f"{fname}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  Saved: {county_dir / fname}.png")
    plt.close(fig)

    return {
        "figure_path": str(county_dir / f"{fname}.png"),
        "county": county_norm,
        "water_year": water_year,
        "uid": uid,
        "method": method.upper(),
        "et_mode": et_mode,
        "n_harvest_dates": len(harvest_dates),
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    result = et_separate_daily_monthly("Fresno", 2022, "1000891", "A", "both")
    print("Created:", result["figure_path"])
