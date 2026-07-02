"""
ET Full Two Panel Plot

Generates a two-panel stacked plot:
TOP: Daily ET + ETof, BOTTOM: Monthly ET bars with CI.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from es_analysis.data_providers.et_provider import (
    compute_daily_and_monthly_for_uid,
    _norm_county_name,
    water_year_bounds,
)
from es_analysis.utils.plotting import save_figure


def _month_centers(month_starts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    centers = []
    for m0 in month_starts:
        m0 = pd.to_datetime(m0).normalize()
        mend = (m0 + pd.offsets.MonthEnd(0)).normalize()
        n_days = (mend - m0).days + 1
        centers.append(m0 + pd.Timedelta(days=n_days / 2.0))
    return pd.DatetimeIndex(centers)


def et_full_two_panel_plot(
    county: str,
    water_year: int,
    uid: str,
    method: str = "A",
    ci_alpha: float = 0.10,
    use_left_daily_axis: bool = True,
    left_daily_axis_offset: float = -0.10,
    cloud_cover_max: float = 20.0,
    output_dir: Path = Path("es_analysis/output/figures"),
) -> Dict[str, str]:
    county_norm = _norm_county_name(county)
    start, end = water_year_bounds(water_year)

    daily_df, monthly_df, harvest_dates, passes_df = compute_daily_and_monthly_for_uid(
        county=county_norm,
        wy=water_year,
        uid=str(uid),
        chosen_method=method.upper(),
        ci_alpha=ci_alpha,
        cloud_cover_max=cloud_cover_max,
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(right=0.78)

    m_starts = pd.DatetimeIndex(monthly_df.index)
    m_centers = _month_centers(m_starts)

    bar_w = pd.Timedelta(days=14)
    off = pd.Timedelta(days=7)

    x_actual = m_centers - off
    x_corr = m_centers + off

    color_actual = "lightgrey"
    color_corr = "#6BCB77"

    ax.bar(
        x_actual,
        monthly_df["ET_open"].to_numpy(float),
        width=bar_w,
        align="center",
        color=color_actual,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        label="Monthly ET (OpenET)",
        zorder=1,
    )

    ax.bar(
        x_corr,
        monthly_df["ET_corr"].to_numpy(float),
        width=bar_w,
        align="center",
        color=color_corr,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.95,
        label=f"Monthly ET (corrected, method {str(method).upper()})",
        zorder=2,
    )

    if monthly_df["ET_corr_ci_low"].notna().any():
        y_mid = monthly_df["ET_corr"].to_numpy(float)
        y_lo = monthly_df["ET_corr_ci_low"].to_numpy(float)
        y_hi = monthly_df["ET_corr_ci_high"].to_numpy(float)
        err_lower = np.clip(y_mid - y_lo, 0.0, np.inf)
        err_upper = np.clip(y_hi - y_mid, 0.0, np.inf)

        ax.errorbar(
            x_corr,
            y_mid,
            yerr=[err_lower, err_upper],
            fmt="none",
            ecolor="black",
            elinewidth=1.8,
            capsize=5,
            capthick=1.8,
            label=f"Corrected ET CI ({int((1-ci_alpha)*100)}%)",
            zorder=3,
        )

    ax.set_xlim(start, end)
    y_max_month = max(
        float(np.nanmax(monthly_df[["ET_open", "ET_corr"]].to_numpy())),
        1.0,
    ) * 1.25
    ax.set_ylim(0, y_max_month)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    if harvest_dates:
        ax.scatter(
            harvest_dates,
            [y_max_month * 0.93] * len(harvest_dates),
            marker="x",
            s=90,
            color="black",
            linewidths=2.0,
            label="Harvest dates (cuts)",
            zorder=5,
        )

    if not passes_df.empty:
        pass_dates = pd.to_datetime(passes_df["date_only"].unique())
        ax.scatter(
            pass_dates,
            [y_max_month * 0.85] * len(pass_dates),
            marker="D",
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            label=f"Landsat passes (≤ {cloud_cover_max:.0f}% CC, single track)",
            zorder=5,
        )

    ax_daily: Optional[plt.Axes] = None
    if use_left_daily_axis:
        ax_daily = ax.twinx()
        ax_daily.spines["right"].set_visible(False)
        ax_daily.spines["left"].set_visible(True)
        ax_daily.spines["left"].set_position(("axes", left_daily_axis_offset))
        ax_daily.yaxis.set_label_position("left")
        ax_daily.yaxis.tick_left()
        ax_daily.patch.set_visible(False)

        ax_daily.scatter(
            daily_df.index,
            daily_df["ET_open"].to_numpy(float),
            s=16,
            marker="o",
            facecolors="red",
            edgecolors="black",
            linewidths=0.2,
            alpha=0.45,
            label="Daily ET (OpenET)",
            zorder=4,
        )

        y_max_daily = max(float(np.nanmax(daily_df["ET_open"].to_numpy())), 0.1) * 1.25
        ax_daily.set_ylim(0, y_max_daily)
        ax_daily.set_ylabel("Daily ET (mm/day)")
    else:
        ax.plot(
            daily_df.index,
            daily_df["ET_open"].to_numpy(float),
            color="red",
            lw=1.1,
            label="Daily ET (OpenET)",
            zorder=4,
        )

    ax2 = ax.twinx()
    ax2.plot(
        daily_df.index,
        daily_df["ETof"].to_numpy(float),
        color="black",
        lw=1.4,
        linestyle="--",
        label="ETof (fraction of ETo)",
    )
    ax2.set_ylabel("Fraction of ETo (–)")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Water year timeline")
    ax.set_ylabel("Monthly ET (mm/month)")
    ax.set_title(
        f"{county_norm} | WY{water_year} | UID {uid} – Off-phase correction (method {str(method).upper()})"
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    if ax_daily is not None:
        linesd, labelsd = ax_daily.get_legend_handles_labels()
    else:
        linesd, labelsd = [], []
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        lines1 + linesd + lines2,
        labels1 + labelsd + labels2,
        loc="center left",
        bbox_to_anchor=(1.08, 0.50),
        borderaxespad=0.0,
        frameon=False,
    )

    output_path = output_dir / "et_corrections" / f"{county_norm}_WY{water_year}_UID{uid}_full_twopanel.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(output_path),
        "county": county_norm,
        "water_year": water_year,
        "uid": uid,
        "method": method.upper(),
        "ci_alpha": ci_alpha,
        "n_harvest_dates": len(harvest_dates),
        "n_landsat_passes": len(passes_df) if not passes_df.empty else 0,
        "figure_type": "full_twopanel",
    }


if __name__ == "__main__":
    result = et_full_two_panel_plot("Fresno", 2020, "1003334", "A", 0.10, True, -0.10, 20.0)
    print("Created:", result["figure_path"])