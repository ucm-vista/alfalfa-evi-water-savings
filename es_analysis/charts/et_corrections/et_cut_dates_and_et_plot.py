"""
ET Cut Dates and ET Plot

Monthly ET (OpenET) side-by-side for water years.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

from es_analysis.data_providers.et_provider import (
    _load_openet_for_wy,
    _norm_county_name,
    water_year_bounds,
    _load_seasonal_csv,
    _parse_cp_dates_iso,
)


def _build_uid_colors(uids: list) -> dict:
    try:
        _get_cmap = mpl.colormaps.get_cmap
    except AttributeError:
        from matplotlib.cm import get_cmap as _get_cmap

    def _darken(c, f=0.90):
        r, g, b, a = mpl.colors.to_rgba(c)
        return (r*f, g*f, b*f, a)

    names = ["tab10", "Dark2", "Set1", "Paired", "tab20", "tab20b", "tab20c"]
    pool: list = []
    for nm in names:
        cmap = _get_cmap(nm)
        cols = getattr(cmap, "colors", None)
        if cols is None:
            cols = [cmap(i / max(1, cmap.N-1)) for i in range(cmap.N)]
        pool.extend([_darken(c, 0.97) for c in cols])
    return {uid: pool[i % len(pool)] for i, uid in enumerate(sorted(map(str, uids)))}


def et_cut_dates_and_et_plot(
    county: str,
    water_year: int,
    uid: str,
    filter_n_cp_season: Optional[int] = None,
    fig_width: float = 12.0,
    fig_height: float = 5.0,
    bar_width_days: float = 4.0,
    bar_dodge_days: float = 2.5,
    cross_y_fraction: float = 0.80,
    cross_stack_step: float = 0.035,
    marker_size: float = 55.0,
    marker_lw: float = 1.2,
    bar_alpha: float = 0.40,
    output_dir: Path = Path("es_analysis/output/figures"),
) -> Dict[str, str]:
    county_norm = _norm_county_name(county)
    start, end = water_year_bounds(int(water_year))

    df = _load_seasonal_csv(county_norm, water_year)

    if filter_n_cp_season is not None:
        m = (pd.to_numeric(df["n_cp_season"], errors="coerce") == int(filter_n_cp_season))
        df = df.loc[m].copy()

    df = df[df["UniqueID"].astype(str) == str(uid)]

    if df.empty:
        raise ValueError(f"No parcel found with UID={uid} in {county_norm}, WY{water_year}")

    cuts_by_uid: Dict[str, list] = {}
    for uid_i, sub in df.groupby("UniqueID", sort=False):
        lst = _parse_cp_dates_iso(sub["season_cp_dates_iso"].iloc[0] if "season_cp_dates_iso" in sub else "")
        lst = [d for d in lst if (d >= start) and (d <= end)]
        if lst:
            cuts_by_uid[str(uid_i)] = sorted(lst)

    if not cuts_by_uid:
        raise ValueError(f"No seasonal cut dates found for UID={uid} in WY{water_year}")

    uids = sorted(cuts_by_uid.keys(), key=lambda s: s)
    uid_colors = _build_uid_colors(uids)

    et_series = _load_openet_for_wy(county_norm, water_year, uids)

    bars = []
    crosses = []
    same_day_counts = {}

    uid_index = {uid: i for i, uid in enumerate(uids)}
    for uid_i in uids:
        series = et_series.get(uid_i)
        if series is None:
            continue
        last = start - pd.Timedelta(days=1)

        for cut_date in cuts_by_uid[uid_i]:
            if series.index.dtype.kind != 'M':
                series.index = pd.to_datetime(series.index)
            m = (series.index > last) & (series.index <= cut_date)
            et_sum = float(series.loc[m].sum()) if m.any() else 0.0

            crosses.append((cut_date, uid_i))
            same_day_counts[cut_date] = same_day_counts.get(cut_date, 0) + 1

            dodge = (uid_index[uid_i] - (len(uids)-1)/2.0) * bar_dodge_days
            bars.append((cut_date + pd.Timedelta(days=dodge), et_sum, uid_i, dodge))
            last = cut_date

        m_tail = (series.index > last) & (series.index <= end)
        et_tail = float(series.loc[m_tail].sum()) if m_tail.any() else 0.0
        if et_tail > 0:
            dodge = (uid_index[uid_i] - (len(uids)-1)/2.0) * bar_dodge_days
            bars.append((end + pd.Timedelta(days=dodge), et_tail, uid_i, dodge))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ymax = max([et for _, et, _, _ in bars] + [1.0]) * 1.15

    for x, et_sum, uid_i, _ in bars:
        color = uid_colors[uid_i]
        ax.bar(x, et_sum, width=pd.Timedelta(days=bar_width_days), color=color, alpha=bar_alpha, align="center", edgecolor="none")

    for x, uid_i in crosses:
        seen = same_day_counts.get(x, 1)
        h = cross_y_fraction + (seen-1)*cross_stack_step
        ax.scatter([x], [ymax*h], marker="x", s=marker_size, linewidths=marker_lw, color=uid_colors[uid_i])
        same_day_counts[x] = seen - 1

    ax.set_ylim(0, ymax)
    ax.set_xlim(start - pd.Timedelta(days=5), end + pd.Timedelta(days=5))
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%b\n%Y"))
    ax.set_xlabel(f"Water Year {water_year} ({start.date()} → {end.date()})")
    ax.set_ylabel("ET between cuts (mm)")

    title_bits = [county_norm, f"WY {water_year}: seasonal cut dates (×) & ET per interval"]
    if filter_n_cp_season is not None:
        title_bits.append(f"| n_cp_season={filter_n_cp_season}")
    ax.set_title("  ".join(title_bits))

    handles = [mpl.lines.Line2D([0],[0], color=uid_colors[uid_i], lw=6) for uid_i in uids]
    ax.legend(handles, uids, title="UniqueID", frameon=False,
              bbox_to_anchor=(1.02, 0.50), loc="center left", borderaxespad=0.0)

    fig.subplots_adjust(right=0.78)

    filename = f"{county_norm}_WY{water_year}_UID{uid}_cuts_x_et.png"
    if filter_n_cp_season is not None:
        filename = f"{county_norm}_WY{water_year}_UID{uid}_ncp{filter_n_cp_season}_cuts_x_et.png"
    output_path = output_dir / "et_corrections" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(output_path),
        "county": county_norm,
        "water_year": water_year,
        "uid": uid,
        "filter_n_cp_season": filter_n_cp_season,
        "n_bar_intervals": len(bars),
        "n_cut_markers": len(crosses),
        "figure_type": "cut_dates_and_et",
    }


if __name__ == "__main__":
    result = et_cut_dates_and_et_plot("Fresno", 2020, "1003334", filter_n_cp_season=10)
    print("Created:", result["figure_path"])