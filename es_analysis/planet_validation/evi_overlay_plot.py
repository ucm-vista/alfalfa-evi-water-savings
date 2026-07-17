#!/usr/bin/env python3
"""Overlay figure: this-study (BEAST) cut trough vs PlanetScope EVI.

One figure per cut cycle: HLS raw + smoothed EVI (from county_year_exports), the
this-study cut date (BEAST trough, solid line), and PlanetScope median-EVI points
with the PlanetScope trough (dashed). Title reports
"PlanetScope date − This study Cut-date = ±N d". Legend sits above the plot.
"""
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from es_analysis.utils.publication_style import apply_style, save_pub_figure

HLS_RAW = "#56B4E9"
HLS_SM = "#0072B2"
PLANET = "#D55E00"
CUT = "#000000"


def overlay_figure(hls: pd.DataFrame, planet: pd.DataFrame, cut_date: pd.Timestamp,
                   planet_trough: Optional[pd.Timestamp],
                   hls_trough: Optional[pd.Timestamp], cross_offset: Optional[int],
                   meta: dict, out_dir: Path, name: str) -> None:
    apply_style()
    lo, hi = cut_date - pd.Timedelta(days=35), cut_date + pd.Timedelta(days=35)
    h = hls[(hls["date"] >= lo) & (hls["date"] <= hi)]
    raw = h.dropna(subset=["original_mean_evi"])

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.plot(h["date"], h["smoothed_mean_evi"], "-", color=HLS_SM, lw=1.6,
            label="HLS EVI (smoothed)", zorder=3)
    ax.scatter(raw["date"], raw["original_mean_evi"], s=15, color=HLS_RAW,
               alpha=0.7, label="HLS EVI (obs)", zorder=2)
    if not planet.empty:
        ax.plot(planet["date"], planet["evi_median"], "-o", color=PLANET, ms=5,
                lw=1.2, label="PlanetScope EVI", zorder=4)

    ax.axvline(cut_date, color=CUT, lw=1.6, label="this study cut date")
    if planet_trough is not None:
        ax.axvline(planet_trough, color=PLANET, ls="--", lw=1.4,
                   label="PlanetScope trough")

    ax.set_ylabel("EVI", fontsize=10)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    for s in ax.spines.values():
        s.set_color("black")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Legend: out of the chart, just above the axes, 2 rows x 3 columns.
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              frameon=False, fontsize=7.6, columnspacing=1.5, handlelength=1.9)

    # Two-line title, right above the legend (small gap).
    off = "" if cross_offset is None else f"{cross_offset:+d} d"
    fig.suptitle(
        f"{meta['region'].title()} — {meta['county']} parcel {meta['UniqueID']} "
        f"(cut {meta['cut_n']})\n"
        f"PlanetScope date − This study Cut-date = {off}",
        fontsize=10, y=0.93)

    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.09, right=0.97)
    save_pub_figure(fig, name, out_dir)
