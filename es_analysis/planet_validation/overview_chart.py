#!/usr/bin/env python3
"""Comprehensive PlanetScope cut-date validation figure (all 12 cuts).

Two panels, read from validation_summary.csv (no API):
  (a) Dumbbell timeline — every cut as this-study date (|) vs PlanetScope date (o),
      joined by a line (the offset), grouped/coloured by county; weak cuts hollow.
  (b) Agreement scatter — this-study cut DOY vs PlanetScope cut DOY, 1:1 line +
      ±5 d band, coloured by county, filled=strong / hollow=weak.

Run from repo root (evi_analysis/):
    python -m es_analysis.planet_validation.overview_chart
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from es_analysis.utils.publication_style import (
    DOUBLE_COL_WIDTH, apply_style, save_pub_figure,
)

OUT_DIR = REPO / "es_analysis" / "planet_validation"
SUMMARY = OUT_DIR / "validation_summary.csv"
FIG_DIR = OUT_DIR / "figures"
WEAK_DROP = 0.13
REGION_ORDER = {"north": 0, "central": 1, "south": 2}
COUNTY_COLOR = {"San Joaquin": "#0072B2", "Kern": "#E69F00", "Imperial": "#009E73"}
COUNTY_SHORT = {"San Joaquin": "SJ", "Kern": "Kern", "Imperial": "Imp"}


def _load() -> pd.DataFrame:
    d = pd.read_csv(SUMMARY)
    d = d[d["status"] == "ok"].copy()
    d["UniqueID"] = d["UniqueID"].astype(float).astype(int).astype(str)
    d["cut_n"] = d["cut_n"].astype(float).astype(int)
    d["beast"] = pd.to_datetime(d["beast_trough_date"])
    d["planet"] = pd.to_datetime(d["planet_trough_date"])
    d["offset"] = d["offset_days"].astype(float)
    d["weak"] = d["planet_evi_drop"].astype(float) < WEAK_DROP
    d["ro"] = d["region"].map(REGION_ORDER)
    return d.sort_values(["ro", "UniqueID", "cut_n"]).reset_index(drop=True)


def main():
    apply_style()
    d = _load()
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH * 1.45, DOUBLE_COL_WIDTH * 0.55),
        gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- Panel (a): dumbbell timeline ----
    y = np.arange(len(d))[::-1]
    for yy, (_, r) in zip(y, d.iterrows()):
        c = COUNTY_COLOR[r["county"]]
        axa.plot([r["beast"], r["planet"]], [yy, yy], "-", color="#b0b0b0",
                 lw=1.3, zorder=1)
        axa.plot(r["beast"], yy, marker="|", color="black", ms=12, mew=2.2, zorder=3)
        if r["weak"]:
            axa.plot(r["planet"], yy, "o", mfc="white", mec=c, mew=1.7, ms=7, zorder=3)
        else:
            axa.plot(r["planet"], yy, "o", color=c, ms=7, zorder=3)
    axa.set_yticks(y)
    axa.set_yticklabels([f"{COUNTY_SHORT[r['county']]} {r['UniqueID']} c{r['cut_n']}"
                         for _, r in d.iterrows()])
    for tick, (_, r) in zip(axa.get_yticklabels(), d.iterrows()):
        tick.set_color(COUNTY_COLOR[r["county"]])
        tick.set_fontsize(7.5)
    axa.set_ylim(-0.7, len(d) - 0.3)
    pad = pd.Timedelta(days=8)
    axa.set_xlim(min(d["beast"].min(), d["planet"].min()) - pad,
                 max(d["beast"].max(), d["planet"].max()) + pad)
    axa.xaxis.set_major_locator(mdates.MonthLocator())
    axa.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axa.tick_params(axis="x", labelsize=8)
    axa.set_title("(a) Cut date: this study ( | ) vs PlanetScope ( ● );  weak = open ○",
                  fontsize=9)
    axa.grid(True, axis="x", alpha=0.25)

    # ---- Panel (b): agreement scatter (day-of-year) ----
    d["bdoy"] = d["beast"].dt.dayofyear
    d["pdoy"] = d["planet"].dt.dayofyear
    lo = int(min(d["bdoy"].min(), d["pdoy"].min())) - 6
    hi = int(max(d["bdoy"].max(), d["pdoy"].max())) + 6
    xs = np.array([lo, hi])
    axb.fill_between(xs, xs - 5, xs + 5, color="#888", alpha=0.15, label="±5 d", zorder=1)
    axb.plot(xs, xs, "-", color="black", lw=1.0, label="1:1", zorder=2)
    for county, c in COUNTY_COLOR.items():
        s = d[(d["county"] == county) & (~d["weak"])]
        w = d[(d["county"] == county) & (d["weak"])]
        axb.scatter(s["bdoy"], s["pdoy"], color=c, s=48, edgecolor="black",
                    linewidth=0.4, label=county, zorder=4)
        axb.scatter(w["bdoy"], w["pdoy"], facecolor="white", edgecolor=c,
                    linewidth=1.6, s=48, zorder=4)
    axb.scatter([], [], facecolor="white", edgecolor="#555", linewidth=1.6, s=48,
                label="weak (open)")
    axb.set_xlim(lo, hi)
    axb.set_ylim(lo, hi)
    axb.set_aspect("equal")
    axb.set_xlabel("This-study cut (day of year, 2022)", fontsize=9)
    axb.set_ylabel("PlanetScope cut (day of year)", fontsize=9)
    axb.set_title("(b) Agreement", fontsize=9.5)
    axb.grid(True, alpha=0.25)
    ab = d["offset"].abs()
    axb.text(0.04, 0.96,
             f"mean |Δ| = {ab.mean():.1f} d\nmedian |Δ| = {ab.median():.0f} d\n"
             f"bias = {d['offset'].mean():+.1f} d\n"
             f"{100*(ab <= 5).mean():.0f}% within ±5 d   (n={len(d)})",
             transform=axb.transAxes, va="top", ha="left", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", lw=0.6))
    axb.legend(fontsize=6.8, loc="lower right", frameon=True, framealpha=0.9)

    for ax in (axa, axb):
        for sp in ax.spines.values():
            sp.set_color("black")

    fig.suptitle(
        "PlanetScope validation of alfalfa cut dates — 6 parcels, 3 counties, WY2022",
        fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_pub_figure(fig, "cutdate_validation_overview", FIG_DIR)
    print(f"mean|offset|={ab.mean():.2f} bias={d['offset'].mean():+.2f} "
          f"within5={100*(ab<=5).mean():.0f}% n={len(d)}")


if __name__ == "__main__":
    main()
