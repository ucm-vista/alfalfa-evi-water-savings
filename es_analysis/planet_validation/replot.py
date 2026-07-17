#!/usr/bin/env python3
"""Regenerate overlay figures from saved CSVs — NO Planet API calls.

Rebuilds each per-cut overlay from planet_evi_points.csv + validation_summary.csv
+ HLS county_year_exports. Default regenerates the two manuscript exemplars;
--all regenerates all 12 for style consistency.

Run from repo root (evi_analysis/):
    python -m es_analysis.planet_validation.replot            # 2 exemplars
    python -m es_analysis.planet_validation.replot --all      # all 12
    python -m es_analysis.planet_validation.replot --only north_San_Joaquin_3901548_cut2
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from es_analysis.planet_validation.evi_overlay_plot import overlay_figure
from es_analysis.planet_validation.run_planet_validation import load_hls, OUT_DIR

SUMMARY = OUT_DIR / "validation_summary.csv"
POINTS = OUT_DIR / "planet_evi_points.csv"
FIG_DIR = OUT_DIR / "figures"
EXEMPLARS = {"south_Imperial_1300103_cut1", "north_San_Joaquin_3901548_cut2"}


def _fig_name(region, county, uid, cut_n):
    return f"{region}_{county.replace(' ', '_')}_{uid}_cut{cut_n}"


def main():
    ap = argparse.ArgumentParser(description="Regenerate overlay figures from saved CSVs.")
    ap.add_argument("--all", action="store_true", help="regenerate all 12 (default: 2 exemplars)")
    ap.add_argument("--only", type=str, default=None, help="single figure name")
    args = ap.parse_args()

    summ = pd.read_csv(SUMMARY)
    summ = summ[summ["status"] == "ok"].copy()
    pts = pd.read_csv(POINTS, parse_dates=["date"])
    pts["UniqueID"] = pts["UniqueID"].astype(str)

    n = 0
    for _, r in summ.iterrows():
        uid = str(int(float(r["UniqueID"])))
        cut_n = int(float(r["cut_n"]))
        county, region = r["county"], r["region"]
        name = _fig_name(region, county, uid, cut_n)
        if args.only:
            if name != args.only:
                continue
        elif not args.all and name not in EXEMPLARS:
            continue

        hls = load_hls(county, uid)
        planet = (pts[(pts["UniqueID"] == uid) & (pts["cut_n"] == cut_n)]
                  [["date", "evi_median"]].sort_values("date").reset_index(drop=True))
        beast_date = pd.Timestamp(r["beast_trough_date"])
        planet_date = (None if pd.isna(r["planet_trough_date"])
                       else pd.Timestamp(r["planet_trough_date"]))
        offset = None if pd.isna(r["offset_days"]) else int(float(r["offset_days"]))
        meta = {"region": region, "county": county, "UniqueID": uid, "cut_n": cut_n}

        overlay_figure(hls, planet, beast_date, planet_date, None, offset, meta, FIG_DIR, name)
        n += 1
    print(f"\nRegenerated {n} figure(s) in {FIG_DIR}")


if __name__ == "__main__":
    main()
