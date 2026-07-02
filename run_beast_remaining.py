#!/usr/bin/env python3
"""Run BEAST 16-run ensemble re-processing for remaining counties.

Processes one county at a time (all 6 WYs sequentially) to avoid CPU overload.
Backs up old CSV as *_old4run.csv before overwriting.
Validates each WY after processing.

Usage:
    python run_beast_remaining.py [--start-county Kings] [--n-jobs 16]
"""
import sys
import os
import shutil
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "es_analysis"))

from es_analysis.data_providers.beast_provider import BEASTDataProvider
import pandas as pd


COUNTIES_IN_ORDER = [
    "Madera",       # 29 parcels
    "Kings",        # 65 parcels
    "Tulare",       # 117 parcels
    "Merced",       # 149 parcels
    "Imperial",     # 167 parcels
    "Kern",         # 175 parcels
    "Stanislaus",   # 194 parcels
    "San Joaquin",  # 375 parcels
    "Riverside",    # 390 parcels
]

WATER_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

NEW_COLS = ["matched_consensus_freq", "matched_timing_sigma_days"]


def backup_old_csv(county: str, wy: int, beast_dir: str) -> None:
    """Back up old CSV if it doesn't already have new columns."""
    fname = f"beast_seasonal_cuts_WY{wy}.csv"
    fpath = os.path.join(beast_dir, county, fname)
    backup = fpath.replace(".csv", "_old4run.csv")
    if os.path.exists(fpath) and not os.path.exists(backup):
        df = pd.read_csv(fpath)
        if not all(c in df.columns for c in NEW_COLS):
            shutil.copy2(fpath, backup)
            print(f"  Backed up: {backup}")


def validate_wy(county: str, wy: int, beast_dir: str) -> dict:
    """Validate a completed WY output."""
    fname = f"beast_seasonal_cuts_WY{wy}.csv"
    fpath = os.path.join(beast_dir, county, fname)
    df = pd.read_csv(fpath)
    result = {
        "county": county,
        "wy": wy,
        "n_parcels": len(df),
        "has_new_cols": all(c in df.columns for c in NEW_COLS),
        "mean_cuts": df["n_cuttings"].mean(),
        "zero_cuts": (df["n_cuttings"] == 0).sum(),
        "fallback": df["fallback_used"].sum() if "fallback_used" in df.columns else -1,
        "mean_cp": df["n_cp_season"].mean(),
    }
    return result


def run_county(county: str, beast_dir: str, n_jobs: int = 16) -> None:
    """Process all 6 WYs for a county."""
    bp = BEASTDataProvider()
    print(f"\n{'='*60}")
    print(f"PROCESSING: {county}")
    print(f"{'='*60}")
    t0_county = time.time()

    for wy in WATER_YEARS:
        # Check if already processed with 16-run
        fname = f"beast_seasonal_cuts_WY{wy}.csv"
        fpath = os.path.join(beast_dir, county, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if all(c in df.columns for c in NEW_COLS):
                print(f"  WY{wy}: Already has 16-run columns, skipping")
                continue

        # Backup old
        backup_old_csv(county, wy, beast_dir)

        # Run
        t0 = time.time()
        print(f"  WY{wy}: Starting...", flush=True)
        bp.run_seasonal_for_year(county, wy, n_jobs=n_jobs)
        elapsed = time.time() - t0

        # Validate
        v = validate_wy(county, wy, beast_dir)
        print(
            f"  WY{wy}: Done in {elapsed:.0f}s | "
            f"parcels={v['n_parcels']} "
            f"mean_cuts={v['mean_cuts']:.2f} "
            f"zero_cuts={v['zero_cuts']} "
            f"fallback={v['fallback']} "
            f"mean_cp={v['mean_cp']:.1f} "
            f"has_new_cols={v['has_new_cols']}",
            flush=True,
        )

    elapsed_county = time.time() - t0_county
    print(f"\n{county} total: {elapsed_county/3600:.1f}h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-county", default=None, help="Skip counties before this one")
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()

    beast_dir = os.path.join(
        os.path.dirname(__file__),
        "beast_outputs_new",
    )

    counties = COUNTIES_IN_ORDER
    if args.start_county:
        try:
            idx = counties.index(args.start_county)
            counties = counties[idx:]
        except ValueError:
            print(f"Unknown county: {args.start_county}")
            print(f"Available: {counties}")
            sys.exit(1)

    print(f"Counties to process: {counties}")
    print(f"Workers per WY: {args.n_jobs}")
    t0_all = time.time()

    for county in counties:
        run_county(county, beast_dir, args.n_jobs)

    elapsed_all = time.time() - t0_all
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total time: {elapsed_all/3600:.1f}h")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
