#!/usr/bin/env python3
"""Run BEAST 16-run ensemble re-processing with concurrent county-WY pairs.

Uses ThreadPoolExecutor to run multiple (county, WY) pairs concurrently.
Each pair spawns its own BEASTDataProvider + joblib Parallel(n_jobs)
internally, so actual CPU work happens in loky subprocesses.

Auto-recovery: if a job fails, it retries with halved n_jobs. If an entire
round has >50% failures, the next round reduces both n_jobs and concurrency.
The outer loop keeps retrying until all jobs complete or n_jobs bottoms out.

Usage:
    python run_beast_parallel.py --dry-run
    python run_beast_parallel.py --max-concurrent 4 --n-jobs 32
    nohup python -u run_beast_parallel.py --max-concurrent 4 --n-jobs 32 > beast_parallel_run.log 2>&1 &
"""
import sys
import os
import shutil
import time
import argparse
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "es_analysis"))

import pandas as pd
from es_analysis.data_providers.beast_provider import BEASTDataProvider


COUNTIES_IN_ORDER = [
    "Tulare",       # 117 parcels
    "Merced",       # 149 parcels
    "Imperial",     # 167 parcels
    "Kern",         # 175 parcels
    "Stanislaus",   # 194 parcels
    "San Joaquin",  # 375 parcels
    "Riverside",    # 390 parcels
    "Fresno",       # 131 parcels
    "Madera",       # 82 parcels
    "Kings",        # 112 parcels
]

WATER_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

NEW_COLS = ["matched_consensus_freq", "matched_timing_sigma_days"]

MIN_N_JOBS = 4  # floor — don't go below this

_print_lock = threading.Lock()


def tprint(msg: str) -> None:
    """Thread-safe timestamped print."""
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] {msg}", flush=True)


def get_beast_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "beast_outputs_new")


def build_work_queue(beast_dir: str, force: bool = False,
                     counties_filter: list = None) -> list:
    """Return (county, wy) pairs that need 16-run processing."""
    counties = counties_filter if counties_filter else COUNTIES_IN_ORDER
    queue = []
    for county in counties:
        for wy in WATER_YEARS:
            csv_path = os.path.join(beast_dir, county, f"beast_seasonal_cuts_WY{wy}.csv")
            if not force and os.path.isfile(csv_path):
                df = pd.read_csv(csv_path, nrows=1)
                if all(c in df.columns for c in NEW_COLS):
                    continue  # already done
            queue.append((county, wy))
    return queue


def backup_old_csv(county: str, wy: int, beast_dir: str) -> None:
    """Back up old CSV before re-run."""
    fname = f"beast_seasonal_cuts_WY{wy}.csv"
    fpath = os.path.join(beast_dir, county, fname)
    # Two backup slots: _old4run (original 4-run) and _pre_rerun (current before force re-run)
    backup_old = fpath.replace(".csv", "_old4run.csv")
    backup_pre = fpath.replace(".csv", "_pre_rerun.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, nrows=1)
        if not all(c in df.columns for c in NEW_COLS):
            if not os.path.exists(backup_old):
                shutil.copy2(fpath, backup_old)
                tprint(f"  Backed up (old4run): {county}/WY{wy}")
        else:
            # Already has new cols — back up as _pre_rerun for force re-runs
            if not os.path.exists(backup_pre):
                shutil.copy2(fpath, backup_pre)
                tprint(f"  Backed up (pre_rerun): {county}/WY{wy}")


def validate_wy(county: str, wy: int, beast_dir: str) -> dict:
    """Validate a completed WY output. Returns None if validation fails."""
    fname = f"beast_seasonal_cuts_WY{wy}.csv"
    fpath = os.path.join(beast_dir, county, fname)
    if not os.path.isfile(fpath):
        return None
    df = pd.read_csv(fpath)
    has_new = all(c in df.columns for c in NEW_COLS)
    if not has_new:
        return None
    # Tier breakdown: 0=consensus, 1=best-run, 2=minima-only
    fb = df["fallback_used"] if "fallback_used" in df.columns else pd.Series([], dtype=int)
    return {
        "county": county,
        "wy": wy,
        "n_parcels": len(df),
        "has_new_cols": has_new,
        "mean_cuts": df["n_cuttings"].mean(),
        "zero_cuts": int((df["n_cuttings"] == 0).sum()),
        "zero_cp": int((df["n_cp_season"] == 0).sum()),
        "fallback_t1": int((fb == 0).sum()),
        "fallback_t2": int((fb == 1).sum()),
        "fallback_t3": int((fb == 2).sum()),
        "fallow": int((df["beast_status"] == "fallow").sum()) if "beast_status" in df.columns else 0,
        "mean_cp": df["n_cp_season"].mean(),
    }


def get_parcel_count(county: str, wy: int, beast_dir: str) -> int:
    """Get parcel count from existing CSV (for display)."""
    fpath = os.path.join(beast_dir, county, f"beast_seasonal_cuts_WY{wy}.csv")
    if os.path.isfile(fpath):
        return len(pd.read_csv(fpath, usecols=["parcel_id"]))
    return 0


def process_one(county: str, wy: int, beast_dir: str, n_jobs: int) -> dict:
    """Process a single (county, WY) pair with automatic n_jobs backoff.

    Tries at n_jobs, then n_jobs//2, then n_jobs//4, down to MIN_N_JOBS.
    Returns result dict with status 'ok' or 'error'.
    """
    attempts = []
    current_n_jobs = n_jobs

    while current_n_jobs >= MIN_N_JOBS:
        attempts.append(current_n_jobs)
        t0 = time.time()
        n_parcels = get_parcel_count(county, wy, beast_dir)
        attempt_label = f"(n_jobs={current_n_jobs})" if current_n_jobs != n_jobs else ""
        tprint(f"START  {county:<14s} WY{wy} ({n_parcels} parcels) {attempt_label}")

        try:
            # Backup old CSV (idempotent)
            backup_old_csv(county, wy, beast_dir)

            # Fresh provider instance per attempt
            bp = BEASTDataProvider()
            bp.run_seasonal_for_year(county, wy, n_jobs=current_n_jobs)

            # Validate output
            v = validate_wy(county, wy, beast_dir)
            elapsed = time.time() - t0

            if v is None:
                raise RuntimeError("Output CSV missing or lacks new columns after run")

            tprint(
                f"DONE   {county:<14s} WY{wy} in {elapsed/60:.1f}m | "
                f"cuts={v['mean_cuts']:.2f} zero={v['zero_cuts']} "
                f"T1={v['fallback_t1']} T2={v['fallback_t2']} T3={v['fallback_t3']} "
                f"fallow={v['fallow']} n_jobs={current_n_jobs}"
            )
            v["elapsed_s"] = elapsed
            v["status"] = "ok"
            v["n_jobs_used"] = current_n_jobs
            v["attempts"] = attempts
            return v

        except Exception as e:
            elapsed = time.time() - t0
            tprint(
                f"FAIL   {county:<14s} WY{wy} in {elapsed/60:.1f}m | "
                f"n_jobs={current_n_jobs} | {type(e).__name__}: {e}"
            )
            # Halve workers for retry
            current_n_jobs = current_n_jobs // 2

            if current_n_jobs >= MIN_N_JOBS:
                tprint(f"RETRY  {county:<14s} WY{wy} with n_jobs={current_n_jobs}")
                time.sleep(5)  # brief cooldown before retry

    # All retries exhausted
    tprint(f"GIVEUP {county:<14s} WY{wy} after attempts {attempts}")
    return {
        "county": county,
        "wy": wy,
        "status": "error",
        "error": f"All retries exhausted: {attempts}",
        "attempts": attempts,
    }


def run_round(queue, beast_dir, max_concurrent, n_jobs):
    """Run one round of processing. Returns (ok_results, fail_results)."""
    ok_results = []
    fail_results = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(process_one, county, wy, beast_dir, n_jobs): (county, wy)
            for county, wy in queue
        }

        for future in as_completed(futures):
            county, wy = futures[future]
            try:
                result = future.result()
            except Exception as e:
                tprint(f"THREAD {county:<14s} WY{wy} | unexpected: {e}")
                result = {
                    "county": county, "wy": wy,
                    "status": "error", "error": str(e),
                }

            if result.get("status") == "ok":
                ok_results.append(result)
            else:
                fail_results.append(result)

    return ok_results, fail_results


def main():
    parser = argparse.ArgumentParser(description="Parallel BEAST 16-run re-processing")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Max concurrent (county, WY) pairs (default: 4)")
    parser.add_argument("--n-jobs", type=int, default=32,
                        help="Joblib workers per pair (default: 32)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show work queue without running")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if new columns already exist")
    parser.add_argument("--counties", nargs="+", default=None,
                        help="Only process these counties (default: all 10)")
    args = parser.parse_args()

    beast_dir = get_beast_dir()
    queue = build_work_queue(beast_dir, force=args.force,
                             counties_filter=args.counties)

    print(f"{'='*60}")
    print(f"BEAST Parallel Re-processing (auto-retry)")
    print(f"{'='*60}")
    print(f"Work queue: {len(queue)} (county, WY) pairs")
    print(f"Concurrency: {args.max_concurrent} pairs x {args.n_jobs} workers = ~{args.max_concurrent * args.n_jobs} cores")
    print(f"Auto-retry: on failure, halve n_jobs down to {MIN_N_JOBS}")
    print()

    if args.dry_run:
        print("Work queue:")
        for i, (county, wy) in enumerate(queue, 1):
            n = get_parcel_count(county, wy, beast_dir)
            print(f"  {i:3d}. {county:<14s} WY{wy}  ({n} parcels)")
        print(f"\nTotal: {len(queue)} jobs")
        return

    t0_all = time.time()
    all_ok = []
    max_concurrent = args.max_concurrent
    n_jobs = args.n_jobs
    round_num = 0
    first_round = True

    while True:
        # First round uses the initial queue (may include --force items);
        # subsequent rounds re-scan without force to pick up remaining work
        if first_round:
            first_round = False
        else:
            queue = build_work_queue(beast_dir, force=False,
                                     counties_filter=args.counties)
        if not queue:
            tprint("All (county, WY) pairs complete!")
            break

        round_num += 1
        tprint(f"{'='*60}")
        tprint(f"ROUND {round_num}: {len(queue)} remaining | "
               f"concurrent={max_concurrent} n_jobs={n_jobs}")
        tprint(f"{'='*60}")

        ok, fail = run_round(queue, beast_dir, max_concurrent, n_jobs)
        all_ok.extend(ok)

        if not fail:
            tprint(f"Round {round_num}: all {len(ok)} succeeded")
            break

        tprint(f"Round {round_num}: {len(ok)} ok, {len(fail)} failed")

        # If >50% failed, reduce concurrency and n_jobs for next round
        if len(fail) > len(ok):
            old_concurrent = max_concurrent
            old_n_jobs = n_jobs
            max_concurrent = max(1, max_concurrent // 2)
            n_jobs = max(MIN_N_JOBS, n_jobs // 2)
            tprint(
                f"High failure rate — reducing: "
                f"concurrent {old_concurrent}->{max_concurrent}, "
                f"n_jobs {old_n_jobs}->{n_jobs}"
            )

        if n_jobs < MIN_N_JOBS:
            tprint(f"n_jobs would go below {MIN_N_JOBS}, giving up on remaining failures")
            break

        # Cooldown between rounds
        tprint("Cooldown 30s before next round...")
        time.sleep(30)

    elapsed_all = time.time() - t0_all

    # Final check
    remaining = build_work_queue(beast_dir)

    print(f"\n{'='*60}")
    print(f"FINISHED in {elapsed_all/3600:.1f}h over {round_num} round(s)")
    print(f"Completed: {len(all_ok)} jobs")
    if remaining:
        print(f"Still incomplete ({len(remaining)}):")
        for county, wy in remaining:
            print(f"  {county} WY{wy}")
    else:
        print("All 39 (county, WY) pairs successfully processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
