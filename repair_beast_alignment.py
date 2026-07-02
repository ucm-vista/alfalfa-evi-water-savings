#!/usr/bin/env python3
"""Repair BEAST CSV alignment without re-running MCMC.

Reads existing BEAST seasonal CSVs, re-loads EVI for each parcel,
re-detects minima, and re-aligns with CP dates using the fixed
tolerance logic (max-matches instead of early-exit).

Usage:
    python repair_beast_alignment.py --county "San Joaquin" --wy all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from es_analysis.data_providers.config import config
from es_analysis.data_providers.beast_provider import BEASTDataProvider
from es_analysis.data_providers.evi_provider import normalize_county_name


def repair_csv(county: str, wy: int, backup: bool = True) -> dict:
    """Re-align one county-WY BEAST CSV."""
    county_norm = normalize_county_name(county)
    csv_path = (
        config.beast_out_root_new
        / county_norm.replace("_", " ")
        / f"beast_seasonal_cuts_WY{wy}.csv"
    )
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return {"status": "skip"}

    beast_df = pd.read_csv(csv_path)
    if beast_df.empty:
        return {"status": "empty"}

    # Backup original
    if backup:
        bak = csv_path.with_name(csv_path.stem + "_pre_realign.csv")
        if not bak.exists():
            beast_df.to_csv(bak, index=False)

    # Load EVI via beast_provider's method
    beast_prov = BEASTDataProvider()
    evi_df = beast_prov.load_county_year_csv(county_norm, wy)

    # Group EVI by parcel
    groups = {str(pid): g for pid, g in evi_df.groupby(evi_df["parcel_id"].astype(str))}

    n_improved = 0
    n_same = 0
    n_no_cp = 0
    n_error = 0

    for idx, row in beast_df.iterrows():
        pid = str(row["parcel_id"])
        old_n = int(row["n_cuttings"])

        # Parse CP dates
        cp_str = row.get("season_cp_dates_iso", "")
        if pd.isna(cp_str) or not str(cp_str).strip():
            n_no_cp += 1
            continue
        cp_dates = pd.DatetimeIndex(pd.to_datetime(str(cp_str).split(";")))

        cp_prob_str = row.get("season_cp_probs", "")
        if pd.isna(cp_prob_str) or not str(cp_prob_str).strip():
            cp_pr = np.ones(len(cp_dates))
        else:
            cp_pr = np.array([float(x) for x in str(cp_prob_str).split(";")])

        try:
            if pid not in groups:
                n_error += 1
                continue

            sub = groups[pid].copy()
            series = beast_prov.series_for_beast(sub, col="smoothed_mean_evi")
            if series is None or series.empty or len(series) < 30:
                n_error += 1
                continue

            # Build minima
            mins_pack = beast_prov.build_inclusive_minima(series)
            minima_union = list(mins_pack["union"])

            # CP-centric boost
            boost_win = int(config.cp_boost_window_days)
            evi_arr = series.to_numpy()
            evi_idx = series.index
            for cpd in cp_dates:
                if any(abs((m - cpd).days) <= boost_win for m in minima_union):
                    continue
                mstar = beast_prov.argmin_in_window(series, cpd, boost_win)
                if mstar is None:
                    continue
                evi_at_min = float(series.get(mstar, np.nan))
                if not np.isfinite(evi_at_min) or evi_at_min > config.max_boost_evi:
                    continue
                if config.require_peak_before_boost:
                    peak_win_start = mstar - pd.Timedelta(days=config.peak_window_days)
                    peak_mask = (evi_idx >= peak_win_start) & (evi_idx < mstar)
                    if peak_mask.any():
                        peak_val = float(np.nanmax(evi_arr[peak_mask]))
                        amp = float(
                            np.nanpercentile(evi_arr[np.isfinite(evi_arr)], 90)
                            - np.nanpercentile(evi_arr[np.isfinite(evi_arr)], 10)
                        )
                        delta_thresh = max(config.delta_min, config.amp_frac_min * amp)
                        if (peak_val - evi_at_min) < delta_thresh:
                            continue
                    else:
                        continue
                minima_union.append(mstar)

            # Dedupe
            minima_union = beast_prov.dedupe_by_nearness(
                sorted(minima_union),
                series,
                min_gap_days=min(config.min_spacing_days_range),
            )

            # FIXED alignment: try all tolerances, pick max matches
            best_cuts, best_probs = [], []
            for tol in config.strict_tol_steps:
                d, p = beast_prov._align_minima_with_beast(
                    minima_union, cp_dates, cp_pr, tol_days=tol,
                )
                if len(d) > len(best_cuts):
                    best_cuts, best_probs = d, p

            new_n = len(best_cuts)

            # Update CSV row
            if best_cuts:
                beast_df.at[idx, "matched_minima_iso"] = ";".join(
                    pd.to_datetime(best_cuts).strftime("%Y-%m-%d").tolist()
                )
                beast_df.at[idx, "matched_minima_probs"] = ";".join(
                    f"{p:.3f}" for p in best_probs
                )
                beast_df.at[idx, "n_cuttings"] = new_n

            if new_n > old_n:
                n_improved += 1
            else:
                n_same += 1

        except Exception as exc:
            n_error += 1
            continue

    # Save repaired CSV
    beast_df.to_csv(csv_path, index=False)

    return {
        "status": "ok",
        "total": len(beast_df),
        "improved": n_improved,
        "same": n_same,
        "no_cp": n_no_cp,
        "error": n_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Repair BEAST CSV alignment")
    parser.add_argument("--county", required=True)
    parser.add_argument("--wy", nargs="+", required=True)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    wys = list(range(2019, 2025)) if args.wy == ["all"] else [int(w) for w in args.wy]

    print(f"Repairing alignment for {args.county}, WY {wys}")
    print(f"Tolerance steps: {config.strict_tol_steps}")
    print()

    total_improved = 0
    for wy in wys:
        print(f"  {args.county} WY{wy} ...", end=" ", flush=True)
        stats = repair_csv(args.county, wy, backup=not args.no_backup)
        if stats["status"] == "ok":
            total_improved += stats["improved"]
            print(
                f"done: {stats['improved']} improved, "
                f"{stats['same']} same, {stats['no_cp']} no-cp, "
                f"{stats['error']} err (of {stats['total']})"
            )
        else:
            print(f"status={stats['status']}")

    print(f"\nTotal improved: {total_improved}")


if __name__ == "__main__":
    main()
