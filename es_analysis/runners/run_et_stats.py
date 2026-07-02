"""Runner: ET correction statistics pipeline.

Usage:
    python -m es_analysis.runners.run_et_stats [options]

Runs the off-phase ET correction across all counties and water years,
producing aggregation tables, descriptive statistics, narrative
summaries, and CSV exports.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run off-phase ET correction statistics pipeline.",
    )
    parser.add_argument(
        "--counties", nargs="+", default=None,
        help="County names to process (default: all 10 counties).",
    )
    parser.add_argument(
        "--wy-start", type=int, default=2019,
        help="First water year (default: 2019).",
    )
    parser.add_argument(
        "--wy-end", type=int, default=2024,
        help="Last water year (default: 2024).",
    )
    parser.add_argument(
        "--max-uids", type=int, default=None,
        help="Max UIDs per county-year (default: all).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=None,
        help="Bootstrap samples (default: from config).",
    )
    parser.add_argument(
        "--method", type=str, default=None, choices=["A", "B"],
        help="Correction method (default: from config).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for CSV exports.",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip CSV export.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers for per-UID processing (default: sequential).",
    )

    args = parser.parse_args()

    from es_analysis.data_providers.et_stats_provider import (
        run_et_correction_stats,
    )

    kwargs = dict(
        wy_start=args.wy_start,
        wy_end=args.wy_end,
        export_csv=not args.no_export,
    )
    if args.counties is not None:
        kwargs["counties"] = args.counties
    if args.max_uids is not None:
        kwargs["max_uids_per_county_wy"] = args.max_uids
    if args.n_boot is not None:
        kwargs["n_boot"] = args.n_boot
    if args.method is not None:
        kwargs["chosen_method"] = args.method
    if args.out_dir is not None:
        kwargs["out_dir"] = Path(args.out_dir)
    if args.workers is not None:
        kwargs["max_workers"] = args.workers

    results = run_et_correction_stats(**kwargs)

    n_py = results["df_parcel_year"].shape[0]
    n_fail = results["df_fail"].shape[0]
    print(f"\nDone. {n_py} parcel-years computed, {n_fail} failures.")


if __name__ == "__main__":
    main()
