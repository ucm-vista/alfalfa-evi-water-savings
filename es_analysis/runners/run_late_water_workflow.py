"""Runner: Late-water savings workflow.

Usage:
    python -m es_analysis.runners.run_late_water_workflow [options]

Builds the parcel-year late-cut dataset, computes cap-based savings
scenarios, and exports CSV files.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run late-water savings workflow.",
    )
    parser.add_argument(
        "--counties", nargs="+", default=None,
        help="County names to process (default: all counties).",
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
        "--cutoff-month", type=int, default=None,
        help="Late-season cutoff month (default: from config, typically 7).",
    )
    parser.add_argument(
        "--cutoff-day", type=int, default=None,
        help="Late-season cutoff day (default: from config, typically 1).",
    )
    parser.add_argument(
        "--cap-values", nargs="+", type=int, default=None,
        help="Cap values for savings scenarios (default: from config).",
    )
    parser.add_argument(
        "--compute-area-acft", action="store_true",
        help="Compute area-based ac-ft values using parcel geometries.",
    )
    parser.add_argument(
        "--evi-mode", type=str, default=None,
        choices=["smoothed", "gapfilled"],
        help="EVI mode (default: from config).",
    )
    parser.add_argument(
        "--evi-required", action="store_true",
        help="Skip parcels without EVI data.",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable ET-rise fallback when EVI is unavailable.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for CSV exports.",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip CSV export.",
    )

    args = parser.parse_args()

    from es_analysis.data_providers.late_water_provider import (
        run_late_water_saving_workflow,
    )

    kwargs = dict(
        wy_start=args.wy_start,
        wy_end=args.wy_end,
        compute_area_acft=args.compute_area_acft,
        evi_required=args.evi_required,
        fallback_to_et_rise=not args.no_fallback,
        export_csv=not args.no_export,
    )
    if args.counties is not None:
        kwargs["counties"] = args.counties
    if args.cutoff_month is not None:
        kwargs["cutoff_month"] = args.cutoff_month
    if args.cutoff_day is not None:
        kwargs["cutoff_day"] = args.cutoff_day
    if args.cap_values is not None:
        kwargs["cap_values"] = tuple(args.cap_values)
    if args.evi_mode is not None:
        kwargs["evi_mode"] = args.evi_mode
    if args.out_dir is not None:
        kwargs["out_dir"] = Path(args.out_dir)

    results = run_late_water_saving_workflow(**kwargs)

    n_py = len(results["df"])
    n_unique = results["df"]["UniqueID"].nunique()
    print(f"\nDone. {n_py:,} parcel-years, {n_unique:,} unique parcels.")


if __name__ == "__main__":
    main()
