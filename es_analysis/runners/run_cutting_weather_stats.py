"""Runner: Cutting-count and weather statistics pipeline.

Usage:
    python -m es_analysis.runners.run_cutting_weather_stats [options]
    python -m es_analysis.runners.run_cutting_weather_stats regenerate [options]

Computes cutting frequency statistics from BEAST seasonal CSVs,
produces coverage tables, descriptive statistics, frequency tables,
county mean-range labels, narrative summaries, and CSV exports.

The ``regenerate`` subcommand rebuilds the parcel-year wide CSVs
using the fixed ET loader (NaN instead of 0.0 for missing parcels).
"""

import argparse
from pathlib import Path


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by the default command and regenerate."""
    parser.add_argument(
        "--counties", nargs="+", default=None,
        help="County names to process (default: all counties).",
    )
    parser.add_argument(
        "--wy-start", type=int, default=None,
        help="First water year (default: auto-detect from data).",
    )
    parser.add_argument(
        "--wy-end", type=int, default=None,
        help="Last water year (default: auto-detect from data).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for CSV exports.",
    )


def _run_stats(args: argparse.Namespace) -> None:
    """Execute the original cutting-stats pipeline."""
    from es_analysis.data_providers.cutting_stats_provider import (
        run_cutting_weather_stats,
    )

    kwargs = dict(
        cut_metric=args.metric,
        export_csv=not args.no_export,
    )
    if args.counties is not None:
        kwargs["counties"] = args.counties
    if args.wy_start is not None:
        kwargs["wy_start"] = args.wy_start
    if args.wy_end is not None:
        kwargs["wy_end"] = args.wy_end
    if args.cut_min is not None:
        kwargs["cut_min"] = args.cut_min
    if args.cut_max is not None:
        kwargs["cut_max"] = args.cut_max
    if args.out_dir is not None:
        kwargs["out_dir"] = Path(args.out_dir)

    results = run_cutting_weather_stats(**kwargs)

    n_tables = len(results["stats"]) - 3  # exclude _wy_min, _wy_max, _cut_metric
    n_csv = len(results["paths"])
    print(f"\nDone. {n_tables} tables computed, {n_csv} CSVs exported.")


def _run_regenerate(args: argparse.Namespace) -> None:
    """Execute the regenerate subcommand."""
    from es_analysis.data_providers.cutting_stats_provider import (
        build_parcel_year_master,
        validate_parcel_year_master,
    )
    from es_analysis.data_providers.config import config
    from es_analysis.data_providers.spatial_provider import COUNTY_ORDER

    counties = args.counties if args.counties else list(COUNTY_ORDER)
    wy_start = args.wy_start if args.wy_start else 2019
    wy_end = args.wy_end if args.wy_end else 2024
    out_root = Path(args.out_dir) if args.out_dir else config.cutting_weather_stats_root

    daymet_vars = []
    if args.daymet_var in ("tmax", "both"):
        daymet_vars.append("tmax")
    if args.daymet_var in ("gdd5", "both"):
        daymet_vars.append("gdd5")

    for dvar in daymet_vars:
        variant_dir = out_root / "gdd5" if dvar == "gdd5" else out_root
        df_master, df_failures = build_parcel_year_master(
            daymet_var=dvar,
            counties=counties,
            wy_start=wy_start,
            wy_end=wy_end,
            out_dir=variant_dir,
            max_workers=args.workers,
        )
        if args.verify:
            validate_parcel_year_master(df_master, dvar)

    print("\nRegeneration complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run cutting-count and weather statistics pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- Default (stats) command args on the top-level parser ---
    _add_common_args(parser)
    parser.add_argument(
        "--metric", type=str, default="n_cp_season",
        choices=["n_cp_season", "n_cuttings"],
        help="Cutting metric column (default: n_cp_season).",
    )
    parser.add_argument(
        "--cut-min", type=float, default=None,
        help="Minimum cut value filter.",
    )
    parser.add_argument(
        "--cut-max", type=float, default=None,
        help="Maximum cut value filter.",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip CSV export.",
    )

    # --- regenerate subcommand ---
    regen = subparsers.add_parser(
        "regenerate",
        help="Rebuild parcel-year wide CSVs with the fixed ET loader.",
    )
    _add_common_args(regen)
    regen.add_argument(
        "--daymet-var", type=str, default="both",
        choices=["tmax", "gdd5", "both"],
        help="Daymet variable to regenerate (default: both).",
    )
    regen.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1).",
    )
    regen.add_argument(
        "--verify", action="store_true",
        help="Run validation checks after generation.",
    )

    args = parser.parse_args()

    if args.command == "regenerate":
        _run_regenerate(args)
    else:
        _run_stats(args)


if __name__ == "__main__":
    main()
