#!/usr/bin/env python3
"""CLI runner for per-parcel ET correction plots (separate daily + monthly).

Examples:
    # Single parcel, single WY, method A
    python -m es_analysis.runners.run_et_correction_plots \
        --county Fresno --water-year 2022 --parcel 1003334 --et-mode both --method A

    # Single parcel, WY range, method B
    python -m es_analysis.runners.run_et_correction_plots \
        --county Fresno --wy-start 2019 --wy-end 2024 --parcel 1003334 --method B

    # Random sample of 3 parcels
    python -m es_analysis.runners.run_et_correction_plots \
        --county Fresno --water-year 2022 --random 3 --et-mode both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_parcels_for_county_wy(county: str, wy: int):
    """Return list of parcel UIDs available for a county/WY."""
    from es_analysis.data_providers.et_provider import _load_seasonal_csv, _norm_county_name
    county_norm = _norm_county_name(county)
    df = _load_seasonal_csv(county_norm, wy)
    return sorted(df["UniqueID"].astype(str).unique().tolist())


def main():
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Generate per-parcel ET correction plots (separate daily + monthly).",
    )
    parser.add_argument("--county", required=True, help="County name (e.g. Fresno)")
    parser.add_argument("--water-year", type=int, default=None,
                        help="Single water year")
    parser.add_argument("--wy-start", type=int, default=None,
                        help="Start water year (inclusive)")
    parser.add_argument("--wy-end", type=int, default=None,
                        help="End water year (inclusive)")
    parser.add_argument("--parcel", type=str, default=None,
                        help="Parcel UniqueID")
    parser.add_argument("--random", type=int, default=None, metavar="N",
                        help="Plot N randomly sampled parcels")
    parser.add_argument("--et-mode", choices=["actual", "corrected", "both"],
                        default="both", help="ET mode (default: both)")
    parser.add_argument("--method", choices=["A", "B"], default="A",
                        help="ET correction method (default: A)")
    parser.add_argument("--n-boot", type=int, default=400,
                        help="Bootstrap replicates for CI (default: 400)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: es_analysis/output/figures/et_corrections/)")

    args = parser.parse_args()

    # Resolve water years
    if args.water_year is not None:
        wys = [args.water_year]
    elif args.wy_start is not None and args.wy_end is not None:
        wys = list(range(args.wy_start, args.wy_end + 1))
    else:
        parser.error("Provide --water-year or both --wy-start and --wy-end")

    output_dir = Path(args.output_dir) if args.output_dir else None

    from es_analysis.charts.et_corrections.et_separate_daily_monthly_plot import (
        et_separate_daily_monthly,
    )

    results = []
    for wy in wys:
        # Determine parcels
        if args.parcel:
            parcels = [args.parcel]
        elif args.random:
            import numpy as np
            all_parcels = _get_parcels_for_county_wy(args.county, wy)
            rng = np.random.default_rng(42)
            n_sample = min(args.random, len(all_parcels))
            parcels = rng.choice(all_parcels, size=n_sample, replace=False).tolist()
            print(f"WY{wy}: sampled {n_sample} parcels: {parcels}")
        else:
            parser.error("Provide --parcel or --random N")

        for uid in parcels:
            print(f"\nPlotting {args.county} WY{wy} UID={uid} "
                  f"(et_mode={args.et_mode}, method={args.method})")
            try:
                result = et_separate_daily_monthly(
                    county=args.county,
                    water_year=wy,
                    uid=uid,
                    method=args.method,
                    et_mode=args.et_mode,
                    n_boot=args.n_boot,
                    output_dir=output_dir,
                )
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\nDone. Generated {len(results)} plot(s).")
    for r in results:
        print(f"  {r['figure_path']}")


if __name__ == "__main__":
    main()
