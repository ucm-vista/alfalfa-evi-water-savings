#!/usr/bin/env python3

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_sample_parcel(county, wy):
    """Pick the first parcel ID from the county-year CSV."""
    import pandas as pd
    from es_analysis.data_providers.config import config
    from es_analysis.data_providers.evi_provider import normalize_county_name
    county_norm = normalize_county_name(county)
    csv_path = config.county_year_root_new / county_norm / f"WY{wy}.csv"
    if not csv_path.exists():
        csv_path = config.county_year_root / county_norm / f"WY{wy}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, usecols=["parcel_id"], nrows=1000)
        parcels = sorted(df["parcel_id"].unique())
        if parcels:
            return str(parcels[0])
    return None


# (module_path, func_name, description, chart_type)
# chart_type: "parcel" needs county+parcel_id+years, "county_year" needs county+wy
CHART_FUNCTIONS = [
    ("es_analysis.charts.evi.evi_raw_plot", "plot_evi_raw", "Original EVI with raw observations", "parcel"),
    ("es_analysis.charts.evi.evi_gapfilled_plot", "plot_evi_gapfilled", "Gap-filled EVI using quartic interpolation", "parcel"),
    ("es_analysis.charts.evi.evi_smoothed_plot", "plot_evi_smoothed", "Smoothed EVI using Savitzky-Golay", "parcel"),
    ("es_analysis.charts.evi.evi_combined_plot", "plot_evi_combined", "Combined EVI (original, gap-filled, smoothed)", "parcel"),
    ("es_analysis.charts.evi.evi_county_year_original_plot", "plot_evi_county_year_original", "County/Year original EVI", "county_year"),
    ("es_analysis.charts.evi.evi_county_year_observations_plot", "plot_evi_county_year_observations", "County/Year observation points", "county_year"),
    ("es_analysis.charts.evi.evi_county_year_gapfilled_plot", "plot_evi_county_year_gapfilled", "County/Year gap-filled EVI", "county_year"),
    ("es_analysis.charts.evi.evi_county_year_smoothed_plot", "plot_evi_county_year_smoothed", "County/Year smoothed EVI", "county_year"),
]


def run_all_evi_plots(county=None, water_year=None, parcel_id=None, output_dir=None):
    county = county or "Kern"
    water_year = water_year or 2021

    if parcel_id is None:
        parcel_id = _get_sample_parcel(county, water_year)

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("es_analysis/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_charts = len(CHART_FUNCTIONS)
    success_count = 0
    failed_charts = []

    print(f"\n{'='*60}")
    print("EVI PLOTS RUNNER")
    print(f"{'='*60}")
    print(f"Total charts to generate: {total_charts}")
    print(f"Parameters: county={county}, water_year={water_year}, parcel_id={parcel_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    for i, (module_path, func_name, func_desc, chart_type) in enumerate(CHART_FUNCTIONS, 1):
        print(f"[{i}/{total_charts}] Generating: {func_desc}")

        try:
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name, None)
            if func is None:
                print(f"    Warning: Function '{func_name}' not found in {module_path}, skipping...")
                failed_charts.append((func_name, "Function not found"))
                continue

            if chart_type == "parcel":
                if parcel_id is None:
                    print(f"    Warning: No parcel_id available, skipping...")
                    failed_charts.append((func_name, "No parcel_id"))
                    continue
                result = func(county=county, parcel_id=str(parcel_id), years=[water_year], output_dir=output_dir)
            elif chart_type == "county_year":
                result = func(county=county, wy=water_year, output_dir=output_dir)
            else:
                result = func(county=county, output_dir=output_dir)

            print(f"    Success: {result}")
            success_count += 1

        except Exception as e:
            print(f"    Failed: {type(e).__name__}: {e}")
            failed_charts.append((func_name, str(e)))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total_charts}")
    print(f"Failed: {len(failed_charts)}/{total_charts}")

    if failed_charts:
        print("\nFailed charts:")
        for name, error in failed_charts:
            print(f"  - {name}: {error}")

    print(f"{'='*60}\n")

    return 1 if failed_charts else 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all EVI-related charts (8 charts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evi_plots.py
  python run_evi_plots.py --county Fresno
  python run_evi_plots.py --county Kern --water-year 2021
  python run_evi_plots.py --county Kern --water-year 2021 --parcel-id 1500041
        """
    )
    parser.add_argument("--county", type=str, default=None, help="County name")
    parser.add_argument("--water-year", type=int, default=None, dest="water_year", help="Water year (2019-2024)")
    parser.add_argument("--parcel-id", type=str, default=None, dest="parcel_id", help="Parcel ID")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir", help="Output directory")

    args = parser.parse_args()
    exit_code = run_all_evi_plots(
        county=args.county, water_year=args.water_year,
        parcel_id=args.parcel_id, output_dir=args.output_dir
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
