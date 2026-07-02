#!/usr/bin/env python3

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_sample_parcel(county, wy):
    """Pick the first parcel ID from the BEAST seasonal CSV."""
    import pandas as pd
    from es_analysis.data_providers import BEASTDataProvider
    provider = BEASTDataProvider()
    df = provider.load_seasonal_cuts_csv(county, wy)
    if df is not None and "parcel_id" in df.columns:
        parcels = sorted(df["parcel_id"].astype(str).unique())
        if parcels:
            return parcels[0]
    return None


# (module, function, description, chart_type)
# Types: "parcel" = per-parcel, "county_agg" = county aggregate, "trend" = trend CPs, "all_counties"
CHART_FUNCTIONS = [
    ("es_analysis.charts.beast.beast_decomposition_plot", "plot_beast_decomposition", "BEAST decomposition with change points", "parcel"),
    ("es_analysis.charts.beast.beast_evi_with_cps_original_plot", "plot_beast_evi_with_cps_original", "EVI with change points - original", "parcel"),
    ("es_analysis.charts.beast.beast_evi_with_cps_observations_plot", "plot_beast_evi_with_cps_observations", "EVI with change points - observations", "parcel"),
    ("es_analysis.charts.beast.beast_evi_with_cps_gapfilled_plot", "plot_beast_evi_with_cps_gapfilled", "EVI with change points - gap-filled", "parcel"),
    ("es_analysis.charts.beast.beast_evi_with_cps_smoothed_plot", "plot_beast_evi_with_cps_smoothed", "EVI with change points - smoothed", "parcel"),
    ("es_analysis.charts.beast.beast_evi_with_cps_combined_plot", "plot_beast_evi_with_cps_combined", "EVI with change points - combined", "parcel"),
    ("es_analysis.charts.beast.beast_n_cuttings_boxplot", "plot_beast_n_cuttings_boxplot", "Number of cuttings distribution", "county_agg"),
    ("es_analysis.charts.beast.beast_n_cp_season_boxplot", "plot_beast_n_cp_season_boxplot", "Seasonal change points distribution", "county_agg"),
    ("es_analysis.charts.beast.beast_trend_cp_medians_yearly", "plot_beast_trend_cp_medians_yearly", "Trend CP medians across years", "trend"),
    ("es_analysis.charts.beast.beast_trend_cp_medians_by_order_histogram", "plot_beast_trend_cp_medians_by_order_histogram", "Trend CP frequency by order", "trend"),
    ("es_analysis.charts.beast.beast_trend_cps_allpoints", "plot_beast_trend_cps_allpoints", "All trend CP scatter points", "trend"),
    ("es_analysis.charts.beast.beast_trend_cp_medians_by_order_with_errors", "plot_beast_trend_cp_medians_by_order_with_errors", "Trend CP medians with error bars", "trend"),
    ("es_analysis.charts.beast.beast_county_year_boxplots", "plot_beast_county_year_boxplots", "Cutting distribution by county and year", "county_agg"),
    ("es_analysis.charts.beast.beast_all_counties_boxplot", "plot_beast_all_counties_boxplot", "All counties comparison boxplot", "all_counties"),
    ("es_analysis.charts.beast.beast_evi_with_cps_and_mins_plot", "plot_beast_evi_with_cps_and_mins", "EVI with CPs and detected minima", "parcel"),
]


def run_all_beast_plots(county=None, water_year=None, parcel_id=None, output_dir=None):
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
    print("BEAST PLOTS RUNNER")
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
                print(f"    Warning: Function '{func_name}' not found, skipping...")
                failed_charts.append((func_name, "Function not found"))
                continue

            if chart_type == "parcel":
                if parcel_id is None:
                    print(f"    Warning: No parcel_id available, skipping...")
                    failed_charts.append((func_name, "No parcel_id"))
                    continue
                result = func(county=county, wy=water_year, parcel=str(parcel_id), output_dir=output_dir)
            elif chart_type == "county_agg":
                result = func(county=county, years=[water_year], output_dir=output_dir)
            elif chart_type == "trend":
                result = func(county=county, output_dir=output_dir, show=False)
            elif chart_type == "all_counties":
                result = func(counties=[county], column="n_cuttings", year=water_year, output_dir=output_dir)
            else:
                result = func(county=county, output_dir=output_dir)

            print(f"    Success")
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
        description="Run all BEAST-related charts (15 charts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--county", type=str, default=None, help="County name")
    parser.add_argument("--water-year", type=int, default=None, dest="water_year", help="Water year")
    parser.add_argument("--parcel-id", type=str, default=None, dest="parcel_id", help="Parcel ID")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir", help="Output directory")

    args = parser.parse_args()
    exit_code = run_all_beast_plots(
        county=args.county, water_year=args.water_year,
        parcel_id=args.parcel_id, output_dir=args.output_dir
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
