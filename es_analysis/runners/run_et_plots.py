#!/usr/bin/env python3

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_sample_uid(county, wy):
    """Pick the first parcel ID (uid) from the BEAST seasonal CSV."""
    import pandas as pd
    from es_analysis.data_providers import BEASTDataProvider
    provider = BEASTDataProvider()
    df = provider.load_seasonal_cuts_csv(county, wy)
    if df is not None and "parcel_id" in df.columns:
        uids = sorted(df["parcel_id"].astype(str).unique())
        if uids:
            return uids[0]
    return None


# (module, function, description, chart_type)
# Types: "single_wy" = county+water_year+uid, "multi_wy" = county+wy_start+wy_end+uid
CHART_FUNCTIONS = [
    ("es_analysis.charts.et_corrections.et_monthly_openet_bar_plot", "et_monthly_openet_bar_plot", "Monthly OpenET bar chart", "single_wy"),
    ("es_analysis.charts.et_corrections.et_monthly_corrected_bar_plot", "et_monthly_corrected_bar_plot", "Monthly corrected ET bar chart", "single_wy"),
    ("es_analysis.charts.et_corrections.et_monthly_ci_errorbars_plot", "et_monthly_ci_errorbars_plot", "Monthly ET with confidence intervals", "single_wy"),
    ("es_analysis.charts.et_corrections.et_harvest_dates_scatter_plot", "et_harvest_dates_scatter_plot", "Harvest dates scatter plot", "single_wy"),
    ("es_analysis.charts.et_corrections.et_landsat_passes_scatter_plot", "et_landsat_passes_scatter_plot", "Landsat passes scatter plot", "single_wy"),
    ("es_analysis.charts.et_corrections.et_daily_scatter_plot", "et_daily_scatter_plot", "Daily ET scatter plot", "single_wy"),
    ("es_analysis.charts.et_corrections.etof_daily_line_plot", "etof_daily_line_plot", "ETof daily line plot", "single_wy"),
    ("es_analysis.charts.et_corrections.et_full_two_panel_plot", "et_full_two_panel_plot", "Full two-panel ET correction plot", "single_wy"),
    ("es_analysis.charts.et_corrections.et_full_multiwy_plot", "et_full_multiwy_plot", "Multi-year ET correction plot", "multi_wy"),
    ("es_analysis.charts.et_corrections.et_cut_dates_and_et_plot", "et_cut_dates_and_et_plot", "Cut dates with ET bars", "single_wy"),
    ("es_analysis.charts.et_corrections.et_cut_dates_and_et_corrected_plot", "et_cut_dates_and_et_corrected_plot", "Cut dates with corrected ET bars", "single_wy"),
]


def _uid_exists(county, wy, uid):
    """Check if a UID exists in the BEAST seasonal CSV."""
    import pandas as pd
    from es_analysis.data_providers import BEASTDataProvider
    provider = BEASTDataProvider()
    df = provider.load_seasonal_cuts_csv(county, wy)
    if df is not None and "parcel_id" in df.columns:
        return str(uid) in df["parcel_id"].astype(str).values
    return False


def run_all_et_plots(county=None, water_year=None, uid=None, output_dir=None):
    county = county or "Kern"
    water_year = water_year or 2021

    if uid is not None and not _uid_exists(county, water_year, uid):
        fallback = _get_sample_uid(county, water_year)
        print(f"    Note: UID {uid} not found in BEAST CSVs for {county} WY{water_year}, falling back to {fallback}")
        uid = fallback

    if uid is None:
        uid = _get_sample_uid(county, water_year)

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("es_analysis/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_charts = len(CHART_FUNCTIONS)
    success_count = 0
    failed_charts = []

    print(f"\n{'='*60}")
    print("ET CORRECTIONS PLOTS RUNNER")
    print(f"{'='*60}")
    print(f"Total charts to generate: {total_charts}")
    print(f"Parameters: county={county}, water_year={water_year}, uid={uid}")
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

            if uid is None:
                print(f"    Warning: No uid available, skipping...")
                failed_charts.append((func_name, "No uid"))
                continue

            if chart_type == "single_wy":
                result = func(county=county, water_year=water_year, uid=str(uid), output_dir=output_dir)
            elif chart_type == "multi_wy":
                result = func(county=county, wy_start=water_year, wy_end=water_year, uid=str(uid), output_dir=output_dir)
            else:
                result = func(county=county, water_year=water_year, uid=str(uid), output_dir=output_dir)

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
        description="Run all ET correction charts (11 charts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--county", type=str, default=None, help="County name")
    parser.add_argument("--water-year", type=int, default=None, dest="water_year", help="Water year")
    parser.add_argument("--uid", type=str, default=None, help="Parcel UID")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir", help="Output directory")

    args = parser.parse_args()
    exit_code = run_all_et_plots(
        county=args.county, water_year=args.water_year,
        uid=args.uid, output_dir=args.output_dir
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
