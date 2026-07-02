#!/usr/bin/env python3

import argparse
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _load_runner_module(module_name):
    return importlib.import_module(f"es_analysis.runners.{module_name}")


def run_evi_plots(county=None, water_year=None, parcel_id=None, output_dir=None):
    evi_runner = _load_runner_module("run_evi_plots")
    return evi_runner.run_all_evi_plots(county, water_year, parcel_id, output_dir)


def run_beast_plots(county=None, water_year=None, parcel_id=None, output_dir=None):
    beast_runner = _load_runner_module("run_beast_plots")
    return beast_runner.run_all_beast_plots(county, water_year, parcel_id, output_dir)


def run_et_plots(county=None, water_year=None, parcel_id=None, output_dir=None):
    et_runner = _load_runner_module("run_et_plots")
    return et_runner.run_all_et_plots(county=county, water_year=water_year, uid=parcel_id, output_dir=output_dir)


def run_statistics_plots(county=None, water_year=None, parcel_id=None, output_dir=None):
    stats_runner = _load_runner_module("run_statistics_plots")
    return stats_runner.run_all_statistics_plots(county=county, water_year=water_year, output_dir=output_dir)


MULTI_PANEL_CHARTS = [
    ("es_analysis.charts.multi_panel.county_points_two_panel_v1_plot", "run_all_counties_county_year_scatter", "Two-panel v1: county-year scatter", "two_panel_daymet"),
    ("es_analysis.charts.multi_panel.county_points_two_panel_v1_plot", "run_all_counties_county_overall_scatter", "Two-panel v1: county-overall scatter", "two_panel_daymet"),
    ("es_analysis.charts.multi_panel.county_points_two_panel_v2_plot", "run_two_panel_scatter", "Two-panel v2 scatter", "two_panel_daymet"),
    ("es_analysis.charts.multi_panel.county_points_four_panel_plot", "run_four_panel_scatter", "Four-panel scatter", "four_panel"),
    ("es_analysis.charts.multi_panel.county_map_and_ridgelines_plot", "plot_county_map_and_ridgelines", "County map and ridgelines", "map_ridge"),
]


def _build_parcel_year_df_for_multi_panel():
    """Build a parcel-year DataFrame with columns needed by multi-panel charts.

    Loads real pre-computed data from Cutting_weather_stats/ (tmax and gdd5
    variants), merges them, and derives per-cut metrics for the four-panel chart.
    """
    import numpy as np
    import pandas as pd
    from es_analysis.data_providers.config import config

    stats_root = config.cutting_weather_stats_root

    # Load tmax variant (has tmax_mean)
    tmax_csv = stats_root / "cut_weather_parcel_year_wide_WY2019-2024.csv"
    # Load gdd5 variant (has gdd5_mean)
    gdd5_csv = stats_root / "gdd5" / "cut_weather_parcel_year_wide_WY2019-2024.csv"

    if not tmax_csv.exists() and not gdd5_csv.exists():
        raise FileNotFoundError(
            f"No pre-computed parcel-year CSVs found at {stats_root}. "
            "Run the monolith build_parcel_summary_wy() first."
        )

    frames = []
    if tmax_csv.exists():
        frames.append(pd.read_csv(tmax_csv))
    if gdd5_csv.exists():
        df_gdd5 = pd.read_csv(gdd5_csv)
        if frames:
            # Merge gdd5_mean onto the tmax frame
            frames[0] = frames[0].merge(
                df_gdd5[["UniqueID", "county", "WY", "gdd5_mean"]],
                on=["UniqueID", "county", "WY"],
                how="outer",
            )
        else:
            frames.append(df_gdd5)

    df = frames[0].copy()

    # Ensure expected columns exist with proper names for charts
    # Two-panel v1/v2 charts expect: et_cum_minET_to_last_cut_mm, {daymet_var}_mean
    # Four-panel chart expects: gdd5_cum, gdd5_mean_per_cut, et_cum_mm, et_mean_per_cut_mm

    # Map pre-computed columns to four-panel column names
    n_cuts = pd.to_numeric(df["n_cp_season"], errors="coerce").clip(lower=1)

    if "gdd5_mean" in df.columns:
        df["gdd5_cum"] = df["gdd5_mean"]  # cumulative GDD5 over segments
        df["gdd5_mean_per_cut"] = df["gdd5_mean"] / n_cuts

    if "et_cum_minET_to_last_cut_mm" in df.columns:
        df["et_cum_mm"] = df["et_cum_minET_to_last_cut_mm"]
        df["et_mean_per_cut_mm"] = df["et_cum_minET_to_last_cut_mm"] / n_cuts

    if df.empty:
        return pd.DataFrame()
    return df


def run_multi_panel_plots(county=None, water_year=None, parcel_id=None, output_dir=None):
    total = len(MULTI_PANEL_CHARTS)
    success_count = 0
    failed_charts = []
    parcel_year_df = None

    print(f"\n{'='*60}")
    print("MULTI-PANEL PLOTS RUNNER")
    print(f"{'='*60}")
    print(f"Total charts to generate: {total}")
    print(f"Parameters: county={county}, water_year={water_year}, parcel_id={parcel_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    for i, (module_path, func_name, func_desc, chart_type) in enumerate(MULTI_PANEL_CHARTS, 1):
        print(f"[{i}/{total}] Generating: {func_desc}")

        try:
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name, None)
            if func is None:
                print(f"    Warning: Function '{func_name}' not found in {module_path}, skipping...")
                failed_charts.append((func_name, "Function not found"))
                continue

            if parcel_year_df is None and chart_type in ("two_panel_daymet", "four_panel", "map_ridge"):
                print("    Building parcel-year DataFrame from BEAST CSVs...")
                parcel_year_df = _build_parcel_year_df_for_multi_panel()
                if parcel_year_df.empty:
                    raise ValueError("Could not build parcel-year DataFrame")
                print(f"    Built DataFrame: {len(parcel_year_df)} rows, counties={parcel_year_df['county'].nunique()}")

            if chart_type == "two_panel_daymet":
                result = func(
                    df_parcel_year=parcel_year_df,
                    daymet_var="tmax",
                    cut_metric="n_cp_season",
                    wy_start=2019, wy_end=2024,
                )
            elif chart_type == "four_panel":
                result = func(
                    df_parcel_year=parcel_year_df,
                    cut_metric="n_cp_season",
                    wy_start=2019, wy_end=2024,
                )
            elif chart_type == "map_ridge":
                result = func(
                    gdf=parcel_year_df,
                    wy_map_start=2019,
                    wy_map_end=2024,
                    cut_metric="n_cp_season",
                )
            else:
                result = func()

            print(f"    Success")
            success_count += 1

        except Exception as e:
            print(f"    Failed: {type(e).__name__}: {e}")
            failed_charts.append((func_name, str(e)))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total}")
    print(f"Failed: {len(failed_charts)}/{total}")

    if failed_charts:
        print("\nFailed charts:")
        for name, error in failed_charts:
            print(f"  - {name}: {error}")

    print(f"{'='*60}\n")

    return 1 if failed_charts else 0


def run_all_plots(county=None, water_year=None, parcel_id=None, output_dir=None, stop_on_error=False):
    execution_stages = [
        ("EVI", run_evi_plots),
        ("BEAST", run_beast_plots),
        ("ET", run_et_plots),
        ("Statistics", run_statistics_plots),
        ("Multi-panel", run_multi_panel_plots),
    ]

    print(f"\n{'='*60}")
    print("MASTER PLOTS RUNNER - ALL CHARTS")
    print(f"{'='*60}")
    print(f"Parameters: county={county}, water_year={water_year}, parcel_id={parcel_id}")
    print(f"Output directory: {output_dir}")
    print(f"Stop on error: {stop_on_error}")
    print(f"{'='*60}\n")

    results = {}

    for stage_name, stage_func in execution_stages:
        print(f"\n>>> RUNNING {stage_name.upper()} STAGE...")
        print("="*60)

        try:
            exit_code = stage_func(
                county=county,
                water_year=water_year,
                parcel_id=parcel_id,
                output_dir=output_dir
            )
            results[stage_name] = exit_code

            if exit_code == 0:
                print(f"  {stage_name} stage completed successfully")
            else:
                print(f"  {stage_name} stage completed with errors (exit code {exit_code})")
                if stop_on_error:
                    print("\nStopping execution due to error (stop-on-error flag set)")
                    break

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  {stage_name} stage failed with exception: {error_msg}")
            results[stage_name] = 1

            if stop_on_error:
                print("\nStopping execution due to error (stop-on-error flag set)")
                break

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for stage_name, exit_code in results.items():
        status = "SUCCESS" if exit_code == 0 else "FAILED"
        print(f"{stage_name:15s} - {status} (exit code: {exit_code})")

    ran = len(results)
    successful_stages = sum(1 for code in results.values() if code == 0)
    failed_stages = sum(1 for code in results.values() if code != 0)
    skipped = len(execution_stages) - ran

    print(f"\nTotal stages: {len(execution_stages)}")
    print(f"Ran: {ran}")
    print(f"Successful: {successful_stages}")
    print(f"Failed: {failed_stages}")
    if skipped:
        print(f"Skipped: {skipped}")
    print(f"{'='*60}\n")

    return 1 if any(results.values()) else 0


def main():
    parser = argparse.ArgumentParser(
        description="Run ALL charts in dependency order: EVI -> BEAST -> ET -> Statistics -> Multi-panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_plots.py
  python run_all_plots.py --county Fresno
  python run_all_plots.py --county Fresno --water-year 2022
  python run_all_plots.py --stop-on-error
  python run_all_plots.py --output-dir /path/to/output

Execution order:
  1. EVI plots (8 charts)
  2. BEAST plots (15 charts)
  3. ET corrections plots (11 charts)
  4. Statistics plots (7 charts)
  5. Multi-panel plots (5 charts)
  Total: 46 charts
        """
    )
    parser.add_argument(
        "--county",
        type=str,
        default=None,
        help="Filter by county name"
    )
    parser.add_argument(
        "--water-year",
        type=int,
        default=None,
        dest="water_year",
        help="Filter by water year (2019-2023)"
    )
    parser.add_argument(
        "--parcel-id",
        type=str,
        default=None,
        dest="parcel_id",
        help="Filter by parcel ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Custom output directory for generated figures"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        dest="stop_on_error",
        help="Stop execution if any stage fails"
    )

    args = parser.parse_args()

    exit_code = run_all_plots(
        county=args.county,
        water_year=args.water_year,
        parcel_id=args.parcel_id,
        output_dir=args.output_dir,
        stop_on_error=args.stop_on_error
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
