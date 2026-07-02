#!/usr/bin/env python3

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _build_parcel_year_df(counties=None, wy_start=2019, wy_end=2024):
    """Build a parcel-year DataFrame from pre-computed Cutting_weather_stats.

    Loads real ET and weather data instead of synthetic values.
    Needed by bar_by_year_plot, bar_by_county_plot, and late_scatter charts.
    """
    import pandas as pd
    from es_analysis.data_providers.config import config

    stats_root = config.cutting_weather_stats_root
    csv_path = stats_root / "cut_weather_parcel_year_wide_WY2019-2024.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Merge GDD5 variant if available
        gdd5_csv = stats_root / "gdd5" / "cut_weather_parcel_year_wide_WY2019-2024.csv"
        if gdd5_csv.exists():
            df_gdd5 = pd.read_csv(gdd5_csv)
            if "gdd5_mean" in df_gdd5.columns:
                df = df.merge(
                    df_gdd5[["UniqueID", "county", "WY", "gdd5_mean"]],
                    on=["UniqueID", "county", "WY"],
                    how="left",
                )
    else:
        # Fallback: load from BEAST CSVs (cutting counts only, no ET)
        from es_analysis.data_providers import BEASTDataProvider
        provider = BEASTDataProvider()
        frames = []
        if counties is None:
            counties = provider.get_selected_counties()
        for county in counties:
            for wy in provider.detect_years_on_disk(county):
                if wy < wy_start or wy > wy_end:
                    continue
                sub = provider.load_seasonal_cuts_csv(county, wy)
                if sub is None or sub.empty:
                    continue
                frames.append(pd.DataFrame({
                    "county": county,
                    "WY": wy,
                    "UniqueID": sub["parcel_id"].astype(str) if "parcel_id" in sub.columns else sub.index.astype(str),
                    "n_cuttings": pd.to_numeric(sub.get("n_cuttings", 0), errors="coerce").fillna(0).astype(int),
                    "n_cp_season": pd.to_numeric(sub.get("n_cp_season", 0), errors="coerce").fillna(0).astype(int),
                }))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

    # Filter by counties and WY range
    if counties is not None:
        df = df[df["county"].isin(counties)]
    df = df[df["WY"].between(wy_start, wy_end)]

    return df


# (module, function, description, chart_type)
# Types:
#   "self_counties" = auto-detects counties/years from disk (metric/years/outfile)
#   "county_wy"     = needs county + wy
#   "county_years"  = needs county + years list
#   "df_by_year"    = needs parcel-year DataFrame + wy_start + wy_end
#   "df_by_county"  = needs parcel-year DataFrame + wy_start + wy_end
#   "df_scatter"    = needs parcel-year DataFrame + wy_start + wy_end (late scatter)
CHART_FUNCTIONS = [
    ("es_analysis.charts.statistics.monthly_by_county_bar_plot", "monthly_by_county_bar_plot", "Monthly statistics by county", "self_counties"),
    ("es_analysis.charts.statistics.yearly_by_county_bar_plot", "yearly_by_county_bar_plot", "Yearly statistics by county", "self_counties"),
    ("es_analysis.charts.statistics.cut_dates_scatter_plot", "cut_dates_scatter_plot", "Cut dates scatter plot", "county_wy"),
    ("es_analysis.charts.statistics.n_cuttings_boxplot", "n_cuttings_boxplot", "Number of cuttings boxplot", "county_years"),
    ("es_analysis.charts.statistics.bar_by_year_plot", "bar_by_year_plot", "Total cuttings by year", "df_by_year"),
    ("es_analysis.charts.statistics.bar_by_county_plot", "bar_by_county_plot", "County comparison of cuttings", "df_by_county"),
    ("es_analysis.charts.statistics.late_scatter_county_color_year_marker_plot", "late_scatter_county_color_year_marker_plot", "Late cuttings vs ET scatter", "df_scatter"),
    ("es_analysis.charts.statistics.multicounty_parcel_scatter_plot", "multicounty_parcel_scatter_plot", "Multi-county parcel scatter", "multicounty_scatter"),
    ("es_analysis.charts.statistics.daymet_scatter_plot", "daymet_scatter_plot", "Daymet vs cuttings scatter", "daymet_scatter"),
    ("es_analysis.charts.statistics.weather_et_scatter_plot", "weather_et_scatter_plot", "Weather & ET three-panel scatter", "weather_et_scatter"),
    ("es_analysis.charts.statistics.savings_bar_by_year_plot", "savings_bar_by_year_plot", "Late-water savings by year", "savings_by_year"),
    ("es_analysis.charts.statistics.savings_bar_by_county_plot", "savings_bar_by_county_plot", "Late-water savings by county", "savings_by_county"),
    ("es_analysis.charts.statistics.debug_parcel_plot", "debug_parcel_plot", "Debug parcel EVI/ET plot", "debug_parcel"),
    ("es_analysis.charts.statistics.marginal_et_per_cutting_plot", "marginal_et_per_cutting_plot", "Marginal ET per late cutting", "late_cut_parcel"),
    ("es_analysis.charts.statistics.county_late_water_bar_plot", "county_late_water_bar_plot", "County late-season water budget", "late_cut_parcel"),
    ("es_analysis.charts.statistics.cuttings_scatter_plot", "cuttings_scatter_by_year", "Cuttings vs ET/GDD5 scatter (per year)", "cuttings_scatter_year"),
    ("es_analysis.charts.statistics.cuttings_scatter_plot", "cuttings_scatter_aggregate", "Cuttings vs ET/GDD5 aggregate scatter", "cuttings_scatter_agg"),
    ("es_analysis.charts.statistics.wy_type_analysis_plot", "wy_type_cuttings_bar", "WY type cuttings bar", "wy_type"),
    ("es_analysis.charts.statistics.wy_type_analysis_plot", "wy_type_annual_et_bar", "WY type annual ET bar", "wy_type"),
    ("es_analysis.charts.statistics.wy_type_analysis_plot", "wy_type_county_et_heatmap", "WY type county ET heatmap", "wy_type"),
    ("es_analysis.charts.statistics.wy_type_analysis_plot", "wy_type_late_cut_pct_bar", "WY type late-cut prevalence", "wy_type"),
    ("es_analysis.charts.statistics.county_wy_et_heatmap_plot", "county_wy_et_triple_heatmap", "County x WY ET triple heatmap", "triple_heatmap"),
    ("es_analysis.charts.statistics.diagnostic_anomaly_plots", "generate_diagnostic_plots", "Diagnostic anomaly plots", "diagnostic"),
]


def run_all_statistics_plots(county=None, water_year=None, output_dir=None):
    county = county or "Kern"
    water_year = water_year or 2021

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("es_analysis/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build parcel-year DataFrame for charts that need it
    parcel_year_df = None

    total_charts = len(CHART_FUNCTIONS)
    success_count = 0
    failed_charts = []

    print(f"\n{'='*60}")
    print("STATISTICS PLOTS RUNNER")
    print(f"{'='*60}")
    print(f"Total charts to generate: {total_charts}")
    print(f"Parameters: county={county}, water_year={water_year}")
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

            if chart_type == "self_counties":
                result = func(metric="n_cuttings", outfile=output_dir / f"{func_name}.png")
            elif chart_type == "county_wy":
                result = func(county=county, wy=water_year, outfile=output_dir / f"{func_name}.png")
            elif chart_type == "county_years":
                result = func(county=county, years=[water_year], outfile=output_dir / f"{func_name}.png")
            elif chart_type == "late_cut_parcel":
                from es_analysis.data_providers.config import config as cfg
                csv_path = cfg.water_saving_out_dir / "late_cut_base_parcel_year.csv"
                if not csv_path.exists():
                    print(f"    Skipping: {csv_path} not found")
                    continue
                import pandas as _pd
                df_late = _pd.read_csv(csv_path)
                result = func(df=df_late, outfile=output_dir / f"{func_name}.png")
            elif chart_type in ("df_by_year", "df_by_county", "df_scatter",
                                "multicounty_scatter", "daymet_scatter",
                                "weather_et_scatter",
                                "savings_by_year", "savings_by_county",
                                "debug_parcel",
                                "cuttings_scatter_year", "cuttings_scatter_agg"):
                if parcel_year_df is None:
                    print("    Building parcel-year DataFrame from BEAST CSVs...")
                    parcel_year_df = _build_parcel_year_df()
                    if parcel_year_df.empty:
                        raise ValueError("Could not build parcel-year DataFrame")
                    print(f"    Built DataFrame: {len(parcel_year_df)} rows, counties={parcel_year_df['county'].nunique()}")

                if chart_type == "df_by_year":
                    result = func(
                        df=parcel_year_df,
                        wy_start=2019, wy_end=2024,
                        value_col="n_cuttings",
                        ylabel="Number of Cuttings",
                        title="Cuttings by Water Year",
                        outfile=output_dir / f"{func_name}.png",
                    )
                elif chart_type == "df_by_county":
                    result = func(
                        df=parcel_year_df,
                        wy_start=2019, wy_end=2024,
                        value_col="n_cuttings",
                        ylabel="Number of Cuttings",
                        title="Cuttings by County",
                        outfile=output_dir / f"{func_name}.png",
                    )
                elif chart_type == "df_scatter":
                    # This chart requires late_et_mm and n_late_cuts columns.
                    # These aren't in the pre-computed CSV; derive from real ET
                    # using cumulative ET and cutting counts as proxies.
                    import numpy as np
                    import pandas as pd
                    df_scatter = parcel_year_df.copy()
                    n_cuts = pd.to_numeric(df_scatter.get("n_cuttings", 0), errors="coerce").fillna(1).clip(lower=1)
                    et_cum = pd.to_numeric(df_scatter.get("et_cum_minET_to_last_cut_mm", 0), errors="coerce").fillna(0)
                    # Approximate: late cuts ~ cuts beyond the 4th, late ET ~ proportional share
                    df_scatter["n_late_cuts"] = (n_cuts - 4).clip(lower=0).astype(int)
                    df_scatter["late_et_mm"] = (et_cum * df_scatter["n_late_cuts"] / n_cuts).round(2)
                    result = func(
                        df=df_scatter,
                        wy_start=2019, wy_end=2024,
                        outfile=output_dir / f"{func_name}.png",
                    )
                elif chart_type == "multicounty_scatter":
                    if parcel_year_df is None:
                        raise ValueError("parcel_year_df not built")
                    if "gdd5_mean" not in parcel_year_df.columns:
                        print("    Skipping multicounty scatter: gdd5_mean column not available")
                        continue
                    result = func(
                        df=parcel_year_df,
                        daymet_var="gdd5",
                        cut_metric="n_cp_season",
                        wy_label="WY 2019-2024",
                        outfile=output_dir / f"{func_name}.png",
                    )
                elif chart_type == "daymet_scatter":
                    if parcel_year_df is None:
                        raise ValueError("parcel_year_df not built")
                    if "gdd5_mean" not in parcel_year_df.columns:
                        print("    Skipping daymet scatter: gdd5_mean column not available")
                        continue
                    sub = parcel_year_df[
                        (parcel_year_df["county"] == county)
                        & (parcel_year_df["WY"] == water_year)
                    ]
                    if sub.empty:
                        print(f"    Skipping daymet scatter: no data for {county} WY {water_year}")
                        continue
                    result = func(
                        df=sub,
                        county=county,
                        wy_label=f"WY {water_year}",
                        daymet_var="gdd5",
                        cut_metric="n_cp_season",
                        outfile=output_dir / f"{func_name}.png",
                    )
                elif chart_type == "weather_et_scatter":
                    if parcel_year_df is None:
                        raise ValueError("parcel_year_df not built")
                    needed = ["tmax_mean", "gdd5_mean", "et_cum_minET_to_last_cut_mm"]
                    missing = [c for c in needed if c not in parcel_year_df.columns]
                    if missing:
                        print(f"    Skipping weather-ET scatter: missing columns {missing}")
                        continue
                    result = func(
                        df=parcel_year_df,
                        cut_metric="n_cp_season",
                        wy_label="WYs 2019-2024",
                        outfile=output_dir / f"{func_name}.png",
                    )
                elif chart_type == "savings_by_year":
                    # Requires late-water savings summary; skip if not available
                    try:
                        from es_analysis.data_providers.late_water_provider import (
                            make_savings_summary,
                            add_cap_savings_columns,
                        )
                        import pandas as pd
                        csv_path = Path("water_saving_scenarios_stats/late_cut_savings_parcel_year_mm.csv")
                        if not csv_path.exists():
                            from es_analysis.data_providers.config import config
                            csv_path = config.water_saving_out_dir / "late_cut_savings_parcel_year_mm.csv"
                        if csv_path.exists():
                            df_sav = pd.read_csv(csv_path)
                            # Reconstruct list column
                            import ast
                            df_sav["late_cycle_et_mm_list"] = df_sav["late_cycle_et_mm_list"].apply(
                                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
                            )
                            if "saved_mm_cap0" not in df_sav.columns:
                                df_sav = add_cap_savings_columns(df_sav, cap_values=(0,), use_acft=False)
                            df_year = make_savings_summary(df_sav, cap_k=0, group_cols=["WY"])
                            result = func(
                                df=df_year,
                                wy_start=2019, wy_end=2024,
                                outfile=output_dir / f"{func_name}.png",
                            )
                        else:
                            print("    Skipping savings by year: no savings CSV found")
                            continue
                    except Exception as e:
                        print(f"    Skipping savings by year: {e}")
                        continue
                elif chart_type == "savings_by_county":
                    try:
                        from es_analysis.data_providers.late_water_provider import (
                            make_savings_summary,
                            add_cap_savings_columns,
                        )
                        import pandas as pd
                        csv_path = Path("water_saving_scenarios_stats/late_cut_savings_parcel_year_mm.csv")
                        if not csv_path.exists():
                            from es_analysis.data_providers.config import config
                            csv_path = config.water_saving_out_dir / "late_cut_savings_parcel_year_mm.csv"
                        if csv_path.exists():
                            df_sav = pd.read_csv(csv_path)
                            import ast
                            df_sav["late_cycle_et_mm_list"] = df_sav["late_cycle_et_mm_list"].apply(
                                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
                            )
                            if "saved_mm_cap0" not in df_sav.columns:
                                df_sav = add_cap_savings_columns(df_sav, cap_values=(0,), use_acft=False)
                            df_cy = make_savings_summary(df_sav, cap_k=0, group_cols=["county", "WY"])
                            result = func(
                                df=df_cy,
                                wy_start=2019, wy_end=2024,
                                outfile=output_dir / f"{func_name}.png",
                            )
                        else:
                            print("    Skipping savings by county: no savings CSV found")
                            continue
                    except Exception as e:
                        print(f"    Skipping savings by county: {e}")
                        continue
                elif chart_type == "debug_parcel":
                    # Debug plot requires specific parcel data; skip in batch mode
                    print("    Skipping debug parcel plot (requires specific parcel data)")
                    continue
                elif chart_type == "cuttings_scatter_year":
                    if parcel_year_df is None:
                        raise ValueError("parcel_year_df not built")
                    result = func(
                        df=parcel_year_df,
                        wy=water_year,
                        outfile=output_dir / f"{func_name}_WY{water_year}.png",
                    )
                elif chart_type == "cuttings_scatter_agg":
                    if parcel_year_df is None:
                        raise ValueError("parcel_year_df not built")
                    result = func(
                        df=parcel_year_df,
                        outfile=output_dir / f"{func_name}.png",
                    )
                elif chart_type == "wy_type":
                    # WY type charts need late_cut_base data
                    from es_analysis.data_providers.config import config as _cfg
                    _lc_csv = _cfg.water_saving_out_dir / "late_cut_base_parcel_year.csv"
                    if not _lc_csv.exists():
                        print(f"    Skipping: {_lc_csv} not found")
                        continue
                    import pandas as _pd2
                    _df_lc = _pd2.read_csv(_lc_csv)
                    wy_out = output_dir / "WY_types"
                    wy_out.mkdir(parents=True, exist_ok=True)
                    result = func(df=_df_lc, output_dir=wy_out)
                elif chart_type == "triple_heatmap":
                    if parcel_year_df is None:
                        print("    Building parcel-year DataFrame from BEAST CSVs...")
                        parcel_year_df = _build_parcel_year_df()
                        if parcel_year_df.empty:
                            raise ValueError("Could not build parcel-year DataFrame")
                    stats_out = output_dir / "statistics"
                    stats_out.mkdir(parents=True, exist_ok=True)
                    result = func(df=parcel_year_df, out_dir=stats_out)
                elif chart_type == "diagnostic":
                    if parcel_year_df is None:
                        print("    Building parcel-year DataFrame from BEAST CSVs...")
                        parcel_year_df = _build_parcel_year_df()
                        if parcel_year_df.empty:
                            raise ValueError("Could not build parcel-year DataFrame")
                    test_out = output_dir / "test"
                    test_out.mkdir(parents=True, exist_ok=True)
                    result = func(df=parcel_year_df, out_dir=test_out)
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
        description="Run all statistics charts (7 charts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--county", type=str, default=None, help="County name")
    parser.add_argument("--water-year", type=int, default=None, dest="water_year", help="Water year")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir", help="Output directory")

    args = parser.parse_args()
    exit_code = run_all_statistics_plots(
        county=args.county, water_year=args.water_year,
        output_dir=args.output_dir
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
