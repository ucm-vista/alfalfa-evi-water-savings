#!/usr/bin/env python3
"""Water Year Type analysis runner.

Generates charts and summary statistics stratified by DWR Water Year
Type (SJV index) with optional USDM drought annotation.

Usage:
    python -m es_analysis.runners.run_wy_type_analysis [OPTIONS]

Examples:
    # All charts, all counties
    python -m es_analysis.runners.run_wy_type_analysis

    # SJ Valley only, specific charts
    python -m es_analysis.runners.run_wy_type_analysis --sj-valley-only --charts cuttings,heatmap

    # Mixed mode (SJV for 8 + USDM for Imperial/Riverside)
    python -m es_analysis.runners.run_wy_type_analysis --class-mode mixed
"""

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from es_analysis.data_providers.config import config
from es_analysis.data_providers.statistics_provider import COUNTY_ORDER
from es_analysis.data_providers.wy_type_provider import (
    SJ_VALLEY_COUNTIES,
    COLORADO_RIVER_COUNTIES,
    add_wy_type_columns,
    add_usdm_columns,
)


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def _load_late_cut_base(counties=None, wy_start=2019, wy_end=2024, data_dir=None):
    """Load late_cut_base_parcel_year.csv.

    Args:
        data_dir: Optional override directory. Falls back to
            config.water_saving_out_dir when None.
    """
    base = Path(data_dir) if data_dir is not None else config.water_saving_out_dir
    csv_path = base / "late_cut_base_parcel_year.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df[df["WY"].between(wy_start, wy_end)]
    if "n_cuttings" in df.columns:
        df = df[df["n_cuttings"] >= config.min_cuttings]
    if counties:
        df = df[df["county"].isin(counties)]
    return df


def _safe_parse_list(x):
    """Parse a string list that may contain nan values."""
    if not isinstance(x, str):
        return []
    try:
        # Replace nan with None for ast.literal_eval
        cleaned = x.replace("nan", "None")
        result = ast.literal_eval(cleaned)
        return [v if v is not None else float("nan") for v in result]
    except (ValueError, SyntaxError):
        return []


def _load_savings(counties=None, wy_start=2019, wy_end=2024, data_dir=None):
    """Load late_cut_savings_parcel_year_mm.csv.

    Args:
        data_dir: Optional override directory. Falls back to
            config.water_saving_out_dir when None.
    """
    base = Path(data_dir) if data_dir is not None else config.water_saving_out_dir
    csv_path = base / "late_cut_savings_parcel_year_mm.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Parse list columns safely
    for col in ["late_cycle_et_mm_list", "late_cycle_et_corrected_mm_list"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)
    df = df[df["WY"].between(wy_start, wy_end)]
    if counties:
        df = df[df["county"].isin(counties)]
    return df


def _load_et_stats(counties=None, wy_start=2019, wy_end=2024, et_dir=None):
    """Load offphase ET correction parcel-year CSV.

    Args:
        et_dir: Optional override directory for ET correction CSVs.
            Falls back to es_analysis/output/et_correction when None.
    """
    search_dir = Path(et_dir) if et_dir is not None else Path("es_analysis/output/et_correction")
    pattern = "offphase_parcel_year_methodA_*boot500*.csv"
    candidates = sorted(search_dir.glob(pattern))
    if not candidates:
        print(f"  WARNING: no ET stats CSV found in {search_dir}")
        return pd.DataFrame()
    csv_path = candidates[-1]  # most recent
    df = pd.read_csv(csv_path)
    df = df[df["WY"].between(wy_start, wy_end)]
    if counties:
        df = df[df["county"].isin(counties)]
    return df


# -----------------------------------------------------------------------
# Chart name → function mapping
# -----------------------------------------------------------------------
CHART_NAMES = [
    "cuttings", "annual_et", "heatmap", "correction",
    "savings", "late_cut", "multipanel",
]


def run_wy_type_analysis(
    counties=None,
    wy_start=2019,
    wy_end=2024,
    class_mode="sji",
    sj_valley_only=False,
    output_dir=None,
    stats_dir=None,
    charts=None,
    data_dir=None,
    et_dir=None,
):
    """Run WY type analysis charts and export stats CSVs.

    Args:
        counties: List of county names (default: all 10 or 8 if sj_valley_only).
        wy_start: Start water year.
        wy_end: End water year.
        class_mode: "sji" | "usdm" | "mixed".
        sj_valley_only: Exclude Imperial and Riverside.
        output_dir: Figures output path.
        stats_dir: Stats CSV output path.
        charts: List of chart names to generate (None = all).
        data_dir: Optional directory to read water-savings CSVs from.
            Falls back to config.water_saving_out_dir when None.
        et_dir: Optional directory to read ET correction CSVs from.
            Falls back to es_analysis/output/et_correction when None.
    """
    from es_analysis.charts.statistics.wy_type_analysis_plot import (
        wy_type_cuttings_bar,
        wy_type_annual_et_bar,
        wy_type_county_et_heatmap,
        wy_type_et_correction_bar,
        wy_type_savings_bar,
        wy_type_late_cut_pct_bar,
        wy_type_summary_multipanel,
    )

    # Resolve counties
    if counties is None:
        if sj_valley_only:
            counties = list(SJ_VALLEY_COUNTIES)
        else:
            counties = list(COUNTY_ORDER)
    elif sj_valley_only:
        counties = [c for c in counties if c not in COLORADO_RIVER_COUNTIES]

    # Resolve output dirs
    if output_dir is None:
        output_dir = Path("es_analysis/output/figures/WY_types")
    else:
        output_dir = Path(output_dir)
    if stats_dir is None:
        stats_dir = output_dir.parent / "WY_stats"
    else:
        stats_dir = Path(stats_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if charts is None:
        charts = list(CHART_NAMES)

    print("=" * 60)
    print("WATER YEAR TYPE ANALYSIS")
    print("=" * 60)
    print(f"  Counties:     {', '.join(counties)} ({len(counties)})")
    print(f"  WY range:     {wy_start}-{wy_end}")
    print(f"  Class mode:   {class_mode}")
    print(f"  Charts:       {', '.join(charts)}")
    print(f"  Figures dir:  {output_dir}")
    print(f"  Stats dir:    {stats_dir}")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    df_base = _load_late_cut_base(counties, wy_start, wy_end, data_dir=data_dir)
    df_sav = _load_savings(counties, wy_start, wy_end, data_dir=data_dir)
    df_et = _load_et_stats(counties, wy_start, wy_end, et_dir=et_dir)

    print(f"  Late-cut base: {len(df_base)} rows")
    print(f"  Savings:       {len(df_sav)} rows")
    print(f"  ET stats:      {len(df_et)} rows")

    if df_base.empty:
        print("\n  ERROR: No late-cut base data. Run late_water workflow first.")
        return 1

    # Apply classification columns
    if class_mode in ("sji", "mixed"):
        add_wy_type_columns(df_base)
        if not df_sav.empty:
            add_wy_type_columns(df_sav)
        if not df_et.empty:
            add_wy_type_columns(df_et)
    if class_mode in ("usdm", "mixed"):
        add_usdm_columns(df_base)
        if not df_sav.empty:
            add_usdm_columns(df_sav)
        if not df_et.empty:
            add_usdm_columns(df_et)

    # Determine ET column (prefer total_et_mm, fallback to late+non-late)
    et_col = "total_et_mm"
    if et_col not in df_base.columns:
        if "ET_open_annual_mm" in df_base.columns:
            et_col = "ET_open_annual_mm"
        else:
            # Compute from late + residual
            et_col = "total_et_mm"
            if "late_et_mm" in df_base.columns:
                df_base[et_col] = df_base.get("total_et_mm", df_base.get("late_et_mm", 0))

    # Run selected charts
    success = 0
    failed = []
    total = len(charts)

    for i, chart_name in enumerate(charts, 1):
        print(f"\n[{i}/{total}] {chart_name}")
        try:
            if chart_name == "cuttings":
                wy_type_cuttings_bar(df_base, output_dir=output_dir)
            elif chart_name == "annual_et":
                if et_col in df_base.columns:
                    wy_type_annual_et_bar(df_base, output_dir=output_dir, et_col=et_col)
                else:
                    print(f"  Skipping: no ET column ({et_col}) in base data")
                    continue
            elif chart_name == "heatmap":
                if et_col in df_base.columns:
                    wy_type_county_et_heatmap(df_base, output_dir=output_dir, et_col=et_col)
                else:
                    print(f"  Skipping: no ET column ({et_col}) in base data")
                    continue
            elif chart_name == "correction":
                if df_et.empty:
                    print("  Skipping: no ET stats data")
                    continue
                wy_type_et_correction_bar(df_et, output_dir=output_dir)
            elif chart_name == "savings":
                if df_sav.empty:
                    print("  Skipping: no savings data")
                    continue
                wy_type_savings_bar(df_sav, output_dir=output_dir)
            elif chart_name == "late_cut":
                wy_type_late_cut_pct_bar(df_base, output_dir=output_dir)
            elif chart_name == "multipanel":
                if df_et.empty or df_sav.empty:
                    print("  Skipping multipanel: missing ET or savings data")
                    continue
                wy_type_summary_multipanel(
                    df_base, df_et, df_sav,
                    output_dir=output_dir, et_col=et_col,
                )
            else:
                print(f"  Unknown chart: {chart_name}")
                continue

            print(f"  Success")
            success += 1

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append((chart_name, str(e)))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {success}/{total} charts generated")
    if failed:
        print("Failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"{'=' * 60}")

    return 1 if failed else 0


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Water Year Type analysis: charts and statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--counties", nargs="+", default=None,
        help="Counties to include (default: all 10)",
    )
    parser.add_argument("--wy-start", type=int, default=2019, help="Start WY")
    parser.add_argument("--wy-end", type=int, default=2024, help="End WY")
    parser.add_argument(
        "--class-mode", choices=["sji", "usdm", "mixed"], default="sji",
        help="Classification mode: sji (SJV index), usdm, or mixed",
    )
    parser.add_argument(
        "--sj-valley-only", action="store_true",
        help="Exclude Imperial and Riverside counties",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Figures output dir")
    parser.add_argument("--stats-dir", type=str, default=None, help="Stats CSV output dir")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory to read water-savings CSVs from (overrides config default)")
    parser.add_argument("--et-dir", type=str, default=None,
                        help="Directory to read ET correction CSVs from (overrides default)")
    parser.add_argument(
        "--charts", type=str, default=None,
        help=f"Comma-separated chart names: {','.join(CHART_NAMES)}",
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Run all 7 charts (default if no --charts specified)",
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Named run (e.g. alfalfa_run_6). Auto-resolves --data-dir, --et-dir, --output-dir, --stats-dir from saved run data.")

    args = parser.parse_args()

    # --run-name sets defaults for data/output paths
    if args.run_name:
        from es_analysis.utils.run_output import get_run_root
        run_root = get_run_root(args.run_name)
        if not args.data_dir:
            args.data_dir = str(run_root / "water_savings")
        if not args.et_dir:
            args.et_dir = str(run_root / "et_correction")
        if not args.output_dir:
            args.output_dir = str(run_root / "WY_types")
        if not args.stats_dir:
            args.stats_dir = str(run_root / "WY_stats")

    # Normalize county names
    counties = args.counties
    if counties:
        from es_analysis.data_providers.evi_provider import normalize_county_name
        counties = [normalize_county_name(c) for c in counties]

    chart_list = None
    if args.charts:
        chart_list = [c.strip() for c in args.charts.split(",")]

    exit_code = run_wy_type_analysis(
        counties=counties,
        wy_start=args.wy_start,
        wy_end=args.wy_end,
        class_mode=args.class_mode,
        sj_valley_only=args.sj_valley_only,
        output_dir=args.output_dir,
        stats_dir=args.stats_dir,
        charts=chart_list,
        data_dir=args.data_dir,
        et_dir=args.et_dir,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
