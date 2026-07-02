"""Versioned run: structured output with comprehensive parquet persistence.

Creates a named run directory, saves all pipeline data as parquet files,
generates ALL plots from saved data into the run directory.

Usage:
    python -m es_analysis.runners.run_alfalfa_run_2
    python -m es_analysis.runners.run_alfalfa_run_2 --plots-only
    python -m es_analysis.runners.run_alfalfa_run_2 --run-name alfalfa_run_3
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.utils.run_output import (
    ensure_run_dirs,
    get_run_root,
    list_saved_data,
    load_dataframe,
    save_dataframe,
)

# Data source paths
_PKG_ROOT = Path(__file__).parent.parent
_REPO_ROOT = _PKG_ROOT.parent
_WATER_SAVING_DIR = _REPO_ROOT / "water_saving_scenarios_stats"
_ET_CORRECTION_DIR = _PKG_ROOT / "output" / "et_correction"
_WY_STATS_DIR = _PKG_ROOT / "output" / "figures" / "WY_stats"
_ANOMALY_CSV = _PKG_ROOT / "output" / "figures" / "test" / "anomaly_parcels.csv"

# Default run name (used by CLI main(); library callers pass run_name explicitly)
_DEFAULT_RUN_NAME = "alfalfa_run_2"


# ──────────────────────────────────────────────────────────────────────
# Helper: CSV → parquet
# ──────────────────────────────────────────────────────────────────────
def _csv_to_parquet(csv_path: Path, name: str, run_name: str) -> bool:
    if not csv_path.exists():
        print(f"    SKIP (not found): {csv_path.name}")
        return False
    df = pd.read_csv(csv_path)
    save_dataframe(df, name, run_name)
    print(f"    {name}.parquet <- {csv_path.name} ({len(df)} rows)")
    return True


def _find_latest_csv(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV found for pattern: {pattern}")
    return Path(matches[-1])


# ──────────────────────────────────────────────────────────────────────
# Step 1: Save all data as parquet
# ──────────────────────────────────────────────────────────────────────
def step_save_all_data(run_name: str, et_mode: str = "actual") -> int:
    """Save all pipeline data as parquet. Returns count of files saved."""
    print("\n=== Step 1: Saving all pipeline data as parquet ===")
    saved = 0

    # 1. multicounty_matched
    print("\n  [Core datasets]")
    try:
        from es_analysis.data_providers.parcel_summary_provider import build_multicounty_matched
        print(f"    Building multicounty_matched (et_mode='{et_mode}')...", end=" ", flush=True)
        df_mc = build_multicounty_matched(et_mode=et_mode)
        save_dataframe(df_mc, "multicounty_matched", run_name)
        print(f"done ({len(df_mc)} rows)")
        saved += 1
    except Exception as exc:
        print(f"FAILED: {exc}")

    # 2-3. late_cut CSVs
    if _csv_to_parquet(_WATER_SAVING_DIR / "late_cut_base_parcel_year.csv", "late_cut_base", run_name):
        saved += 1
    if _csv_to_parquet(_WATER_SAVING_DIR / "late_cut_savings_parcel_year_mm.csv", "late_cut_savings", run_name):
        saved += 1

    # 4-5. ET correction CSVs — prefer run-specific dir, fall back to global
    from es_analysis.utils.run_output import get_run_root
    run_et_dir = get_run_root(run_name) / "et_correction"
    for csv_glob, pq_name in [
        ("offphase_parcel_year_*.csv", "et_stats_parcel_year"),
        ("offphase_county_WY_annual_totals_*.csv", "et_stats_county_wy"),
    ]:
        csv_path = None
        for search_dir in [run_et_dir, _ET_CORRECTION_DIR]:
            try:
                csv_path = _find_latest_csv(str(search_dir / csv_glob))
                break
            except FileNotFoundError:
                continue
        if csv_path and _csv_to_parquet(csv_path, pq_name, run_name):
            saved += 1
        elif csv_path is None:
            print(f"    SKIP: no CSV found for {csv_glob}")

    # 6-8. Water saving summaries
    print("\n  [Water saving summaries]")
    for csv_name, pq_name in [
        ("water_saving_summary_county_WY_by_cap.csv", "savings_summary_county_wy_cap"),
        ("water_saving_summary_county_by_cap.csv", "savings_summary_county_cap"),
        ("water_saving_summary_WY_by_cap.csv", "savings_summary_wy_cap"),
    ]:
        if _csv_to_parquet(_WATER_SAVING_DIR / csv_name, pq_name, run_name):
            saved += 1

    # 9-14. WY type analysis
    print("\n  [WY type analysis]")
    for stem in ["cuttings", "annual_et", "county_et_heatmap", "et_correction", "savings", "late_cut_pct"]:
        if _csv_to_parquet(_WY_STATS_DIR / f"wy_type_{stem}.csv", f"wy_type_{stem}", run_name):
            saved += 1

    # 15. anomaly_parcels
    print("\n  [Anomaly parcels]")
    if _csv_to_parquet(_ANOMALY_CSV, "anomaly_parcels", run_name):
        saved += 1

    print(f"\n  Saved {saved} parquet files to {get_run_root(run_name) / 'data'}")
    return saved


# ──────────────────────────────────────────────────────────────────────
# Step 2: Cuttings Analysis Plots (from parquet — no rebuild)
# ──────────────────────────────────────────────────────────────────────
def step_cuttings_plots(run_name: str, et_mode: str = "actual") -> int:
    """Generate per-county and all-counties cuttings plots from parquet."""
    from es_analysis.charts.statistics.cuttings_matched_plots import (
        scatter_all_years,
        scatter_by_year,
        boxplot_year_colored,
    )
    from es_analysis.data_providers.evi_provider import normalize_county_name
    from es_analysis.data_providers.spatial_provider import COUNTY_ORDER

    print("\n=== Step 2: Cuttings Analysis Plots ===")

    df = load_dataframe("multicounty_matched", run_name)
    if df is None or df.empty:
        print("  SKIP: multicounty_matched.parquet not found or empty")
        return 0

    root = get_run_root(run_name)
    base_out = root / "cuttings_analysis"
    n_plots = 0

    # Per-county
    for county in COUNTY_ORDER:
        county_norm = normalize_county_name(county)
        sub = df[df["county"] == county_norm]
        if sub.empty:
            print(f"  {county_norm}: no data, skipping")
            continue

        county_dir = county_norm.replace(" ", "_")
        out_dir = base_out / county_dir

        print(f"  {county_norm}: {len(sub)} rows", end=" ... ", flush=True)
        try:
            scatter_all_years(sub, county_norm, out_dir, et_mode=et_mode)
            scatter_by_year(sub, county_norm, out_dir, et_mode=et_mode)
            boxplot_year_colored(sub, county_norm, out_dir, et_mode=et_mode)
            n_plots += 3
            print("done (3 plots)")
        except Exception as exc:
            print(f"FAILED: {exc}")

    # All-counties aggregate
    out_all = base_out / "all_counties"
    print(f"  all_counties: {len(df)} rows", end=" ... ", flush=True)
    try:
        scatter_all_years(df, "All Counties", out_all, et_mode=et_mode)
        scatter_by_year(df, "All Counties", out_all, et_mode=et_mode)
        boxplot_year_colored(df, "All Counties", out_all, et_mode=et_mode)
        n_plots += 3
        print("done (3 plots)")
    except Exception as exc:
        print(f"FAILED: {exc}")

    print(f"  Total cuttings plots: {n_plots}")
    return n_plots


# ──────────────────────────────────────────────────────────────────────
# Step 3: Water Savings Plots
# ──────────────────────────────────────────────────────────────────────
def step_water_savings_plots(run_name: str, et_mode: str = "actual") -> int:
    """Generate water savings charts (load from CSV, redirect output)."""
    from es_analysis.runners.run_water_savings_plots import run_aggregate_plots

    print("\n=== Step 3: Water Savings Plots ===")
    root = get_run_root(run_name)
    out_dir = root / "water_savings"
    try:
        run_aggregate_plots(out_dir, cap_values=(0, 1, 2, 3), et_mode=et_mode)
        n = len(list(out_dir.rglob("*.png")))
        print(f"  Generated {n} water savings PNGs")
        return n
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 0


# ──────────────────────────────────────────────────────────────────────
# Step 4: WY Type Analysis Plots
# ──────────────────────────────────────────────────────────────────────
def step_wy_type_plots(run_name: str, class_mode: str = "sji") -> int:
    """Generate WY type analysis charts."""
    from es_analysis.runners.run_wy_type_analysis import run_wy_type_analysis

    print("\n=== Step 4: WY Type Analysis Plots ===")
    root = get_run_root(run_name)
    out_dir = root / "WY_types"
    stats_dir = root / "WY_stats"
    try:
        run_wy_type_analysis(
            output_dir=out_dir,
            stats_dir=stats_dir,
            class_mode=class_mode,
        )
        n = len(list(out_dir.rglob("*.png")))
        print(f"  Generated {n} WY type PNGs")
        return n
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 0


# ──────────────────────────────────────────────────────────────────────
# Step 5: ET Correction Summary Plot
# ──────────────────────────────────────────────────────────────────────
def step_et_correction_summary(run_name: str) -> int:
    """Generate the aggregate ET correction summary plot."""
    from es_analysis.charts.et_corrections.et_correction_summary_plot import (
        et_correction_summary,
    )

    print("\n=== Step 5: ET Correction Summary ===")
    root = get_run_root(run_name)
    out_dir = root / "et_corrections"
    try:
        _, _, summary = et_correction_summary(out_dir=out_dir)
        print(f"  {summary.get('n_counties', '?')} counties, "
              f"mean reduction: {summary.get('mean_pct_reduction_county', '?')}")
        return 1
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 0


# ──────────────────────────────────────────────────────────────────────
# Step 6: Dual Heatmap
# ──────────────────────────────────────────────────────────────────────
def step_dual_heatmap(run_name: str) -> int:
    """Generate the dual heatmap and save pivot tables."""
    from es_analysis.charts.statistics.county_wy_et_heatmap_plot import (
        county_wy_et_dual_heatmap,
        county_wy_et_triple_heatmap,
    )

    print("\n=== Step 6: Heatmaps (dual + triple) ===")

    df = load_dataframe("multicounty_matched", run_name)
    if df is None:
        print("  SKIP: multicounty_matched.parquet not found")
        return 0

    root = get_run_root(run_name)
    out_dir = root / "statistics"
    n = 0

    # Dual heatmap (new)
    try:
        _, _, summary = county_wy_et_dual_heatmap(df, out_dir=out_dir)
        if "pivot_annual" in summary:
            save_dataframe(summary["pivot_annual"].reset_index(), "heatmap_pivot_annual", run_name)
        if "pivot_segment" in summary:
            save_dataframe(summary["pivot_segment"].reset_index(), "heatmap_pivot_segment", run_name)
        print(f"  Dual heatmap: annual={summary['mean_annual_et_mm']:.0f} mm, "
              f"segment={summary['mean_segment_et_mm']:.0f} mm")
        n += 1
    except Exception as exc:
        print(f"  Dual heatmap FAILED: {exc}")

    # Triple heatmap (original, for reference)
    try:
        county_wy_et_triple_heatmap(df, out_dir=out_dir)
        print("  Triple heatmap: done")
        n += 1
    except Exception as exc:
        print(f"  Triple heatmap FAILED: {exc}")

    return n


# ──────────────────────────────────────────────────────────────────────
# Step 7: Anomaly EVI Plots
# ──────────────────────────────────────────────────────────────────────
def step_anomaly_evi(run_name: str) -> int:
    """Generate anomaly EVI plots."""
    from es_analysis.charts.evi.evi_anomaly_plot import generate_anomaly_evi_plots

    print("\n=== Step 7: Anomaly EVI Plots ===")
    root = get_run_root(run_name)
    out_dir = root / "test"
    result = generate_anomaly_evi_plots(anomaly_csv=_ANOMALY_CSV, out_dir=out_dir)
    return result.get("n_success", 0)


# ──────────────────────────────────────────────────────────────────────
# Step 8: Diagnostic Anomaly Plots (ET per-parcel)
# ──────────────────────────────────────────────────────────────────────
def step_diagnostic_plots(run_name: str) -> int:
    """Generate diagnostic anomaly detection plots from parquet."""
    print("\n=== Step 8: Diagnostic Anomaly Plots ===")

    df = load_dataframe("multicounty_matched", run_name)
    if df is None or df.empty:
        print("  SKIP: multicounty_matched.parquet not found")
        return 0

    root = get_run_root(run_name)
    out_dir = root / "test"
    try:
        from es_analysis.charts.statistics.diagnostic_anomaly_plots import generate_diagnostic_plots
        result = generate_diagnostic_plots(df=df, out_dir=out_dir)
        n = result.get("n_plots_generated", 0)
        print(f"  Generated {n} diagnostic plots")
        return n
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 0


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
def step_print_summary(run_name: str) -> None:
    """Print manifest of all saved data and generated plots."""
    print("\n=== Summary ===")
    root = get_run_root(run_name)

    parquets = list_saved_data(run_name)
    total_size = sum(
        (root / "data" / f"{name}.parquet").stat().st_size
        for name in parquets
        if (root / "data" / f"{name}.parquet").exists()
    )
    print(f"\n  Parquet files: {len(parquets)} ({total_size / 1024 / 1024:.1f} MB)")
    for name in parquets:
        p = root / "data" / f"{name}.parquet"
        if p.exists():
            kb = p.stat().st_size / 1024
            print(f"    {name}.parquet  ({kb:.0f} KB)")

    # Count plots per subdirectory
    print(f"\n  Plot files by directory:")
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name != "data":
            pngs = list(subdir.rglob("*.png"))
            pdfs = list(subdir.rglob("*.pdf"))
            if pngs or pdfs:
                print(f"    {subdir.name}/: {len(pngs)} PNG, {len(pdfs)} PDF")

    total_png = len(list(root.rglob("*.png")))
    total_pdf = len(list(root.rglob("*.pdf")))
    print(f"\n  Total: {total_png} PNG, {total_pdf} PDF")
    print(f"\n  All output in: {root}")


# ──────────────────────────────────────────────────────────────────────
# Library entry point (called by pipeline and CLI)
# ──────────────────────────────────────────────────────────────────────
def run_versioned(
    run_name: str = _DEFAULT_RUN_NAME,
    plots_only: bool = False,
    et_mode: str = "actual",
    class_mode: str = "sji",
) -> dict:
    """Run the full versioned output pipeline.

    Creates ``output/figures/{run_name}/`` with all data + all plots.
    Returns a dict of plot counts per step.
    """
    t0 = time.time()
    print("=" * 60)
    print(f"  Versioned Run: {run_name}")
    print("=" * 60)

    root = ensure_run_dirs(run_name)
    print(f"\n  Output root: {root}")
    print(f"  ET mode: {et_mode}, WY class mode: {class_mode}")

    # Step 1: Save data (unless plots_only)
    if not plots_only:
        step_save_all_data(run_name, et_mode=et_mode)
    else:
        parquets = list_saved_data(run_name)
        print(f"\n  --plots-only: using {len(parquets)} existing parquet files")

    # Steps 2-8: Generate plots
    counts = {}
    counts["cuttings"] = step_cuttings_plots(run_name, et_mode=et_mode)
    counts["water_savings"] = step_water_savings_plots(run_name, et_mode=et_mode)
    counts["wy_type"] = step_wy_type_plots(run_name, class_mode=class_mode)
    counts["et_correction"] = step_et_correction_summary(run_name)
    counts["heatmaps"] = step_dual_heatmap(run_name)
    counts["anomaly_evi"] = step_anomaly_evi(run_name)
    counts["diagnostic"] = step_diagnostic_plots(run_name)

    # Summary
    step_print_summary(run_name)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    total_plots = sum(counts.values())
    print(f"  Plot counts: {counts}")
    print(f"  Total plots generated: {total_plots}")

    return counts


# ──────────────────────────────────────────────────────────────────────
# CLI Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Versioned Run Master Runner")
    parser.add_argument("--run-name", default=_DEFAULT_RUN_NAME,
                        help=f"Run directory name (default: {_DEFAULT_RUN_NAME})")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip data save step, regenerate plots from existing parquet")
    parser.add_argument("--et-mode", default="actual",
                        choices=["actual", "corrected", "both"],
                        help="ET mode for data and plots (default: actual)")
    parser.add_argument("--class-mode", default="sji",
                        choices=["sji", "usdm", "mixed"],
                        help="WY type classification mode (default: sji)")
    args = parser.parse_args()

    run_versioned(
        run_name=args.run_name,
        plots_only=args.plots_only,
        et_mode=args.et_mode,
        class_mode=args.class_mode,
    )


if __name__ == "__main__":
    main()
