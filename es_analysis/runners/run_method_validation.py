"""Runner: Method validation (Phase 4).

Usage:
    python -m es_analysis.runners.run_method_validation [options]

Runs the Method A vs B comparison (ET-correction method selection evidence).
(The earlier OAT sensitivity sweep and economic-breakeven analyses were removed
with the simulation layer.)

All outputs saved to es_analysis/output/method_validation/.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default paths (relative to working directory)
# ---------------------------------------------------------------------------

_DEFAULT_METHOD_A_CSV = (
    "statistics_exports/"
    "offphase_parcel_year_methodA_cc20_ci90_boot500_WY2019-2024.csv"
)
_DEFAULT_OUT_DIR = "es_analysis/output/method_validation"


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------

def _export_csv(df_or_dict, path: Path, label: str) -> None:
    """Write a DataFrame, Series, or dict to CSV and print confirmation."""
    if isinstance(df_or_dict, dict):
        df_or_dict = pd.DataFrame([df_or_dict])
    if isinstance(df_or_dict, pd.Series):
        df_or_dict = df_or_dict.to_frame()
    df_or_dict.to_csv(path, index=False)
    size_kb = path.stat().st_size / 1024
    print(f"  {label:35s} -> {path.name} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the method validation runner.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase 4 method validation and robustness analyses.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--max-uids", type=int, default=None,
        help="Max UIDs per county-year for pipeline runs (default: all).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=100,
        help="Bootstrap samples for Method B comparison run (default: 100).",
    )
    parser.add_argument(
        "--n-boot-sensitivity", type=int, default=10,
        help="Bootstrap samples for sensitivity runs (default: 10).",
    )
    parser.add_argument(
        "--method-a-csv", type=str, default=_DEFAULT_METHOD_A_CSV,
        help=f"Path to Method A parcel-year CSV (default: {_DEFAULT_METHOD_A_CSV}).",
    )
    parser.add_argument(
        "--method-b-csv", type=str, default=None,
        help="Path to pre-computed Method B CSV (skip running pipeline).",
    )
    parser.add_argument(
        "--late-water-csv", type=str, default=None,
        help="Path to late-water savings CSV for breakeven analysis.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers for sensitivity runs (default: CPU count).",
    )
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip Method A vs B comparison.",
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="Skip sensitivity analysis.",
    )
    parser.add_argument(
        "--skip-breakeven", action="store_true",
        help="Skip breakeven analysis.",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip all plot generation.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for figure exports (default: 300).",
    )

    args = parser.parse_args()

    t0 = time.time()

    out_dir = Path(args.out_dir)
    comparison_dir = out_dir / "method_comparison"
    sensitivity_dir = out_dir / "sensitivity"
    breakeven_dir = out_dir / "breakeven"

    for d in [out_dir, comparison_dir, sensitivity_dir, breakeven_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("METHOD VALIDATION RUNNER (Phase 4)")
    print(f"{'='*60}")
    print(f"Output:          {out_dir}")
    print(f"Max UIDs:        {args.max_uids or 'all'}")
    print(f"N-boot (comp):   {args.n_boot}")
    print(f"N-boot (sens):   {args.n_boot_sensitivity}")
    print(f"Skip comparison: {args.skip_comparison}")
    print(f"Skip sensitivity:{args.skip_sensitivity}")
    print(f"Skip breakeven:  {args.skip_breakeven}")
    workers = args.workers or os.cpu_count() or 4
    print(f"Workers:         {workers}")
    print(f"Skip plots:      {args.skip_plots}")
    print(f"DPI:             {args.dpi}")

    try:
        # ---------------------------------------------------------------
        # 1. Method A vs B comparison
        # ---------------------------------------------------------------
        if not args.skip_comparison:
            print(f"\n{'='*60}")
            print("SECTION 1: METHOD A vs B COMPARISON")
            print(f"{'='*60}")

            from ..data_providers.method_comparison_provider import (
                run_method_comparison,
            )

            method_a_csv = Path(args.method_a_csv)
            method_b_csv = Path(args.method_b_csv) if args.method_b_csv else None
            run_method_b = method_b_csv is None

            comp_result = run_method_comparison(
                method_a_csv=method_a_csv,
                method_b_csv=method_b_csv,
                run_method_b=run_method_b,
                n_boot=args.n_boot,
                max_uids_per_county_wy=args.max_uids,
                out_dir=comparison_dir,
            )

            evidence = comp_result["evidence"]
            merged_df = comp_result["merged_df"]

            # Export CSVs (provider already saves paired CSV + plot,
            # but we export the summary dict too)
            _export_csv(
                evidence,
                comparison_dir / "method_comparison_summary.csv",
                "Method comparison summary",
            )

            print(f"\n  Evidence summary:")
            print(f"    N matched:         {evidence['n_matched']:,}")
            print(f"    Median A:          {evidence['method_a_median']:.2f} mm")
            print(f"    Median B:          {evidence['method_b_median']:.2f} mm")
            print(f"    Paired diff median: {evidence['paired_diff_median']:.2f} mm")
            if np.isfinite(evidence["wilcoxon_p"]):
                print(f"    Wilcoxon p:        {evidence['wilcoxon_p']:.2e}")
        else:
            print(f"\n  [skipped] Method A vs B comparison")

        # ---------------------------------------------------------------
        # (Sensitivity and economic-breakeven analyses removed with the
        #  simulation layer; this runner now performs only the Method A/B
        #  ET-correction comparison.)

        # ---------------------------------------------------------------
        # Final summary
        # ---------------------------------------------------------------
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Elapsed:  {elapsed:.1f}s")
        print(f"Output:   {out_dir}")

        for label, subdir in [
            ("Comparison", comparison_dir),
        ]:
            n_csv = len(list(subdir.glob("*.csv")))
            n_png = len(list(subdir.glob("*.png")))
            print(f"  {label:15s}: {n_csv} CSVs, {n_png} PNGs")

        return 0

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
