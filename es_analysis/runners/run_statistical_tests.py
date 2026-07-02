"""Runner: Statistical tests for ET correction analysis.

Usage:
    python -m es_analysis.runners.run_statistical_tests [options]

Runs all Phase 3 statistical tests (Wilcoxon, Cohen's d, Hodges-Lehmann,
Kruskal-Wallis, Dunn's post-hoc CLD, normality, sensitivity, failure
breakdown) on the Phase 2 parcel-year CSV output.  Produces CSV exports
and optional distribution plots in a single invocation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..data_providers.statistical_tests_provider import (
    run_all_statistical_tests,
    load_parcel_year_data,
)
from ..data_providers.spatial_provider import COUNTY_ORDER
from ..utils.units import mm_to_acft_per_acre


# ---------------------------------------------------------------------------
# Default paths (relative to working directory)
# ---------------------------------------------------------------------------

_DEFAULT_PARCEL_YEAR_CSV = (
    "statistics_exports/"
    "offphase_parcel_year_methodA_cc20_ci90_boot500_WY2019-2024.csv"
)
_DEFAULT_FAILURES_CSV = (
    "statistics_exports/"
    "offphase_failures_methodA_cc20_ci90_boot500_WY2019-2024.csv"
)
_DEFAULT_OUT_DIR = "es_analysis/output/statistical_tests"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_correction_distribution(
    df: pd.DataFrame,
    cld_dict: dict,
    out_dir: Path,
    dpi: int = 300,
) -> Path:
    """Histogram of correction magnitudes with a log-scale inset.

    Main panel: linear histogram of delta_annual_mm, 0 to 95th percentile.
    Inset: same data (non-zero) on log10 x-axis.

    Args:
        df: DataFrame with 'delta_annual_mm' column.
        cld_dict: CLD letter mapping (unused here, kept for API consistency).
        out_dir: Output directory.
        dpi: Figure DPI.

    Returns:
        Path to saved PNG.
    """
    d = df["delta_annual_mm"].dropna().values
    n_total = len(d)
    n_zero = int(np.sum(d == 0))
    n_pos = n_total - n_zero
    mean_val = float(np.mean(d))
    median_val = float(np.median(d))
    p95 = float(np.percentile(d, 95))

    fig, ax_main = plt.subplots(figsize=(8, 5))

    # Main histogram: 0 to p95 in 50 bins + overflow bin
    bin_edges = np.linspace(0, p95, 51)
    # Separate zero vs positive for colour distinction
    d_zero = d[d == 0]
    d_pos = d[d > 0]

    # Positive corrections
    ax_main.hist(
        d_pos, bins=bin_edges, color="#4C72B0", edgecolor="white",
        linewidth=0.3, alpha=0.85, label=f"Non-zero (n={n_pos:,})",
    )
    # Zero bar
    if n_zero > 0:
        zero_width = bin_edges[1] - bin_edges[0]
        ax_main.bar(
            0, n_zero, width=zero_width, color="#C0C0C0", edgecolor="white",
            linewidth=0.3, alpha=0.85, label=f"Zero correction (n={n_zero:,})",
        )

    # Overflow annotation
    n_overflow = int(np.sum(d > p95))
    if n_overflow > 0:
        ax_main.annotate(
            f"{n_overflow:,} values > {p95:.0f} mm\n(not shown)",
            xy=(p95 * 0.95, 0), xytext=(p95 * 0.75, ax_main.get_ylim()[1] * 0.7),
            fontsize=8, ha="right",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # Mean and median lines
    ax_main.axvline(mean_val, color="red", linestyle="--", linewidth=1.2,
                    label=f"Mean = {mean_val:.1f} mm")
    ax_main.axvline(median_val, color="black", linestyle="--", linewidth=1.2,
                    label=f"Median = {median_val:.1f} mm")

    ax_main.set_xlabel("ET correction magnitude (mm/year)")
    ax_main.set_ylabel("Number of parcel-years")
    ax_main.set_title("Distribution of annual ET corrections")
    ax_main.legend(fontsize=8, loc="upper right")

    # Inset: log-scale (non-zero only)
    ax_inset = fig.add_axes([0.55, 0.45, 0.33, 0.35])  # [left, bottom, w, h]
    d_log = d_pos[d_pos > 0]
    if len(d_log) > 0:
        log_bins = np.logspace(np.log10(max(0.1, d_log.min())),
                               np.log10(d_log.max()), 40)
        ax_inset.hist(d_log, bins=log_bins, color="#4C72B0", edgecolor="white",
                      linewidth=0.3, alpha=0.7)
        ax_inset.set_xscale("log")
        ax_inset.set_xlabel("mm/year (log scale)", fontsize=7)
        ax_inset.set_ylabel("Count", fontsize=7)
        ax_inset.set_title("Log scale (non-zero)", fontsize=8)
        ax_inset.tick_params(labelsize=6)

    # Skip tight_layout when inset axes are present (causes warning)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.12)
    outpath = Path(out_dir) / "correction_distribution.png"
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {outpath} ({outpath.stat().st_size / 1024:.0f} KB)")
    return outpath


def plot_correction_violin_by_county(
    df: pd.DataFrame,
    cld_dict: dict,
    out_dir: Path,
    dpi: int = 300,
) -> Path:
    """Violin plot of ET correction by county with CLD letters.

    One violin per county, ordered by median correction.  CLD letters
    annotated above each violin.  Sample size in x-axis labels.

    Args:
        df: DataFrame with 'county' and 'delta_annual_mm' columns.
        cld_dict: Dict mapping county name -> CLD letter string.
        out_dir: Output directory.
        dpi: Figure DPI.

    Returns:
        Path to saved PNG.
    """
    # Build per-county data, ordered by median
    county_medians = (
        df.groupby("county")["delta_annual_mm"]
        .median()
        .sort_values(ascending=True)
    )
    ordered_counties = county_medians.index.tolist()

    # Collect data arrays and labels
    data_arrays = []
    labels = []
    for c in ordered_counties:
        vals = df.loc[df["county"] == c, "delta_annual_mm"].dropna().values
        data_arrays.append(vals)
        n = len(vals)
        labels.append(f"{c}\n(n={n:,})")

    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(
        data_arrays,
        positions=range(len(ordered_counties)),
        showmedians=False,
        showextrema=False,
        points=200,
    )

    # Style violins
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_edgecolor("#2B4570")
        pc.set_alpha(0.7)

    # Median dots
    for i, vals in enumerate(data_arrays):
        med = np.median(vals)
        ax.scatter(i, med, color="white", s=30, zorder=5, edgecolors="black",
                   linewidth=0.8)

    # CLD letters above each violin
    for i, c in enumerate(ordered_counties):
        letters = cld_dict.get(c, "?")
        ymax = np.percentile(data_arrays[i], 99)
        ax.text(i, ymax * 1.05 + 10, letters, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333")

    ax.set_xticks(range(len(ordered_counties)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("ET correction magnitude (mm/year)")
    ax.set_title("ET correction by county (Compact Letter Display)")

    # Secondary y-axis: ac-ft/acre
    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(mm_to_acft_per_acre(ymin), mm_to_acft_per_acre(ymax))
    ax2.set_ylabel("ET correction (ac-ft/acre)")

    fig.tight_layout()
    outpath = Path(out_dir) / "correction_violin_by_county.png"
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {outpath} ({outpath.stat().st_size / 1024:.0f} KB)")
    return outpath


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def _export_csv(df_or_series, path: Path, label: str) -> None:
    """Write a DataFrame or Series to CSV and print confirmation."""
    if isinstance(df_or_series, pd.Series):
        df_or_series = df_or_series.to_frame()
    df_or_series.to_csv(path)
    size_kb = path.stat().st_size / 1024
    print(f"  {label:35s} -> {path.name} ({size_kb:.1f} KB)")


def _build_sensitivity_rows(sens: dict) -> list:
    """Extract summary-table rows from sensitivity results."""
    rows = []
    wr = sens["wilcoxon_result"]
    kr = sens["kw_result"]
    for r in [wr, kr]:
        rows.append({
            "test_name": r["test_name"],
            "scope": r["scope"],
            "statistic_name": r["statistic_name"],
            "statistic": r["statistic"],
            "p_value": r["p_value"],
            "effect_size": r["effect_size"],
            "effect_size_name": r["effect_size_name"],
            "effect_size_label": r["effect_size_label"],
            "n": r["n"],
            "notes": r.get("notes", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the statistical tests runner.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase 3 statistical tests and export results.",
    )
    parser.add_argument(
        "--parcel-year-csv", type=str, default=_DEFAULT_PARCEL_YEAR_CSV,
        help=f"Path to parcel-year CSV (default: {_DEFAULT_PARCEL_YEAR_CSV}).",
    )
    parser.add_argument(
        "--failures-csv", type=str, default=_DEFAULT_FAILURES_CSV,
        help=f"Path to failures CSV (default: {_DEFAULT_FAILURES_CSV}).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--min-county-n", type=int, default=30,
        help="Minimum county sample size for KW test (default: 30).",
    )
    parser.add_argument(
        "--skip-hl", action="store_true",
        help="Skip Hodges-Lehmann computation (~30s, ~1GB RAM).",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip distribution plot generation.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for figure exports (default: 300).",
    )

    args = parser.parse_args()

    t0 = time.time()

    parcel_year_csv = Path(args.parcel_year_csv)
    failures_csv = Path(args.failures_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("STATISTICAL TESTS RUNNER")
    print(f"{'='*60}")
    print(f"Input:          {parcel_year_csv}")
    print(f"Failures:       {failures_csv}")
    print(f"Output:         {out_dir}")
    print(f"Min county N:   {args.min_county_n}")
    print(f"Skip HL:        {args.skip_hl}")
    print(f"Skip plots:     {args.skip_plots}")
    print(f"DPI:            {args.dpi}")

    try:
        # ---------------------------------------------------------------
        # Run all tests
        # ---------------------------------------------------------------
        results = run_all_statistical_tests(
            parcel_year_csv=parcel_year_csv,
            failures_csv=failures_csv,
            min_county_n=args.min_county_n,
            skip_hl=args.skip_hl,
        )

        # ---------------------------------------------------------------
        # CSV exports
        # ---------------------------------------------------------------
        print(f"\n{'='*60}")
        print("CSV EXPORTS")
        print(f"{'='*60}")

        # 1. Summary table (all results)
        summary = results["summary_table"]
        # Add notes column if not present
        if "notes" not in summary.columns:
            all_results = results["pooled_results"] + results["per_year_results"]
            notes_map = {
                (r["test_name"], r["scope"]): r.get("notes", "")
                for r in all_results
            }
            summary["notes"] = summary.apply(
                lambda row: notes_map.get(
                    (row["test_name"], row["scope"]), ""
                ),
                axis=1,
            )
        _export_csv(summary, out_dir / "summary_table.csv", "Summary table")

        # 2. Post-hoc p-value matrix
        p_matrix = results["posthoc"]["p_matrix"]
        _export_csv(p_matrix, out_dir / "posthoc_pvalues.csv", "Post-hoc p-values")

        # 3. CLD table
        cld_dict = results["posthoc"]["cld"]
        df = results["df"]
        county_medians = (
            df.groupby("county")["delta_annual_mm"].median()
        )
        county_ns = df.groupby("county")["delta_annual_mm"].count()
        cld_rows = []
        for county in sorted(cld_dict.keys()):
            cld_rows.append({
                "county": county,
                "cld_letters": cld_dict[county],
                "median_delta_mm": county_medians.get(county, np.nan),
                "n": county_ns.get(county, 0),
            })
        cld_df = pd.DataFrame(cld_rows)
        _export_csv(cld_df, out_dir / "cld_table.csv", "CLD table")

        # 4. Failure breakdown
        fb = results["failure_breakdown"]
        _export_csv(fb, out_dir / "failure_breakdown.csv", "Failure breakdown")

        # 5. Per-year results
        per_year_rows = []
        for r in results["per_year_results"]:
            per_year_rows.append({
                "test_name": r["test_name"],
                "scope": r["scope"],
                "statistic_name": r["statistic_name"],
                "statistic": r["statistic"],
                "p_value": r["p_value"],
                "effect_size": r["effect_size"],
                "effect_size_name": r["effect_size_name"],
                "effect_size_label": r["effect_size_label"],
                "n": r["n"],
                "notes": r.get("notes", ""),
            })
        per_year_df = pd.DataFrame(per_year_rows)
        _export_csv(per_year_df, out_dir / "per_year_results.csv", "Per-year results")

        # 6. Sensitivity results
        sens = results["sensitivity"]
        sens_rows = _build_sensitivity_rows(sens)
        # Add metadata row
        sens_meta = {
            "test_name": "Sensitivity metadata",
            "scope": f"delta >= {sens['threshold_mm']} mm",
            "statistic_name": "n_excluded",
            "statistic": sens["n_excluded"],
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_size_name": "pct_excluded",
            "effect_size_label": f"{sens['pct_excluded']:.1f}%",
            "n": sens["n_remaining"],
            "notes": sens["notes"],
        }
        sens_rows.insert(0, sens_meta)
        sens_df = pd.DataFrame(sens_rows)
        _export_csv(sens_df, out_dir / "sensitivity_results.csv", "Sensitivity results")

        # ---------------------------------------------------------------
        # Distribution plots
        # ---------------------------------------------------------------
        if not args.skip_plots:
            print(f"\n{'='*60}")
            print("DISTRIBUTION PLOTS")
            print(f"{'='*60}")

            plot_correction_distribution(df, cld_dict, out_dir, dpi=args.dpi)
            plot_correction_violin_by_county(df, cld_dict, out_dir, dpi=args.dpi)

        # ---------------------------------------------------------------
        # Final summary
        # ---------------------------------------------------------------
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Elapsed:  {elapsed:.1f}s")
        print(f"Output:   {out_dir}")
        n_csv = len(list(out_dir.glob("*.csv")))
        n_png = len(list(out_dir.glob("*.png")))
        print(f"Files:    {n_csv} CSVs, {n_png} PNGs")

        return 0

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
