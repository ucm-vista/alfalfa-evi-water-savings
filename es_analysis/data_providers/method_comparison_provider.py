"""Method A vs Method B comparison provider.

Loads paired parcel-year ET correction results from Methods A and B,
computes comparison statistics (median, mean, Wilcoxon signed-rank),
and produces a violin plot showing side-by-side correction distributions.

Exports:
    run_method_comparison  -- main entry point
    plot_method_comparison -- standalone plotting function
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import config
from ..utils.units import mm_to_acft_per_acre


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MERGE_KEYS = ["county", "WY", "UniqueID"]

_REQUIRED_COLUMNS = [
    "county", "WY", "UniqueID",
    "ET_open_annual_mm", "ET_corr_annual_mm", "delta_annual_mm",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_method_csv(csv_path: Path, label: str) -> pd.DataFrame:
    """Load and validate a parcel-year CSV for one method.

    Args:
        csv_path: Path to the parcel-year CSV.
        label: Human-readable label (e.g. "Method A").

    Returns:
        DataFrame with NaN delta rows removed.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} CSV missing required columns: {sorted(missing)}"
        )

    n_before = len(df)
    df = df.dropna(subset=["delta_annual_mm"]).copy()
    n_dropped = n_before - len(df)

    print(f"  {label}: loaded {n_before:,} rows, dropped {n_dropped:,} NaN "
          f"delta rows, {len(df):,} valid")
    return df


# ---------------------------------------------------------------------------
# Paired merge
# ---------------------------------------------------------------------------

def _paired_merge(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> tuple:
    """Merge Method A and B on (county, WY, UniqueID) with indicator.

    Args:
        df_a: Method A parcel-year DataFrame.
        df_b: Method B parcel-year DataFrame.

    Returns:
        Tuple of (merged_both_df, n_matched, n_a_only, n_b_only).
    """
    # Coerce merge keys to consistent types (UniqueID may be int64
    # from CSV but object from in-memory pipeline DataFrames)
    for key in _MERGE_KEYS:
        if df_a[key].dtype != df_b[key].dtype:
            df_a = df_a.copy()
            df_b = df_b.copy()
            df_a[key] = df_a[key].astype(str)
            df_b[key] = df_b[key].astype(str)

    merged = df_a.merge(
        df_b,
        on=_MERGE_KEYS,
        how="outer",
        suffixes=("_A", "_B"),
        indicator=True,
    )
    n_matched = int((merged["_merge"] == "both").sum())
    n_a_only = int((merged["_merge"] == "left_only").sum())
    n_b_only = int((merged["_merge"] == "right_only").sum())

    print(f"\n  Paired merge results:")
    print(f"    Matched (both):   {n_matched:,}")
    print(f"    A-only:           {n_a_only:,}")
    print(f"    B-only:           {n_b_only:,}")

    # Keep only matched rows for comparison
    matched = merged[merged["_merge"] == "both"].copy()
    matched = matched.drop(columns=["_merge"])

    return matched, n_matched, n_a_only, n_b_only


# ---------------------------------------------------------------------------
# Comparison statistics
# ---------------------------------------------------------------------------

def _compute_comparison_stats(
    matched: pd.DataFrame,
    n_matched: int,
    n_a_only: int,
    n_b_only: int,
) -> dict:
    """Compute comparison statistics on matched pairs.

    Args:
        matched: DataFrame with _A and _B suffixed columns.
        n_matched: Count of matched parcel-years.
        n_a_only: Count of A-only parcel-years.
        n_b_only: Count of B-only parcel-years.

    Returns:
        Evidence summary dict.
    """
    delta_a = matched["delta_annual_mm_A"].values
    delta_b = matched["delta_annual_mm_B"].values
    paired_diff = delta_a - delta_b

    # Medians and means
    method_a_median = float(np.nanmedian(delta_a))
    method_a_mean = float(np.nanmean(delta_a))
    method_b_median = float(np.nanmedian(delta_b))
    method_b_mean = float(np.nanmean(delta_b))

    # Paired Wilcoxon signed-rank test
    # Remove pairs where difference is exactly zero
    nonzero_mask = paired_diff != 0
    n_nonzero = int(nonzero_mask.sum())

    if n_nonzero >= 10:
        wil_result = wilcoxon(
            paired_diff[nonzero_mask],
            zero_method="wilcox",
            alternative="two-sided",
        )
        wilcoxon_stat = float(wil_result.statistic)
        wilcoxon_p = float(wil_result.pvalue)
    else:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan
        print(f"  WARNING: Only {n_nonzero} non-zero differences; "
              f"Wilcoxon test skipped")

    # Paired difference stats
    paired_diff_median = float(np.nanmedian(paired_diff))
    paired_diff_mean = float(np.nanmean(paired_diff))

    # CI width comparison
    ci_width_mean_a = np.nan
    ci_width_mean_b = np.nan
    if "annual_ci_width_mm_A" in matched.columns:
        ci_width_mean_a = float(
            matched["annual_ci_width_mm_A"].dropna().mean()
        )
    if "annual_ci_width_mm_B" in matched.columns:
        ci_width_mean_b = float(
            matched["annual_ci_width_mm_B"].dropna().mean()
        )

    # Annual ET comparison
    annual_et_median_a = np.nan
    annual_et_median_b = np.nan
    if "ET_corr_annual_mm_A" in matched.columns:
        annual_et_median_a = float(
            matched["ET_corr_annual_mm_A"].dropna().median()
        )
    if "ET_corr_annual_mm_B" in matched.columns:
        annual_et_median_b = float(
            matched["ET_corr_annual_mm_B"].dropna().median()
        )

    evidence = {
        "method_a_median": method_a_median,
        "method_a_mean": method_a_mean,
        "method_b_median": method_b_median,
        "method_b_mean": method_b_mean,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "paired_diff_median": paired_diff_median,
        "paired_diff_mean": paired_diff_mean,
        "n_matched": n_matched,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_nonzero_pairs": n_nonzero,
        "ci_width_mean_a": ci_width_mean_a,
        "ci_width_mean_b": ci_width_mean_b,
        "annual_et_median_a": annual_et_median_a,
        "annual_et_median_b": annual_et_median_b,
    }

    # Print summary
    print(f"\n  Comparison statistics (matched N={n_matched:,}):")
    print(f"    Method A median delta: {method_a_median:.2f} mm "
          f"({mm_to_acft_per_acre(method_a_median):.4f} ac-ft/acre)")
    print(f"    Method B median delta: {method_b_median:.2f} mm "
          f"({mm_to_acft_per_acre(method_b_median):.4f} ac-ft/acre)")
    print(f"    Paired diff median (A-B): {paired_diff_median:.2f} mm")
    if np.isfinite(wilcoxon_p):
        print(f"    Wilcoxon T={wilcoxon_stat:.1f}, "
              f"p={wilcoxon_p:.2e}")
    if np.isfinite(ci_width_mean_a) and np.isfinite(ci_width_mean_b):
        print(f"    Mean CI width: A={ci_width_mean_a:.2f} mm, "
              f"B={ci_width_mean_b:.2f} mm")
    if np.isfinite(annual_et_median_a) and np.isfinite(annual_et_median_b):
        print(f"    Median corrected annual ET: A={annual_et_median_a:.1f} mm, "
              f"B={annual_et_median_b:.1f} mm")

    return evidence


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_method_comparison(
    merged_df: pd.DataFrame,
    evidence: dict,
    out_dir: Path,
    dpi: int = 300,
) -> Path:
    """Create side-by-side violin plot of Method A vs B corrections.

    Args:
        merged_df: Paired DataFrame with _A and _B columns.
        evidence: Evidence summary dict from comparison statistics.
        out_dir: Output directory for the plot.
        dpi: Plot resolution.

    Returns:
        Path to saved violin plot PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_a = merged_df["delta_annual_mm_A"].dropna().values
    delta_b = merged_df["delta_annual_mm_B"].dropna().values

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Violin plot
    parts = ax1.violinplot(
        [delta_a, delta_b],
        positions=[1, 2],
        showmedians=False,
        showextrema=False,
    )

    # Style violins
    colors = ["#4C72B0", "#DD8452"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    # Median dots
    for i, data in enumerate([delta_a, delta_b], start=1):
        med = np.median(data)
        ax1.scatter(
            i, med, color="white", edgecolor="black",
            s=60, zorder=5, linewidths=1.5,
        )

    # Labels
    n_a = evidence["n_matched"]
    n_b = evidence["n_matched"]
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([
        f"Method A\n(N={n_a:,})",
        f"Method B\n(N={n_b:,})",
    ])
    ax1.set_ylabel("Annual ET correction (mm)")
    ax1.set_title("Method A vs Method B: Correction Magnitude Distribution")

    # Secondary y-axis in ac-ft/acre
    ax2 = ax1.twinx()
    ymin, ymax = ax1.get_ylim()
    ax2.set_ylim(mm_to_acft_per_acre(ymin), mm_to_acft_per_acre(ymax))
    ax2.set_ylabel("Annual ET correction (ac-ft/acre)")

    # Annotation box
    med_a = evidence["method_a_median"]
    med_b = evidence["method_b_median"]
    wilcoxon_p = evidence["wilcoxon_p"]
    if np.isfinite(wilcoxon_p):
        p_str = f"p = {wilcoxon_p:.2e}" if wilcoxon_p >= 1e-4 else f"p < 1e-4"
    else:
        p_str = "p = N/A"

    annotation = (
        f"Median A: {med_a:.1f} mm\n"
        f"Median B: {med_b:.1f} mm\n"
        f"Wilcoxon {p_str}"
    )
    ax1.text(
        0.98, 0.97, annotation,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()
    plot_path = out_dir / "method_comparison_violin.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  [saved] Violin plot: {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_method_comparison(
    method_a_csv: Path,
    method_b_csv: Optional[Path] = None,
    *,
    run_method_b: bool = False,
    n_boot: int = 100,
    max_uids_per_county_wy: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> dict:
    """Compare Method A and Method B ET corrections on paired parcel-years.

    Loads Method A data from a pre-existing CSV, loads or runs Method B,
    merges on (county, WY, UniqueID), computes comparison statistics,
    and generates a violin plot.

    Args:
        method_a_csv: Path to the Method A parcel-year CSV.
        method_b_csv: Optional path to a pre-existing Method B CSV.
            If None and run_method_b is True, Method B will be computed.
        run_method_b: If True and method_b_csv is None, run the ET
            correction pipeline with chosen_method="B".
        n_boot: Number of bootstrap samples for Method B run.
        max_uids_per_county_wy: Optional cap on UIDs per county-year
            for Method B run.
        out_dir: Output directory for plots and CSVs. Defaults to
            config.statistics_export_dir / "method_comparison".

    Returns:
        Dict with keys:
            evidence: Summary statistics dict.
            merged_df: Paired DataFrame with _A and _B columns.
            plot_path: Path to the violin plot PNG.

    Raises:
        ValueError: If neither method_b_csv nor run_method_b is provided.
    """
    if out_dir is None:
        out_dir = config.statistics_export_dir / "method_comparison"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("METHOD A vs B COMPARISON")
    print(f"{'='*60}")

    # --- Load Method A ---
    print(f"\n{'~'*40}")
    print("Loading Method A data")
    print(f"{'~'*40}")
    df_a = _load_method_csv(method_a_csv, "Method A")

    # --- Load or run Method B ---
    print(f"\n{'~'*40}")
    print("Loading / running Method B data")
    print(f"{'~'*40}")

    if method_b_csv is not None:
        df_b = _load_method_csv(method_b_csv, "Method B")
    elif run_method_b:
        print("  Running ET correction pipeline with chosen_method='B'...")
        from .et_stats_provider import run_et_correction_stats

        result_b = run_et_correction_stats(
            chosen_method="B",
            n_boot=n_boot,
            max_uids_per_county_wy=max_uids_per_county_wy,
            export_csv=True,
        )
        df_b = result_b["df_parcel_year"]
        print(f"  Method B pipeline complete: {len(df_b):,} parcel-years")
    else:
        raise ValueError(
            "Must provide method_b_csv or set run_method_b=True "
            "to generate Method B data."
        )

    # --- Paired merge ---
    print(f"\n{'~'*40}")
    print("Paired merge")
    print(f"{'~'*40}")
    matched, n_matched, n_a_only, n_b_only = _paired_merge(df_a, df_b)

    if n_matched == 0:
        raise ValueError(
            "No matched parcel-years found between Method A and B. "
            "Check that both datasets cover the same counties/WY/UIDs."
        )

    # --- Comparison statistics ---
    print(f"\n{'~'*40}")
    print("Comparison statistics")
    print(f"{'~'*40}")
    evidence = _compute_comparison_stats(
        matched, n_matched, n_a_only, n_b_only,
    )

    # --- Violin plot ---
    print(f"\n{'~'*40}")
    print("Generating violin plot")
    print(f"{'~'*40}")
    plot_path = plot_method_comparison(matched, evidence, out_dir)

    # --- Export merged CSV ---
    merged_csv = out_dir / "method_comparison_paired.csv"
    matched.to_csv(merged_csv, index=False)
    print(f"  [saved] Paired data: {merged_csv}")

    print(f"\n{'='*60}")
    print("METHOD COMPARISON COMPLETE")
    print(f"{'='*60}")

    return {
        "evidence": evidence,
        "merged_df": matched,
        "plot_path": plot_path,
    }
