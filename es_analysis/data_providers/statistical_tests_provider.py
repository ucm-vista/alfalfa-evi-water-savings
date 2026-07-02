"""Statistical tests provider for ET correction analysis.

Implements all hypothesis tests, effect sizes, post-hoc comparisons,
and sensitivity checks required for Phase 3 (Statistical Tests).
Each test function takes a DataFrame and returns a standardized results
dict. The orchestrator function ``run_all_statistical_tests`` calls
all tests and assembles results into a single dict.

Tests included:
- Paired Wilcoxon signed-rank test (pooled and per-year)
- Cohen's d for paired differences
- Hodges-Lehmann estimator with 95% CI
- Kruskal-Wallis test for county variation
- Dunn's post-hoc with Holm correction and compact letter display
- D'Agostino-Pearson normality check (+ conditional paired t-test)
- Sensitivity check excluding small corrections
- Failure breakdown table (county x reason)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.stats import wilcoxon, kruskal, normaltest, ttest_rel, skew, kurtosis
from scipy.stats import norm
import scikit_posthocs as sp
from compactletterdisplay import compact_letter_display

from ..utils.units import mm_to_acft_per_acre


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_COUNTY_N_DEFAULT = 30
SENSITIVITY_THRESHOLD_MM = 5.0

_REQUIRED_COLUMNS = [
    "county", "WY", "UniqueID",
    "ET_open_annual_mm", "ET_corr_annual_mm", "delta_annual_mm",
]


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_parcel_year_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the Phase 2 parcel-year output.

    Reads CSV, validates expected columns exist, drops rows where
    ET_open_annual_mm or ET_corr_annual_mm is NaN.

    Args:
        csv_path: Path to the parcel-year CSV file.

    Returns:
        Cleaned DataFrame with NaN ET rows removed.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Parcel-year CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    total_rows = len(df)
    df = df.dropna(subset=["ET_open_annual_mm", "ET_corr_annual_mm"]).copy()
    dropped = total_rows - len(df)

    print(f"{'='*60}")
    print("STATISTICAL TESTS: DATA LOADED")
    print(f"{'='*60}")
    print(f"Total rows in CSV:  {total_rows:,}")
    print(f"Dropped NaN ET:     {dropped:,} ({100*dropped/total_rows:.1f}%)")
    print(f"Valid rows:         {len(df):,}")
    print(f"Counties:           {df['county'].nunique()}")
    print(f"Water years:        {sorted(df['WY'].unique())}")
    print(f"{'='*60}")

    return df


# ---------------------------------------------------------------------------
# 1. Paired Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def paired_wilcoxon_test(
    df: pd.DataFrame,
    scope: str = "Pooled (all years)",
) -> dict:
    """Paired Wilcoxon signed-rank test on ET correction magnitude.

    Uses delta_annual_mm (= ET_open - ET_corr) as the paired difference.
    Zero differences are discarded (zero_method='wilcox').

    Args:
        df: DataFrame with 'delta_annual_mm' column.
        scope: Label for the scope of this test (e.g. "Pooled (all years)").

    Returns:
        Standardized result dict with keys: test_name, scope, statistic,
        statistic_name, p_value, effect_size, effect_size_name,
        effect_size_label, n, notes.
    """
    d = df["delta_annual_mm"].dropna().values
    n_total = len(d)
    n_nonzero = int(np.sum(d != 0))

    result = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    T = result.statistic
    p_val = result.pvalue

    # Rank-biserial correlation (nonparametric effect size)
    denom = n_nonzero * (n_nonzero + 1) / 2.0
    r_rb = 1.0 - (2.0 * T) / denom if denom > 0 else np.nan

    # Label rank-biserial
    abs_r = abs(r_rb)
    if abs_r < 0.1:
        r_label = "negligible"
    elif abs_r < 0.3:
        r_label = "small"
    elif abs_r < 0.5:
        r_label = "medium"
    else:
        r_label = "large"

    return {
        "test_name": "Paired Wilcoxon signed-rank",
        "scope": scope,
        "statistic": float(T),
        "statistic_name": "T",
        "p_value": float(p_val),
        "effect_size": float(r_rb),
        "effect_size_name": "rank-biserial r",
        "effect_size_label": r_label,
        "n": n_total,
        "notes": (
            f"zero_method='wilcox' discards {n_total - n_nonzero} zero-difference "
            f"pairs; effective Wilcoxon N={n_nonzero}. "
            f"T is the smaller of T+ and T-."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Cohen's d for paired differences
# ---------------------------------------------------------------------------

def cohens_d_effect_size(
    df: pd.DataFrame,
    scope: str = "Pooled (all years)",
) -> dict:
    """Cohen's d for the paired ET correction differences.

    d = mean(delta) / std(delta, ddof=1). Also reports practical units.

    Args:
        df: DataFrame with 'delta_annual_mm' and 'ET_open_annual_mm' columns.
        scope: Label for the scope of this test.

    Returns:
        Standardized result dict.
    """
    d_vals = df["delta_annual_mm"].dropna().values
    n = len(d_vals)
    mean_delta = float(np.mean(d_vals))
    std_delta = float(np.std(d_vals, ddof=1))
    d = mean_delta / std_delta if std_delta > 0 else np.nan

    # Label
    abs_d = abs(d)
    if abs_d < 0.2:
        d_label = "negligible"
    elif abs_d < 0.5:
        d_label = "small"
    elif abs_d < 0.8:
        d_label = "medium"
    else:
        d_label = "large"

    # Practical units
    mean_delta_acft = float(mm_to_acft_per_acre(mean_delta))
    mean_et_open = float(df["ET_open_annual_mm"].dropna().mean())
    pct_of_et = 100.0 * mean_delta / mean_et_open if mean_et_open > 0 else np.nan
    skewness = float(skew(d_vals))
    kurt = float(kurtosis(d_vals))

    return {
        "test_name": "Cohen's d (paired differences)",
        "scope": scope,
        "statistic": float(d),
        "statistic_name": "d",
        "p_value": np.nan,
        "effect_size": float(d),
        "effect_size_name": "Cohen's d",
        "effect_size_label": d_label,
        "n": n,
        "notes": (
            f"mean delta={mean_delta:.2f} mm ({mean_delta_acft:.4f} ac-ft/acre, "
            f"{pct_of_et:.2f}% of mean ET_open={mean_et_open:.1f} mm). "
            f"CAVEAT: skewness={skewness:.2f}, kurtosis={kurt:.1f}; "
            f"Cohen's d assumes symmetry."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Hodges-Lehmann estimator with 95% CI
# ---------------------------------------------------------------------------

def hodges_lehmann_ci(
    d: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Hodges-Lehmann estimator and CI for median paired difference.

    Uses Walsh averages: (d_i + d_j) / 2 for all i <= j.
    CI from order statistics with normal approximation.

    Args:
        d: Array of paired differences.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (hl_estimate, ci_lower, ci_upper).
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)

    n_walsh = n * (n + 1) / 2
    print(
        f"Computing Hodges-Lehmann estimator for N={n} "
        f"({n_walsh/1e6:.1f}M Walsh averages)..."
    )

    # Walsh averages: (d_i + d_j) / 2 for all i <= j
    walsh = np.add.outer(d, d) / 2.0
    idx = np.triu_indices(n)
    walsh_avgs = walsh[idx]
    walsh_sorted = np.sort(walsh_avgs)

    hl = float(np.median(walsh_avgs))

    # CI from order statistics
    alpha = 1.0 - confidence
    z = norm.ppf(1.0 - alpha / 2.0)
    N_w = len(walsh_avgs)
    sigma = np.sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 6.0)

    k_lower = max(0, int(np.floor(N_w / 2.0 - z * sigma / 2.0)))
    k_upper = min(N_w - 1, int(np.ceil(N_w / 2.0 + z * sigma / 2.0)))

    ci_lo = float(walsh_sorted[k_lower])
    ci_hi = float(walsh_sorted[k_upper])

    print(f"  HL estimate: {hl:.4f} mm, 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}] mm")

    return hl, ci_lo, ci_hi


def hodges_lehmann_result(
    df: pd.DataFrame,
    scope: str = "Pooled (all years)",
) -> dict:
    """Hodges-Lehmann estimator result dict for delta_annual_mm.

    Wrapper around hodges_lehmann_ci that returns a standardized dict.

    Args:
        df: DataFrame with 'delta_annual_mm' column.
        scope: Label for the scope.

    Returns:
        Standardized result dict.
    """
    d = df["delta_annual_mm"].dropna().values
    n = len(d)
    hl, ci_lo, ci_hi = hodges_lehmann_ci(d, confidence=0.95)

    hl_acft = float(mm_to_acft_per_acre(hl))
    ci_lo_acft = float(mm_to_acft_per_acre(ci_lo))
    ci_hi_acft = float(mm_to_acft_per_acre(ci_hi))

    return {
        "test_name": "Hodges-Lehmann estimator",
        "scope": scope,
        "statistic": hl,
        "statistic_name": "HL median",
        "p_value": np.nan,
        "effect_size": hl,
        "effect_size_name": "HL estimate (mm)",
        "effect_size_label": (
            f"{hl:.2f} mm ({hl_acft:.4f} ac-ft/acre), "
            f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}] mm "
            f"([{ci_lo_acft:.4f}, {ci_hi_acft:.4f}] ac-ft/acre)"
        ),
        "n": n,
        "notes": (
            f"Walsh averages median. N(N+1)/2 = {n*(n+1)//2:,} Walsh averages."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Kruskal-Wallis test
# ---------------------------------------------------------------------------

def kruskal_wallis_test(
    df: pd.DataFrame,
    scope: str = "Pooled (all years)",
    min_county_n: int = MIN_COUNTY_N_DEFAULT,
) -> dict:
    """Kruskal-Wallis test for county variation in correction magnitude.

    Groups by county, excludes counties with fewer than min_county_n
    observations. Reports eta-squared effect size.

    Args:
        df: DataFrame with 'county' and 'delta_annual_mm' columns.
        scope: Label for the scope.
        min_county_n: Minimum observations per county.

    Returns:
        Standardized result dict.
    """
    # Group by county, filter small groups
    county_counts = df.groupby("county")["delta_annual_mm"].count()
    included = county_counts[county_counts >= min_county_n].index.tolist()
    excluded = county_counts[county_counts < min_county_n].index.tolist()

    df_filtered = df[df["county"].isin(included)].copy()
    groups = [
        sub["delta_annual_mm"].dropna().values
        for _, sub in df_filtered.groupby("county")
    ]
    k = len(groups)
    N = sum(len(g) for g in groups)

    if k < 2:
        return {
            "test_name": "Kruskal-Wallis",
            "scope": scope,
            "statistic": np.nan,
            "statistic_name": "H",
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_size_name": "eta-squared",
            "effect_size_label": "insufficient groups",
            "n": N,
            "notes": f"Only {k} county group(s) with >= {min_county_n} observations.",
        }

    stat, pval = kruskal(*groups, nan_policy="omit")

    # Eta-squared: (H - k + 1) / (N - k)
    eta_sq = (stat - k + 1) / (N - k) if N > k else np.nan

    # Label
    if eta_sq < 0.01:
        eta_label = "negligible"
    elif eta_sq < 0.06:
        eta_label = "small"
    elif eta_sq < 0.14:
        eta_label = "medium"
    else:
        eta_label = "large"

    included_str = ", ".join(sorted(included))
    excluded_str = ", ".join(sorted(excluded)) if excluded else "none"

    return {
        "test_name": "Kruskal-Wallis",
        "scope": scope,
        "statistic": float(stat),
        "statistic_name": "H",
        "p_value": float(pval),
        "effect_size": float(eta_sq),
        "effect_size_name": "eta-squared",
        "effect_size_label": eta_label,
        "n": N,
        "notes": (
            f"k={k} counties (min N>={min_county_n}). "
            f"Included: {included_str}. Excluded: {excluded_str}."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Dunn's post-hoc with CLD
# ---------------------------------------------------------------------------

def dunn_posthoc_with_cld(
    df: pd.DataFrame,
    min_county_n: int = MIN_COUNTY_N_DEFAULT,
) -> Tuple[pd.DataFrame, dict]:
    """Dunn's post-hoc test with Holm correction and compact letter display.

    Runs scikit_posthocs.posthoc_dunn with Holm correction, then
    extracts significant pairs and produces CLD groupings.

    Args:
        df: DataFrame with 'county' and 'delta_annual_mm' columns.
        min_county_n: Minimum observations per county.

    Returns:
        Tuple of (p_value_matrix_df, cld_dict) where cld_dict maps
        county name to letter(s).
    """
    county_counts = df.groupby("county")["delta_annual_mm"].count()
    included = county_counts[county_counts >= min_county_n].index.tolist()
    df_filtered = df[df["county"].isin(included)].copy()

    # Dunn's test
    p_matrix = sp.posthoc_dunn(
        df_filtered,
        val_col="delta_annual_mm",
        group_col="county",
        p_adjust="holm",
    )

    # Extract significant pairs for CLD
    groups = sorted(p_matrix.index.tolist())
    sig_pairs = [
        (a, b) for a, b in combinations(groups, 2)
        if p_matrix.loc[a, b] < 0.05
    ]

    # Compact letter display
    letters = compact_letter_display(sig_pairs, groups)
    cld_dict = dict(zip(groups, letters))

    print(f"\nDunn's post-hoc CLD ({len(groups)} counties, {len(sig_pairs)} significant pairs):")
    for county in groups:
        print(f"  {county:20s}  {cld_dict[county]}")

    return p_matrix, cld_dict


# ---------------------------------------------------------------------------
# 6. Normality check + conditional paired t-test
# ---------------------------------------------------------------------------

def normality_check(df: pd.DataFrame) -> dict:
    """D'Agostino-Pearson normality test on paired differences.

    If approximately normal (p > 0.05), also runs paired t-test as
    confirmation alongside the Wilcoxon test.

    Args:
        df: DataFrame with 'ET_open_annual_mm' and 'ET_corr_annual_mm' columns.

    Returns:
        Result dict with normality test and optional t-test results.
    """
    d = df["delta_annual_mm"].dropna().values
    n = len(d)
    sk = float(skew(d))
    kt = float(kurtosis(d))

    stat, pval = normaltest(d)
    is_normal = bool(pval > 0.05)

    notes = (
        f"D'Agostino-Pearson test. N={n}, skewness={sk:.2f}, kurtosis={kt:.1f}. "
    )

    result = {
        "test_name": "Normality (D'Agostino-Pearson)",
        "scope": "Pooled (all years)",
        "statistic": float(stat),
        "statistic_name": "K^2",
        "p_value": float(pval),
        "effect_size": np.nan,
        "effect_size_name": "N/A",
        "effect_size_label": "normal" if is_normal else "non-normal",
        "n": n,
        "notes": notes,
        "is_normal": is_normal,
        "skewness": sk,
        "kurtosis": kt,
    }

    if is_normal:
        # Paired t-test as confirmation
        et_open = df["ET_open_annual_mm"].dropna().values
        et_corr = df["ET_corr_annual_mm"].dropna().values
        # Align lengths
        min_len = min(len(et_open), len(et_corr))
        t_stat, t_pval = ttest_rel(et_open[:min_len], et_corr[:min_len])
        result["t_test_statistic"] = float(t_stat)
        result["t_test_p_value"] = float(t_pval)
        result["notes"] += (
            f"Approximately normal: paired t-test t={t_stat:.2f}, p={t_pval:.2e}."
        )
    else:
        result["notes"] += (
            "Non-normal distribution confirmed; Wilcoxon is the primary test."
        )

    return result


# ---------------------------------------------------------------------------
# 7. Sensitivity check
# ---------------------------------------------------------------------------

def sensitivity_check(
    df: pd.DataFrame,
    threshold_mm: float = SENSITIVITY_THRESHOLD_MM,
    min_county_n: int = MIN_COUNTY_N_DEFAULT,
) -> dict:
    """Sensitivity check: re-run main tests excluding small corrections.

    Filters to delta_annual_mm >= threshold_mm, then re-runs Wilcoxon
    and KW tests. Reports whether conclusions change.

    Args:
        df: DataFrame with standard columns.
        threshold_mm: Minimum correction threshold in mm.
        min_county_n: Minimum county N for KW.

    Returns:
        Dict with sensitivity analysis results.
    """
    n_total = len(df)
    df_sens = df[df["delta_annual_mm"] >= threshold_mm].copy()
    n_remaining = len(df_sens)
    n_excluded = n_total - n_remaining

    wilcoxon_result = paired_wilcoxon_test(
        df_sens, scope=f"Sensitivity (delta >= {threshold_mm} mm)"
    )
    kw_result = kruskal_wallis_test(
        df_sens, scope=f"Sensitivity (delta >= {threshold_mm} mm)",
        min_county_n=min_county_n,
    )

    # Check if conclusions changed (significance direction)
    wilcoxon_sig = wilcoxon_result["p_value"] < 0.05
    kw_sig = kw_result["p_value"] < 0.05 if np.isfinite(kw_result["p_value"]) else False

    return {
        "threshold_mm": threshold_mm,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_remaining": n_remaining,
        "pct_excluded": 100.0 * n_excluded / n_total if n_total > 0 else 0.0,
        "wilcoxon_result": wilcoxon_result,
        "kw_result": kw_result,
        "wilcoxon_significant": wilcoxon_sig,
        "kw_significant": kw_sig,
        "conclusions_changed": False,  # Updated below
        "notes": "",
    }


# ---------------------------------------------------------------------------
# 8. Failure breakdown
# ---------------------------------------------------------------------------

def failure_breakdown(failures_csv_path: Path) -> pd.DataFrame:
    """Failure breakdown table: county x reason with WY columns.

    Categorizes errors into 'No harvest dates' or 'Negative daily ET'
    (or first 60 chars of other errors). Produces a crosstab with margins.

    Args:
        failures_csv_path: Path to the failures CSV.

    Returns:
        Crosstab DataFrame (county, reason) x WY with margins.
    """
    failures_csv_path = Path(failures_csv_path)
    if not failures_csv_path.exists():
        raise FileNotFoundError(f"Failures CSV not found: {failures_csv_path}")

    df_fail = pd.read_csv(failures_csv_path)

    def categorize_error(e: str) -> str:
        e = str(e)
        if "No harvest" in e:
            return "No harvest dates"
        if "Negative daily ET" in e:
            return "Negative daily ET"
        return e[:60]

    df_fail["reason"] = df_fail["error"].apply(categorize_error)

    breakdown = pd.crosstab(
        [df_fail["county"], df_fail["reason"]],
        df_fail["WY"],
        margins=True,
    )

    return breakdown


# ---------------------------------------------------------------------------
# 9. Per-year runner
# ---------------------------------------------------------------------------

def run_per_year_tests(
    df: pd.DataFrame,
    min_county_n: int = MIN_COUNTY_N_DEFAULT,
) -> List[dict]:
    """Run Wilcoxon, Cohen's d, and KW tests for each water year.

    Args:
        df: DataFrame with standard columns.
        min_county_n: Minimum county N for KW.

    Returns:
        List of result dicts (3 per year).
    """
    results = []
    for wy in sorted(df["WY"].unique()):
        df_year = df[df["WY"] == wy].copy()
        scope = f"WY{wy}"

        results.append(paired_wilcoxon_test(df_year, scope=scope))
        results.append(cohens_d_effect_size(df_year, scope=scope))
        results.append(kruskal_wallis_test(
            df_year, scope=scope, min_county_n=min_county_n,
        ))

    return results


# ---------------------------------------------------------------------------
# 10. Orchestrator
# ---------------------------------------------------------------------------

def run_all_statistical_tests(
    parcel_year_csv: Path,
    failures_csv: Path,
    min_county_n: int = MIN_COUNTY_N_DEFAULT,
    skip_hl: bool = False,
) -> dict:
    """Run all statistical tests on the Phase 2 parcel-year data.

    Orchestrator function that loads data, runs all pooled and per-year
    tests, sensitivity check, and failure breakdown.

    Args:
        parcel_year_csv: Path to the parcel-year CSV.
        failures_csv: Path to the failures CSV.
        min_county_n: Minimum county observations for KW test.
        skip_hl: If True, skip the Hodges-Lehmann computation
            (~30s, ~1GB RAM for N=9131).

    Returns:
        Dict with keys: pooled_results, per_year_results, posthoc,
        sensitivity, failure_breakdown, summary_table.
    """
    # Load data
    df = load_parcel_year_data(parcel_year_csv)

    # --- Pooled tests ---
    print(f"\n{'='*60}")
    print("POOLED STATISTICAL TESTS")
    print(f"{'='*60}")

    pooled_results = []

    # Wilcoxon
    print(f"\n{'~'*40}")
    print("1. Paired Wilcoxon signed-rank test")
    print(f"{'~'*40}")
    wilcoxon_res = paired_wilcoxon_test(df)
    pooled_results.append(wilcoxon_res)
    print(f"  T = {wilcoxon_res['statistic']:.0f}")
    print(f"  p = {wilcoxon_res['p_value']:.2e}")
    print(f"  rank-biserial r = {wilcoxon_res['effect_size']:.4f} ({wilcoxon_res['effect_size_label']})")
    print(f"  N = {wilcoxon_res['n']}")
    print(f"  {wilcoxon_res['notes']}")

    # Cohen's d
    print(f"\n{'~'*40}")
    print("2. Cohen's d (paired differences)")
    print(f"{'~'*40}")
    cohens_res = cohens_d_effect_size(df)
    pooled_results.append(cohens_res)
    print(f"  d = {cohens_res['statistic']:.4f} ({cohens_res['effect_size_label']})")
    print(f"  {cohens_res['notes']}")

    # Hodges-Lehmann
    print(f"\n{'~'*40}")
    print("3. Hodges-Lehmann estimator with 95% CI")
    print(f"{'~'*40}")
    if skip_hl:
        print("  SKIPPED (--skip-hl flag)")
    else:
        hl_res = hodges_lehmann_result(df)
        pooled_results.append(hl_res)
        print(f"  {hl_res['effect_size_label']}")

    # Kruskal-Wallis
    print(f"\n{'~'*40}")
    print("4. Kruskal-Wallis test (county variation)")
    print(f"{'~'*40}")
    kw_res = kruskal_wallis_test(df, min_county_n=min_county_n)
    pooled_results.append(kw_res)
    print(f"  H = {kw_res['statistic']:.2f}")
    print(f"  p = {kw_res['p_value']:.2e}")
    print(f"  eta-squared = {kw_res['effect_size']:.4f} ({kw_res['effect_size_label']})")
    print(f"  {kw_res['notes']}")

    # Dunn's post-hoc
    print(f"\n{'~'*40}")
    print("5. Dunn's post-hoc with Holm correction + CLD")
    print(f"{'~'*40}")
    p_matrix, cld_dict = dunn_posthoc_with_cld(df, min_county_n=min_county_n)

    # Normality
    print(f"\n{'~'*40}")
    print("6. Normality check (D'Agostino-Pearson)")
    print(f"{'~'*40}")
    norm_res = normality_check(df)
    pooled_results.append(norm_res)
    print(f"  K^2 = {norm_res['statistic']:.2f}")
    print(f"  p = {norm_res['p_value']:.2e}")
    print(f"  Distribution: {norm_res['effect_size_label']}")
    print(f"  {norm_res['notes']}")

    # --- Per-year tests ---
    print(f"\n{'='*60}")
    print("PER-YEAR STATISTICAL TESTS")
    print(f"{'='*60}")
    per_year_results = run_per_year_tests(df, min_county_n=min_county_n)
    for res in per_year_results:
        print(f"  [{res['scope']}] {res['test_name']}: "
              f"{res['statistic_name']}={res['statistic']:.2f}, "
              f"p={res['p_value']:.2e}" if np.isfinite(res['p_value']) else
              f"  [{res['scope']}] {res['test_name']}: "
              f"{res['statistic_name']}={res['statistic']:.4f}")

    # --- Sensitivity check ---
    print(f"\n{'='*60}")
    print("SENSITIVITY CHECK")
    print(f"{'='*60}")
    sens = sensitivity_check(df, min_county_n=min_county_n)

    # Determine if conclusions changed relative to pooled
    pooled_wilcoxon_sig = wilcoxon_res["p_value"] < 0.05
    pooled_kw_sig = kw_res["p_value"] < 0.05
    sens_wilcoxon_changed = pooled_wilcoxon_sig != sens["wilcoxon_significant"]
    sens_kw_changed = pooled_kw_sig != sens["kw_significant"]
    sens["conclusions_changed"] = sens_wilcoxon_changed or sens_kw_changed

    if sens["conclusions_changed"]:
        sens["notes"] = "CONCLUSIONS CHANGED after excluding small corrections."
    else:
        sens["notes"] = (
            "Conclusions robust: significance unchanged after excluding "
            f"corrections < {SENSITIVITY_THRESHOLD_MM} mm."
        )

    print(f"  Excluded: {sens['n_excluded']:,} parcel-years "
          f"(delta < {SENSITIVITY_THRESHOLD_MM} mm)")
    print(f"  Remaining: {sens['n_remaining']:,} parcel-years")
    print(f"  Wilcoxon p={sens['wilcoxon_result']['p_value']:.2e} "
          f"(significant={sens['wilcoxon_significant']})")
    print(f"  KW p={sens['kw_result']['p_value']:.2e} "
          f"(significant={sens['kw_significant']})" if np.isfinite(sens['kw_result']['p_value']) else
          f"  KW: insufficient groups")
    print(f"  {sens['notes']}")

    # --- Failure breakdown ---
    print(f"\n{'='*60}")
    print("FAILURE BREAKDOWN")
    print(f"{'='*60}")
    fb = failure_breakdown(failures_csv)
    print(fb.to_string())

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")

    all_results = pooled_results + per_year_results
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "test_name": r["test_name"],
            "scope": r["scope"],
            "statistic_name": r["statistic_name"],
            "statistic": r["statistic"],
            "p_value": r["p_value"],
            "effect_size": r["effect_size"],
            "effect_size_name": r["effect_size_name"],
            "effect_size_label": r["effect_size_label"],
            "n": r["n"],
        })
    summary_table = pd.DataFrame(summary_rows)
    print(summary_table.to_string(index=False))

    return {
        "df": df,
        "pooled_results": pooled_results,
        "per_year_results": per_year_results,
        "posthoc": {
            "p_matrix": p_matrix,
            "cld": cld_dict,
        },
        "sensitivity": sens,
        "failure_breakdown": fb,
        "summary_table": summary_table,
    }
