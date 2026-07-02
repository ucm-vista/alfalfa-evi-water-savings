"""Sensitivity analysis provider for ET correction parameters.

Runs a one-at-a-time (OAT) sensitivity analysis across 5 pipeline
parameters, producing summary statistics, a tornado plot (sorted by
impact on median correction), and line plots with IQR bands.

Each sensitivity run calls ``run_et_correction_stats`` with a single
parameter varied while holding all others at their default values.
Full DataFrames are discarded after extracting summary statistics to
keep memory usage bounded.

Phase 4 (Method Validation): demonstrates that ET correction results
are robust to reasonable parameter choices.
"""

import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULTS = {
    "r_days": 8,
    "pre_window": 5,
    "post_window": 5,
    "low_quantile": 0.25,
    "cloud_cover_max": 20.0,
}

SENSITIVITY_GRID = {
    "r_days":          [4, 6, 8, 10, 14],
    "pre_window":      [2, 3, 5, 7, 10],
    "post_window":     [2, 3, 5, 7, 10],
    "low_quantile":    [0.10, 0.15, 0.25, 0.35, 0.50],
    "cloud_cover_max": [10.0, 15.0, 20.0, 30.0, 50.0],
}

PARAM_LABELS = {
    "r_days": "Recovery days (r)",
    "pre_window": "Pre-harvest window (days)",
    "post_window": "Post-harvest window (days)",
    "low_quantile": "Low quantile (f_min)",
    "cloud_cover_max": "Max cloud cover (%)",
}


# ---------------------------------------------------------------------------
# Summary extraction helper
# ---------------------------------------------------------------------------

def _extract_stats(results_run: dict) -> dict:
    """Extract summary statistics from a single pipeline run.

    Args:
        results_run: Dict returned by ``run_et_correction_stats``.

    Returns:
        Dict with median, mean, p25, p75, and n_valid for
        ``delta_annual_mm``.
    """
    df = results_run["df_parcel_year"]
    delta = pd.to_numeric(df["delta_annual_mm"], errors="coerce").dropna()
    n_valid = int(delta.size)
    if n_valid == 0:
        return {
            "median_delta_mm": np.nan,
            "mean_delta_mm": np.nan,
            "p25_delta_mm": np.nan,
            "p75_delta_mm": np.nan,
            "n_valid": 0,
        }
    return {
        "median_delta_mm": float(np.nanmedian(delta)),
        "mean_delta_mm": float(np.nanmean(delta)),
        "p25_delta_mm": float(np.nanquantile(delta, 0.25)),
        "p75_delta_mm": float(np.nanquantile(delta, 0.75)),
        "n_valid": n_valid,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tornado(
    results_df: pd.DataFrame,
    impact: Dict[str, float],
    out_dir: Path,
    dpi: int = 300,
) -> Path:
    """Create a tornado plot showing sensitivity impact per parameter.

    Args:
        results_df: Summary DataFrame from the OAT analysis.
        impact: Dict mapping parameter name to max absolute percent
            change from default.
        out_dir: Output directory for the saved figure.
        dpi: Resolution in dots per inch.

    Returns:
        Path to the saved tornado plot PNG.
    """
    # Sort parameters by descending impact
    sorted_params = sorted(impact.keys(), key=lambda p: impact[p])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y_positions = np.arange(len(sorted_params))
    for i, param in enumerate(sorted_params):
        pdf = results_df[results_df["parameter"] == param]
        pct_vals = pdf["pct_change_from_default"].values
        lo = float(np.nanmin(pct_vals))
        hi = float(np.nanmax(pct_vals))

        # Draw left (negative) portion in blue, right (positive) in red
        if lo < 0:
            ax.barh(i, lo, left=0, height=0.6, color="#4393c3",
                    edgecolor="white", linewidth=0.5)
        if hi > 0:
            ax.barh(i, hi, left=0, height=0.6, color="#d6604d",
                    edgecolor="white", linewidth=0.5)
        # If both sides exist, fill the gap
        if lo >= 0:
            ax.barh(i, hi - lo, left=lo, height=0.6, color="#d6604d",
                    edgecolor="white", linewidth=0.5)
        elif hi <= 0:
            ax.barh(i, hi - lo, left=lo, height=0.6, color="#4393c3",
                    edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in sorted_params])
    ax.set_xlabel("Change in median correction (%)")
    ax.set_title("Sensitivity of ET Correction to Parameter Choices")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_dir) / "sensitivity_tornado.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] tornado plot: {out_path}")
    return out_path


def plot_sensitivity_lines(
    results_df: pd.DataFrame,
    defaults: Dict,
    out_dir: Path,
    dpi: int = 300,
) -> Path:
    """Create line plots showing correction magnitude vs parameter value.

    One subplot per parameter in a 2x3 grid layout. Each subplot shows
    the median correction with an IQR shaded band, plus reference lines
    at the default parameter value and default median correction.

    Args:
        results_df: Summary DataFrame from the OAT analysis.
        defaults: Dict of default parameter values.
        out_dir: Output directory for the saved figure.
        dpi: Resolution in dots per inch.

    Returns:
        Path to the saved line plots PNG.
    """
    params = list(SENSITIVITY_GRID.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes_flat[idx]
        pdf = results_df[results_df["parameter"] == param].sort_values("value")

        x = pdf["value"].values
        y_med = pdf["median_delta_mm"].values
        y_p25 = pdf["p25_delta_mm"].values
        y_p75 = pdf["p75_delta_mm"].values

        ax.fill_between(x, y_p25, y_p75, alpha=0.25, color="#4393c3",
                        label="IQR (p25-p75)")
        ax.plot(x, y_med, "o-", color="#2166ac", markersize=5,
                linewidth=1.5, label="Median")

        # Default value reference lines
        default_val = defaults[param]
        default_row = pdf[pdf["value"] == default_val]
        if not default_row.empty:
            default_median = float(default_row["median_delta_mm"].iloc[0])
            ax.axhline(default_median, color="gray", linestyle=":",
                       linewidth=0.8, alpha=0.7)
        ax.axvline(default_val, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.7, label="Default")

        ax.set_title(PARAM_LABELS.get(param, param), fontsize=10)
        ax.set_xlabel(param)
        ax.tick_params(labelsize=8)

    # Remove unused 6th subplot
    axes_flat[5].set_visible(False)

    # Shared y-label via figure text
    fig.text(0.02, 0.5, "Median ET correction (mm)",
             va="center", rotation="vertical", fontsize=11)
    fig.suptitle("Sensitivity Analysis: One-at-a-Time Parameter Variation",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.04, 0.0, 1.0, 0.95])

    out_path = Path(out_dir) / "sensitivity_line_plots.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] line plots: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Parallel worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _run_one_config(
    kwargs: dict,
    n_boot: int,
    max_uids_per_county_wy: Optional[int],
    param: str,
    value,
) -> Tuple[str, float, dict]:
    """Run a single pipeline configuration and return summary stats.

    This is a top-level function so it can be pickled by ProcessPoolExecutor.
    """
    from .et_stats_provider import run_et_correction_stats

    results_run = run_et_correction_stats(
        **kwargs,
        n_boot=n_boot,
        max_uids_per_county_wy=max_uids_per_county_wy,
        export_csv=False,
    )
    stats = _extract_stats(results_run)
    del results_run
    gc.collect()
    return param, value, stats


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    *,
    n_boot: int = 10,
    max_uids_per_county_wy: Optional[int] = None,
    grid: Optional[Dict] = None,
    defaults: Optional[Dict] = None,
    out_dir: Optional[Path] = None,
    dpi: int = 300,
    max_workers: int = 1,
) -> dict:
    """Run OAT sensitivity analysis on ET correction parameters.

    Varies each of 5 parameters independently while holding others at
    their default values. For each configuration, runs the full ET
    correction pipeline and extracts summary statistics (median, mean,
    IQR of delta_annual_mm). Full DataFrames are discarded after each
    run to keep memory bounded.

    Produces:
    - A tornado plot showing parameters sorted by sensitivity impact.
    - Line plots with IQR bands for each parameter.

    Args:
        n_boot: Number of bootstrap replicates per run (low default
            for speed during sensitivity sweeps).
        max_uids_per_county_wy: Optional cap on UIDs per county-year
            for faster iteration.
        grid: Override for SENSITIVITY_GRID (dict of param -> list
            of values).
        defaults: Override for DEFAULTS (dict of param -> default
            value).
        out_dir: Output directory for plots and CSV. Defaults to
            ``statistics_export_dir / sensitivity``.
        dpi: Plot resolution in dots per inch.

    Returns:
        Dict with keys: ``results_df`` (summary DataFrame),
        ``impact`` (dict of param -> max abs pct change),
        ``tornado_path`` (Path), ``line_plots_path`` (Path).
    """
    # Lazy import to avoid circular dependency and slow module load
    from .et_stats_provider import run_et_correction_stats
    from .config import config

    grid = grid or SENSITIVITY_GRID
    defs = defaults or DEFAULTS

    if out_dir is None:
        out_dir = Path(config.statistics_export_dir) / "sensitivity"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count total unique runs (excluding 4 duplicate default runs)
    total_unique = sum(
        1 for vals in grid.values() for v in vals
        if v != defs.get(list(grid.keys())[list(grid.values()).index(vals)])
    )
    # Simpler: total grid points minus (N_params - 1) duplicate defaults
    # But we need the default run too, so total = total_unique + 1
    total_runs = total_unique + 1

    effective_workers = max(1, min(max_workers, total_runs))

    print("=" * 60)
    print("SENSITIVITY ANALYSIS: ONE-AT-A-TIME PARAMETER VARIATION")
    print("-" * 60)
    print(f"Parameters: {len(grid)}")
    print(f"Grid points: {sum(len(v) for v in grid.values())} "
          f"({total_unique} unique non-default + 1 default = {total_runs} runs)")
    print(f"Bootstrap replicates: {n_boot}")
    print(f"Workers: {effective_workers}")
    if max_uids_per_county_wy is not None:
        print(f"Max UIDs per county-year: {max_uids_per_county_wy}")
    print("=" * 60)

    # ---- Build work items ----
    work_items: List[Tuple[str, float, dict]] = []
    for param, values in grid.items():
        for value in values:
            if value != defs[param]:
                kwargs = dict(defs)
                kwargs[param] = value
                work_items.append((param, value, kwargs))

    # ---- Run default configuration (always sequential) ----
    print("\n--- Running default configuration ---")
    _, _, default_stats = _run_one_config(
        dict(defs), n_boot, max_uids_per_county_wy, "default", 0,
    )
    print(f"  [default] median={default_stats['median_delta_mm']:.2f} mm "
          f"(n={default_stats['n_valid']})")

    # ---- Run non-default configs ----
    results_map: Dict[Tuple[str, float], dict] = {}

    if effective_workers > 1:
        print(f"\n--- Running {len(work_items)} configs in parallel "
              f"({effective_workers} workers) ---")
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(
                    _run_one_config,
                    kwargs, n_boot, max_uids_per_county_wy, param, value,
                ): (param, value)
                for param, value, kwargs in work_items
            }
            done_count = 0
            for future in as_completed(futures):
                param, value, stats = future.result()
                results_map[(param, value)] = stats
                done_count += 1
                print(f"  [{done_count}/{len(work_items)}] {param}={value}: "
                      f"median={stats['median_delta_mm']:.2f} mm "
                      f"(n={stats['n_valid']})")
    else:
        print(f"\n--- Running {len(work_items)} configs sequentially ---")
        for i, (param, value, kwargs) in enumerate(work_items, 1):
            print(f"  Running {param}={value} ...", end=" ", flush=True)
            _, _, stats = _run_one_config(
                kwargs, n_boot, max_uids_per_county_wy, param, value,
            )
            results_map[(param, value)] = stats
            print(f"[{i}/{len(work_items)}] "
                  f"median={stats['median_delta_mm']:.2f} mm "
                  f"(n={stats['n_valid']})")

    # ---- Assemble rows in grid order ----
    rows = []
    for param, values in grid.items():
        for value in values:
            if value == defs[param]:
                rows.append({"parameter": param, "value": value, **default_stats})
            else:
                rows.append({
                    "parameter": param, "value": value,
                    **results_map[(param, value)],
                })

    # ---- Step 3: Build results DataFrame ----
    results_df = pd.DataFrame(rows)

    # Compute pct_change_from_default for each parameter
    pct_changes = []
    for _, row in results_df.iterrows():
        default_median = default_stats["median_delta_mm"]
        if default_median != 0 and np.isfinite(default_median):
            pct = 100.0 * (row["median_delta_mm"] - default_median) / abs(default_median)
        else:
            pct = np.nan
        pct_changes.append(pct)
    results_df["pct_change_from_default"] = pct_changes

    # ---- Step 4: Compute sensitivity impact per parameter ----
    impact = {}
    for param in grid:
        pdf = results_df[
            (results_df["parameter"] == param)
            & (results_df["value"] != defs[param])
        ]
        if pdf.empty or pdf["pct_change_from_default"].isna().all():
            impact[param] = 0.0
        else:
            impact[param] = float(
                pdf["pct_change_from_default"].abs().max()
            )

    print("\n" + "=" * 60)
    print("SENSITIVITY IMPACT (max |% change| from default)")
    print("-" * 60)
    for param in sorted(impact, key=impact.get, reverse=True):
        print(f"  {PARAM_LABELS.get(param, param):35s}: "
              f"{impact[param]:+.2f}%")
    print("=" * 60)

    # ---- Step 5: Export results CSV ----
    csv_path = out_dir / "sensitivity_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[saved] results CSV: {csv_path}")

    # ---- Step 6: Plots ----
    tornado_path = plot_tornado(results_df, impact, out_dir, dpi=dpi)
    line_plots_path = plot_sensitivity_lines(
        results_df, defs, out_dir, dpi=dpi,
    )

    return {
        "results_df": results_df,
        "impact": impact,
        "tornado_path": tornado_path,
        "line_plots_path": line_plots_path,
    }
