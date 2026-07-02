"""Diagnostic anomaly plots.

Identifies anomalous parcels and generates per-parcel ET correction
plots for investigation. Saves to output/figures/test/.

Anomaly criteria:
  1. Max-cuttings parcels with low ET (< median for that n_cuttings level)
  2. Low-cuttings parcels (1-2) with very low ET (< 50 mm)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def find_anomalous_parcels(
    df: pd.DataFrame,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    cut_col: str = "n_cuttings",
    n_max_cuts_anomalies: int = 5,
    n_low_cut_anomalies: int = 5,
    low_et_threshold_mm: float = 50.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find anomalous parcels for diagnostic investigation.

    Returns:
        (high_cut_low_et, low_cut_low_et) DataFrames.
    """
    work = df.dropna(subset=[et_col, cut_col]).copy()
    work[cut_col] = work[cut_col].astype(int)

    # 1. Max cuttings with below-median ET for that cutting level
    medians = work.groupby(cut_col)[et_col].median()
    work["_et_median_for_cuts"] = work[cut_col].map(medians)
    work["_below_median"] = work[et_col] < work["_et_median_for_cuts"]

    # Sort by n_cuttings desc, then ET asc to get the most surprising anomalies
    high_cut_candidates = (
        work[work["_below_median"]]
        .sort_values([cut_col, et_col], ascending=[False, True])
        .head(n_max_cuts_anomalies)
    )

    # 2. Low cuttings (1-2) with very low ET
    low_cut_candidates = (
        work[(work[cut_col] <= 2) & (work[et_col] < low_et_threshold_mm)]
        .sort_values(et_col, ascending=True)
        .head(n_low_cut_anomalies)
    )

    return high_cut_candidates, low_cut_candidates


def generate_diagnostic_plots(
    df: pd.DataFrame,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    cut_col: str = "n_cuttings",
    n_max_cuts_anomalies: int = 5,
    n_low_cut_anomalies: int = 5,
    low_et_threshold_mm: float = 50.0,
    method: str = "A",
    n_boot: int = 50,
    out_dir: Optional[Path] = None,
) -> Dict:
    """Generate ET correction plots for anomalous parcels.

    Args:
        df: Parcel-year DataFrame with UniqueID, county, WY, et/cut columns.
        et_col: ET column name.
        cut_col: Cutting count column.
        n_max_cuts_anomalies: Number of high-cut anomalies to plot.
        n_low_cut_anomalies: Number of low-cut anomalies to plot.
        low_et_threshold_mm: Threshold for low-ET anomaly.
        method: ET correction method.
        n_boot: Bootstrap replicates for per-parcel plots.
        out_dir: Output directory for plots.

    Returns:
        Summary dict with anomaly counts and plot paths.
    """
    import matplotlib
    matplotlib.use("Agg")

    if out_dir is None:
        out_dir = Path(__file__).parent.parent.parent / "output" / "figures" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    high_cut, low_cut = find_anomalous_parcels(
        df, et_col=et_col, cut_col=cut_col,
        n_max_cuts_anomalies=n_max_cuts_anomalies,
        n_low_cut_anomalies=n_low_cut_anomalies,
        low_et_threshold_mm=low_et_threshold_mm,
    )

    print(f"  Found {len(high_cut)} high-cut/low-ET anomalies")
    print(f"  Found {len(low_cut)} low-cut/low-ET anomalies")

    from es_analysis.charts.et_corrections.et_separate_daily_monthly_plot import (
        et_separate_daily_monthly,
    )

    plot_paths = []

    for label, anomaly_df in [("high_cut_low_et", high_cut), ("low_cut_low_et", low_cut)]:
        for _, row in anomaly_df.iterrows():
            uid = str(row["UniqueID"])
            county = str(row["county"])
            wy = int(row["WY"])
            n_cuts = int(row[cut_col])
            et_val = float(row[et_col])

            plot_name = f"anomaly_{label}_{county.replace(' ', '_')}_WY{wy}_{uid}"
            print(f"    Plotting {plot_name} ({n_cuts} cuts, {et_val:.0f} mm)")

            try:
                result = et_separate_daily_monthly(
                    county=county, water_year=wy, uid=uid,
                    method=method, n_boot=n_boot,
                    output_dir=out_dir,
                )
                plot_paths.append(str(out_dir / plot_name))
            except Exception as e:
                print(f"    FAILED: {e}")

    summary = {
        "chart": "diagnostic_anomaly_plots",
        "n_high_cut_anomalies": len(high_cut),
        "n_low_cut_anomalies": len(low_cut),
        "n_plots_generated": len(plot_paths),
        "out_dir": str(out_dir),
    }

    # Save anomaly list CSV
    all_anomalies = pd.concat([
        high_cut.assign(anomaly_type="high_cut_low_et"),
        low_cut.assign(anomaly_type="low_cut_low_et"),
    ], ignore_index=True)
    csv_path = out_dir / "anomaly_parcels.csv"
    cols_keep = ["UniqueID", "county", "WY", cut_col, et_col, "anomaly_type"]
    cols_keep = [c for c in cols_keep if c in all_anomalies.columns]
    all_anomalies[cols_keep].to_csv(csv_path, index=False)
    print(f"  Anomaly list saved: {csv_path}")
    summary["anomaly_csv"] = str(csv_path)

    return summary


if __name__ == "__main__":
    from es_analysis.data_providers.parcel_summary_provider import build_multicounty_matched

    df = build_multicounty_matched()
    summary = generate_diagnostic_plots(df)
    print(f"Summary: {summary}")
