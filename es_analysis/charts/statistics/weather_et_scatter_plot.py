"""Three-panel weather-ET scatter plot (Tmax, GDD5, ETa).

Side-by-side scatter of three x-variables vs cutting metric,
colored by county, with per-parcel means across water years.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data_providers.spatial_provider import COUNTY_ORDER
from ...utils.validation import MIN_CUMULATIVE_ET_MM
from .multicounty_parcel_scatter_plot import (
    _county_color_map,
    _add_regression,
    _metric_label,
    _style_axes_full_border,
)


def weather_et_scatter_plot(
    df: pd.DataFrame,
    cut_metric: str = "n_cp_season",
    counties: Optional[List[str]] = None,
    wy_label: str = "WYs 2019-2024",
    min_et_mm: float = MIN_CUMULATIVE_ET_MM,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Create a three-panel scatter: Tmax, GDD5, ETa vs cutting metric.

    Each panel shows per-county colored scatter with a pooled OLS
    regression line.  Data is aggregated per parcel (mean across WYs).

    Args:
        df: DataFrame with columns: UniqueID, county, WY,
            n_cp_season, tmax_mean, gdd5_mean,
            et_cum_minET_to_last_cut_mm.
        cut_metric: Column for y-axis ("n_cp_season" or "n_cuttings").
        counties: Optional list of counties to include (default: all).
        wy_label: Label for the water year range shown in titles.
        min_et_mm: Minimum cumulative ET threshold (mm) for filtering.
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes_array, summary_dict).
    """
    tmax_col = "tmax_mean"
    gdd5_col = "gdd5_mean"
    et_col = "et_cum_minET_to_last_cut_mm"
    required = ["UniqueID", "county", cut_metric, tmax_col, gdd5_col, et_col]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # --- Filter to requested counties before aggregation ---
    work = df.copy()
    if counties is not None:
        work = work[work["county"].isin(counties)]

    work = work.dropna(subset=[cut_metric, tmax_col, gdd5_col, et_col])
    if work.empty:
        raise ValueError("No data after dropping NaNs.")

    # --- Aggregate per parcel (mean across WYs) ---
    agg = (
        work.groupby("UniqueID")
        .agg(
            county=("county", "first"),
            **{
                cut_metric: (cut_metric, "mean"),
                tmax_col: (tmax_col, "mean"),
                gdd5_col: (gdd5_col, "mean"),
                et_col: (et_col, "mean"),
            },
        )
        .reset_index()
    )

    agg[cut_metric] = agg[cut_metric].round()

    # Filter low-quality ET rows
    n_before = len(agg)
    agg = agg[np.isfinite(agg[et_col].to_numpy(float))]
    agg = agg[agg[et_col] >= min_et_mm]
    n_filtered_low_et = n_before - len(agg)
    if agg.empty:
        raise ValueError(f"No data after filtering ET < {min_et_mm} mm.")

    # --- County ordering and colors ---
    present = agg["county"].unique().tolist()
    ordered = [c for c in COUNTY_ORDER if c in present]
    others = sorted([c for c in present if c not in ordered])
    county_list = ordered + others

    ccolors = _county_color_map(county_list)
    ylab = _metric_label(cut_metric)

    # --- Three-panel figure ---
    panel_specs = [
        (tmax_col, "Mean Daymet Tmax", "Tmax"),
        (gdd5_col, "Mean GDD per cutting (base 5 \u00b0C)", "GDD5"),
        (et_col, "Mean OpenET ET cumulative (mm)\n(parcel-specific cut-cycle windows)", "ETa"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    reg_results = {}

    for ax, (col, xlabel, panel_name) in zip(axes, panel_specs):
        # Per-county scatter
        for c in county_list:
            sub = agg[agg["county"] == c]
            ax.scatter(
                sub[col].to_numpy(float),
                sub[cut_metric].to_numpy(float),
                alpha=0.6, edgecolor="none", c=[ccolors[c]],
            )

        ax.set_xlabel(xlabel)
        ax.set_title(
            f"{ylab} vs {panel_name}\n{wy_label} (mean per parcel)"
        )
        _style_axes_full_border(ax)

        # Pooled regression
        x_all = agg[col].to_numpy(float)
        y_all = agg[cut_metric].to_numpy(float)
        reg_results[panel_name] = _add_regression(ax, x_all, y_all)

    # Only first panel gets y-label (shared y-axis)
    axes[0].set_ylabel(ylab)

    # --- Legend to right of third panel ---
    handles = []
    for c in county_list:
        n_parcels = int(agg.loc[agg["county"] == c, "UniqueID"].nunique())
        handles.append(
            mpl.lines.Line2D(
                [0], [0], marker="o", linestyle="",
                color=ccolors[c], label=f"{c} (n={n_parcels})",
            )
        )
    axes[-1].legend(
        handles=handles, title="County", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()

    summary = {
        "cut_metric": cut_metric,
        "wy_label": wy_label,
        "n_counties": len(county_list),
        "n_parcels": int(agg["UniqueID"].nunique()),
        "n_rows_after_agg": len(agg),
        "n_filtered_low_et": n_filtered_low_et,
        "min_et_mm": min_et_mm,
        "regressions": reg_results,
    }

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        summary["outfile"] = str(outfile)
        print(f"Weather-ET scatter saved: {outfile}")

    return fig, axes, summary
