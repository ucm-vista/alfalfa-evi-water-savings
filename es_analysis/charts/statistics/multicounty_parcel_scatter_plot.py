"""Multi-county parcel scatter plot.

Side-by-side scatter of Daymet variable vs cutting metric (left)
and ET vs cutting metric (right), colored by county.

Source: alfalfa_evi_jovyan.py lines 10185-10447
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data_providers.evi_provider import normalize_county_name
from ...data_providers.spatial_provider import COUNTY_ORDER
from ...utils.validation import MIN_CUMULATIVE_ET_MM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _county_color_map(counties: List[str]) -> Dict[str, tuple]:
    """Assign tab20 colors to counties in COUNTY_ORDER order."""
    ordered = [c for c in COUNTY_ORDER if c in counties]
    others = [c for c in counties if c not in ordered]
    final = ordered + sorted(others)
    cmap = mpl.colormaps.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(final)}


def _metric_label(cut_metric: str) -> str:
    if cut_metric == "n_cp_season":
        return "Number of seasonal change points"
    if cut_metric == "n_cuttings":
        return "Number of cuttings"
    return cut_metric


def _style_axes_full_border(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _add_regression(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> Dict:
    """Draw OLS regression line and annotate with n, R², slope."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 2:
        return {"n": n, "r2": np.nan, "slope": np.nan}
    r = (
        float(np.corrcoef(x, y)[0, 1])
        if (np.std(x) > 0 and np.std(y) > 0)
        else np.nan
    )
    r2 = float(r * r) if np.isfinite(r) else np.nan
    slope = np.nan
    if np.std(x) > 0:
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coeffs, xs), "r--", linewidth=1.5)
    ax.annotate(
        f"n = {n}\nslope = {slope:.3g}\nR\u00b2 = {r2:.3f}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    return {"n": n, "r2": r2, "slope": slope}


# ---------------------------------------------------------------------------
# Main chart function
# ---------------------------------------------------------------------------

def multicounty_parcel_scatter_plot(
    df: pd.DataFrame,
    daymet_var: str,
    cut_metric: str,
    wy_label: str,
    show_count_n_in_legend: bool = True,
    daymet_label_window: Optional[str] = None,
    min_et_mm: float = MIN_CUMULATIVE_ET_MM,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create a two-panel scatter plot colored by county.

    Left panel:  Daymet variable vs cut metric
    Right panel: ET vs cut metric

    Args:
        df: DataFrame with columns: county, {cut_metric},
            {daymet_var}_mean, et_cum_minET_to_last_cut_mm.
        daymet_var: Daymet variable name (e.g. "gdd5", "tmax").
        cut_metric: "n_cp_season" or "n_cuttings".
        wy_label: Label for the water year range.
        show_count_n_in_legend: Show parcel count in legend.
        daymet_label_window: Optional window label for x-axis.
        min_et_mm: Minimum cumulative ET threshold (mm).
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    dm_col = f"{daymet_var}_mean"
    et_col = "et_cum_minET_to_last_cut_mm"
    et_label = (
        "Cumulative OpenET ET (mm)\n"
        "(parcel-specific cut-cycle windows)"
    )

    dfp = df.dropna(subset=["county", cut_metric, dm_col, et_col]).copy()
    if dfp.empty:
        raise ValueError("No data after dropping NaNs for scatter.")

    dfp[cut_metric] = pd.to_numeric(
        dfp[cut_metric], errors="coerce"
    ).round()

    # Filter low-quality ET rows
    n_before = len(dfp)
    dfp = dfp[np.isfinite(dfp[et_col].to_numpy(float))]
    dfp = dfp[dfp[et_col] >= min_et_mm]
    n_filtered_low_et = n_before - len(dfp)
    if dfp.empty:
        raise ValueError(f"No data after filtering ET < {min_et_mm} mm.")

    counties = [c for c in COUNTY_ORDER if c in dfp["county"].unique()]
    if not counties:
        counties = sorted(dfp["county"].unique().tolist())

    ccolors = _county_color_map(counties)
    ylab = _metric_label(cut_metric)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Left: Daymet vs cut metric
    ax = axes[0]
    for c in counties:
        sub = dfp[dfp["county"] == c]
        ax.scatter(
            sub[dm_col].to_numpy(float),
            sub[cut_metric].to_numpy(float),
            alpha=0.6, edgecolor="none", c=[ccolors[c]],
        )
    if daymet_var.lower() == "gdd5":
        xlab = "Mean GDD per cutting (base 5 \u00b0C)"
    else:
        xlab = f"Mean Daymet {daymet_var}"
    if daymet_label_window:
        xlab += f"\n{daymet_label_window}"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{ylab} vs {daymet_var}\n{wy_label}".strip())
    _style_axes_full_border(ax)

    # Pooled regression on left panel
    x_dm_all = dfp[dm_col].to_numpy(float)
    y_all = dfp[cut_metric].to_numpy(float)
    reg_daymet = _add_regression(ax, x_dm_all, y_all)

    # Right: ET vs cut metric
    ax2 = axes[1]
    for c in counties:
        sub = dfp[dfp["county"] == c]
        ax2.scatter(
            sub[et_col].to_numpy(float),
            sub[cut_metric].to_numpy(float),
            alpha=0.6, edgecolor="none", c=[ccolors[c]],
        )
    ax2.set_xlabel(et_label)
    ax2.set_title(f"{ylab} vs ET\n{wy_label}".strip())
    _style_axes_full_border(ax2)

    # Pooled regression on right panel
    x_et_all = dfp[et_col].to_numpy(float)
    reg_et = _add_regression(ax2, x_et_all, y_all)

    # Legend
    handles = []
    for c in counties:
        n_parcels = int(dfp.loc[dfp["county"] == c, "UniqueID"].nunique())
        label = (
            f"{c} (n={n_parcels})" if show_count_n_in_legend else c
        )
        handles.append(
            mpl.lines.Line2D(
                [0], [0], marker="o", linestyle="",
                color=ccolors[c], label=label,
            )
        )
    ax2.legend(
        handles=handles, title="County", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()

    summary = {
        "daymet_var": daymet_var,
        "cut_metric": cut_metric,
        "wy_label": wy_label,
        "n_counties": len(counties),
        "n_parcels": int(dfp["UniqueID"].nunique()),
        "n_rows": len(dfp),
        "n_filtered_low_et": n_filtered_low_et,
        "min_et_mm": min_et_mm,
        "reg_daymet": reg_daymet,
        "reg_et": reg_et,
    }

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        summary["outfile"] = str(outfile)
        print(f"Multicounty scatter saved: {outfile}")

    return fig, axes, summary
