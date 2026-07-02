"""Single-county Daymet vs cuttings scatter plot.

Side-by-side scatter of Daymet variable vs cutting metric (left)
and ET vs cutting metric (right) for one county.

Source: alfalfa_evi_jovyan.py lines 9989-10058
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data_providers.evi_provider import normalize_county_name
from ...utils.validation import MIN_CUMULATIVE_ET_MM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def daymet_scatter_plot(
    df: pd.DataFrame,
    county: str,
    wy_label: str,
    daymet_var: str,
    cut_metric: str = "n_cuttings",
    daymet_label_window: Optional[str] = None,
    min_et_mm: float = MIN_CUMULATIVE_ET_MM,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create a two-panel scatter for one county.

    Left panel:  Daymet variable vs cut metric
    Right panel: ET vs cut metric

    Args:
        df: DataFrame with columns: {cut_metric}, {daymet_var}_mean,
            et_cum_minET_to_last_cut_mm.
        county: County name.
        wy_label: Water year label for title.
        daymet_var: Daymet variable name (e.g. "gdd5", "tmax").
        cut_metric: "n_cp_season" or "n_cuttings".
        daymet_label_window: Optional window label for x-axis.
        min_et_mm: Minimum cumulative ET threshold (mm).
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    if cut_metric not in {"n_cuttings", "n_cp_season"}:
        raise ValueError("cut_metric must be 'n_cuttings' or 'n_cp_season'.")
    dm_col = f"{daymet_var}_mean"
    if dm_col not in df.columns:
        raise ValueError(f"DataFrame is missing column '{dm_col}'.")

    et_col = "et_cum_minET_to_last_cut_mm"
    et_label = (
        "Cumulative OpenET ET (mm)\n"
        "(parcel-specific cut-cycle windows)"
    )

    df_plot = df.dropna(subset=[cut_metric, dm_col, et_col]).copy()
    if df_plot.empty:
        raise ValueError("No data after dropping NaNs for chosen variables.")

    df_plot[cut_metric] = pd.to_numeric(
        df_plot[cut_metric], errors="coerce"
    ).round()

    # Filter low-quality ET rows
    n_before = len(df_plot)
    df_plot = df_plot[np.isfinite(df_plot[et_col].to_numpy(float))]
    df_plot = df_plot[df_plot[et_col] >= min_et_mm]
    n_filtered_low_et = n_before - len(df_plot)
    if df_plot.empty:
        raise ValueError(f"No data after filtering ET < {min_et_mm} mm.")

    county_norm = normalize_county_name(county)
    ylab = _metric_label(cut_metric)

    x_dm = df_plot[dm_col].to_numpy(float)
    x_et = df_plot[et_col].to_numpy(float)
    y = df_plot[cut_metric].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left: Daymet vs cut metric
    ax = axes[0]
    ax.scatter(x_dm, y, alpha=0.75, edgecolor="none")
    if daymet_var.lower() == "gdd5":
        xlab = "Mean GDD per cutting (base 5 \u00b0C)"
    else:
        xlab = f"Mean Daymet {daymet_var}"
    if daymet_label_window:
        xlab += f"\n{daymet_label_window}"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{county_norm} {wy_label}\n{ylab} vs {daymet_var}")
    _style_axes_full_border(ax)
    reg_daymet = _add_regression(ax, x_dm, y)

    # Right: ET vs cut metric
    ax2 = axes[1]
    ax2.scatter(x_et, y, alpha=0.75, edgecolor="none")
    ax2.set_xlabel(et_label)
    ax2.set_title(f"{county_norm} {wy_label}\n{ylab} vs ET")
    _style_axes_full_border(ax2)
    reg_et = _add_regression(ax2, x_et, y)

    fig.suptitle(
        f"Parcel-level scatter plots: {county_norm} {wy_label}",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    summary = {
        "county": county_norm,
        "wy_label": wy_label,
        "daymet_var": daymet_var,
        "cut_metric": cut_metric,
        "n_parcels": int(df_plot["UniqueID"].nunique())
            if "UniqueID" in df_plot.columns else len(df_plot),
        "n_rows": len(df_plot),
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
        print(f"Daymet scatter saved: {outfile}")

    return fig, axes, summary
