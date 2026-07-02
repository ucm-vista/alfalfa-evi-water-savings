"""
Statistics Data Provider

Handles aggregated statistics and summaries for county-level analysis.

Source: alfalfa_evi_jovyan.py lines 5219-5600+, 12243-12318
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


COUNTY_ORDER = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]


def aggregate_to_county_year_points(
    df_parcel_year: pd.DataFrame,
    *,
    daymet_var: str,
    cut_metric: str,
    counties: Optional[List[str]] = None,
    wy_start: Optional[int] = None,
    wy_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate parcel-year data to county-year points.

    Parameters
    ----------
    df_parcel_year : pd.DataFrame
        Parcel-year level data with columns: county, WY, UniqueID,
        {cut_metric}, et_cum_minET_to_last_cut_mm, {daymet_var}_mean
    daymet_var : str
        Daymet variable name (e.g., "tmax", "tmin", "gdd5")
    cut_metric : str
        Cutting metric name (e.g., "n_cp_season", "n_cuttings")
    counties : list of str, optional
        Filter to specific counties
    wy_start : int, optional
        Filter starting water year
    wy_end : int, optional
        Filter ending water year

    Returns
    -------
    pd.DataFrame
        County-year aggregated data with columns: county, WY, n_parcels,
        {cut_metric}, et_cum_minET_to_last_cut_mm, {daymet_var}_mean
    """
    dm_col = f"{daymet_var}_mean"
    need = {"county", "WY", "UniqueID", cut_metric, "et_cum_minET_to_last_cut_mm", dm_col}
    missing = [c for c in need if c not in df_parcel_year.columns]
    if missing:
        raise ValueError(f"df_parcel_year missing required columns: {missing}")

    df = df_parcel_year.copy()
    if counties is not None:
        df = df[df["county"].isin(counties)]
    if wy_start is not None and wy_end is not None:
        df = df[df["WY"].between(wy_start, wy_end)]
    if df.empty:
        raise ValueError("No data after filtering counties/WY range.")

    agg = (
        df.groupby(["county", "WY"], as_index=False)
          .agg(
              n_parcels=("UniqueID", "nunique"),
              cut_mean=(cut_metric, "mean"),
              et_mean=("et_cum_minET_to_last_cut_mm", "mean"),
              dm_mean=(dm_col, "mean"),
          )
    )
    agg[cut_metric] = agg["cut_mean"].round().astype("Int64")
    agg["et_cum_minET_to_last_cut_mm"] = agg["et_mean"]
    agg[dm_col] = agg["dm_mean"]

    return agg[["county", "WY", "n_parcels", cut_metric, "et_cum_minET_to_last_cut_mm", dm_col]]


def aggregate_to_county_overall_points(
    df_parcel_year: pd.DataFrame,
    *,
    daymet_var: str,
    cut_metric: str,
    counties: Optional[List[str]] = None,
    wy_start: Optional[int] = None,
    wy_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate parcel-year data to county-level overall points.

    Parameters
    ----------
    df_parcel_year : pd.DataFrame
        Parcel-year level data
    daymet_var : str
        Daymet variable name
    cut_metric : str
        Cutting metric name
    counties : list of str, optional
        Filter to specific counties
    wy_start : int, optional
        Filter starting water year
    wy_end : int, optional
        Filter ending water year

    Returns
    -------
    pd.DataFrame
        County level aggregated data with columns: county, n_parcels, n_years,
        {cut_metric}, et_cum_minET_to_last_cut_mm, {daymet_var}_mean
    """
    dm_col = f"{daymet_var}_mean"
    need = {"county", "WY", "UniqueID", cut_metric, "et_cum_minET_to_last_cut_mm", dm_col}
    missing = [c for c in need if c not in df_parcel_year.columns]
    if missing:
        raise ValueError(f"df_parcel_year missing required columns: {missing}")

    df = df_parcel_year.copy()
    if counties is not None:
        df = df[df["county"].isin(counties)]
    if wy_start is not None and wy_end is not None:
        df = df[df["WY"].between(wy_start, wy_end)]
    if df.empty:
        raise ValueError("No data after filtering counties/WY range.")

    agg = (
        df.groupby("county", as_index=False)
          .agg(
              n_parcels=("UniqueID", "nunique"),
              n_years=("WY", "nunique"),
              cut_mean=(cut_metric, "mean"),
              et_mean=("et_cum_minET_to_last_cut_mm", "mean"),
              dm_mean=(dm_col, "mean"),
          )
    )
    agg[cut_metric] = agg["cut_mean"].round().astype("Int64")
    agg["et_cum_minET_to_last_cut_mm"] = agg["et_mean"]
    agg[dm_col] = agg["dm_mean"]

    return agg[["county", "n_parcels", "n_years", cut_metric, "et_cum_minET_to_last_cut_mm", dm_col]]


def aggregate_to_county_year_points_four(
    df_parcel_year: pd.DataFrame,
    *,
    counties: Optional[List[str]] = None,
    wy_start: Optional[int] = None,
    wy_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate to county-year points for four-panel plot with all required metrics.

    Required columns in df_parcel_year:
    - county, WY
    - gdd5_cum, gdd5_mean_per_cut
    - et_cum_mm, et_mean_per_cut_mm
    - n_cp_season (or other cut metric)

    Parameters
    ----------
    df_parcel_year : pd.DataFrame
        Parcel-year level data
    counties : list of str, optional
        Filter to specific counties
    wy_start : int, optional
        Filter starting water year
    wy_end : int, optional
        Filter ending water year

    Returns
    -------
    pd.DataFrame
        County-year aggregated data with columns: county, WY, gdd5_cum,
        gdd5_mean_per_cut, et_cum_mm, et_mean_per_cut_mm, n_cp_season
    """
    cols = ["county", "WY", "gdd5_cum", "gdd5_mean_per_cut", "et_cum_mm", "et_mean_per_cut_mm", "n_cp_season"]
    missing = [c for c in cols if c not in df_parcel_year.columns]
    if missing:
        raise ValueError(f"df_parcel_year missing required columns: {missing}")

    df = df_parcel_year.copy()
    if counties is not None:
        df = df[df["county"].isin(counties)]
    if wy_start is not None and wy_end is not None:
        df = df[df["WY"].between(wy_start, wy_end)]
    if df.empty:
        raise ValueError("No data after filtering counties/WY range.")

    agg = (
        df.groupby(["county", "WY"], as_index=False)
          .agg(
              gdd5_cum=("gdd5_cum", "mean"),
              gdd5_mean_per_cut=("gdd5_mean_per_cut", "mean"),
              et_cum_mm=("et_cum_mm", "mean"),
              et_mean_per_cut_mm=("et_mean_per_cut_mm", "mean"),
              n_cp_season=("n_cp_season", "mean"),
          )
    )
    agg["n_cp_season"] = agg["n_cp_season"].round().astype("Int64")

    return agg[["county", "WY", "gdd5_cum", "gdd5_mean_per_cut", "et_cum_mm", "et_mean_per_cut_mm", "n_cp_season"]]


def _compute_regression_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute linear regression statistics.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable

    Returns
    -------
    dict
        Dictionary with keys: n, slope, intercept, r, r2
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = x.size
    if n < 2:
        return {"n": int(n), "slope": np.nan, "intercept": np.nan, "r": np.nan, "r2": np.nan}
    xm, ym = x.mean(), y.mean()
    cov = float(np.sum((x - xm) * (y - ym)))
    varx = float(np.sum((x - xm) ** 2))
    slope = np.nan if varx == 0 else cov / varx
    intercept = np.nan if not np.isfinite(slope) else (ym - slope * xm)
    r = np.nan if (np.std(x) == 0 or np.std(y) == 0) else float(np.corrcoef(x, y)[0, 1])
    r2 = float(r**2) if np.isfinite(r) else np.nan
    return {"n": int(n), "slope": slope, "intercept": intercept, "r": r, "r2": r2}


def _style_axes_full_border(ax: plt.Axes) -> None:
    """
    Style axes with full border and no grid.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to style
    """
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))


def _add_stats_text_top_left(ax: plt.Axes, stats: Dict[str, float]) -> None:
    """
    Add regression statistics text to top-left of axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    stats : dict
        Dictionary with keys: n, slope, r, r2
    """
    txt = (
        f"n = {stats['n']}\n"
        f"slope = {stats['slope']:.3g}\n"
        f"r = {stats['r']:.3f}\n"
        f"R² = {stats['r2']:.3f}"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )


def _county_color_map(counties: List[str]) -> Dict[str, tuple]:
    """
    Create a color map for counties.

    Parameters
    ----------
    counties : list of str
        List of county names

    Returns
    -------
    dict
        Dictionary mapping county names to RGB tuples
    """
    cmap = mpl.colormaps.get_cmap("tab10")
    return {c: cmap(i % 10) for i, c in enumerate(counties)}


def _marker_for_wy(wy: int) -> str:
    """
    Get marker symbol for water year.

    Parameters
    ----------
    wy : int
        Water year

    Returns
    -------
    str
        Matplotlib marker symbol
    """
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H", "d", "p", "8"]
    return marker_cycle[(wy - 2000) % len(marker_cycle)]