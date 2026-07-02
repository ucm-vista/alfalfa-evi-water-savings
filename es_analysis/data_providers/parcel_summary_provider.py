"""Per-parcel per-WY summary provider.

Orchestrates EVI, ET, Daymet, and GDD data into a unified parcel-year
DataFrame suitable for scatter plots, regression, and aggregation.

Source: alfalfa_evi_jovyan.py lines 10544-10665, 11588-11598,
        11999-12187
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import config
from .evi_provider import normalize_county_name, water_year_bounds
from .et_provider import (
    _load_seasonal_csv,
    _load_openet_for_wy,
    _parse_cp_dates_iso,
    compute_daily_and_monthly_for_uid,
)
from .evi_cut_window_provider import (
    load_evi_for_wy,
    compute_cut_cycle_segments,
    merge_segment_gaps,
    segment_et_sums_mm,
    filter_segments_by_et,
    resolve_evi_col,
)
from ..utils.whittaker import whittaker_smooth_series
from .spatial_provider import load_parcels_for_county, COUNTY_ORDER
from .daymet_provider import (
    compute_daymet_mean_for_parcels,
    compute_gdd5_for_parcels_cut_window,
    compute_daily_gdd5_for_parcels,
)
from .evi_cut_window_provider import calibrate_cutting_thresholds


# ---------------------------------------------------------------------------
# Segment ET summation (total across all segments)
# ---------------------------------------------------------------------------

def _sum_over_segments(
    series: pd.Series,
    segs: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    """Sum a daily series over a list of (start, end) segments.

    Any NaN day within a segment makes the entire result NaN,
    since a partial sum would understate cumulative ET.

    After merge_segment_gaps, consecutive segments share a boundary day
    (seg_i ends at seg_{i+1}'s start).  To avoid double-counting that
    day, each segment after the first uses an exclusive start bound.

    Args:
        series: Daily time series (e.g., ET in mm/day).
        segs: List of (start, end) tuples.

    Returns:
        Total sum across all segments, or NaN if any day is missing.
    """
    if series is None or series.empty:
        return np.nan
    if not segs:
        return np.nan
    total = 0.0
    prev_end = None
    for s, e in segs:
        if pd.isna(s) or pd.isna(e) or s > e:
            continue
        # If this segment starts exactly where the previous one ended,
        # skip the boundary day to avoid double-counting.
        if prev_end is not None and pd.to_datetime(s) <= pd.to_datetime(prev_end):
            s = pd.to_datetime(s) + pd.Timedelta(days=1)
            if s > pd.to_datetime(e):
                prev_end = e
                continue
        seg_data = series.loc[s:e]
        if seg_data.isna().any():
            return np.nan
        total += float(seg_data.sum())
        prev_end = e
    return float(total) if total > 0 else np.nan


# ---------------------------------------------------------------------------
# Three-layer cut recovery
# ---------------------------------------------------------------------------

def _spacing_filter(
    dates: List[pd.Timestamp],
    min_days: int = 22,
) -> List[pd.Timestamp]:
    """Keep only dates separated by at least *min_days*."""
    if not dates:
        return []
    out = [dates[0]]
    for d in dates[1:]:
        if (d - out[-1]).days >= min_days:
            out.append(d)
    return out


def recover_cut_dates(
    beast_row: pd.Series,
    county: str,
    wy: int,
    evi_series: Optional[pd.Series] = None,
    min_spacing_days: int = 22,
) -> List[pd.Timestamp]:
    """Recover cutting dates using a three-layer strategy.

    Layer 1 — Hybrid detection (from BEAST CSV columns):
      * Consensus parcels (fallback_used == 0): use ALL BEAST CPs
        directly, skipping the EVI-minimum alignment that drops CPs
        without a nearby trough.
      * Fallback parcels (fallback_used >= 1): UNION of BEAST CPs
        and EVI fallback minima (``matched_minima_iso``).

    Layer 2 — Runtime EVI re-detection (for fallback parcels only):
      Re-detect local minima in the smoothed EVI series with
      county-specific relaxed thresholds (lower ``delta_min`` and
      higher ``min_evi_max`` for desert counties).  Unions the
      result with Layer 1 dates.

    Both layers share a final deduplication pass that enforces
    *min_spacing_days* between consecutive cut dates.

    Args:
        beast_row: One row from the BEAST seasonal CSV (must contain
            ``season_cp_dates_iso``, ``matched_minima_iso``,
            ``fallback_used``).
        county: County name (used to select desert vs CV thresholds).
        wy: Water year.
        evi_series: Optional daily smoothed EVI (DatetimeIndex).
            Required for Layer 2.
        min_spacing_days: Minimum days between kept cut dates.

    Returns:
        Sorted list of ``pd.Timestamp`` cut dates.
    """
    from scipy.signal import find_peaks

    def _parse_dates(col_name: str) -> List[pd.Timestamp]:
        raw = str(beast_row.get(col_name, ""))
        if raw in ("", "nan", "None") or pd.isna(beast_row.get(col_name)):
            return []
        out = []
        for x in raw.split(";"):
            x = x.strip()
            if x:
                try:
                    out.append(pd.Timestamp(x))
                except Exception:
                    pass
        return sorted(out)

    fallback = int(beast_row.get("fallback_used", 2))

    # ------------------------------------------------------------------
    # Layer 1: hybrid detection
    # ------------------------------------------------------------------
    if fallback == 0:
        # Consensus parcel — trust ALL BEAST CPs directly
        candidates = set(_parse_dates("season_cp_dates_iso"))
    else:
        # Fallback parcel — union of CPs + EVI minima
        candidates = set(_parse_dates("season_cp_dates_iso"))
        candidates |= set(_parse_dates("matched_minima_iso"))

    # ------------------------------------------------------------------
    # Layer 2: runtime EVI re-detection (fallback parcels only)
    # ------------------------------------------------------------------
    if fallback >= 1 and evi_series is not None and len(evi_series) > 30:
        county_norm = normalize_county_name(county)
        is_desert = county_norm in [
            normalize_county_name(c) for c in config.desert_counties
        ]
        delta = config.desert_delta_min if is_desert else config.cv_delta_min
        evi_max = config.desert_min_evi_max if is_desert else config.cv_min_evi_max

        evi_vals = evi_series.values.astype(float)
        evi_idx = evi_series.index
        if np.isfinite(evi_vals).sum() > 20:
            min_idx, _ = find_peaks(-evi_vals)
            max_idx, _ = find_peaks(evi_vals)
            for i in min_idx:
                y_min = evi_vals[i]
                if not np.isfinite(y_min) or y_min > evi_max:
                    continue
                # Require a preceding peak with sufficient drop
                window_start = evi_idx[i] - pd.Timedelta(days=config.peak_window_days)
                prev_peaks = [
                    j for j in max_idx
                    if evi_idx[j] < evi_idx[i] and evi_idx[j] >= window_start
                ]
                if not prev_peaks:
                    continue
                j_peak = max(prev_peaks)
                y_peak = evi_vals[j_peak]
                if not np.isfinite(y_peak) or (y_peak - y_min) < delta:
                    continue
                candidates.add(evi_idx[i])

    # ------------------------------------------------------------------
    # Deduplicate with minimum spacing
    # ------------------------------------------------------------------
    if not candidates:
        return []
    return _spacing_filter(sorted(candidates), min_days=min_spacing_days)


# ---------------------------------------------------------------------------
# Core: build parcel-year DataFrame for one county + WY
# ---------------------------------------------------------------------------

def build_parcel_summary_wy(
    county: str,
    wy: int,
    daymet_var: str,
    month_start: int,
    month_end: int,
    filter_n_cuttings: Optional[Iterable[int]] = None,
    filter_n_cp_season: Optional[Iterable[int]] = None,
    apply_month_window_to_et_gdd: bool = False,
    *,
    evi_mode: str = None,
    summer_lookback_range_days: Tuple[int, int] = None,
    winter_lookback_range_days: Tuple[int, int] = None,
    min_gap_before_cut_days: int = None,
    min_segment_days: int = None,
) -> pd.DataFrame:
    """Build a parcel-year DataFrame for one county and water year.

    For each parcel in the BEAST seasonal CSV:
      1. Parse cut dates (prefer matched_minima_iso, fall back to season_cp_dates_iso)
      2. Load OpenET ETa and EVI for the water year
      3. Compute cut-cycle segments (EVI-preferred lookback)
      4. Sum ET over segments
      5. Compute Daymet variable (or GDD5) over the same segments

    Args:
        county: County name (will be normalized).
        wy: Water year.
        daymet_var: Daymet variable name (e.g., "tmax", "gdd5").
        month_start: Start month for windowing (1-12).
        month_end: End month for windowing (1-12).
        filter_n_cuttings: If set, keep only parcels with n_cuttings in this set.
        filter_n_cp_season: If set, keep only parcels with n_cp_season in this set.
        apply_month_window_to_et_gdd: If True, intersect segments with month window.
        evi_mode: EVI column to use ("smoothed" or "gapfilled").
        summer_lookback_range_days: (min, max) days for spring/summer lookback.
        winter_lookback_range_days: (min, max) days for winter lookback.
        min_gap_before_cut_days: Minimum gap before cut date in lookback.
        min_segment_days: Minimum segment length.

    Returns:
        DataFrame with columns: UniqueID, county, WY, n_cuttings,
        n_cp_season, et_cum_minET_to_last_cut_mm, {daymet_var}_mean.
    """
    if evi_mode is None:
        evi_mode = config.evi_mode
    if summer_lookback_range_days is None:
        summer_lookback_range_days = config.summer_lookback_range_days
    if winter_lookback_range_days is None:
        winter_lookback_range_days = config.winter_lookback_range_days
    if min_gap_before_cut_days is None:
        min_gap_before_cut_days = config.min_gap_before_cut_days
    if min_segment_days is None:
        min_segment_days = config.min_segment_days

    county_norm = normalize_county_name(county)
    wy_start, wy_end = water_year_bounds(wy)

    # 1. Load BEAST seasonal CSV
    df_seasonal = _load_seasonal_csv(county_norm, wy)
    df_seasonal["UniqueID"] = df_seasonal["UniqueID"].astype(str)

    if filter_n_cuttings is not None:
        fc = set(int(x) for x in filter_n_cuttings)
        df_seasonal = df_seasonal[df_seasonal["n_cuttings"].isin(fc)]
    if filter_n_cp_season is not None:
        fn = set(int(x) for x in filter_n_cp_season)
        df_seasonal = df_seasonal[df_seasonal["n_cp_season"].isin(fn)]

    if df_seasonal.empty:
        raise ValueError("No parcels in seasonal CSV after applying filters.")

    # 2. Load parcel geometries (needed for Daymet spatial mapping)
    gdf_parcels = load_parcels_for_county(county_norm)
    uids_seasonal = set(df_seasonal["UniqueID"].astype(str))
    gdf_parcels = gdf_parcels[gdf_parcels["UniqueID"].isin(uids_seasonal)].copy()
    if gdf_parcels.empty:
        raise ValueError(
            "No parcels in shapefile match seasonal CSV (after filters)."
        )

    uids = sorted(gdf_parcels["UniqueID"].unique().tolist())

    # 3. Load OpenET ETa and EVI for the water year
    et_series_dict = _load_openet_for_wy(county_norm, wy, uids)
    evi_series_dict = load_evi_for_wy(county_norm, wy, uids, evi_mode=evi_mode)

    # Compute Whittaker-smoothed EVI for adaptive interval estimation
    # Uses lighter lambda than BEAST smoothing to preserve cutting troughs
    whittaker_evi_dict = {}
    if config.use_whittaker_interval:
        for uid_str, evi_s in evi_series_dict.items():
            if evi_s is not None and not evi_s.empty and evi_s.notna().any():
                whittaker_evi_dict[uid_str] = whittaker_smooth_series(
                    evi_s, lmbda=config.whittaker_interval_lambda, d=config.whittaker_order,
                )
            else:
                whittaker_evi_dict[uid_str] = None

    # 4. Compute ET segments per parcel
    parcel_windows: Dict[
        str,
        Union[
            Tuple[pd.Timestamp, pd.Timestamp],
            List[Tuple[pd.Timestamp, pd.Timestamp]],
        ],
    ] = {}
    et_sums: Dict[str, float] = {}

    rows_base: List[Dict[str, object]] = []
    for uid, sub in df_seasonal.groupby("UniqueID", sort=False):
        uid_str = str(uid)

        # Three-layer cut recovery: hybrid detection + EVI re-detection
        beast_row = sub.iloc[0]
        et_series = et_series_dict.get(uid_str)
        evi_series = evi_series_dict.get(uid_str)

        cp_dates = recover_cut_dates(
            beast_row, county_norm, wy,
            evi_series=evi_series,
            min_spacing_days=config.min_spacing_days,
        )
        cp_dates = sorted([d for d in cp_dates if (wy_start <= d <= wy_end)])

        n_cuttings = len(cp_dates)
        n_cp_season = beast_row.get("n_cp_season", 0)

        if et_series is None or not cp_dates:
            parcel_windows[uid_str] = []
            et_sums[uid_str] = np.nan
        else:
            segs = compute_cut_cycle_segments(
                cut_dates=cp_dates,
                wy=wy,
                month_start=month_start,
                month_end=month_end,
                apply_month_window_to_et_gdd=apply_month_window_to_et_gdd,
                evi=evi_series,
                whittaker_evi=whittaker_evi_dict.get(uid_str),
                summer_lookback_range_days=summer_lookback_range_days,
                winter_lookback_range_days=winter_lookback_range_days,
                min_gap_before_cut_days=min_gap_before_cut_days,
                min_segment_days=min_segment_days,
                rise_days=config.rise_days,
                rise_eps=config.rise_eps,
            )
            # Layer 3: close post-harvest gaps
            segs = merge_segment_gaps(
                segs,
                max_extension_days=config.max_segment_gap_extension_days,
            )
            parcel_windows[uid_str] = segs
            et_sums[uid_str] = _sum_over_segments(et_series, segs)

        rows_base.append({
            "UniqueID": uid_str,
            "county": county_norm,
            "WY": int(wy),
            "n_cuttings": n_cuttings,
            "n_cp_season": n_cp_season,
        })

    df_base = pd.DataFrame(rows_base)
    df_base["et_cum_minET_to_last_cut_mm"] = df_base["UniqueID"].map(et_sums)

    # 5. Daymet variable aligned to the same segment dates as ET
    dm_col = f"{daymet_var}_mean"
    if daymet_var.lower() == "gdd5":
        gdd_vals = compute_gdd5_for_parcels_cut_window(
            wy=wy,
            parcels=gdf_parcels,
            parcel_windows=parcel_windows,
        )
        df_base[dm_col] = df_base["UniqueID"].map(gdd_vals)
    else:
        daymet_means = compute_daymet_mean_for_parcels(
            var=daymet_var,
            wy=wy,
            parcels=gdf_parcels,
            month_start=month_start,
            month_end=month_end,
            parcel_windows=parcel_windows,
        )
        df_base[dm_col] = df_base["UniqueID"].map(daymet_means)

    df_base = df_base.dropna(
        subset=[dm_col, "et_cum_minET_to_last_cut_mm"], how="any"
    )
    return df_base


# ---------------------------------------------------------------------------
# Matched-minima variant: segments built from matched_minima_iso
# ---------------------------------------------------------------------------

def build_parcel_summary_matched(
    county: str,
    wy: int,
    daymet_var: str = "gdd5",
    month_start: int = 10,
    month_end: int = 9,
    evi_mode: str = None,
    et_mode: str = "actual",
    method: str = "A",
) -> pd.DataFrame:
    """Build parcel-year summary using matched_minima_iso for segments.

    Unlike build_parcel_summary_wy() which uses season_cp_dates_iso (all BEAST
    change points), this function uses matched_minima_iso (physical cutting
    dates) for segment construction.  This ensures ET and GDD5 are summed
    over exactly the cutting cycles that correspond to n_cuttings.

    Args:
        et_mode: "actual" (default), "corrected", or "both".
            When "corrected" or "both", computes corrected ET per parcel
            using compute_daily_and_monthly_for_uid() and sums over the
            same segments.
        method: ET correction method "A" or "B" (only used when et_mode
            is "corrected" or "both").

    Returns:
        DataFrame with columns: UniqueID, county, WY, n_cuttings,
        n_cp_season, et_cum_minET_to_last_cut_mm, {daymet_var}_mean.
        When et_mode is "corrected" or "both", also includes
        et_cum_corrected_mm.
    """
    if evi_mode is None:
        evi_mode = config.evi_mode

    county_norm = normalize_county_name(county)
    wy_start, wy_end = water_year_bounds(wy)

    df_seasonal = _load_seasonal_csv(county_norm, wy)
    df_seasonal["UniqueID"] = df_seasonal["UniqueID"].astype(str)

    if df_seasonal.empty:
        raise ValueError(f"No parcels in seasonal CSV for {county_norm} WY{wy}.")

    gdf_parcels = load_parcels_for_county(county_norm)
    uids_seasonal = set(df_seasonal["UniqueID"].astype(str))
    gdf_parcels = gdf_parcels[gdf_parcels["UniqueID"].isin(uids_seasonal)].copy()
    uids = sorted(gdf_parcels["UniqueID"].unique().tolist())

    et_series_dict = _load_openet_for_wy(county_norm, wy, uids)
    evi_series_dict = load_evi_for_wy(county_norm, wy, uids, evi_mode=evi_mode)

    whittaker_evi_dict = {}
    if config.use_whittaker_interval:
        for uid_str, evi_s in evi_series_dict.items():
            if evi_s is not None and not evi_s.empty and evi_s.notna().any():
                whittaker_evi_dict[uid_str] = whittaker_smooth_series(
                    evi_s, lmbda=config.whittaker_interval_lambda, d=config.whittaker_order,
                )

    # Load daily GDD5 for thermal-time segment estimation
    daily_gdd5_dict: Dict[str, pd.Series] = {}
    gdd5_thresholds = None
    if config.use_thermal_time_interval:
        try:
            _, daily_gdd5_dict = compute_daily_gdd5_for_parcels(
                wy=wy, parcels=gdf_parcels,
            )
        except Exception:
            daily_gdd5_dict = {}

        # Parse all cut dates for calibration
        if daily_gdd5_dict:
            cut_dates_by_parcel: Dict[str, List] = {}
            for uid, sub in df_seasonal.groupby("UniqueID", sort=False):
                uid_str = str(uid)
                cut_s = ""
                if "matched_minima_iso" in sub.columns:
                    v = sub["matched_minima_iso"].iloc[0]
                    if not pd.isna(v) and str(v).strip():
                        cut_s = str(v)
                if not cut_s.strip():
                    cut_s = str(sub.get("season_cp_dates_iso", pd.Series([""])).iloc[0])
                dates = _parse_cp_dates_iso(cut_s)
                dates = sorted([d for d in dates if (wy_start <= d <= wy_end)])
                if len(dates) >= 2:
                    cut_dates_by_parcel[uid_str] = dates

            gdd5_thresholds = calibrate_cutting_thresholds(
                cut_dates_by_parcel=cut_dates_by_parcel,
                daily_gdd5_by_parcel=daily_gdd5_dict,
                daily_et_by_parcel=et_series_dict,
                trusted_summer_range=config.thermal_time_trusted_summer_range,
                trusted_winter_range=config.thermal_time_trusted_winter_range,
            )

    parcel_windows: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    et_sums: Dict[str, float] = {}
    rows_base: List[Dict[str, object]] = []

    for uid, sub in df_seasonal.groupby("UniqueID", sort=False):
        uid_str = str(uid)

        # Three-layer cut recovery
        beast_row = sub.iloc[0]
        et_series = et_series_dict.get(uid_str)
        evi_series = evi_series_dict.get(uid_str)

        cp_dates = recover_cut_dates(
            beast_row, county_norm, wy,
            evi_series=evi_series,
            min_spacing_days=config.min_spacing_days,
        )
        cp_dates = sorted([d for d in cp_dates if (wy_start <= d <= wy_end)])

        n_cuttings = len(cp_dates)
        n_cp_season = beast_row.get("n_cp_season", 0)

        if et_series is None or not cp_dates:
            parcel_windows[uid_str] = []
            et_sums[uid_str] = np.nan
        else:
            segs = compute_cut_cycle_segments(
                cut_dates=cp_dates,
                wy=wy,
                month_start=month_start,
                month_end=month_end,
                apply_month_window_to_et_gdd=False,
                evi=evi_series,
                whittaker_evi=whittaker_evi_dict.get(uid_str),
                daily_gdd5=daily_gdd5_dict.get(uid_str),
                gdd5_thresholds=gdd5_thresholds,
                summer_lookback_range_days=config.summer_lookback_range_days,
                winter_lookback_range_days=config.winter_lookback_range_days,
                min_gap_before_cut_days=config.min_gap_before_cut_days,
                min_segment_days=config.min_segment_days,
                rise_days=config.rise_days,
                rise_eps=config.rise_eps,
            )
            # Layer 3: close post-harvest gaps
            segs = merge_segment_gaps(
                segs,
                max_extension_days=config.max_segment_gap_extension_days,
            )
            parcel_windows[uid_str] = segs
            et_sums[uid_str] = _sum_over_segments(et_series, segs)

        rows_base.append({
            "UniqueID": uid_str,
            "county": county_norm,
            "WY": int(wy),
            "n_cuttings": n_cuttings,
            "n_cp_season": n_cp_season,
        })

    df_base = pd.DataFrame(rows_base)
    if config.min_cuttings > 0:
        df_base = df_base[df_base["n_cuttings"] >= config.min_cuttings].copy()
    df_base["et_cum_minET_to_last_cut_mm"] = df_base["UniqueID"].map(et_sums)

    # Compute corrected ET when requested
    if et_mode in ("corrected", "both"):
        corr_sums: Dict[str, float] = {}
        method_upper = str(method).strip().upper()
        for uid_str in df_base["UniqueID"].unique():
            segs = parcel_windows.get(uid_str, [])
            if not segs:
                corr_sums[uid_str] = np.nan
                continue
            try:
                daily_df, _, _, _ = compute_daily_and_monthly_for_uid(
                    county=county_norm, wy=wy, uid=uid_str,
                    chosen_method=method_upper, n_boot=50,
                )
                corr_series = (daily_df["ET_open"] - daily_df["delta_corr"]).clip(lower=0.0)
                corr_sums[uid_str] = _sum_over_segments(corr_series, segs)
            except Exception as exc:
                print(f"  [warn] corrected ET failed for {uid_str}: {exc}")
                corr_sums[uid_str] = np.nan
        df_base["et_cum_corrected_mm"] = df_base["UniqueID"].map(corr_sums)

    # Flag parcels where matched cuttings are much fewer than total CPs
    df_base["cut_match_ratio"] = np.where(
        df_base["n_cp_season"] > 0,
        df_base["n_cuttings"] / df_base["n_cp_season"],
        np.nan,
    )

    dm_col = f"{daymet_var}_mean"
    if daymet_var.lower() == "gdd5":
        gdd_vals = compute_gdd5_for_parcels_cut_window(
            wy=wy, parcels=gdf_parcels, parcel_windows=parcel_windows,
        )
        df_base[dm_col] = df_base["UniqueID"].map(gdd_vals)
    else:
        daymet_means = compute_daymet_mean_for_parcels(
            var=daymet_var, wy=wy, parcels=gdf_parcels,
            month_start=month_start, month_end=month_end,
            parcel_windows=parcel_windows,
        )
        df_base[dm_col] = df_base["UniqueID"].map(daymet_means)

    drop_cols = [dm_col, "et_cum_minET_to_last_cut_mm"]
    if et_mode in ("corrected", "both"):
        drop_cols.append("et_cum_corrected_mm")
    df_base = df_base.dropna(subset=drop_cols, how="any")
    return df_base


def build_multicounty_matched(
    *,
    daymet_var: str = "gdd5",
    month_start: int = 10,
    month_end: int = 9,
    wy_start: int = 2019,
    wy_end: int = 2024,
    counties: Optional[List[str]] = None,
    evi_mode: str = None,
    et_mode: str = "actual",
    method: str = "A",
) -> pd.DataFrame:
    """Multi-county wrapper for build_parcel_summary_matched."""
    if counties is None:
        counties = list(COUNTY_ORDER)
    frames: List[pd.DataFrame] = []
    for c in counties:
        for wy in range(wy_start, wy_end + 1):
            try:
                df = build_parcel_summary_matched(
                    county=c, wy=wy, daymet_var=daymet_var,
                    month_start=month_start, month_end=month_end,
                    evi_mode=evi_mode, et_mode=et_mode, method=method,
                )
                frames.append(df)
            except (FileNotFoundError, ValueError) as e:
                print(f"[info] Skipping {c}, WY{wy}: {e}")
    if not frames:
        raise ValueError("No data available for requested counties/WY range.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Multi-county, multi-year wrapper
# ---------------------------------------------------------------------------

def build_multicounty_df_parcel_year(
    *,
    daymet_var: str,
    month_start: int,
    month_end: int,
    wy_start: int,
    wy_end: int,
    counties: Optional[List[str]] = None,
    filter_n_cuttings: Optional[Iterable[int]] = None,
    filter_n_cp_season: Optional[Iterable[int]] = None,
    apply_month_window_to_et_gdd: bool = False,
    evi_mode: str = None,
    summer_lookback_range_days: Tuple[int, int] = None,
    winter_lookback_range_days: Tuple[int, int] = None,
    min_gap_before_cut_days: int = None,
    min_segment_days: int = None,
) -> pd.DataFrame:
    """Build parcel-year DataFrame across multiple counties and water years.

    Iterates over all county/WY combinations and concatenates results.
    Skips county/WY pairs where data is missing or filtering yields no parcels.

    Args:
        daymet_var: Daymet variable name (e.g., "tmax", "gdd5").
        month_start: Start month for windowing (1-12).
        month_end: End month for windowing (1-12).
        wy_start: First water year (inclusive).
        wy_end: Last water year (inclusive).
        counties: List of county names. Defaults to COUNTY_ORDER.
        filter_n_cuttings: Optional set of n_cuttings values to keep.
        filter_n_cp_season: Optional set of n_cp_season values to keep.
        apply_month_window_to_et_gdd: If True, intersect segments with month window.
        evi_mode: EVI column to use.
        summer_lookback_range_days: Lookback band for spring/summer.
        winter_lookback_range_days: Lookback band for winter.
        min_gap_before_cut_days: Minimum gap before cut in lookback.
        min_segment_days: Minimum segment length.

    Returns:
        Concatenated parcel-year DataFrame.

    Raises:
        ValueError: If wy_start > wy_end or no data is available.
    """
    if wy_start > wy_end:
        raise ValueError("wy_start must be <= wy_end")
    if counties is None:
        counties = list(COUNTY_ORDER)

    frames: List[pd.DataFrame] = []
    for c in counties:
        for wy in range(wy_start, wy_end + 1):
            try:
                df = build_parcel_summary_wy(
                    county=c,
                    wy=wy,
                    daymet_var=daymet_var,
                    month_start=month_start,
                    month_end=month_end,
                    filter_n_cuttings=filter_n_cuttings,
                    filter_n_cp_season=filter_n_cp_season,
                    apply_month_window_to_et_gdd=apply_month_window_to_et_gdd,
                    evi_mode=evi_mode,
                    summer_lookback_range_days=summer_lookback_range_days,
                    winter_lookback_range_days=winter_lookback_range_days,
                    min_gap_before_cut_days=min_gap_before_cut_days,
                    min_segment_days=min_segment_days,
                )
                frames.append(df)
            except (FileNotFoundError, ValueError) as e:
                print(f"[info] Skipping {c}, WY{wy}: {e}")
                continue

    if not frames:
        raise ValueError(
            "No parcel-year data available for requested counties/WY range."
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Debug helper: single-parcel segment inspection
# ---------------------------------------------------------------------------

def debug_one_parcel(
    county: str,
    wy: int,
    unique_id: str,
    evi_mode: str = None,
    month_start: int = 10,
    month_end: int = 9,
    apply_month_window_to_et_gdd: bool = False,
    summer_lookback_range_days: Tuple[int, int] = None,
    winter_lookback_range_days: Tuple[int, int] = None,
    min_gap_before_cut_days: int = None,
    min_segment_days: int = None,
    segment_et_filter_mode: str = None,
    segment_et_abs_min_mm: float = None,
    segment_et_rel_min_frac: float = None,
    do_plot: bool = True,
) -> pd.DataFrame:
    """Print and return a per-segment diagnostic table for a single parcel.

    For each raw segment, shows: index, start, end, days, seg_et_mm,
    and whether it passes the ET filter.

    Optionally plots EVI and ET time series with segment shading.

    Args:
        county: County name.
        wy: Water year.
        unique_id: Parcel UniqueID.
        evi_mode: EVI column to use.
        month_start: Start month for windowing.
        month_end: End month for windowing.
        apply_month_window_to_et_gdd: Intersect segments with month window.
        summer_lookback_range_days: Lookback band for spring/summer.
        winter_lookback_range_days: Lookback band for winter.
        min_gap_before_cut_days: Minimum gap before cut.
        min_segment_days: Minimum segment length.
        segment_et_filter_mode: ET filter mode ("none"/"absolute"/"relative"/"both").
        segment_et_abs_min_mm: Absolute ET threshold for filtering.
        segment_et_rel_min_frac: Relative ET threshold (fraction of median).
        do_plot: If True, plot EVI and ET with segment shading.

    Returns:
        DataFrame with one row per raw segment, including 'kept' flag.
    """
    if evi_mode is None:
        evi_mode = config.evi_mode
    if summer_lookback_range_days is None:
        summer_lookback_range_days = config.summer_lookback_range_days
    if winter_lookback_range_days is None:
        winter_lookback_range_days = config.winter_lookback_range_days
    if min_gap_before_cut_days is None:
        min_gap_before_cut_days = config.min_gap_before_cut_days
    if min_segment_days is None:
        min_segment_days = config.min_segment_days
    if segment_et_filter_mode is None:
        segment_et_filter_mode = config.segment_et_filter_mode
    if segment_et_abs_min_mm is None:
        segment_et_abs_min_mm = config.segment_et_abs_min_mm
    if segment_et_rel_min_frac is None:
        segment_et_rel_min_frac = config.segment_et_rel_min_frac

    county_norm = normalize_county_name(county)
    uid = str(unique_id)

    # Load seasonal CSV and find this parcel
    df_seas = _load_seasonal_csv(county_norm, wy)
    sub = df_seas[df_seas["UniqueID"].astype(str) == uid].copy()
    if sub.empty:
        raise ValueError(
            f"UniqueID={uid} not found in seasonal CSV for "
            f"{county_norm} WY{wy}"
        )

    wy_start, wy_end = water_year_bounds(wy)

    # Parse cut dates: prefer matched_minima_iso, then season_cp_dates_iso
    cut_dates_s = ""
    if "matched_minima_iso" in sub.columns:
        v = sub["matched_minima_iso"].iloc[0]
        if (not pd.isna(v)) and str(v).strip():
            cut_dates_s = v
    if not str(cut_dates_s).strip():
        cut_dates_s = (
            sub["season_cp_dates_iso"].iloc[0]
            if "season_cp_dates_iso" in sub.columns
            else ""
        )
    cut_dates = sorted([
        d for d in _parse_cp_dates_iso(cut_dates_s)
        if wy_start <= d <= wy_end
    ])

    # Load time series
    et = _load_openet_for_wy(county_norm, wy, [uid]).get(uid)
    evi = load_evi_for_wy(county_norm, wy, [uid], evi_mode=evi_mode).get(uid)

    # Compute Whittaker-smoothed EVI for adaptive intervals (light lambda)
    w_evi = None
    if config.use_whittaker_interval and evi is not None and not evi.empty and evi.notna().any():
        w_evi = whittaker_smooth_series(
            evi, lmbda=config.whittaker_interval_lambda, d=config.whittaker_order,
        )

    if et is None:
        raise ValueError("ET series not found/loaded for this parcel.")
    if not cut_dates:
        raise ValueError(
            "No cut dates found for this parcel (matched minima / CP dates)."
        )

    # Compute raw segments
    segs_raw = compute_cut_cycle_segments(
        cut_dates=cut_dates,
        wy=wy,
        month_start=month_start,
        month_end=month_end,
        apply_month_window_to_et_gdd=apply_month_window_to_et_gdd,
        evi=evi,
        whittaker_evi=w_evi,
        summer_lookback_range_days=summer_lookback_range_days,
        winter_lookback_range_days=winter_lookback_range_days,
        min_gap_before_cut_days=min_gap_before_cut_days,
        min_segment_days=min_segment_days,
    )
    seg_et_raw = segment_et_sums_mm(et, segs_raw)

    # Apply ET filter
    segs_kept, diag = filter_segments_by_et(
        et=et,
        segs=segs_raw,
        mode=segment_et_filter_mode,
        abs_min_mm=segment_et_abs_min_mm,
        rel_min_frac_of_median=segment_et_rel_min_frac,
    )

    # Build diagnostic table
    raw_rows = []
    for i, (s, e) in enumerate(segs_raw):
        raw_rows.append({
            "i": i,
            "start": pd.to_datetime(s),
            "end": pd.to_datetime(e),
            "days": int((pd.to_datetime(e) - pd.to_datetime(s)).days) + 1,
            "seg_et_mm": (
                float(seg_et_raw[i]) if i < len(seg_et_raw) else np.nan
            ),
        })
    dfr = pd.DataFrame(raw_rows)

    kept_set = set(
        (pd.to_datetime(s).normalize(), pd.to_datetime(e).normalize())
        for s, e in segs_kept
    )
    dfr["kept"] = [
        (r["start"].normalize(), r["end"].normalize()) in kept_set
        for _, r in dfr.iterrows()
    ]

    print(f"== DEBUG ONE PARCEL: {county_norm} WY{wy} UniqueID={uid} ==")
    print(
        f"cut_dates ({len(cut_dates)}): "
        f"{[d.date().isoformat() for d in cut_dates]}"
    )
    print("filter diag:", diag)
    print(dfr)

    if do_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # EVI panel
        axes[0].plot(evi.index, evi.values, linewidth=1.2)
        axes[0].set_ylabel(f"EVI ({resolve_evi_col(evi_mode)})")
        axes[0].set_title("EVI with segment windows (kept shaded)")

        # ET panel
        axes[1].plot(et.index, et.values, linewidth=1.2)
        axes[1].set_ylabel("OpenET ET (mm/day)")
        axes[1].set_title("ET with segment windows (kept shaded)")

        for (s, e) in segs_kept:
            axes[0].axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.15)
            axes[1].axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.15)

        for c in cut_dates:
            axes[0].axvline(pd.to_datetime(c), linestyle="--", linewidth=1.0)
            axes[1].axvline(pd.to_datetime(c), linestyle="--", linewidth=1.0)

        for ax in axes:
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.0)

        fig.tight_layout()
        plt.show()

    return dfr
