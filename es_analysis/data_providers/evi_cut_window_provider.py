"""EVI-based cut-window estimation provider.

Determines segment START dates using EVI lookback bands and constructs
cut-cycle segments for ET and Daymet/GDD aggregation.

Source: alfalfa_evi_jovyan.py lines 4054-4223, 9160-9478, 11364-11586
Uses the latest/most refined version of each function.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import config
from .evi_provider import normalize_county_name, water_year_bounds
from ..utils.whittaker import whittaker_smooth_series


# ---------------------------------------------------------------------------
# EVI column resolution
# ---------------------------------------------------------------------------

def resolve_evi_col(evi_mode: str) -> str:
    """Map EVI mode keyword to the actual column name in county-year CSVs.

    Args:
        evi_mode: One of "smoothed", "gapfilled", or an explicit column name.

    Returns:
        Column name string.
    """
    m = str(evi_mode).strip().lower()
    if m in {"smoothed", "smoothed_mean_evi"}:
        return "smoothed_mean_evi"
    if m in {"gapfilled", "gapfilled_mean_evi", "gap_filled"}:
        return "gapfilled_mean_evi"
    return evi_mode


# ---------------------------------------------------------------------------
# EVI loading for cut-window analysis
# ---------------------------------------------------------------------------

def load_evi_for_uid_wy(
    county: str,
    wy: int,
    uid: str,
    *,
    evi_mode: str = None,
) -> pd.Series:
    """Load daily EVI for a single UID for a water year.

    Reads from county_year_root_new/{County}/WY{wy}.csv and returns
    a daily series indexed Oct 1 .. Sep 30 with interpolation/clipping.

    Args:
        county: County name.
        wy: Water year.
        uid: Parcel unique identifier.
        evi_mode: EVI column to use ("smoothed" or "gapfilled").

    Returns:
        Daily EVI series indexed by date. All-NaN if file/column missing.
    """
    if evi_mode is None:
        evi_mode = config.evi_mode

    county_norm = normalize_county_name(county)
    uid_str = str(uid)
    wy_start, wy_end = water_year_bounds(wy)
    idx = pd.date_range(wy_start, wy_end, freq="D")

    p = Path(config.county_year_root_new) / county_norm / f"WY{wy}.csv"
    if not p.exists():
        return pd.Series(np.nan, index=idx, dtype=float)

    df = pd.read_csv(p, parse_dates=["date"])
    need = {"date", "parcel_id", "gapfilled_mean_evi", "smoothed_mean_evi"}
    miss = need - set(df.columns)
    if miss:
        return pd.Series(np.nan, index=idx, dtype=float)

    col = resolve_evi_col(evi_mode)
    if col not in df.columns:
        return pd.Series(np.nan, index=idx, dtype=float)

    df["UniqueID"] = df["parcel_id"].astype(str)
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= wy_start) & (df["date"] <= wy_end)]
    df = df[df["UniqueID"] == uid_str].copy()
    if df.empty:
        return pd.Series(np.nan, index=idx, dtype=float)

    df[col] = pd.to_numeric(df[col], errors="coerce")
    s = df.groupby("date", sort=True)[col].mean().astype(float).sort_index()
    s = s.reindex(idx)

    try:
        s = s.interpolate(method="time", limit=90, limit_direction="both")
    except Exception:
        s = s.interpolate(limit=90, limit_direction="both")
    s = s.clip(-0.05, 1.05)
    if s.isna().any():
        s = s.ffill().bfill()

    return s


def load_evi_for_wy(
    county: str,
    wy: int,
    uids: List[str],
    evi_mode: str = None,
) -> Dict[str, pd.Series]:
    """Load daily EVI for multiple UIDs for a water year.

    Args:
        county: County name.
        wy: Water year.
        uids: List of parcel unique identifiers.
        evi_mode: EVI column to use.

    Returns:
        Dict mapping uid -> daily EVI series.
    """
    if evi_mode is None:
        evi_mode = config.evi_mode

    county_norm = normalize_county_name(county)
    wy_start, wy_end = water_year_bounds(wy)
    idx = pd.date_range(wy_start, wy_end, freq="D")

    p = Path(config.county_year_root_new) / county_norm / f"WY{wy}.csv"
    if not p.exists():
        raise FileNotFoundError(f"County-year EVI CSV not found: {p}")

    df = pd.read_csv(p, parse_dates=["date"])
    need = {"date", "parcel_id", "gapfilled_mean_evi", "smoothed_mean_evi"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{p} missing columns: {sorted(miss)}")

    col = resolve_evi_col(evi_mode)
    if col not in df.columns:
        raise ValueError(
            f"{p} does not have EVI column '{col}'. Available: {sorted(df.columns)}"
        )

    df["UniqueID"] = df["parcel_id"].astype(str)
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= wy_start) & (df["date"] <= wy_end)]

    if uids:
        df = df[df["UniqueID"].isin([str(u) for u in uids])]

    df[col] = pd.to_numeric(df[col], errors="coerce")

    out: Dict[str, pd.Series] = {}
    for uid, g in df.groupby("UniqueID", sort=False):
        s = g.groupby("date", sort=True)[col].mean().astype(float).sort_index()
        s = s.reindex(idx)
        try:
            s = s.interpolate(method="time", limit=90, limit_direction="both")
        except Exception:
            s = s.interpolate(limit=90, limit_direction="both")
        s = s.clip(-0.05, 1.05)
        if s.isna().any():
            s = s.ffill().bfill()
        out[str(uid)] = s

    # ensure keys for all requested uids
    for uid in uids:
        if str(uid) not in out:
            out[str(uid)] = pd.Series(np.nan, index=idx, dtype=float)

    return out


# ---------------------------------------------------------------------------
# Seasonal classification
# ---------------------------------------------------------------------------

def is_spring_summer_main_season(d: pd.Timestamp) -> bool:
    """Return True if date falls in the main growing season (Mar-Sep)."""
    return int(d.month) in {3, 4, 5, 6, 7, 8, 9}


# ---------------------------------------------------------------------------
# EVI-based segment start estimation
# ---------------------------------------------------------------------------

def find_pre_cut_min_start_date_from_evi(
    evi: pd.Series,
    cut_date: pd.Timestamp,
    lower_bound: pd.Timestamp,
    summer_lookback_range_days: Tuple[int, int] = None,
    winter_lookback_range_days: Tuple[int, int] = None,
    min_gap_before_cut_days: int = None,
) -> pd.Timestamp:
    """Find segment start as the date of minimum EVI in a seasonal lookback band.

    For a cut at date D:
      - Mar-Sep (spring/summer): search [D - max_days, D - min_days]
      - Oct-Feb (winter):        search [D - max_days, D - min_days]
    Bounded below by lower_bound (previous cut or WY start).
    min_gap_before_cut_days prevents the window from reaching into
    the cut-date trough itself.

    Args:
        evi: Daily EVI series with DatetimeIndex.
        cut_date: Date of the cutting event.
        lower_bound: Earliest allowable segment start (previous cut or WY start).
        summer_lookback_range_days: (min_days, max_days) for spring/summer.
        winter_lookback_range_days: (min_days, max_days) for winter.
        min_gap_before_cut_days: Minimum days before cut to stop the window.

    Returns:
        Segment start date (Timestamp).
    """
    if summer_lookback_range_days is None:
        summer_lookback_range_days = config.summer_lookback_range_days
    if winter_lookback_range_days is None:
        winter_lookback_range_days = config.winter_lookback_range_days
    if min_gap_before_cut_days is None:
        min_gap_before_cut_days = config.min_gap_before_cut_days

    if evi is None or evi.empty:
        return lower_bound
    if pd.isna(cut_date) or pd.isna(lower_bound):
        return pd.NaT

    idx = evi.index

    # align to nearest available EVI date
    if cut_date not in idx:
        pos = idx.get_indexer([cut_date], method="nearest")[0]
        if pos < 0:
            return lower_bound
        cut_date = idx[pos]
    if lower_bound not in idx:
        pos = idx.get_indexer([lower_bound], method="nearest")[0]
        if pos < 0:
            lower_bound = idx[0]
        else:
            lower_bound = idx[pos]

    if cut_date < lower_bound:
        return lower_bound

    if is_spring_summer_main_season(pd.to_datetime(cut_date)):
        mn, mx = summer_lookback_range_days
    else:
        mn, mx = winter_lookback_range_days

    mn = int(mn)
    mx = int(mx)
    if mx < mn:
        mn, mx = mx, mn

    win_start = pd.to_datetime(cut_date) - pd.Timedelta(days=mx)
    win_end = pd.to_datetime(cut_date) - pd.Timedelta(days=mn)
    win_end = min(
        win_end,
        pd.to_datetime(cut_date) - pd.Timedelta(days=int(min_gap_before_cut_days)),
    )

    win_start = max(win_start, pd.to_datetime(lower_bound))
    if win_end < win_start:
        return lower_bound

    w = evi.loc[win_start:win_end].dropna()
    if w.empty:
        return lower_bound

    return pd.to_datetime(w.idxmin()).normalize()


# ---------------------------------------------------------------------------
# Whittaker-based adaptive segment start estimation
# ---------------------------------------------------------------------------

def find_pre_cut_min_whittaker(
    whittaker_evi: pd.Series,
    cut_date: pd.Timestamp,
    lower_bound: pd.Timestamp,
    season: str,
    summer_max_lookback_days: int = None,
    winter_max_lookback_days: int = None,
    regrowth_slope_threshold: float = None,
) -> pd.Timestamp:
    """Find segment start using Whittaker-smoothed EVI derivative analysis.

    Summer (Mar-Sep) - trough detection:
        Walk backward from cut_date to find a local minimum where EVI is
        below the parcel's seasonal mean and the derivative changes sign.

    Winter (Oct-Feb) - regrowth onset detection:
        Walk backward from cut_date to find where the derivative drops
        below a threshold, indicating dormancy-to-growth transition.

    Args:
        whittaker_evi: Whittaker-smoothed daily EVI series.
        cut_date: Date of the cutting event.
        lower_bound: Earliest allowable segment start.
        season: "summer" (Mar-Sep) or "winter" (Oct-Feb).
        summer_max_lookback_days: Max backward search for summer.
        winter_max_lookback_days: Max backward search for winter.
        regrowth_slope_threshold: EVI/day threshold for winter regrowth onset.

    Returns:
        Segment start date (Timestamp).
    """
    if summer_max_lookback_days is None:
        summer_max_lookback_days = config.whittaker_summer_max_lookback_days
    if winter_max_lookback_days is None:
        winter_max_lookback_days = config.whittaker_winter_max_lookback_days
    if regrowth_slope_threshold is None:
        regrowth_slope_threshold = config.whittaker_regrowth_slope_threshold

    if whittaker_evi is None or whittaker_evi.empty:
        return lower_bound
    if pd.isna(cut_date) or pd.isna(lower_bound):
        return pd.NaT

    cut_date = pd.to_datetime(cut_date).normalize()
    lower_bound = pd.to_datetime(lower_bound).normalize()

    if cut_date < lower_bound:
        return lower_bound

    # Determine lookback window
    if season == "summer":
        max_lookback = summer_max_lookback_days
    else:
        max_lookback = winter_max_lookback_days

    win_start = max(cut_date - pd.Timedelta(days=max_lookback), lower_bound)
    window = whittaker_evi.loc[win_start:cut_date].dropna()

    if len(window) < 3:
        return lower_bound

    # Compute first derivative (daily EVI change)
    deriv = window.diff()

    if season == "summer":
        # Trough detection: walk backward for local minimum
        seasonal_mean = float(window.mean())
        vals = window.values
        deriv_vals = deriv.values
        dates = window.index

        best_date = lower_bound
        # Walk backward from cut_date (skip last few days near cut trough)
        for i in range(len(vals) - 4, 0, -1):
            evi_val = vals[i]
            # Check: EVI below seasonal mean (confirms trough, not plateau)
            if evi_val < seasonal_mean:
                # Check: derivative changes from negative to positive (true minimum)
                if i < len(deriv_vals) - 1 and deriv_vals[i] <= 0 and deriv_vals[i + 1] > 0:
                    best_date = dates[i]
                    break

        # Fallback: EVI peak in window (start of the decline = start of cutting cycle)
        if best_date == lower_bound:
            best_date = window.idxmax()

    else:
        # Winter: regrowth onset detection
        deriv_vals = deriv.values
        dates = window.index

        best_date = lower_bound
        # Walk backward from cut_date to find where derivative drops below threshold
        for i in range(len(deriv_vals) - 4, 0, -1):
            if abs(deriv_vals[i]) < regrowth_slope_threshold:
                best_date = dates[i]
                break

        # Fallback: EVI peak in window (start of the decline = start of cutting cycle)
        if best_date == lower_bound:
            best_date = window.idxmax()

    result = pd.to_datetime(best_date).normalize()
    result = max(result, lower_bound)

    # Sanity check: if result is too close to cut_date, walk back to min_segment_days
    min_seg = config.min_segment_days
    if (cut_date - result).days < min_seg:
        result = max(lower_bound, cut_date - pd.Timedelta(days=min_seg))

    return result


# ---------------------------------------------------------------------------
# Thermal-time calibration and segment start estimation
# ---------------------------------------------------------------------------

def calibrate_cutting_thresholds(
    cut_dates_by_parcel: Dict[str, List[pd.Timestamp]],
    daily_gdd5_by_parcel: Dict[str, pd.Series],
    daily_et_by_parcel: Dict[str, pd.Series],
    trusted_summer_range: Tuple[int, int] = (20, 45),
    trusted_winter_range: Tuple[int, int] = (50, 150),
) -> Dict[str, float]:
    """Calibrate GDD5 and ET thresholds from trusted inter-cut intervals.

    Pass 1 of the thermal-time method: examine all consecutive matched-cut
    pairs, keep those with agronomically plausible intervals, and compute
    the median GDD5 and ET accumulated between cuts per season.

    Args:
        cut_dates_by_parcel: Dict mapping uid -> sorted list of matched cut dates.
        daily_gdd5_by_parcel: Dict mapping uid -> daily GDD5 pd.Series.
        daily_et_by_parcel: Dict mapping uid -> daily ET pd.Series.
        trusted_summer_range: (min_days, max_days) for trusted summer intervals.
        trusted_winter_range: (min_days, max_days) for trusted winter intervals.

    Returns:
        Dict with keys: gdd5_summer, gdd5_winter, et_summer, et_winter,
        n_summer, n_winter (sample sizes).
    """
    summer_gdd5s, summer_ets = [], []
    winter_gdd5s, winter_ets = [], []

    for uid, cuts in cut_dates_by_parcel.items():
        gdd5_s = daily_gdd5_by_parcel.get(uid)
        et_s = daily_et_by_parcel.get(uid)
        if gdd5_s is None or et_s is None:
            continue
        if len(cuts) < 2:
            continue

        for i in range(1, len(cuts)):
            prev_cut = cuts[i - 1]
            this_cut = cuts[i]
            interval_days = (this_cut - prev_cut).days
            if interval_days < 5:
                continue

            # Classify by season of the current cut
            is_summer = is_spring_summer_main_season(this_cut)

            if is_summer:
                mn, mx = trusted_summer_range
            else:
                mn, mx = trusted_winter_range

            if not (mn <= interval_days <= mx):
                continue

            # Accumulate GDD5 between cuts
            gdd_seg = gdd5_s.loc[prev_cut:this_cut]
            if gdd_seg.empty or gdd_seg.isna().all():
                continue
            gdd_total = float(gdd_seg.sum())

            # Accumulate ET between cuts
            et_seg = et_s.loc[prev_cut:this_cut]
            if et_seg.empty or et_seg.isna().all():
                continue
            et_total = float(et_seg.sum())

            if is_summer:
                summer_gdd5s.append(gdd_total)
                summer_ets.append(et_total)
            else:
                winter_gdd5s.append(gdd_total)
                winter_ets.append(et_total)

    result = {
        "gdd5_summer": float(np.median(summer_gdd5s)) if summer_gdd5s else 440.0,
        "gdd5_winter": float(np.median(winter_gdd5s)) if winter_gdd5s else 300.0,
        "et_summer": float(np.median(summer_ets)) if summer_ets else 150.0,
        "et_winter": float(np.median(winter_ets)) if winter_ets else 60.0,
        "n_summer": len(summer_gdd5s),
        "n_winter": len(winter_gdd5s),
    }
    return result


def find_segment_start_thermal_time(
    daily_gdd5: pd.Series,
    cut_date: pd.Timestamp,
    lower_bound: pd.Timestamp,
    gdd5_threshold: float,
    daily_et: Optional[pd.Series] = None,
    et_threshold: Optional[float] = None,
) -> pd.Timestamp:
    """Find segment start by backward GDD5 accumulation.

    Walk backward from cut_date accumulating daily GDD5. The segment
    starts at the date where cumulative GDD5 first reaches the threshold.
    This gives a physiologically-grounded interval: the crop needed this
    much thermal time to reach cutting maturity.

    Optionally cross-validates with ET: if the ET over the found segment
    differs by more than 2x from et_threshold, flags but still returns
    the GDD5-based date.

    Args:
        daily_gdd5: Daily GDD5 series with DatetimeIndex.
        cut_date: Date of the cutting event.
        lower_bound: Earliest allowable segment start (previous cut or WY start).
        gdd5_threshold: Target cumulative GDD5 for a complete cutting cycle.
        daily_et: Optional daily ET series for cross-validation.
        et_threshold: Optional expected ET for cross-validation.

    Returns:
        Segment start date (Timestamp).
    """
    if daily_gdd5 is None or daily_gdd5.empty:
        return lower_bound
    if pd.isna(cut_date) or pd.isna(lower_bound):
        return pd.NaT

    cut_date = pd.to_datetime(cut_date).normalize()
    lower_bound = pd.to_datetime(lower_bound).normalize()

    if cut_date <= lower_bound:
        return lower_bound

    # Get GDD5 values in [lower_bound, cut_date], reversed for backward walk
    window = daily_gdd5.loc[lower_bound:cut_date].dropna()
    if window.empty:
        return lower_bound

    # Backward cumulative sum
    gdd_reversed = window.iloc[::-1].values
    cum_gdd = np.cumsum(gdd_reversed)
    dates_reversed = window.index[::-1]

    # Find where cumulative GDD5 first exceeds threshold
    exceed_mask = cum_gdd >= gdd5_threshold
    if exceed_mask.any():
        idx = np.argmax(exceed_mask)
        result = pd.to_datetime(dates_reversed[idx]).normalize()
    else:
        # Threshold not reached — use lower_bound (all available thermal time)
        result = lower_bound

    return max(result, lower_bound)


# ---------------------------------------------------------------------------
# Legacy ET-rise-based segment start (fallback)
# ---------------------------------------------------------------------------

def find_cycle_start_local_min(
    et: pd.Series,
    cut_date: pd.Timestamp,
    lower_bound: pd.Timestamp,
    rise_days: int = None,
    rise_eps: float = None,
) -> pd.Timestamp:
    """Legacy ET-based segment start finder (fallback when EVI unavailable).

    Walks backward from cut_date to find where ET stops rising, indicating
    the start of the growth cycle.

    Args:
        et: Daily ET series with DatetimeIndex.
        cut_date: Date of the cutting event.
        lower_bound: Earliest allowable start.
        rise_days: Consecutive rising days to trigger stop.
        rise_eps: Minimum rise magnitude to count as "rising".

    Returns:
        Segment start date (Timestamp).
    """
    if rise_days is None:
        rise_days = config.rise_days
    if rise_eps is None:
        rise_eps = config.rise_eps

    if pd.isna(cut_date) or pd.isna(lower_bound):
        return pd.NaT
    if cut_date < lower_bound:
        return lower_bound

    # ensure datetime index
    if et.index.dtype.kind != "M":
        et = et.copy()
        et.index = pd.to_datetime(et.index)
    et = et.sort_index()

    idx = et.index
    if cut_date not in idx:
        cut_date = idx[idx.get_indexer([cut_date], method="nearest")[0]]
    if lower_bound not in idx:
        lower_bound = idx[idx.get_indexer([lower_bound], method="nearest")[0]]

    w = et.loc[lower_bound:cut_date]
    if w.empty:
        return lower_bound

    min_val = np.inf
    min_date = cut_date
    consecutive_rise = 0

    for d in reversed(w.index):
        v = float(w.loc[d])
        if v <= min_val + 1e-12:
            min_val = v
            min_date = d
            consecutive_rise = 0
        else:
            if v > min_val + rise_eps:
                consecutive_rise += 1
            else:
                consecutive_rise = 0
            if consecutive_rise >= rise_days:
                break

    return pd.to_datetime(min_date).normalize()


# ---------------------------------------------------------------------------
# Month-window date computation (for Daymet/ET windowing)
# ---------------------------------------------------------------------------

def compute_daymet_window_dates(
    wy: int,
    month_start: int,
    month_end: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Convert month_start/month_end (1-12) into calendar dates within a WY.

    Months 10-12 belong to the prior calendar year (WY-1);
    months 1-9 belong to the WY calendar year.

    Args:
        wy: Water year.
        month_start: Start month (1-12).
        month_end: End month (1-12).

    Returns:
        Tuple of (start_date, end_date).
    """
    if not (1 <= month_start <= 12 and 1 <= month_end <= 12):
        raise ValueError("month_start and month_end must be between 1 and 12.")
    wy_start, wy_end = water_year_bounds(wy)

    year_start = wy - 1 if month_start >= 10 else wy
    year_end = wy - 1 if month_end >= 10 else wy

    start_date = pd.Timestamp(year_start, month_start, 1)
    end_date = pd.Timestamp(year_end, month_end, 1) + pd.offsets.MonthEnd(0)

    if start_date < wy_start or end_date > wy_end:
        raise ValueError(
            f"Requested window {start_date.date()}-{end_date.date()} is outside WY{wy} "
            f"({wy_start.date()}-{wy_end.date()})."
        )
    if start_date > end_date:
        raise ValueError(
            f"Requested window start {start_date.date()} is after end {end_date.date()}."
        )
    return start_date, end_date


_MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def daymet_window_label(month_start: int, month_end: int) -> str:
    """Format a month-window as a human-readable label."""
    return f"{_MONTH_ABBR[month_start - 1]}-{_MONTH_ABBR[month_end - 1]} (WY window)"


# ---------------------------------------------------------------------------
# Cut-cycle segment construction
# ---------------------------------------------------------------------------

def compute_cut_cycle_segments(
    cut_dates: List[pd.Timestamp],
    wy: int,
    month_start: int,
    month_end: int,
    apply_month_window_to_et_gdd: bool,
    *,
    et: Optional[pd.Series] = None,
    evi: Optional[pd.Series] = None,
    whittaker_evi: Optional[pd.Series] = None,
    daily_gdd5: Optional[pd.Series] = None,
    gdd5_thresholds: Optional[Dict[str, float]] = None,
    summer_lookback_range_days: Tuple[int, int] = None,
    winter_lookback_range_days: Tuple[int, int] = None,
    min_gap_before_cut_days: int = None,
    min_segment_days: int = None,
    rise_days: int = None,
    rise_eps: float = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Build (seg_start, seg_end) pairs for each cut cycle.

    For each cut date:
      - lower_bound = previous cut (or WY start)
      - seg_start determined by fallback chain:
        1. Thermal-time (GDD5 backward walk) — primary when available
        2. Whittaker EVI derivative
        3. Legacy EVI lookback band
        4. ET-rise method
        5. Fixed lookback (min_segment_days)
      - seg_end = cut date

    Enforces minimum segment length. Optionally intersects with month window.

    Args:
        cut_dates: List of cutting event dates.
        wy: Water year.
        month_start: Start month for windowing (1-12).
        month_end: End month for windowing (1-12).
        apply_month_window_to_et_gdd: If True, intersect segments with month window.
        et: Optional daily ET series (used as fallback if EVI missing).
        evi: Optional daily EVI series (preferred for segment start).
        whittaker_evi: Optional Whittaker-smoothed daily EVI series.
        daily_gdd5: Optional daily GDD5 series for thermal-time estimation.
        gdd5_thresholds: Optional dict with 'gdd5_summer' and 'gdd5_winter' keys.
        summer_lookback_range_days: (min_days, max_days) for spring/summer.
        winter_lookback_range_days: (min_days, max_days) for winter.
        min_gap_before_cut_days: Minimum gap before cut date.
        min_segment_days: Minimum segment length.
        rise_days: Legacy ET fallback parameter.
        rise_eps: Legacy ET fallback parameter.

    Returns:
        List of (seg_start, seg_end) tuples.
    """
    if summer_lookback_range_days is None:
        summer_lookback_range_days = config.summer_lookback_range_days
    if winter_lookback_range_days is None:
        winter_lookback_range_days = config.winter_lookback_range_days
    if min_gap_before_cut_days is None:
        min_gap_before_cut_days = config.min_gap_before_cut_days
    if min_segment_days is None:
        min_segment_days = config.min_segment_days
    if rise_days is None:
        rise_days = config.rise_days
    if rise_eps is None:
        rise_eps = config.rise_eps

    wy_start, wy_end = water_year_bounds(wy)
    cds = sorted([
        d.normalize() for d in cut_dates
        if wy_start <= d.normalize() <= wy_end
    ])
    if not cds:
        return []

    if apply_month_window_to_et_gdd:
        mw_start, mw_end = compute_daymet_window_dates(wy, month_start, month_end)
    else:
        mw_start, mw_end = (wy_start, wy_end)

    segs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    prev_cut = wy_start

    et_idx = et.index if (et is not None and not et.empty) else None
    use_evi = (
        evi is not None
        and not evi.empty
        and np.isfinite(evi.to_numpy()).any()
    )
    use_whittaker = (
        config.use_whittaker_interval
        and whittaker_evi is not None
        and not whittaker_evi.empty
        and np.isfinite(whittaker_evi.to_numpy()).any()
    )
    use_thermal = (
        config.use_thermal_time_interval
        and daily_gdd5 is not None
        and not daily_gdd5.empty
        and gdd5_thresholds is not None
    )

    for c in cds:
        lo = prev_cut

        # Determine segment start with fallback chain:
        # Thermal-time → Whittaker → legacy EVI → ET rise → fixed lookback
        start_i = lo

        if use_thermal:
            # Primary: GDD5 backward accumulation
            is_summer = is_spring_summer_main_season(pd.to_datetime(c))
            gdd_thr = gdd5_thresholds.get(
                "gdd5_summer" if is_summer else "gdd5_winter", 440.0
            )
            start_i = find_segment_start_thermal_time(
                daily_gdd5=daily_gdd5,
                cut_date=c,
                lower_bound=lo,
                gdd5_threshold=gdd_thr,
                daily_et=et,
                et_threshold=gdd5_thresholds.get(
                    "et_summer" if is_summer else "et_winter"
                ),
            )

            # If thermal-time gives too-short segment, try Whittaker
            if (pd.to_datetime(c) - pd.to_datetime(start_i)).days < int(min_segment_days) and use_whittaker:
                season = "summer" if is_summer else "winter"
                start_i = find_pre_cut_min_whittaker(
                    whittaker_evi=whittaker_evi,
                    cut_date=c,
                    lower_bound=lo,
                    season=season,
                )

        elif use_whittaker:
            season = "summer" if is_spring_summer_main_season(pd.to_datetime(c)) else "winter"
            start_i = find_pre_cut_min_whittaker(
                whittaker_evi=whittaker_evi,
                cut_date=c,
                lower_bound=lo,
                season=season,
            )

        # Fallback chain for short segments (shared by all primary methods)
        if (pd.to_datetime(c) - pd.to_datetime(start_i)).days < int(min_segment_days) and use_evi:
            start_i = find_pre_cut_min_start_date_from_evi(
                evi=evi,
                cut_date=c,
                lower_bound=lo,
                summer_lookback_range_days=summer_lookback_range_days,
                winter_lookback_range_days=winter_lookback_range_days,
                min_gap_before_cut_days=min_gap_before_cut_days,
            )

        if (pd.to_datetime(c) - pd.to_datetime(start_i)).days < int(min_segment_days) and et is not None and not et.empty:
            start_i = find_cycle_start_local_min(
                et=et,
                cut_date=c,
                lower_bound=lo,
                rise_days=rise_days,
                rise_eps=rise_eps,
            )

        if (pd.to_datetime(c) - pd.to_datetime(start_i)).days < int(min_segment_days):
            start_i = max(lo, pd.to_datetime(c) - pd.Timedelta(days=int(min_segment_days)))

        s = pd.to_datetime(start_i).normalize()
        e = pd.to_datetime(c).normalize()

        # bounds
        s = max(s, lo, wy_start)
        e = min(e, wy_end)

        # enforce minimum segment length
        if (e - s).days < int(min_segment_days):
            s = max(lo, e - pd.Timedelta(days=int(min_segment_days)))

        # align to ET index (daily) if available
        if et_idx is not None:
            if s not in et_idx:
                pos = et_idx.get_indexer([s], method="nearest")[0]
                if pos >= 0:
                    s = et_idx[pos]
            if e not in et_idx:
                pos = et_idx.get_indexer([e], method="nearest")[0]
                if pos >= 0:
                    e = et_idx[pos]

        # intersect with month window
        s2 = max(s, mw_start)
        e2 = min(e, mw_end)

        if s2 <= e2:
            segs.append((
                pd.to_datetime(s2).normalize(),
                pd.to_datetime(e2).normalize(),
            ))

        prev_cut = c

    return segs


# ---------------------------------------------------------------------------
# Post-harvest gap merge (Layer 3)
# ---------------------------------------------------------------------------

def merge_segment_gaps(
    segs: List[Tuple[pd.Timestamp, pd.Timestamp]],
    max_extension_days: int = 75,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Close gaps between consecutive segments by extending each segment end.

    For each pair of consecutive segments, if the gap between seg_i.end and
    seg_{i+1}.start is <= max_extension_days, extend seg_i.end to
    seg_{i+1}.start.  This captures the post-harvest decline ET that
    belongs to the cutting cycle that caused the harvest.

    The segment starts are unchanged — the EVI-minimum lookback logic
    is fully preserved.  Only the segment *ends* are extended.

    Args:
        segs: List of (start, end) segment tuples (sorted by start).
        max_extension_days: Maximum gap to close.  Gaps larger than this
            are assumed to be fallow/dormant and are left open.

    Returns:
        New list of (start, end) tuples with gaps closed.
    """
    if len(segs) <= 1:
        return list(segs)

    merged = []
    for i in range(len(segs) - 1):
        s_i, e_i = segs[i]
        s_next, _ = segs[i + 1]
        gap_days = (pd.to_datetime(s_next) - pd.to_datetime(e_i)).days
        if 0 < gap_days <= max_extension_days:
            merged.append((s_i, s_next))
        else:
            merged.append((s_i, e_i))
    # Last segment keeps its original end
    merged.append(segs[-1])
    return merged


# ---------------------------------------------------------------------------
# Segment ET summation
# ---------------------------------------------------------------------------

def segment_et_sums_mm(
    et: pd.Series,
    segs: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    """Compute per-segment ET totals in mm.

    Args:
        et: Daily ET series.
        segs: List of (start, end) segment tuples.

    Returns:
        Array of per-segment ET sums.
    """
    if et is None or et.empty or not segs:
        return np.array([], dtype=float)
    sums = []
    for s, e in segs:
        if pd.isna(s) or pd.isna(e) or s > e:
            sums.append(np.nan)
            continue
        sums.append(float(et.loc[s:e].sum()))
    return np.asarray(sums, dtype=float)


# ---------------------------------------------------------------------------
# Segment-level ET filtering
# ---------------------------------------------------------------------------

def filter_segments_by_et(
    et: pd.Series,
    segs: List[Tuple[pd.Timestamp, pd.Timestamp]],
    mode: str = None,
    abs_min_mm: float = None,
    rel_min_frac_of_median: float = None,
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp]], Dict[str, float]]:
    """Filter segments based on per-segment ET totals.

    Modes:
      - "none": no filtering
      - "absolute": keep if seg_et >= abs_min_mm
      - "relative": keep if seg_et >= rel_min_frac * median(all seg_et)
      - "both": keep if passes BOTH tests

    Args:
        et: Daily ET series.
        segs: List of (start, end) segment tuples.
        mode: Filtering mode.
        abs_min_mm: Absolute minimum ET threshold.
        rel_min_frac_of_median: Relative threshold as fraction of median.

    Returns:
        Tuple of (filtered_segments, diagnostics_dict).
    """
    if mode is None:
        mode = config.segment_et_filter_mode
    if abs_min_mm is None:
        abs_min_mm = config.segment_et_abs_min_mm
    if rel_min_frac_of_median is None:
        rel_min_frac_of_median = config.segment_et_rel_min_frac

    m = str(mode).strip().lower()
    if m not in {"none", "absolute", "relative", "both"}:
        raise ValueError("mode must be one of: none, absolute, relative, both")

    seg_et = segment_et_sums_mm(et, segs)
    n_raw = int(len(segs))
    if n_raw == 0:
        return [], {
            "n_raw": 0,
            "n_kept": 0,
            "median_seg_et_mm": np.nan,
            "abs_thr_mm": float(abs_min_mm),
            "rel_thr_mm": np.nan,
        }

    finite = np.isfinite(seg_et)
    med = float(np.nanmedian(seg_et[finite])) if finite.any() else float("nan")
    rel_thr_mm = (
        float(rel_min_frac_of_median * med)
        if (np.isfinite(med) and med > 0)
        else float("nan")
    )

    keep = np.ones(n_raw, dtype=bool)

    if m in {"absolute", "both"}:
        keep &= np.isfinite(seg_et) & (seg_et >= float(abs_min_mm))

    if m in {"relative", "both"}:
        if np.isfinite(rel_thr_mm):
            keep &= np.isfinite(seg_et) & (seg_et >= rel_thr_mm)

    segs_f = [segs[i] for i in range(n_raw) if keep[i]]
    return segs_f, {
        "n_raw": n_raw,
        "n_kept": int(len(segs_f)),
        "median_seg_et_mm": med,
        "abs_thr_mm": float(abs_min_mm),
        "rel_thr_mm": rel_thr_mm,
    }
