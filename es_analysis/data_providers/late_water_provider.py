"""Late-water savings provider.

Builds parcel-year late-cut datasets, computes cap-based savings
scenarios, produces group summaries with area-normalized intensity
metrics, and orchestrates the full workflow with CSV export.

Source: alfalfa_evi_jovyan.py lines 19323-19863, 20100-20173
(latest/most refined versions of all duplicated functions).
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import config
from .evi_provider import normalize_county_name, water_year_bounds
from .evi_cut_window_provider import (
    find_pre_cut_min_start_date_from_evi,
    load_evi_for_wy,
    compute_cut_cycle_segments,
    segment_et_sums_mm,
)
from ..utils.whittaker import whittaker_smooth_series
from .et_provider import (
    _load_seasonal_csv,
    _load_openet_for_wy,
    _parse_cp_dates_iso,
)
from .spatial_provider import COUNTY_ORDER, load_parcels_area_acres
from ..utils.units import mm_to_acft_per_acre, mm_to_acft_total, acft_per_acre_to_mm
from ..utils.validation import validate_cycle_et_mm, validate_per_cutting_acft_per_acre


# ---------------------------------------------------------------------------
# Helper: water-year date
# ---------------------------------------------------------------------------

def wy_date(wy: int, month: int, day: int) -> pd.Timestamp:
    """Convert (month, day) into a calendar date inside a water year.

    Months 10-12 belong to calendar year (wy - 1);
    months 1-9 belong to calendar year wy.
    """
    if month < 1 or month > 12:
        raise ValueError("month must be 1..12")
    year = wy - 1 if month >= 10 else wy
    return pd.Timestamp(year, month, day)


# ---------------------------------------------------------------------------
# Legacy ET-rise fallback for cycle start detection
# ---------------------------------------------------------------------------

def find_pre_cut_min_start_date_et_rise(
    et: pd.Series,
    cut_date: pd.Timestamp,
    lower_bound: pd.Timestamp,
    rise_days: int = 5,
    rise_eps: float = 0.02,
) -> pd.Timestamp:
    """Legacy heuristic: walk backward from cut_date to find minimum ET
    date before a sustained rise.

    Starting at *cut_date*, move backward day-by-day down to
    *lower_bound*.  Track the running minimum ET and its date.
    Once a minimum is found and ET stays > (min + rise_eps) for
    *rise_days* consecutive days, stop: we have passed the first
    local minimum.
    """
    if cut_date < lower_bound:
        return lower_bound

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

    return min_date


# ---------------------------------------------------------------------------
# Per-cut-cycle ET computation
# ---------------------------------------------------------------------------

def compute_et_by_cut_cycles(
    et: pd.Series,
    cut_dates: List[pd.Timestamp],
    wy: int,
    *,
    evi: Optional[pd.Series] = None,
    summer_lookback_range_days: Tuple[int, int] = (28, 32),
    winter_lookback_range_days: Tuple[int, int] = (90, 120),
    min_gap_before_cut_days: int = 3,
    min_segment_days: int = 7,
    fallback_to_et_rise: bool = True,
    rise_days: int = 5,
    rise_eps: float = 0.02,
) -> List[Dict[str, object]]:
    """Compute per-cut-cycle ET totals.

    Primary cycle start: EVI min in seasonal lookback band.
    Fallback (if EVI missing/invalid and *fallback_to_et_rise*):
    legacy ET-rise heuristic.

    Returns list of dicts with keys:
        cut_index, cut_date, cycle_start, cycle_et_mm
    """
    wy_start, wy_end = water_year_bounds(wy)
    cds = sorted(
        d.normalize() for d in cut_dates
        if wy_start <= d.normalize() <= wy_end
    )
    if not cds:
        return []

    out: List[Dict[str, object]] = []
    prev_cut = wy_start
    et_idx = et.index

    for i, c in enumerate(cds):
        lo = prev_cut

        # 1) EVI-based start
        start_i = find_pre_cut_min_start_date_from_evi(
            evi=evi,
            cut_date=c,
            lower_bound=lo,
            summer_lookback_range_days=summer_lookback_range_days,
            winter_lookback_range_days=winter_lookback_range_days,
            min_gap_before_cut_days=min_gap_before_cut_days,
        )

        # 2) Fallback if EVI produced no movement (or NaT)
        if (pd.isna(start_i) or pd.to_datetime(start_i) == pd.to_datetime(lo)) and fallback_to_et_rise:
            start_i = find_pre_cut_min_start_date_et_rise(
                et=et, cut_date=c, lower_bound=lo,
                rise_days=rise_days, rise_eps=rise_eps,
            )

        start_i = pd.to_datetime(start_i).normalize()
        end_i = pd.to_datetime(c).normalize()

        # Boundaries
        start_i = max(start_i, lo, wy_start)
        end_i = min(end_i, wy_end)

        # Enforce minimum segment length
        if (end_i - start_i).days < int(min_segment_days):
            start_i = max(lo, end_i - pd.Timedelta(days=int(min_segment_days)))

        # Align to ET index (nearest)
        if start_i not in et_idx:
            pos = et_idx.get_indexer([start_i], method="nearest")[0]
            if pos >= 0:
                start_i = et_idx[pos]
        if end_i not in et_idx:
            pos = et_idx.get_indexer([end_i], method="nearest")[0]
            if pos >= 0:
                end_i = et_idx[pos]

        if pd.isna(start_i) or pd.isna(end_i) or start_i > end_i:
            et_i = np.nan
        else:
            slice_ = et.loc[start_i:end_i]
            if slice_.notna().any():
                et_i = float(slice_.sum(min_count=1))
            else:
                et_i = np.nan

        # Validate cycle ET (ETFIX-03)
        validate_cycle_et_mm(et_i, cut_index=i + 1)

        out.append({
            "cut_index": i + 1,
            "cut_date": pd.to_datetime(c).normalize(),
            "cycle_start": pd.to_datetime(start_i).normalize(),
            "cycle_et_mm": et_i,
        })
        prev_cut = pd.to_datetime(c).normalize()

    return out


# ---------------------------------------------------------------------------
# Late-cut dataset builder
# ---------------------------------------------------------------------------

def build_late_cut_dataset(
    wy_start: int,
    wy_end: int,
    counties: Optional[Iterable[str]] = None,
    cut_metric: str = "n_cp_season",
    cutoff_month: int = 7,
    cutoff_day: int = 1,
    compute_area_acft: bool = False,
    *,
    evi_mode: str = None,  # None = use config.evi_mode (same as parcel_summary)
    summer_lookback_range_days: Tuple[int, int] = (28, 32),
    winter_lookback_range_days: Tuple[int, int] = (90, 120),
    min_gap_before_cut_days: int = 3,
    min_segment_days: int = 7,
    evi_required: bool = False,
    fallback_to_et_rise: bool = True,
    rise_days: int = 5,
    rise_eps: float = 0.02,
    et_mode: str = "actual",
    method: str = "A",
) -> pd.DataFrame:
    """Build parcel-year dataset with late-cut metrics.

    For each parcel-year, computes:
      - n_late_cuts: number of cuts after cutoff date
      - late_et_mm: sum of per-cut cycle ET for late cuts
      - late_et_acft: (optional) volume using parcel area
      - late_cycle_et_mm_list: per-cycle ET values
      - late_cut_dates: timestamps of late cuts

    Args:
        wy_start: First water year.
        wy_end: Last water year.
        counties: County names (defaults to COUNTY_ORDER).
        cut_metric: "n_cp_season" or "n_cuttings".
        cutoff_month: Month component of late-season cutoff.
        cutoff_day: Day component of late-season cutoff.
        compute_area_acft: Whether to compute area-based ac-ft values.
        evi_mode: "smoothed" or "gapfilled".
        summer_lookback_range_days: EVI lookback band for Mar-Sep.
        winter_lookback_range_days: EVI lookback band for Oct-Feb.
        min_gap_before_cut_days: Buffer days before cut date.
        min_segment_days: Minimum cycle length.
        evi_required: If True, skip parcels without EVI.
        fallback_to_et_rise: Use ET-rise heuristic when EVI unavailable.
        rise_days: Consecutive days for ET-rise detection.
        rise_eps: Rise threshold for ET-rise detection.
        et_mode: "actual", "corrected", or "both".
            When "corrected" or "both", also computes corrected late-cycle ET
            using the ETof-based correction from et_provider.
        method: ET correction method "A" or "B" (only used when et_mode != "actual").

    Returns:
        DataFrame with one row per parcel-year.
    """
    if cut_metric not in {"n_cp_season", "n_cuttings"}:
        raise ValueError("cut_metric must be 'n_cp_season' or 'n_cuttings'.")

    if evi_mode is None:
        evi_mode = config.evi_mode

    counties_use = [
        normalize_county_name(c)
        for c in (counties if counties is not None else COUNTY_ORDER)
    ]

    areas = None
    if compute_area_acft:
        areas = load_parcels_area_acres(counties_use)

    rows: List[Dict[str, object]] = []

    for county in counties_use:
        for wy in range(wy_start, wy_end + 1):
            try:
                df_seasonal = _load_seasonal_csv(county, wy)
            except FileNotFoundError as e:
                print(f"[info] Skipping {county}, WY{wy}: {e}")
                continue

            wy_s, wy_e = water_year_bounds(wy)
            cutoff = wy_date(wy, cutoff_month, cutoff_day)

            # Collapse to one record per parcel
            agg_cols = ["n_cuttings", "n_cp_season", "season_cp_dates_iso"]
            if "matched_minima_iso" in df_seasonal.columns:
                agg_cols.append("matched_minima_iso")
            if "fallback_used" in df_seasonal.columns:
                agg_cols.append("fallback_used")
            agg = (
                df_seasonal
                .groupby("UniqueID", as_index=False)[agg_cols]
                .first()
            )

            uids = agg["UniqueID"].astype(str).tolist()
            if not uids:
                continue

            try:
                et_dict = _load_openet_for_wy(county, wy, uids)
            except FileNotFoundError as e:
                print(f"[info] Skipping {county}, WY{wy}: {e}")
                continue

            # EVI dict for EVI-based cycle start
            evi_dict: Dict[str, pd.Series] = {}
            try:
                evi_dict = load_evi_for_wy(county, wy, uids, evi_mode=evi_mode)
            except (FileNotFoundError, ValueError) as e:
                msg = f"[info] EVI unavailable for {county}, WY{wy}: {e}"
                if evi_required:
                    print(msg + " -> skipping WY due to evi_required=True")
                    continue
                print(msg + " -> proceeding (may fallback to ET-rise)")

            # Whittaker-smoothed EVI for derivative-based segment estimation
            whittaker_evi_dict: Dict[str, pd.Series] = {}
            if config.use_whittaker_interval:
                for uid_str, evi_s in evi_dict.items():
                    if evi_s is not None and not evi_s.empty and evi_s.notna().any():
                        whittaker_evi_dict[uid_str] = whittaker_smooth_series(
                            evi_s, lmbda=config.whittaker_interval_lambda,
                            d=config.whittaker_order,
                        )

            for _, r in agg.iterrows():
                n_cuts = r.get("n_cuttings", 0)
                if pd.notna(n_cuts) and int(n_cuts) < config.min_cuttings:
                    continue
                uid = str(r["UniqueID"])
                et = et_dict.get(uid)
                if et is None or et.empty:
                    continue

                evi = evi_dict.get(uid)

                if evi_required and (evi is None or evi.isna().all()):
                    continue

                # Three-layer cut recovery (same as parcel_summary)
                from .parcel_summary_provider import recover_cut_dates
                cut_dates = recover_cut_dates(
                    r, county, wy,
                    evi_series=evi,
                    min_spacing_days=config.min_spacing_days,
                )
                cut_dates = [d for d in cut_dates if wy_s <= d <= wy_e]
                if not cut_dates:
                    continue

                # Use the same 5-level fallback chain as parcel_summary:
                # GDD5 → Whittaker derivative → Legacy EVI → ET rise → Fixed lookback
                segs = compute_cut_cycle_segments(
                    cut_dates=cut_dates,
                    wy=wy,
                    month_start=1,
                    month_end=12,
                    apply_month_window_to_et_gdd=False,
                    et=et,
                    evi=evi,
                    whittaker_evi=whittaker_evi_dict.get(uid),
                    summer_lookback_range_days=summer_lookback_range_days,
                    winter_lookback_range_days=winter_lookback_range_days,
                    min_gap_before_cut_days=min_gap_before_cut_days,
                    min_segment_days=min_segment_days,
                    rise_days=rise_days,
                    rise_eps=rise_eps,
                )
                # Layer 3: close post-harvest gaps
                from .evi_cut_window_provider import merge_segment_gaps
                segs = merge_segment_gaps(
                    segs,
                    max_extension_days=config.max_segment_gap_extension_days,
                )
                if not segs:
                    continue

                # Build per-cycle dicts matching the old format
                per_seg_et = segment_et_sums_mm(et, segs)
                cycles = []
                for idx, ((seg_s, seg_e), et_val) in enumerate(
                    zip(segs, per_seg_et)
                ):
                    validate_cycle_et_mm(et_val, cut_index=idx + 1)
                    cycles.append({
                        "cut_index": idx + 1,
                        "cut_date": seg_e,
                        "cycle_start": seg_s,
                        "cycle_et_mm": et_val,
                    })
                if not cycles:
                    continue

                # Corrected ET: sum corrected daily ET over same cycle windows
                if et_mode in ("corrected", "both"):
                    try:
                        from .et_provider import compute_daily_and_monthly_for_uid
                        daily_df, _, _, _ = compute_daily_and_monthly_for_uid(
                            county=county, wy=wy, uid=uid,
                            chosen_method=str(method).strip().upper(),
                            n_boot=50,
                        )
                        corr_series = (
                            daily_df["ET_open"] - daily_df["delta_corr"]
                        ).clip(lower=0.0)
                        for c in cycles:
                            s, e = c["cycle_start"], c["cut_date"]
                            sl = corr_series.loc[s:e]
                            c["cycle_et_corrected_mm"] = (
                                float(sl.sum(min_count=1))
                                if sl.notna().any()
                                else np.nan
                            )
                    except Exception:
                        for c in cycles:
                            c["cycle_et_corrected_mm"] = np.nan

                late_cycles = [
                    c for c in cycles if c["cut_date"] >= cutoff
                ]
                n_late = len(late_cycles)
                late_et_mm = (
                    float(np.nansum([c["cycle_et_mm"] for c in late_cycles]))
                    if n_late
                    else 0.0
                )
                total_et_mm = float(np.nansum(
                    [c["cycle_et_mm"] for c in cycles]
                ))

                row_data = {
                    "UniqueID": uid,
                    "county": county,
                    "WY": int(wy),
                    "n_cuttings": float(len(cut_dates)),
                    "n_cp_season": (
                        float(r["n_cp_season"])
                        if pd.notna(r["n_cp_season"])
                        else np.nan
                    ),
                    "cutoff_date": cutoff,
                    "n_late_cuts": int(n_late),
                    "late_et_mm": float(late_et_mm),
                    "total_et_mm": total_et_mm,
                    "late_cycle_et_mm_list": [
                        float(c["cycle_et_mm"]) for c in late_cycles
                    ],
                    "late_cut_dates": [
                        c["cut_date"] for c in late_cycles
                    ],
                }

                if et_mode in ("corrected", "both"):
                    late_et_corr = (
                        float(np.nansum([
                            c.get("cycle_et_corrected_mm", np.nan)
                            for c in late_cycles
                        ]))
                        if n_late else 0.0
                    )
                    total_et_corr = float(np.nansum([
                        c.get("cycle_et_corrected_mm", np.nan)
                        for c in cycles
                    ]))
                    row_data["late_et_corrected_mm"] = late_et_corr
                    row_data["total_et_corrected_mm"] = total_et_corr
                    row_data["late_cycle_et_corrected_mm_list"] = [
                        float(c.get("cycle_et_corrected_mm", np.nan))
                        for c in late_cycles
                    ]

                rows.append(row_data)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No parcel-year rows were built. "
            "Check file paths and WY/county inputs."
        )

    df["n_cuttings"] = pd.to_numeric(df["n_cuttings"], errors="coerce")
    df["n_cp_season"] = pd.to_numeric(df["n_cp_season"], errors="coerce")

    # Optional area / ac-ft conversion
    if compute_area_acft and areas is not None and not areas.empty:
        df = df.merge(areas, on=["UniqueID", "county"], how="left")
        df["late_et_acft"] = mm_to_acft_total(df["late_et_mm"], df["area_acres"])
    else:
        df["area_acres"] = np.nan
        df["late_et_acft"] = np.nan

    df[cut_metric] = df[cut_metric].round().astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Cap-based savings computation
# ---------------------------------------------------------------------------

def compute_cap_savings_per_row(
    late_cycle_et_list: List[float],
    cap_late_cuts: int,
) -> float:
    """Savings under a "cap late cuttings" policy.

    - cap=0: remove all late cycles -> savings = sum(all late ET)
    - cap=k: keep first k late cycles -> savings = sum(remaining)
    """
    if late_cycle_et_list is None:
        return np.nan
    vals = [float(v) for v in late_cycle_et_list if np.isfinite(v)]
    if not vals:
        return 0.0
    if cap_late_cuts <= 0:
        return float(np.sum(vals))
    if len(vals) <= cap_late_cuts:
        return 0.0
    return float(np.sum(vals[cap_late_cuts:]))


def add_cap_savings_columns(
    df: pd.DataFrame,
    cap_values: Iterable[int] = (0, 1, 2, 3),
    use_acft: bool = False,
) -> pd.DataFrame:
    """Add per-parcel-year savings columns for each cap value.

    Creates ``saved_mm_cap{k}`` columns from actual ET.
    If ``late_cycle_et_corrected_mm_list`` exists, also creates
    ``saved_corrected_mm_cap{k}`` from corrected ET.
    If *use_acft* is True and ``area_acres`` exists, also creates
    ``saved_acft_cap{k}``.
    """
    out = df.copy()
    has_corrected = "late_cycle_et_corrected_mm_list" in out.columns

    for k in cap_values:
        # Actual ET savings
        col_mm = f"saved_mm_cap{k}"
        out[col_mm] = out["late_cycle_et_mm_list"].apply(
            lambda lst, _k=k: compute_cap_savings_per_row(lst, cap_late_cuts=int(_k))
        )
        if use_acft:
            if "area_acres" not in out.columns:
                out[f"saved_acft_cap{k}"] = np.nan
            else:
                out[f"saved_acft_cap{k}"] = mm_to_acft_total(
                    out[col_mm], out["area_acres"]
                )

        # Corrected ET savings
        if has_corrected:
            col_corr = f"saved_corrected_mm_cap{k}"
            out[col_corr] = out["late_cycle_et_corrected_mm_list"].apply(
                lambda lst, _k=k: compute_cap_savings_per_row(lst, cap_late_cuts=int(_k))
            )

    return out


# ---------------------------------------------------------------------------
# Normalized saving columns (intensity metrics)
# ---------------------------------------------------------------------------

def add_normalized_saving_columns(
    df: pd.DataFrame,
    *,
    area_acres_col: str = "area_acres",
    saved_total_acft_col: str = "water_saved_total_acft",
    saved_total_mm_col: str = "water_saved_total_mm_area_weighted",
    cuts_removed_col: str = "late_cuts_removed",
    baseline_total_acft_col: str = "baseline_late_et_total_acft",
    baseline_total_mm_col: str = "baseline_late_et_total_mm_area_weighted",
    prefer_acft: bool = True,
) -> pd.DataFrame:
    """Add comparable/intensity metrics to a savings summary.

    Creates area-normalized columns (ac-ft/acre, mm/acre, per-cut)
    and sets ``water_saved`` to the preferred comparable metric.
    """
    out = df.copy()

    if area_acres_col not in out.columns:
        out[area_acres_col] = np.nan
    area_acres = pd.to_numeric(out[area_acres_col], errors="coerce")

    # Path 1: totals in ac-ft (volume)
    if saved_total_acft_col in out.columns:
        saved_acft = pd.to_numeric(
            out[saved_total_acft_col], errors="coerce"
        )

        out["water_saved_acft_per_acre"] = saved_acft / area_acres
        out["water_saved_mm_per_acre"] = (
            acft_per_acre_to_mm(out["water_saved_acft_per_acre"])
        )
        out["water_saved_acft_per_parcel"] = saved_acft / pd.to_numeric(
            out.get("n_parcels", np.nan), errors="coerce"
        )

        if cuts_removed_col in out.columns:
            cuts_removed = pd.to_numeric(
                out[cuts_removed_col], errors="coerce"
            )
            # Keep the total-level per-cut ratio (valid as area-weighted metric)
            out["water_saved_acft_per_cut_removed"] = saved_acft / cuts_removed

            # Per-cut metrics: prefer pre-computed per-parcel means (ETFIX-02)
            if "mean_saved_acft_per_acre_per_cut" in out.columns:
                out["water_saved_acft_per_acre_per_cut_removed"] = out[
                    "mean_saved_acft_per_acre_per_cut"
                ]
            else:
                # Fallback: per-cut savings = total per-acre savings / n_cuts
                # (not dividing depth by group-level cut count)
                out["water_saved_acft_per_acre_per_cut_removed"] = np.where(
                    cuts_removed > 0,
                    out["water_saved_acft_per_acre"] / out.get(
                        "n_parcels", pd.Series([1] * len(out))
                    ).clip(lower=1),
                    0.0,
                )

        if baseline_total_acft_col in out.columns:
            base_acft = pd.to_numeric(
                out[baseline_total_acft_col], errors="coerce"
            )
            out["water_saved_pct_of_baseline_late_et"] = saved_acft / base_acft

        out["water_saved"] = (
            out["water_saved_acft_per_acre"]
            if prefer_acft
            else out["water_saved_mm_per_acre"]
        )

    # Path 2: only depth-like totals (mm)
    elif saved_total_mm_col in out.columns:
        saved_mm = pd.to_numeric(out[saved_total_mm_col], errors="coerce")
        out["water_saved_mm_per_acre"] = saved_mm
        out["water_saved_acft_per_acre"] = mm_to_acft_per_acre(
            out["water_saved_mm_per_acre"]
        )

        if cuts_removed_col in out.columns:
            cuts_removed = pd.to_numeric(
                out[cuts_removed_col], errors="coerce"
            )
            # Per-cut metrics: prefer pre-computed per-parcel means (ETFIX-02)
            if "mean_saved_mm_per_cut" in out.columns:
                out["water_saved_mm_per_acre_per_cut_removed"] = out[
                    "mean_saved_mm_per_cut"
                ]
            else:
                # Fallback: group-level ratio (legacy behavior)
                out["water_saved_mm_per_acre_per_cut_removed"] = (
                    out["water_saved_mm_per_acre"] / cuts_removed
                )

        if baseline_total_mm_col in out.columns:
            base_mm = pd.to_numeric(
                out[baseline_total_mm_col], errors="coerce"
            )
            out["water_saved_pct_of_baseline_late_et"] = saved_mm / base_mm

        out["water_saved"] = (
            out["water_saved_mm_per_acre"]
            if not prefer_acft
            else out["water_saved_acft_per_acre"]
        )

    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarize_late_et_by_late_cut_count(
    df: pd.DataFrame,
    et_col: str = "late_et_mm",
) -> pd.DataFrame:
    """Frequency table: n, mean, median, p25, p75 of late ET grouped by
    n_late_cuts."""
    d = df.copy()
    d = d[np.isfinite(d[et_col].values)]
    g = (
        d.groupby("n_late_cuts")[et_col]
        .agg(
            n="size",
            mean="mean",
            median="median",
            p25=lambda x: np.nanpercentile(x, 25),
            p75=lambda x: np.nanpercentile(x, 75),
        )
        .reset_index()
        .sort_values("n_late_cuts")
    )
    return g


# ---------------------------------------------------------------------------
# Savings summary (county-year / year-level aggregation)
# ---------------------------------------------------------------------------

def _has_valid_area(
    df: pd.DataFrame,
    area_col: str = "area_acres",
) -> bool:
    return (
        area_col in df.columns
        and df[area_col].notna().any()
        and (df[area_col].fillna(0) > 0).any()
    )


def _sum_unique_area_acres(
    sub: pd.DataFrame,
    area_col: str = "area_acres",
) -> float:
    if area_col not in sub.columns:
        return np.nan
    d0 = sub.dropna(subset=[area_col]).drop_duplicates(subset=["UniqueID"])
    if d0.empty:
        return np.nan
    return float(d0[area_col].sum())


def make_savings_summary(
    df_savings: pd.DataFrame,
    *,
    cap_k: int,
    group_cols: List[str],
    prefer_acft: bool = True,
) -> pd.DataFrame:
    """Build a grouped savings summary with totals and intensity metrics.

    Requires *df_savings* to contain ``late_et_mm``, ``n_late_cuts``,
    and ``saved_mm_cap{cap_k}``.  Optionally uses ``area_acres`` and
    ``late_et_acft`` for volume-based metrics.

    Args:
        df_savings: Parcel-year DataFrame with savings columns.
        cap_k: Cap value (number of allowed late cuts).
        group_cols: Grouping columns (e.g. ["WY"] or ["county", "WY"]).
        prefer_acft: Set ``water_saved`` to ac-ft/acre (True) or mm.

    Returns:
        Summary DataFrame with one row per group.
    """
    saved_mm_col = f"saved_mm_cap{int(cap_k)}"
    if saved_mm_col not in df_savings.columns:
        raise ValueError(
            f"Missing savings column '{saved_mm_col}'. "
            "Call add_cap_savings_columns() first."
        )

    d = df_savings.copy()
    for c in group_cols:
        if c not in d.columns:
            raise ValueError(f"group col '{c}' not found in df_savings.")

    d["late_et_mm"] = pd.to_numeric(d["late_et_mm"], errors="coerce")
    d[saved_mm_col] = pd.to_numeric(
        d[saved_mm_col], errors="coerce"
    ).fillna(0.0)
    d["n_late_cuts"] = pd.to_numeric(d["n_late_cuts"], errors="coerce")
    d["late_cuts_removed_row"] = np.maximum(
        0.0, d["n_late_cuts"] - float(cap_k)
    )

    # -- Per-parcel per-cut intensity (ETFIX-02: compute BEFORE groupby) --
    mask_cuts = d["late_cuts_removed_row"] > 0

    # Per-parcel saved mm per cut removed (depth, no area needed)
    d["saved_mm_per_cut_parcel"] = np.where(
        mask_cuts,
        d[saved_mm_col] / d["late_cuts_removed_row"],
        np.nan,
    )

    # Per-parcel saved ac-ft/acre per cut removed (depth conversion)
    d["saved_acft_per_acre_per_cut_parcel"] = np.where(
        mask_cuts,
        mm_to_acft_per_acre(d[saved_mm_col] / d["late_cuts_removed_row"]),
        np.nan,
    )

    # Validate per-cutting ac-ft/acre values (PIPE-01: deferred from Phase 1)
    import logging
    _logger = logging.getLogger(__name__)
    _valid_mask = d["saved_acft_per_acre_per_cut_parcel"].notna()
    for _idx in d.index[_valid_mask]:
        try:
            validate_per_cutting_acft_per_acre(
                float(d.loc[_idx, "saved_acft_per_acre_per_cut_parcel"]),
                uid=str(d.loc[_idx, "UniqueID"]) if "UniqueID" in d.columns else "",
                cut_index=0,
            )
        except ValueError as e:
            _logger.warning(str(e))

    has_area = _has_valid_area(d, "area_acres")
    if has_area:
        d["area_acres"] = pd.to_numeric(d["area_acres"], errors="coerce")

        if "late_et_acft" in d.columns:
            d["late_et_acft"] = pd.to_numeric(
                d["late_et_acft"], errors="coerce"
            )
            baseline_acft_row = d["late_et_acft"]
        else:
            baseline_acft_row = mm_to_acft_total(
                d["late_et_mm"], d["area_acres"]
            )

        saved_acft_row = mm_to_acft_total(d[saved_mm_col], d["area_acres"])

        d["_baseline_acft_row"] = baseline_acft_row
        d["_saved_acft_row"] = saved_acft_row
        d["_baseline_mm_x_area"] = d["late_et_mm"] * d["area_acres"]
        d["_saved_mm_x_area"] = d[saved_mm_col] * d["area_acres"]

    def _agg(sub: pd.DataFrame) -> pd.Series:
        n_parcels = (
            int(sub["UniqueID"].nunique())
            if "UniqueID" in sub.columns
            else int(sub.shape[0])
        )
        n_years = (
            int(sub["WY"].nunique()) if "WY" in sub.columns else 1
        )

        result: Dict[str, object] = {
            "n_parcels": n_parcels,
            "n_years": n_years,
            "late_cuts_removed": float(
                np.nansum(sub["late_cuts_removed_row"].values)
            ),
            "cut_cap_k": int(cap_k),
        }

        if has_area:
            area_acre_year = float(
                np.nansum(
                    pd.to_numeric(sub["area_acres"], errors="coerce").values
                )
            )
            result["area_acres"] = area_acre_year
            result["area_acres_unique"] = _sum_unique_area_acres(
                sub, "area_acres"
            )

            base_acft = float(
                np.nansum(sub["_baseline_acft_row"].values)
            )
            saved_acft = float(
                np.nansum(sub["_saved_acft_row"].values)
            )

            result["baseline_late_et_total_acft"] = base_acft
            result["water_saved_total_acft"] = saved_acft
            result["scenario_late_et_total_acft"] = base_acft - saved_acft

            if area_acre_year > 0:
                result["baseline_late_et_total_mm_area_weighted"] = float(
                    np.nansum(sub["_baseline_mm_x_area"].values)
                    / area_acre_year
                )
                result["water_saved_total_mm_area_weighted"] = float(
                    np.nansum(sub["_saved_mm_x_area"].values)
                    / area_acre_year
                )
                result["scenario_late_et_total_mm_area_weighted"] = (
                    result["baseline_late_et_total_mm_area_weighted"]
                    - result["water_saved_total_mm_area_weighted"]
                )
            else:
                result["baseline_late_et_total_mm_area_weighted"] = np.nan
                result["water_saved_total_mm_area_weighted"] = np.nan
                result["scenario_late_et_total_mm_area_weighted"] = np.nan
        else:
            result["area_acres"] = np.nan
            result["area_acres_unique"] = np.nan
            result["baseline_late_et_total_mm_area_weighted"] = float(
                np.nanmean(sub["late_et_mm"].values)
            )
            result["water_saved_total_mm_area_weighted"] = float(
                np.nanmean(sub[saved_mm_col].values)
            )
            result["scenario_late_et_total_mm_area_weighted"] = (
                result["baseline_late_et_total_mm_area_weighted"]
                - result["water_saved_total_mm_area_weighted"]
            )

        # Per-parcel-averaged intensity metrics (ETFIX-02)
        result["mean_saved_mm_per_cut"] = float(
            np.nanmean(sub["saved_mm_per_cut_parcel"].values)
        ) if sub["saved_mm_per_cut_parcel"].notna().any() else np.nan

        result["mean_saved_acft_per_acre_per_cut"] = float(
            np.nanmean(sub["saved_acft_per_acre_per_cut_parcel"].values)
        ) if sub["saved_acft_per_acre_per_cut_parcel"].notna().any() else np.nan

        return pd.Series(result)

    summary = d.groupby(group_cols, dropna=False).apply(_agg).reset_index()

    summary = add_normalized_saving_columns(
        summary,
        area_acres_col="area_acres",
        saved_total_acft_col="water_saved_total_acft",
        saved_total_mm_col="water_saved_total_mm_area_weighted",
        cuts_removed_col="late_cuts_removed",
        baseline_total_acft_col="baseline_late_et_total_acft",
        baseline_total_mm_col="baseline_late_et_total_mm_area_weighted",
        prefer_acft=(prefer_acft and has_area),
    )

    return summary


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_workflow_outputs(
    out: Dict[str, object],
    out_dir: Path,
) -> Dict[str, Path]:
    """Save workflow outputs to CSV files.

    Args:
        out: Output dict from run_late_water_saving_workflow().
        out_dir: Output directory.

    Returns:
        Dict mapping key to file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    p = out_dir / "late_cut_base_parcel_year.csv"
    out["df"].to_csv(p, index=False)
    paths["df"] = p
    print(f"[saved] {p}")

    p = out_dir / "late_cut_freq_by_n_late_cuts.csv"
    out["df_freq_mm"].to_csv(p, index=False)
    paths["df_freq_mm"] = p
    print(f"[saved] {p}")

    p = out_dir / "late_cut_savings_parcel_year_mm.csv"
    out["df_savings_mm"].to_csv(p, index=False)
    paths["df_savings_mm"] = p
    print(f"[saved] {p}")

    if "df_savings_acft" in out:
        p = out_dir / "late_cut_savings_parcel_year_acft.csv"
        out["df_savings_acft"].to_csv(p, index=False)
        paths["df_savings_acft"] = p
        print(f"[saved] {p}")

    return paths


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_late_water_saving_workflow(
    wy_start: int,
    wy_end: int,
    counties: Optional[Iterable[str]] = None,
    cutoff_month: Optional[int] = None,
    cutoff_day: Optional[int] = None,
    compute_area_acft: bool = False,
    cap_values: Optional[Iterable[int]] = None,
    *,
    evi_mode: Optional[str] = None,
    summer_lookback_range_days: Optional[Tuple[int, int]] = None,
    winter_lookback_range_days: Optional[Tuple[int, int]] = None,
    min_gap_before_cut_days: Optional[int] = None,
    min_segment_days: Optional[int] = None,
    evi_required: bool = False,
    fallback_to_et_rise: bool = True,
    out_dir: Optional[Path] = None,
    export_csv: bool = True,
    et_mode: str = "actual",
    method: str = "A",
) -> Dict[str, object]:
    """Run the full late-water savings workflow.

    Builds the parcel-year late-cut dataset, computes cap savings,
    optionally exports CSVs.

    Args:
        wy_start: First water year.
        wy_end: Last water year.
        counties: County names (defaults to COUNTY_ORDER).
        cutoff_month: Month for late-season cutoff (default: from config).
        cutoff_day: Day for late-season cutoff (default: from config).
        compute_area_acft: Include area-based ac-ft conversions.
        cap_values: Cap values for savings scenarios (default: from config).
        evi_mode: "smoothed" or "gapfilled" (default: from config).
        summer_lookback_range_days: EVI lookback for Mar-Sep.
        winter_lookback_range_days: EVI lookback for Oct-Feb.
        min_gap_before_cut_days: Buffer days before cut.
        min_segment_days: Minimum cycle length.
        evi_required: Skip parcels without EVI.
        fallback_to_et_rise: ET-rise heuristic when EVI missing.
        out_dir: Output directory for CSVs.
        export_csv: Whether to export CSVs.

    Returns:
        Dict with keys: df, df_freq_mm, df_savings_mm,
        and optionally df_savings_acft, paths.
    """
    # Defaults from config
    if cutoff_month is None:
        cutoff_month = config.late_cutoff_month
    if cutoff_day is None:
        cutoff_day = config.late_cutoff_day
    if cap_values is None:
        cap_values = config.cap_values
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
    if out_dir is None:
        out_dir = config.water_saving_out_dir

    cap_values = tuple(cap_values)

    print(f"Late-water savings workflow: WY {wy_start}-{wy_end}")
    print(f"Cutoff: month={cutoff_month}, day={cutoff_day}")
    print(f"Cap values: {cap_values}")
    print(f"EVI mode: {evi_mode}, ET mode: {et_mode}, Method: {method}")
    print(f"EVI required={evi_required}, fallback_to_et_rise={fallback_to_et_rise}")

    df = build_late_cut_dataset(
        wy_start=wy_start,
        wy_end=wy_end,
        counties=counties,
        cutoff_month=cutoff_month,
        cutoff_day=cutoff_day,
        compute_area_acft=compute_area_acft,
        evi_mode=evi_mode,
        summer_lookback_range_days=summer_lookback_range_days,
        winter_lookback_range_days=winter_lookback_range_days,
        min_gap_before_cut_days=min_gap_before_cut_days,
        min_segment_days=min_segment_days,
        evi_required=evi_required,
        fallback_to_et_rise=fallback_to_et_rise,
        et_mode=et_mode,
        method=method,
    )

    print(f"Built dataset: {len(df):,} parcel-years, "
          f"{df['UniqueID'].nunique():,} unique parcels, "
          f"{df['county'].nunique()} counties")

    df_freq_mm = summarize_late_et_by_late_cut_count(df, et_col="late_et_mm")
    df_sav_mm = add_cap_savings_columns(
        df, cap_values=cap_values, use_acft=False
    )

    result: Dict[str, object] = {
        "df": df,
        "df_freq_mm": df_freq_mm,
        "df_savings_mm": df_sav_mm,
    }

    if compute_area_acft:
        result["df_savings_acft"] = add_cap_savings_columns(
            df, cap_values=cap_values, use_acft=True
        )

    # Export
    if export_csv:
        paths = save_workflow_outputs(result, out_dir)
        result["paths"] = paths

    return result
