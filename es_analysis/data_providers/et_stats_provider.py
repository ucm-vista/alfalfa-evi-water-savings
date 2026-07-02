"""ET correction statistics provider.

Runs the off-phase ET correction pipeline across counties and water years,
producing aggregation tables, descriptive statistics, narrative paragraphs,
and CSV exports.

Source: alfalfa_evi_jovyan.py lines 5238-5692
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import config
from .evi_provider import normalize_county_name
from .et_provider import (
    list_uids_for_county_wy,
    compute_daily_and_monthly_for_uid,
)
from .spatial_provider import COUNTY_ORDER
from ..utils.validation import validate_annual_et_mm, validate_per_cutting_acft_per_acre


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    """Element-wise division returning NaN where b is zero or non-finite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[m] = a[m] / b[m]
    return out


def _desc(x: pd.Series) -> pd.Series:
    """Descriptive statistics for a numeric series."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return pd.Series({
            "n": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
            "p10": np.nan, "p25": np.nan, "median": np.nan,
            "p75": np.nan, "p90": np.nan, "max": np.nan,
        })
    q = np.nanquantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    return pd.Series({
        "n": int(x.size),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.nanmin(x)),
        "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "p90": float(q[4]),
        "max": float(np.nanmax(x)),
    })


def _month_to_wy_month(dt: pd.Timestamp) -> int:
    """Convert calendar month to water-year month (Oct=1 .. Sep=12)."""
    return int(((dt.month - 10) % 12) + 1)


def _wy_month_label(wym: int) -> str:
    """Convert water-year month number to abbreviated label."""
    labs = [
        "Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
        "Apr", "May", "Jun", "Jul", "Aug", "Sep",
    ]
    return labs[wym - 1]


# ---------------------------------------------------------------------------
# Parallel worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _process_single_uid(args_tuple):
    """Process one (county, wy, uid) tuple. Top-level for ProcessPoolExecutor.

    Returns (result_type, data) where result_type is 'success' or 'fail'.
    """
    (county, wy, uid, cloud_cover_max, n_cp_season_filter, r_days,
     pre_window, post_window, low_quantile, chosen_method, ci_alpha,
     n_boot, inflate_by_cloud_gap) = args_tuple

    try:
        daily_df, monthly_df, harvest_dates, passes_df = (
            compute_daily_and_monthly_for_uid(
                county=county,
                wy=int(wy),
                uid=str(uid),
                cloud_cover_max=float(cloud_cover_max),
                n_cp_season_filter=n_cp_season_filter,
                r_days=int(r_days),
                pre_window_days=int(pre_window),
                post_window_days=int(post_window),
                low_quantile=float(low_quantile),
                chosen_method=str(chosen_method),
                ci_alpha=float(ci_alpha),
                n_boot=int(n_boot),
                inflate_by_cloud_gap=bool(inflate_by_cloud_gap),
            )
        )
    except Exception as e:
        return ("fail", {
            "county": county, "WY": wy,
            "UniqueID": str(uid),
            "stage": "compute_daily_and_monthly_for_uid",
            "error": str(e),
        })

    # Enrich monthly_df
    mdf = monthly_df.copy().reset_index()
    if "index" in mdf.columns:
        mdf = mdf.rename(columns={"index": "month_start"})
    if "month" in mdf.columns and "month_start" not in mdf.columns:
        mdf = mdf.rename(columns={"month": "month_start"})
    mdf["month_start"] = pd.to_datetime(mdf["month_start"]).dt.normalize()

    for c in [
        "ET_open", "ET_corr", "delta_corr",
        "ET_corr_ci_low", "ET_corr_ci_high",
    ]:
        if c not in mdf.columns:
            mdf[c] = np.nan

    mdf["delta_mm"] = (mdf["ET_open"] - mdf["ET_corr"]).astype(float)
    mdf["pct_reduction"] = 100.0 * _safe_div(
        mdf["delta_mm"].to_numpy(float),
        mdf["ET_open"].to_numpy(float),
    )
    mdf["ci_width_mm"] = (
        mdf["ET_corr_ci_high"] - mdf["ET_corr_ci_low"]
    ).astype(float)
    mdf["ci_halfwidth_mm"] = 0.5 * mdf["ci_width_mm"]
    mdf["ci_rel_width_pct_of_corr"] = 100.0 * _safe_div(
        mdf["ci_width_mm"].to_numpy(float),
        mdf["ET_corr"].to_numpy(float),
    )

    mdf["wy_month"] = mdf["month_start"].apply(_month_to_wy_month).astype(int)
    mdf["wy_month_label"] = mdf["wy_month"].apply(_wy_month_label)

    mdf["county"] = county
    mdf["WY"] = int(wy)
    mdf["UniqueID"] = str(uid)
    mdf["method"] = str(chosen_method).upper()
    mdf["ci_alpha"] = float(ci_alpha)
    mdf["n_boot"] = int(n_boot)
    mdf["r_days"] = int(r_days)
    mdf["pre_window_days"] = int(pre_window)
    mdf["post_window_days"] = int(post_window)
    mdf["low_quantile"] = float(low_quantile)
    mdf["inflate_by_cloud_gap"] = bool(inflate_by_cloud_gap)

    keep = [
        "county", "WY", "UniqueID", "month_start",
        "wy_month", "wy_month_label",
        "ET_open", "ET_corr", "delta_corr", "delta_mm",
        "pct_reduction",
        "ET_corr_ci_low", "ET_corr_ci_high",
        "ci_width_mm", "ci_halfwidth_mm",
        "ci_rel_width_pct_of_corr",
        "method", "ci_alpha", "n_boot", "r_days",
        "pre_window_days", "post_window_days",
        "low_quantile", "inflate_by_cloud_gap",
    ]
    mdf = mdf[[c for c in keep if c in mdf.columns]]

    # Parcel-year summary
    py = {
        "county": county,
        "WY": int(wy),
        "UniqueID": str(uid),
        "months_count": int(mdf.shape[0]),
        "harvest_events_n": (
            int(len(harvest_dates))
            if harvest_dates is not None else np.nan
        ),
        "landsat_clear_passes_n": (
            int(passes_df["date_only"].nunique())
            if passes_df is not None and not passes_df.empty
            else 0
        ),
        "ET_open_annual_mm": (
            float(np.nansum(mdf["ET_open"].to_numpy(float)))
            if mdf["ET_open"].notna().any() else np.nan
        ),
        "ET_corr_annual_mm": (
            float(np.nansum(mdf["ET_corr"].to_numpy(float)))
            if mdf["ET_corr"].notna().any() else np.nan
        ),
        "delta_annual_mm": (
            float(np.nansum(mdf["delta_mm"].to_numpy(float)))
            if mdf["delta_mm"].notna().any() else np.nan
        ),
    }
    py["annual_pct_reduction"] = float(
        100.0 * _safe_div(
            py["delta_annual_mm"], py["ET_open_annual_mm"],
        )
    )
    py["ET_corr_ci_low_annual_mm"] = (
        float(np.nansum(mdf["ET_corr_ci_low"].to_numpy(float)))
        if mdf["ET_corr_ci_low"].notna().any() else np.nan
    )
    py["ET_corr_ci_high_annual_mm"] = (
        float(np.nansum(mdf["ET_corr_ci_high"].to_numpy(float)))
        if mdf["ET_corr_ci_high"].notna().any() else np.nan
    )
    py["annual_ci_width_mm"] = (
        py["ET_corr_ci_high_annual_mm"]
        - py["ET_corr_ci_low_annual_mm"]
    )
    py["annual_ci_rel_width_pct_of_corr"] = float(
        100.0 * _safe_div(
            py["annual_ci_width_mm"],
            py["ET_corr_annual_mm"],
        )
    )
    py["method"] = str(chosen_method).upper()
    py["ci_alpha"] = float(ci_alpha)

    # Validate (warnings only, don't fail)
    validate_annual_et_mm(
        py["ET_open_annual_mm"], uid=str(uid), wy=int(wy),
    )
    n_cuts = py.get("harvest_events_n", 0)
    if (not np.isnan(py["ET_open_annual_mm"])
            and isinstance(n_cuts, (int, float))
            and not np.isnan(n_cuts)
            and n_cuts > 0):
        per_cut_acft = py["ET_open_annual_mm"] / 304.8 / n_cuts
        try:
            validate_per_cutting_acft_per_acre(
                per_cut_acft, uid=str(uid), cut_index=0,
            )
        except ValueError:
            pass  # warning already emitted

    return ("success", {"monthly": mdf, "parcel_year": py})


# ---------------------------------------------------------------------------
# Aggregation builders
# ---------------------------------------------------------------------------

def _build_county_wy_table(df_parcel_year: pd.DataFrame) -> pd.DataFrame:
    """County x WY annual totals and percent reduction."""
    cwy = (
        df_parcel_year.groupby(["county", "WY"], as_index=False)
        .agg(
            parcel_years=("UniqueID", "size"),
            unique_parcels=("UniqueID", "nunique"),
            ET_open_total_mm=("ET_open_annual_mm", "sum"),
            ET_corr_total_mm=("ET_corr_annual_mm", "sum"),
            delta_total_mm=("delta_annual_mm", "sum"),
            ET_corr_ci_low_total_mm=("ET_corr_ci_low_annual_mm", "sum"),
            ET_corr_ci_high_total_mm=("ET_corr_ci_high_annual_mm", "sum"),
        )
    )
    cwy["pct_reduction_total"] = 100.0 * _safe_div(
        cwy["delta_total_mm"].to_numpy(float),
        cwy["ET_open_total_mm"].to_numpy(float),
    )
    cwy["total_ci_width_mm"] = (
        cwy["ET_corr_ci_high_total_mm"] - cwy["ET_corr_ci_low_total_mm"]
    )
    return cwy


def _build_county_annual_table(df_parcel_year: pd.DataFrame) -> pd.DataFrame:
    """County pooled across years."""
    ca = (
        df_parcel_year.groupby("county", as_index=False)
        .agg(
            parcel_years=("UniqueID", "size"),
            unique_parcels=("UniqueID", "nunique"),
            ET_open_total_mm=("ET_open_annual_mm", "sum"),
            ET_corr_total_mm=("ET_corr_annual_mm", "sum"),
            delta_total_mm=("delta_annual_mm", "sum"),
            annual_pct_reduction_mean=("annual_pct_reduction", "mean"),
            annual_pct_reduction_median=("annual_pct_reduction", "median"),
            annual_ci_rel_width_pct_median=(
                "annual_ci_rel_width_pct_of_corr", "median"
            ),
        )
    )
    ca["pct_reduction_total"] = 100.0 * _safe_div(
        ca["delta_total_mm"].to_numpy(float),
        ca["ET_open_total_mm"].to_numpy(float),
    )
    return ca


def _build_wy_annual_table(df_parcel_year: pd.DataFrame) -> pd.DataFrame:
    """WY pooled across counties."""
    wa = (
        df_parcel_year.groupby("WY", as_index=False)
        .agg(
            parcel_years=("UniqueID", "size"),
            unique_parcels=("UniqueID", "nunique"),
            ET_open_total_mm=("ET_open_annual_mm", "sum"),
            ET_corr_total_mm=("ET_corr_annual_mm", "sum"),
            delta_total_mm=("delta_annual_mm", "sum"),
            annual_pct_reduction_mean=("annual_pct_reduction", "mean"),
            annual_pct_reduction_median=("annual_pct_reduction", "median"),
            annual_ci_rel_width_pct_median=(
                "annual_ci_rel_width_pct_of_corr", "median"
            ),
        )
    )
    wa["pct_reduction_total"] = 100.0 * _safe_div(
        wa["delta_total_mm"].to_numpy(float),
        wa["ET_open_total_mm"].to_numpy(float),
    )
    return wa


def _build_monthly_pool_table(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """Monthly seasonality pooled (Oct..Sep)."""
    mp = (
        df_monthly.groupby(["wy_month", "wy_month_label"], as_index=False)
        .agg(
            parcel_months=("UniqueID", "size"),
            unique_parcels=("UniqueID", "nunique"),
            ET_open_total_mm=("ET_open", "sum"),
            ET_corr_total_mm=("ET_corr", "sum"),
            delta_total_mm=("delta_mm", "sum"),
            pct_reduction_mean=("pct_reduction", "mean"),
            pct_reduction_median=("pct_reduction", "median"),
            ci_rel_width_pct_median=(
                "ci_rel_width_pct_of_corr", "median"
            ),
        )
    )
    mp["pct_reduction_total"] = 100.0 * _safe_div(
        mp["delta_total_mm"].to_numpy(float),
        mp["ET_open_total_mm"].to_numpy(float),
    )
    return mp.sort_values("wy_month")


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def _build_descriptive_stats(
    df_monthly: pd.DataFrame,
    df_parcel_year: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build descriptive-statistics tables at month and year scales."""
    metrics_month = {
        "ET_open_mm_month": df_monthly["ET_open"],
        "ET_corr_mm_month": df_monthly["ET_corr"],
        "delta_mm_month": df_monthly["delta_mm"],
        "pct_reduction_month": df_monthly["pct_reduction"],
        "ci_width_mm_month": df_monthly["ci_width_mm"],
        "ci_rel_width_pct_month": df_monthly["ci_rel_width_pct_of_corr"],
    }
    desc_month = (
        pd.DataFrame({k: _desc(v) for k, v in metrics_month.items()})
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    desc_month["scale"] = "parcel-month (overall)"

    metrics_year = {
        "ET_open_mm_year": df_parcel_year["ET_open_annual_mm"],
        "ET_corr_mm_year": df_parcel_year["ET_corr_annual_mm"],
        "delta_mm_year": df_parcel_year["delta_annual_mm"],
        "pct_reduction_year": df_parcel_year["annual_pct_reduction"],
        "annual_ci_rel_width_pct": df_parcel_year[
            "annual_ci_rel_width_pct_of_corr"
        ],
    }
    desc_year = (
        pd.DataFrame({k: _desc(v) for k, v in metrics_year.items()})
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    desc_year["scale"] = "parcel-year (overall)"

    return desc_month, desc_year


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

def _generate_narrative(
    df_parcel_year: pd.DataFrame,
    df_monthly: pd.DataFrame,
    county_annual: pd.DataFrame,
    *,
    n_counties: int,
    wy_start: int,
    wy_end: int,
    n_parcel_years: int,
    n_unique_parcels: int,
    attempted: int,
    succeeded: int,
) -> Tuple[str, str]:
    """Generate two Results/Discussion-ready paragraphs.

    Returns:
        Tuple of (paragraph_1, paragraph_2).
    """
    success_rate = (
        100.0 * succeeded / attempted if attempted > 0 else float("nan")
    )

    tot_open = float(df_parcel_year["ET_open_annual_mm"].sum())
    tot_corr = float(df_parcel_year["ET_corr_annual_mm"].sum())
    tot_delta = float(df_parcel_year["delta_annual_mm"].sum())
    tot_pct_red = float(100.0 * _safe_div(tot_delta, tot_open))

    med_ann_unc = float(np.nanmedian(
        df_parcel_year["annual_ci_rel_width_pct_of_corr"].to_numpy(float)
    ))
    med_mon_unc = float(np.nanmedian(
        df_monthly["ci_rel_width_pct_of_corr"].to_numpy(float)
    ))

    if not county_annual.empty:
        hi = county_annual.sort_values(
            "pct_reduction_total", ascending=False
        ).iloc[0]
        lo = county_annual.sort_values(
            "pct_reduction_total", ascending=True
        ).iloc[0]
        hi_txt = f"{hi['county']} ({hi['pct_reduction_total']:.2f}%)"
        lo_txt = f"{lo['county']} ({lo['pct_reduction_total']:.2f}%)"
    else:
        hi_txt, lo_txt = "N/A", "N/A"

    p1 = (
        f"Across the {n_counties} counties and WY {wy_start}\u2013{wy_end}, "
        f"the off-phase correction workflow produced {n_parcel_years:,} "
        f"parcel-year realizations ({n_unique_parcels:,} unique parcels), "
        f"with an overall processing success rate of {success_rate:.1f}% "
        f"({succeeded:,} of {attempted:,} attempted UID\u00d7WY "
        f"computations). Summed over all successful parcel-years, annual "
        f"actual ET from OpenET totaled {tot_open:,.1f} mm, whereas the "
        f"corrected ET totaled {tot_corr:,.1f} mm, implying an aggregate "
        f"reduction of {tot_delta:,.1f} mm ({tot_pct_red:.2f}%)."
    )

    p2 = (
        f"Spatially, the aggregate percent reduction (computed from "
        f"county-level totals) varied across the corridor, with the highest "
        f"reduction observed in {hi_txt} and the lowest in {lo_txt}, "
        f"reflecting systematic differences in the magnitude and timing of "
        f"post-harvest off-phase behavior. Uncertainty was summarized using "
        f"the relative confidence interval width of corrected ET; the median "
        f"relative uncertainty was {med_mon_unc:.2f}% at the monthly scale "
        f"and {med_ann_unc:.2f}% at the annual scale (annual bounds computed "
        f"conservatively by summing monthly CI limits). Together, these "
        f"results indicate that the correction consistently reduces OpenET "
        f"totals at corridor scale while providing a transparent uncertainty "
        f"envelope suitable for inter-county and interannual comparisons."
    )

    return p1, p2


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_stats_csvs(
    *,
    df_monthly: pd.DataFrame,
    df_parcel_year: pd.DataFrame,
    df_fail: pd.DataFrame,
    county_wy: pd.DataFrame,
    county_annual: pd.DataFrame,
    wy_annual: pd.DataFrame,
    monthly_pool: pd.DataFrame,
    desc_month: pd.DataFrame,
    desc_year: pd.DataFrame,
    out_dir: Path,
    method: str,
    cloud_cover: float,
    ci_alpha: float,
    n_boot: int,
    wy_start: int,
    wy_end: int,
) -> Dict[str, Path]:
    """Export all statistics tables to CSV files.

    Returns:
        Dict mapping table name to file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"method{str(method).upper()}"
        f"_cc{float(cloud_cover):.0f}"
        f"_ci{int((1 - float(ci_alpha)) * 100)}"
        f"_boot{int(n_boot)}"
        f"_WY{int(wy_start)}-{int(wy_end)}"
    )

    paths = {}
    tables = {
        "monthly_long": (df_monthly, f"offphase_monthly_long_{tag}.csv"),
        "parcel_year": (df_parcel_year, f"offphase_parcel_year_{tag}.csv"),
        "county_wy": (county_wy, f"offphase_county_WY_annual_totals_{tag}.csv"),
        "county_annual": (county_annual, f"offphase_county_annual_summary_{tag}.csv"),
        "wy_annual": (wy_annual, f"offphase_WY_annual_summary_{tag}.csv"),
        "monthly_pool": (monthly_pool, f"offphase_monthly_seasonality_{tag}.csv"),
        "desc_month": (desc_month, f"offphase_desc_overall_parcel_month_{tag}.csv"),
        "desc_year": (desc_year, f"offphase_desc_overall_parcel_year_{tag}.csv"),
        "failures": (df_fail, f"offphase_failures_{tag}.csv"),
    }

    for name, (df, fname) in tables.items():
        fp = out_dir / fname
        df.to_csv(fp, index=False)
        paths[name] = fp
        print(f"[saved] {name}: {fp}")

    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_et_correction_stats(
    *,
    counties: Optional[List[str]] = None,
    wy_start: int = 2019,
    wy_end: int = 2024,
    n_cp_season_filter: Optional[int] = None,
    max_uids_per_county_wy: Optional[int] = None,
    uid_sample_seed: int = 42,
    cloud_cover_max: float = None,
    r_days: int = None,
    pre_window: int = None,
    post_window: int = None,
    low_quantile: float = None,
    chosen_method: str = None,
    ci_alpha: float = None,
    n_boot: int = None,
    inflate_by_cloud_gap: bool = None,
    out_dir: Optional[Path] = None,
    export_csv: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full ET correction statistics pipeline.

    Iterates over counties and water years, computing per-parcel
    monthly ET corrections, then builds aggregation tables,
    descriptive statistics, and narrative summaries.

    Args:
        counties: List of county names. Defaults to COUNTY_ORDER.
        wy_start: First water year.
        wy_end: Last water year.
        n_cp_season_filter: Optional filter on n_cp_season.
        max_uids_per_county_wy: Optional cap on UIDs per county-year.
        uid_sample_seed: Random seed for UID sampling.
        cloud_cover_max: Max cloud cover for Landsat passes.
        r_days: Off-phase recovery days.
        pre_window: Pre-harvest averaging window (days).
        post_window: Post-harvest averaging window (days).
        low_quantile: Low quantile for f_min estimation.
        chosen_method: "A" or "B" correction method.
        ci_alpha: Confidence interval alpha.
        n_boot: Number of bootstrap samples.
        inflate_by_cloud_gap: Inflate correction by cloud gap.
        out_dir: Output directory for CSVs.
        export_csv: Whether to export CSV files.

    Returns:
        Dict of DataFrames: df_monthly, df_parcel_year, df_fail,
        county_wy, county_annual, wy_annual, monthly_pool,
        desc_month, desc_year.
    """
    # Resolve defaults from config
    if counties is None:
        counties = list(COUNTY_ORDER)
    if cloud_cover_max is None:
        cloud_cover_max = config.cloud_cover_max
    if r_days is None:
        r_days = config.r_days
    if pre_window is None:
        pre_window = config.pre_window
    if post_window is None:
        post_window = config.post_window
    if low_quantile is None:
        low_quantile = config.low_quantile
    if chosen_method is None:
        chosen_method = config.chosen_method
    if ci_alpha is None:
        ci_alpha = config.ci_alpha
    if n_boot is None:
        n_boot = config.n_boot_bulk
    if inflate_by_cloud_gap is None:
        inflate_by_cloud_gap = config.inflate_by_cloud_gap
    if max_uids_per_county_wy is None:
        max_uids_per_county_wy = config.max_uids_per_county_wy
    if out_dir is None:
        out_dir = config.statistics_export_dir

    counties_norm = [normalize_county_name(c) for c in counties]
    years = list(range(int(wy_start), int(wy_end) + 1))
    rng = np.random.default_rng(uid_sample_seed)

    rows_monthly: List[pd.DataFrame] = []
    rows_parcel_year: List[dict] = []
    rows_fail: List[dict] = []

    # Build work items: list all (county, wy, uid) tuples
    work_items = []
    for county in counties_norm:
        for wy in years:
            try:
                uids = list_uids_for_county_wy(
                    county, wy,
                    n_cp_season_filter=n_cp_season_filter,
                )
            except Exception as e:
                rows_fail.append({
                    "county": county, "WY": wy, "UniqueID": None,
                    "stage": "list_uids_for_county_wy", "error": str(e),
                })
                continue

            if not uids:
                rows_fail.append({
                    "county": county, "WY": wy, "UniqueID": None,
                    "stage": "list_uids_for_county_wy",
                    "error": "No UIDs returned",
                })
                continue

            if (max_uids_per_county_wy is not None
                    and len(uids) > int(max_uids_per_county_wy)):
                uids = rng.choice(
                    uids, size=int(max_uids_per_county_wy), replace=False,
                ).tolist()

            for uid in uids:
                work_items.append((
                    county, wy, uid, cloud_cover_max, n_cp_season_filter,
                    r_days, pre_window, post_window, low_quantile,
                    chosen_method, ci_alpha, n_boot, inflate_by_cloud_gap,
                ))

    attempted = len(work_items)
    succeeded = 0

    if max_workers is not None and max_workers > 1:
        # Parallel execution
        import sys
        print(f"Processing {attempted:,} parcel-years with {max_workers} workers...")
        sys.stdout.flush()
        done_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_single_uid, item): item
                for item in work_items
            }
            for future in as_completed(futures):
                done_count += 1
                if done_count % 500 == 0:
                    print(f"  {done_count:,}/{attempted:,} complete...")
                    sys.stdout.flush()
                result_type, data = future.result()
                if result_type == "fail":
                    rows_fail.append(data)
                else:
                    succeeded += 1
                    rows_monthly.append(data["monthly"])
                    rows_parcel_year.append(data["parcel_year"])
    else:
        # Sequential execution (original behavior)
        for item in work_items:
            result_type, data = _process_single_uid(item)
            if result_type == "fail":
                rows_fail.append(data)
            else:
                succeeded += 1
                rows_monthly.append(data["monthly"])
                rows_parcel_year.append(data["parcel_year"])

    # Concatenate
    if not rows_monthly:
        raise ValueError(
            "No successful parcel computations. "
            "Check failures table for details."
        )

    df_monthly = pd.concat(rows_monthly, ignore_index=True)
    df_parcel_year = pd.DataFrame(rows_parcel_year)
    df_fail = pd.DataFrame(rows_fail)

    # Coverage
    n_parcel_years = int(df_parcel_year.shape[0])
    n_unique_parcels = int(df_parcel_year["UniqueID"].nunique())
    n_counties_out = int(df_parcel_year["county"].nunique())
    success_rate = (
        100.0 * succeeded / attempted if attempted > 0 else float("nan")
    )

    print("=" * 60)
    print("OFF-PHASE CORRECTION STATS SUMMARY (corridor-scale)")
    print("-" * 60)
    print(f"WY range: {wy_start}\u2013{wy_end}")
    print(f"Counties processed: {n_counties_out:,}")
    print(f"Parcel-years computed (successful): {n_parcel_years:,}")
    print(f"Unique parcels represented: {n_unique_parcels:,}")
    print(
        f"Attempted UID runs: {attempted:,} | Succeeded: {succeeded:,} | "
        f"Success rate: {success_rate:.1f}%"
    )
    print("=" * 60)

    # Aggregation tables
    county_wy = _build_county_wy_table(df_parcel_year)
    county_annual = _build_county_annual_table(df_parcel_year)
    wy_annual = _build_wy_annual_table(df_parcel_year)
    monthly_pool = _build_monthly_pool_table(df_monthly)

    # Descriptive statistics
    desc_month, desc_year = _build_descriptive_stats(
        df_monthly, df_parcel_year,
    )

    # Narrative
    p1, p2 = _generate_narrative(
        df_parcel_year, df_monthly, county_annual,
        n_counties=n_counties_out,
        wy_start=wy_start, wy_end=wy_end,
        n_parcel_years=n_parcel_years,
        n_unique_parcels=n_unique_parcels,
        attempted=attempted, succeeded=succeeded,
    )
    print("\n=== Two-paragraph Results/Discussion-ready summary ===\n")
    print(p1 + "\n")
    print(p2 + "\n")

    # Export
    paths = {}
    if export_csv:
        paths = export_stats_csvs(
            df_monthly=df_monthly,
            df_parcel_year=df_parcel_year,
            df_fail=df_fail,
            county_wy=county_wy,
            county_annual=county_annual,
            wy_annual=wy_annual,
            monthly_pool=monthly_pool,
            desc_month=desc_month,
            desc_year=desc_year,
            out_dir=out_dir,
            method=chosen_method,
            cloud_cover=cloud_cover_max,
            ci_alpha=ci_alpha,
            n_boot=n_boot,
            wy_start=wy_start,
            wy_end=wy_end,
        )

    return {
        "df_monthly": df_monthly,
        "df_parcel_year": df_parcel_year,
        "df_fail": df_fail,
        "county_wy": county_wy,
        "county_annual": county_annual,
        "wy_annual": wy_annual,
        "monthly_pool": monthly_pool,
        "desc_month": desc_month,
        "desc_year": desc_year,
        "narrative": (p1, p2),
        "paths": paths,
    }
