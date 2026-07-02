"""Cutting-count statistics provider with narrative generation.

Computes coverage, descriptive statistics, frequency tables,
county mean-range labels, and auto-generated Results/Discussion
paragraphs from parcel-year cutting data.

Also provides ``build_parcel_year_master()`` for regenerating the
parcel-year wide CSVs with the fixed ET loader, and
``validate_parcel_year_master()`` for post-generation sanity checks.

Source: alfalfa_evi_jovyan.py lines 14337-14391, 16336-16680
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import config
from .evi_provider import normalize_county_name
from .spatial_provider import COUNTY_ORDER


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _to_num(x):
    """Coerce to numeric, returning NaN for non-numeric values."""
    return pd.to_numeric(x, errors="coerce")


def _reg_stats(x, y) -> Dict[str, float]:
    """Linear regression statistics between x and y."""
    x = np.asarray(_to_num(x), dtype=float)
    y = np.asarray(_to_num(y), dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 2:
        return {
            "n": n, "slope": np.nan, "intercept": np.nan,
            "r": np.nan, "r2": np.nan,
        }
    xm, ym = float(np.mean(x)), float(np.mean(y))
    cov = float(np.sum((x - xm) * (y - ym)))
    varx = float(np.sum((x - xm) ** 2))
    slope = np.nan if varx == 0 else cov / varx
    intercept = np.nan if not np.isfinite(slope) else (ym - slope * xm)
    r = np.nan
    if np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    r2 = float(r ** 2) if np.isfinite(r) else np.nan
    return {"n": n, "slope": slope, "intercept": intercept, "r": r, "r2": r2}


def _fmt(x, nd: int = 2) -> str:
    """Format a number for display, returning 'NA' for non-finite."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{float(x):.{nd}f}"


def _quantile_bins(
    df: pd.DataFrame, xcol: str, ycol: str, q: int = 4,
) -> pd.DataFrame:
    """Bin x into quantile groups and aggregate y."""
    d = df[[xcol, ycol]].copy()
    d[xcol] = _to_num(d[xcol])
    d[ycol] = _to_num(d[ycol])
    d = d.dropna()
    if len(d) < q + 2:
        return pd.DataFrame()
    try:
        d["x_bin"] = pd.qcut(d[xcol], q=q, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    return (
        d.groupby("x_bin", as_index=False)
        .agg(
            n=(ycol, "size"),
            y_mean=(ycol, "mean"),
            y_median=(ycol, "median"),
            x_min=(xcol, "min"),
            x_max=(xcol, "max"),
        )
    )


# ---------------------------------------------------------------------------
# Cutting-specific helpers
# ---------------------------------------------------------------------------

def _desc_series(x: pd.Series) -> pd.Series:
    """Descriptive statistics for a cutting-count series (includes IQR)."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return pd.Series({
            "n": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
            "p10": np.nan, "p25": np.nan, "median": np.nan,
            "p75": np.nan, "p90": np.nan, "max": np.nan, "iqr": np.nan,
        })
    q10, q25, q50, q75, q90 = np.nanquantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    return pd.Series({
        "n": int(x.size),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.nanmin(x)),
        "p10": float(q10), "p25": float(q25), "median": float(q50),
        "p75": float(q75), "p90": float(q90),
        "max": float(np.nanmax(x)),
        "iqr": float(q75 - q25),
    })


def _apply_desc(
    df_in: pd.DataFrame,
    group_cols: List[str],
    cut_metric: str,
) -> pd.DataFrame:
    """Apply descriptive stats, optionally grouped."""
    d = df_in.dropna(subset=[cut_metric]).copy()
    if not group_cols:
        return _desc_series(d[cut_metric]).to_frame().T
    tmp = d.groupby(group_cols)[cut_metric].apply(_desc_series)
    tmp = tmp.unstack() if isinstance(tmp, pd.Series) else tmp
    return tmp.reset_index()


def _counts_table(
    df_in: pd.DataFrame,
    group_cols: List[str],
    cut_metric: str,
) -> pd.DataFrame:
    """Coverage / missingness table."""
    if not group_cols:
        pyr = int(df_in.shape[0])
        up = int(df_in["UniqueID"].nunique())
        vr = int(df_in[cut_metric].notna().sum())
        mr = int(df_in[cut_metric].isna().sum())
        mp = 100.0 * mr / pyr if pyr > 0 else np.nan
        return pd.DataFrame([{
            "parcel_year_rows": pyr, "unique_parcels": up,
            "valid_rows": vr, "missing_rows": mr, "missing_pct": mp,
        }])

    out = (
        df_in.groupby(group_cols)
        .agg(
            parcel_year_rows=("UniqueID", "size"),
            unique_parcels=("UniqueID", "nunique"),
            valid_rows=(cut_metric, lambda s: int(s.notna().sum())),
            missing_rows=(cut_metric, lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    out["missing_pct"] = np.where(
        out["parcel_year_rows"] > 0,
        100.0 * out["missing_rows"] / out["parcel_year_rows"],
        np.nan,
    )
    return out


def _freq_table(
    df_in: pd.DataFrame,
    cut_metric: str,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Frequency table of integer cutting values."""
    d = df_in.dropna(subset=[cut_metric]).copy()
    d["cut_value"] = (
        pd.to_numeric(d[cut_metric], errors="coerce")
        .round()
        .astype("Int64")
    )

    if group_cols is None or len(group_cols) == 0:
        t = (
            d["cut_value"]
            .value_counts(dropna=True)
            .sort_index()
            .rename("count")
            .to_frame()
        )
        t["percent"] = 100.0 * t["count"] / t["count"].sum()
        t = t.reset_index().rename(columns={"index": "cut_value"})
        return t

    t = (
        d.groupby(group_cols + ["cut_value"])
        .size()
        .rename("count")
        .reset_index()
    )
    t["percent_within_group"] = (
        t.groupby(group_cols)["count"]
        .transform(lambda s: 100.0 * s / s.sum())
    )
    return t


def _round_for_display(tbl: pd.DataFrame) -> pd.DataFrame:
    """Round float columns for display."""
    tbl = tbl.copy()
    for c in tbl.columns:
        if c in {"county"}:
            continue
        if pd.api.types.is_float_dtype(tbl[c]):
            tbl[c] = tbl[c].round(2)
    return tbl


# ---------------------------------------------------------------------------
# Data loading from BEAST seasonal CSVs
# ---------------------------------------------------------------------------

def load_cutting_data_from_beast(
    counties: Optional[List[str]] = None,
    wy_start: Optional[int] = None,
    wy_end: Optional[int] = None,
) -> pd.DataFrame:
    """Load cutting data from BEAST seasonal CSVs.

    Reads all available BEAST seasonal CSVs and returns a DataFrame
    with columns: UniqueID, county, WY, n_cp_season, n_cuttings.

    Args:
        counties: List of county names. Defaults to COUNTY_ORDER.
        wy_start: Optional start WY filter.
        wy_end: Optional end WY filter.

    Returns:
        DataFrame with parcel-year cutting data.
    """
    from .beast_provider import BEASTDataProvider

    if counties is None:
        counties = list(COUNTY_ORDER)

    provider = BEASTDataProvider()
    frames = []

    for county in counties:
        county_norm = normalize_county_name(county)
        years = provider.detect_years_on_disk(county_norm)
        for wy in years:
            if wy_start is not None and wy < wy_start:
                continue
            if wy_end is not None and wy > wy_end:
                continue
            df = provider.load_seasonal_cuts_csv(county_norm, wy)
            if df is None or df.empty:
                continue

            # Harmonize column names
            if "parcel_id" in df.columns and "UniqueID" not in df.columns:
                df["UniqueID"] = df["parcel_id"].astype(str)
            elif "UniqueID" in df.columns:
                df["UniqueID"] = df["UniqueID"].astype(str)
            else:
                continue

            if "n_cp_season" not in df.columns:
                if "n_change_points" in df.columns:
                    df["n_cp_season"] = df["n_change_points"]
                else:
                    df["n_cp_season"] = np.nan

            if "n_cuttings" not in df.columns:
                df["n_cuttings"] = np.nan

            df["county"] = county_norm
            df["WY"] = int(wy)

            keep = ["UniqueID", "county", "WY", "n_cp_season", "n_cuttings"]
            keep = [c for c in keep if c in df.columns]
            frames.append(df[keep].copy())

    if not frames:
        raise ValueError("No BEAST seasonal CSVs found.")
    df = pd.concat(frames, ignore_index=True)
    if config.min_cuttings > 0:
        df = df[df["n_cuttings"] >= config.min_cuttings].copy()
    return df


# ---------------------------------------------------------------------------
# Main statistics computation
# ---------------------------------------------------------------------------

def compute_cutting_statistics(
    df: pd.DataFrame,
    cut_metric: str = "n_cp_season",
    wy_start: Optional[int] = None,
    wy_end: Optional[int] = None,
    cut_min: Optional[float] = None,
    cut_max: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute cutting frequency statistics from a parcel-year DataFrame.

    Args:
        df: DataFrame with columns: UniqueID, county, WY, {cut_metric}.
        cut_metric: Metric column ("n_cp_season" or "n_cuttings").
        wy_start: Optional start WY filter.
        wy_end: Optional end WY filter.
        cut_min: Optional minimum cut value filter.
        cut_max: Optional maximum cut value filter.

    Returns:
        Dict of DataFrames with keys: cov_overall, cov_county, cov_wy,
        cov_county_wy, desc_overall, desc_county, desc_wy, desc_county_wy,
        freq_overall, freq_county, freq_wy, county_mean_range.
    """
    if cut_metric not in df.columns:
        raise ValueError(f"Column '{cut_metric}' not found in DataFrame.")

    df = df.copy()
    df["UniqueID"] = df["UniqueID"].astype(str)
    df["county"] = df["county"].astype(str).str.strip().str.title()
    df["WY"] = pd.to_numeric(df["WY"], errors="coerce").astype("Int64")
    df[cut_metric] = pd.to_numeric(df[cut_metric], errors="coerce")

    # Determine WY bounds
    wy_min_data = int(df["WY"].min())
    wy_max_data = int(df["WY"].max())
    wy_min = wy_min_data if wy_start is None else int(wy_start)
    wy_max = wy_max_data if wy_end is None else int(wy_end)

    df = df[df["WY"].between(wy_min, wy_max)].copy()

    if cut_min is not None:
        df = df[
            df[cut_metric].isna() | (df[cut_metric] >= float(cut_min))
        ]
    if cut_max is not None:
        df = df[
            df[cut_metric].isna() | (df[cut_metric] <= float(cut_max))
        ]

    if df.empty:
        raise ValueError("No rows remain after applying WY/cut filters.")

    # County ordering
    county_order = [
        c for c in COUNTY_ORDER if c in df["county"].unique()
    ]
    if not county_order:
        county_order = sorted(df["county"].unique().tolist())

    # Coverage
    cov_overall = _counts_table(df, [], cut_metric)
    cov_county = _counts_table(df, ["county"], cut_metric)
    cov_wy = _counts_table(df, ["WY"], cut_metric)
    cov_cwy = _counts_table(df, ["county", "WY"], cut_metric)

    # Descriptive stats
    desc_overall = _apply_desc(df, [], cut_metric)
    desc_county = _apply_desc(df, ["county"], cut_metric)
    desc_wy = _apply_desc(df, ["WY"], cut_metric)
    desc_cwy = _apply_desc(df, ["county", "WY"], cut_metric)

    # Merge counts into desc
    desc_overall = pd.concat([desc_overall, cov_overall], axis=1)
    if "county" in desc_county.columns:
        desc_county = desc_county.merge(cov_county, on="county", how="left")
    if "WY" in desc_wy.columns:
        desc_wy = desc_wy.merge(cov_wy, on="WY", how="left")
    if {"county", "WY"}.issubset(desc_cwy.columns):
        desc_cwy = desc_cwy.merge(cov_cwy, on=["county", "WY"], how="left")

    # Order counties
    if "county" in desc_county.columns:
        desc_county["county"] = pd.Categorical(
            desc_county["county"], categories=county_order, ordered=True,
        )
        desc_county = desc_county.sort_values("county")
    if "county" in desc_cwy.columns:
        desc_cwy["county"] = pd.Categorical(
            desc_cwy["county"], categories=county_order, ordered=True,
        )
        desc_cwy = desc_cwy.sort_values(["county", "WY"])

    # Frequency tables
    freq_overall = _freq_table(df, cut_metric)
    freq_county = _freq_table(df, cut_metric, ["county"])
    freq_wy = _freq_table(df, cut_metric, ["WY"])

    # County mean range (map-label equivalent)
    tmp = (
        df.dropna(subset=[cut_metric])
        .groupby(["county", "WY"])[cut_metric]
        .mean()
        .reset_index(name="mean_per_wy")
    )
    county_mean_range = (
        tmp.groupby("county")["mean_per_wy"]
        .agg(mean_min="min", mean_max="max", mean_avg="mean")
        .reset_index()
    )
    county_mean_range["mean_min_round"] = (
        county_mean_range["mean_min"].round().astype("Int64")
    )
    county_mean_range["mean_max_round"] = (
        county_mean_range["mean_max"].round().astype("Int64")
    )
    county_mean_range["label_like_map"] = np.where(
        county_mean_range["mean_min_round"]
        == county_mean_range["mean_max_round"],
        county_mean_range["mean_min_round"].astype(str),
        (
            county_mean_range["mean_min_round"].astype(str)
            + "\u2013"
            + county_mean_range["mean_max_round"].astype(str)
        ),
    )
    county_mean_range["county"] = pd.Categorical(
        county_mean_range["county"], categories=county_order, ordered=True,
    )
    county_mean_range = county_mean_range.sort_values("county")

    return {
        "cov_overall": cov_overall,
        "cov_county": cov_county,
        "cov_wy": cov_wy,
        "cov_county_wy": cov_cwy,
        "desc_overall": desc_overall,
        "desc_county": desc_county,
        "desc_wy": desc_wy,
        "desc_county_wy": desc_cwy,
        "freq_overall": freq_overall,
        "freq_county": freq_county,
        "freq_wy": freq_wy,
        "county_mean_range": county_mean_range,
        "_wy_min": wy_min,
        "_wy_max": wy_max,
        "_cut_metric": cut_metric,
        "_df_filtered": df,
    }


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

def generate_cutting_narrative(
    stats: Dict[str, pd.DataFrame],
) -> Tuple[str, str]:
    """Generate two Results/Discussion-ready paragraphs.

    Args:
        stats: Output dict from compute_cutting_statistics().

    Returns:
        Tuple of (paragraph_1, paragraph_2).
    """
    cut_metric = stats["_cut_metric"]
    wy_min = stats["_wy_min"]
    wy_max = stats["_wy_max"]
    df = stats["_df_filtered"]

    metric_label = (
        "number of cuttings"
        if cut_metric == "n_cuttings"
        else "number of BEAST change-points (n_cp_season)"
    )

    d_valid = df.dropna(subset=[cut_metric]).copy()
    d_valid["cut_int"] = (
        pd.to_numeric(d_valid[cut_metric], errors="coerce")
        .round()
        .astype("Int64")
    )

    overall_n = int(d_valid.shape[0])
    overall_unique = int(d_valid["UniqueID"].nunique())
    overall_mean = float(d_valid[cut_metric].mean())
    overall_median = float(d_valid[cut_metric].median())
    overall_std = (
        float(d_valid[cut_metric].std(ddof=1)) if overall_n > 1 else 0.0
    )
    overall_min = float(d_valid[cut_metric].min())
    overall_max = float(d_valid[cut_metric].max())

    share_ge8 = 100.0 * (d_valid["cut_int"] >= 8).mean()
    share_ge9 = 100.0 * (d_valid["cut_int"] >= 9).mean()
    share_le6 = 100.0 * (d_valid["cut_int"] <= 6).mean()

    county_mean_tbl = (
        d_valid.groupby("county")[cut_metric]
        .mean()
        .reset_index(name="mean")
    )
    hi_county = county_mean_tbl.loc[county_mean_tbl["mean"].idxmax()]
    lo_county = county_mean_tbl.loc[county_mean_tbl["mean"].idxmin()]

    wy_mean_tbl = (
        d_valid.groupby("WY")[cut_metric]
        .mean()
        .reset_index(name="mean")
    )
    hi_wy = wy_mean_tbl.loc[wy_mean_tbl["mean"].idxmax()]
    lo_wy = wy_mean_tbl.loc[wy_mean_tbl["mean"].idxmin()]

    p1 = (
        f"Across WY {wy_min}\u2013{wy_max}, the dataset contained "
        f"{overall_n:,} parcel-year observations ({overall_unique:,} unique "
        f"parcels) with valid {cut_metric} estimates. The {metric_label} "
        f"averaged {overall_mean:.2f} (median {overall_median:.2f}; "
        f"SD {overall_std:.2f}) and ranged from {overall_min:.0f} to "
        f"{overall_max:.0f}. In pooled frequencies, {share_ge8:.1f}% of "
        f"parcel-years exhibited \u22658 events, {share_ge9:.1f}% exhibited "
        f"\u22659, and {share_le6:.1f}% were \u22646, indicating that "
        f"high-frequency harvest/regrowth dynamics were common over the "
        f"study corridor."
    )

    p2 = (
        f"Spatially, county means revealed systematic differences in "
        f"{metric_label} across the corridor: the highest mean occurred in "
        f"{str(hi_county['county'])} (mean {float(hi_county['mean']):.2f}), "
        f"whereas the lowest mean occurred in "
        f"{str(lo_county['county'])} (mean {float(lo_county['mean']):.2f}). "
        f"Temporally, interannual variability was evident, with the highest "
        f"water-year mean in WY {int(hi_wy['WY'])} "
        f"({float(hi_wy['mean']):.2f}) and the lowest in "
        f"WY {int(lo_wy['WY'])} ({float(lo_wy['mean']):.2f}). "
        f"County-level ranges of annual mean values (map-equivalent labels) "
        f"indicate that some counties maintain consistently high event "
        f"frequencies across years, while others exhibit broader "
        f"year-to-year shifts."
    )

    return p1, p2


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_cutting_stats_csvs(
    stats: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> Dict[str, Path]:
    """Export cutting statistics tables to CSV files.

    Args:
        stats: Output dict from compute_cutting_statistics().
        out_dir: Output directory.

    Returns:
        Dict mapping table name to file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cut_metric = stats["_cut_metric"]
    wy_min = stats["_wy_min"]
    wy_max = stats["_wy_max"]
    suffix = f"{cut_metric}_WY{wy_min}-{wy_max}"

    export_keys = [
        ("cov_overall", "coverage_overall"),
        ("cov_county", "coverage_by_county"),
        ("cov_wy", "coverage_by_wy"),
        ("cov_county_wy", "coverage_by_county_wy"),
        ("desc_overall", "desc_overall"),
        ("desc_county", "desc_by_county"),
        ("desc_wy", "desc_by_wy"),
        ("desc_county_wy", "desc_by_county_wy"),
        ("county_mean_range", "county_annual_mean_range"),
        ("freq_overall", "freq_overall"),
        ("freq_county", "freq_by_county"),
        ("freq_wy", "freq_by_wy"),
    ]

    paths = {}
    for key, name in export_keys:
        if key in stats and isinstance(stats[key], pd.DataFrame):
            fp = out_dir / f"{name}_{suffix}.csv"
            stats[key].to_csv(fp, index=False)
            paths[key] = fp
            print(f"[saved] {name}: {fp}")

    return paths


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_cutting_weather_stats(
    *,
    counties: Optional[List[str]] = None,
    wy_start: Optional[int] = None,
    wy_end: Optional[int] = None,
    cut_metric: str = "n_cp_season",
    cut_min: Optional[float] = None,
    cut_max: Optional[float] = None,
    out_dir: Optional[Path] = None,
    export_csv: bool = True,
    df_input: Optional[pd.DataFrame] = None,
) -> Dict:
    """Run the full cutting-weather statistics pipeline.

    Either loads data from BEAST seasonal CSVs or uses a provided
    DataFrame.

    Args:
        counties: Counties to process. Defaults to COUNTY_ORDER.
        wy_start: First water year.
        wy_end: Last water year.
        cut_metric: "n_cp_season" or "n_cuttings".
        cut_min: Optional minimum cut value filter.
        cut_max: Optional maximum cut value filter.
        out_dir: Output directory for CSVs.
        export_csv: Whether to export CSVs.
        df_input: Optional pre-built DataFrame.

    Returns:
        Dict with keys: stats (all tables), narrative (two paragraphs),
        paths (exported file paths).
    """
    if out_dir is None:
        out_dir = config.cutting_weather_stats_root

    # Load data
    if df_input is not None:
        df = df_input.copy()
    else:
        df = load_cutting_data_from_beast(
            counties=counties,
            wy_start=wy_start,
            wy_end=wy_end,
        )

    # Compute statistics
    stats = compute_cutting_statistics(
        df,
        cut_metric=cut_metric,
        wy_start=wy_start,
        wy_end=wy_end,
        cut_min=cut_min,
        cut_max=cut_max,
    )

    # Print summary
    wy_min = stats["_wy_min"]
    wy_max = stats["_wy_max"]
    df_filt = stats["_df_filtered"]
    print(f"Metric: {cut_metric}")
    print(f"WY range: {wy_min}\u2013{wy_max}")
    print(f"Rows (parcel-years): {len(df_filt):,}")
    print(f"Unique parcels: {df_filt['UniqueID'].nunique():,}")
    print(f"Counties: {df_filt['county'].nunique():,}")

    # Narrative
    p1, p2 = generate_cutting_narrative(stats)
    print("\n=== Two-paragraph Results/Discussion-ready summary ===\n")
    print(p1 + "\n")
    print(p2 + "\n")

    # Export
    paths = {}
    if export_csv:
        paths = export_cutting_stats_csvs(stats, out_dir)

    return {
        "stats": stats,
        "narrative": (p1, p2),
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# Parcel-year master regeneration (fixed ET)
# ---------------------------------------------------------------------------

def _compute_one_chunk(args_tuple):
    """Process one (county, wy) chunk via build_parcel_summary_wy().

    Top-level function so ProcessPoolExecutor can pickle it.

    Returns:
        (status, county, wy, df_or_error) where status is 'ok' or 'fail'.
    """
    from .parcel_summary_provider import build_parcel_summary_wy

    county, wy, daymet_var, month_start, month_end = args_tuple
    try:
        df = build_parcel_summary_wy(
            county=county,
            wy=wy,
            daymet_var=daymet_var,
            month_start=month_start,
            month_end=month_end,
        )
        return ("ok", county, wy, df)
    except Exception as e:
        return ("fail", county, wy, str(e))


def build_parcel_year_master(
    daymet_var: str,
    counties: List[str],
    wy_start: int,
    wy_end: int,
    out_dir: Path,
    max_workers: int = 1,
    month_start: int = 10,
    month_end: int = 9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the parcel-year wide CSV across all county x WY combos.

    Calls ``build_parcel_summary_wy()`` (which uses the fixed ET loader)
    for every county/WY pair, optionally in parallel, saves per-chunk
    CSVs for resumability, and concatenates into a master CSV.

    Args:
        daymet_var: "tmax" or "gdd5".
        counties: List of county names.
        wy_start: First water year (inclusive).
        wy_end: Last water year (inclusive).
        out_dir: Root output directory for this variant.
        max_workers: Number of parallel processes (1 = sequential).
        month_start: Water-year start month (default 10 = October).
        month_end: Water-year end month (default 9 = September).

    Returns:
        (df_master, df_failures) — the concatenated master DataFrame and
        a DataFrame of failed county/WY pairs.
    """
    out_dir = Path(out_dir)
    chunk_dir = out_dir / "_chunks_county_wy"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    counties_norm = [normalize_county_name(c) for c in counties]
    years = list(range(int(wy_start), int(wy_end) + 1))

    # Build work items
    work_items = [
        (county, wy, daymet_var, month_start, month_end)
        for county in counties_norm
        for wy in years
    ]
    total = len(work_items)
    print(f"[regenerate] {daymet_var}: {total} county×WY chunks, "
          f"workers={max_workers}")
    sys.stdout.flush()

    frames: List[pd.DataFrame] = []
    failures: List[Dict] = []

    if max_workers > 1:
        done_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_one_chunk, item): item
                for item in work_items
            }
            for future in as_completed(futures):
                done_count += 1
                status, county, wy, payload = future.result()
                if status == "fail":
                    print(f"  [FAIL] {county} WY{wy}: {payload}")
                    failures.append({
                        "county": county, "WY": wy, "error": payload,
                    })
                else:
                    df_chunk = payload
                    fp = chunk_dir / f"cut_weather_chunk_{county}_{wy}.csv"
                    df_chunk.to_csv(fp, index=False)
                    frames.append(df_chunk)
                    if done_count % 10 == 0 or done_count == total:
                        print(f"  {done_count}/{total} done "
                              f"({len(frames)} ok, {len(failures)} fail)")
                        sys.stdout.flush()
    else:
        for i, item in enumerate(work_items, 1):
            status, county, wy, payload = _compute_one_chunk(item)
            if status == "fail":
                print(f"  [FAIL] {county} WY{wy}: {payload}")
                failures.append({
                    "county": county, "WY": wy, "error": payload,
                })
            else:
                df_chunk = payload
                fp = chunk_dir / f"cut_weather_chunk_{county}_{wy}.csv"
                df_chunk.to_csv(fp, index=False)
                frames.append(df_chunk)
            if i % 10 == 0 or i == total:
                print(f"  {i}/{total} done "
                      f"({len(frames)} ok, {len(failures)} fail)")
                sys.stdout.flush()

    if not frames:
        raise ValueError("All county×WY chunks failed. See failures.")

    df_master = pd.concat(frames, ignore_index=True)

    # Save master CSV
    master_path = (
        out_dir
        / f"cut_weather_parcel_year_wide_WY{wy_start}-{wy_end}.csv"
    )
    df_master.to_csv(master_path, index=False)
    print(f"[saved] master CSV ({len(df_master):,} rows): {master_path}")

    df_failures = pd.DataFrame(failures)
    if not df_failures.empty:
        fail_path = out_dir / f"regenerate_failures_WY{wy_start}-{wy_end}.csv"
        df_failures.to_csv(fail_path, index=False)
        print(f"[saved] failures ({len(df_failures)} rows): {fail_path}")

    return df_master, df_failures


def validate_parcel_year_master(
    df: pd.DataFrame,
    daymet_var: str,
) -> bool:
    """Run sanity checks on a regenerated parcel-year master DataFrame.

    Checks:
      1. Zero ET count == 0
      2. ET < 10mm count < 5% of rows
      3. Mean ET in plausible range (50-500mm for segment sums)
      4. Row count > 8000

    Args:
        df: The master DataFrame (output of build_parcel_year_master).
        daymet_var: "tmax" or "gdd5" (for reporting only).

    Returns:
        True if all checks pass, False otherwise.
    """
    et_col = "et_cum_minET_to_last_cut_mm"
    print(f"\n[validate] daymet_var={daymet_var}, rows={len(df):,}")

    passed = True

    # Check 1: zero ET
    n_zero = int((df[et_col] == 0.0).sum())
    if n_zero > 0:
        print(f"  FAIL: {n_zero} rows have ET == 0.0 (expected 0)")
        passed = False
    else:
        print(f"  OK: no rows with ET == 0.0")

    # Check 2: ET < 10mm
    n_low = int((df[et_col] < 10).sum())
    pct_low = 100.0 * n_low / len(df) if len(df) > 0 else 0
    if pct_low >= 5.0:
        print(f"  FAIL: {n_low} rows ({pct_low:.1f}%) have ET < 10mm "
              f"(expected < 5%)")
        passed = False
    else:
        print(f"  OK: {n_low} rows ({pct_low:.1f}%) have ET < 10mm")

    # Check 3: mean ET plausible
    mean_et = float(df[et_col].mean())
    if not (50 <= mean_et <= 500):
        print(f"  FAIL: mean ET = {mean_et:.1f}mm "
              f"(expected 50-500mm)")
        passed = False
    else:
        print(f"  OK: mean ET = {mean_et:.1f}mm")

    # Check 4: row count
    if len(df) < 8000:
        print(f"  WARN: only {len(df):,} rows (expected > 8000)")
        passed = False
    else:
        print(f"  OK: {len(df):,} rows")

    status = "PASSED" if passed else "FAILED"
    print(f"[validate] {daymet_var}: {status}\n")
    return passed
