"""
ET Data Provider

Handles OpenET, ETo, ETof data loading and ET corrections.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import config
from ..utils.validation import validate_daily_et

# Path configuration -- sourced from config.py (centralized)
BEAST_OUT_ROOT = config.beast_out_root_new
OPENET_ROOT = config.openet_root
LANDSAT_META_CSV = config.landsat_meta_csv
ETO_ETOF_ROOT = config.eto_etof_root


def _norm_county_name(name: str) -> str:
    s = str(name).replace("_", " ").strip()
    return " ".join(s.split()).title()


def water_year_bounds(wy: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=wy-1, month=10, day=1), pd.Timestamp(year=wy, month=9, day=30)


def _load_seasonal_csv(county: str, wy: int) -> pd.DataFrame:
    county_norm = _norm_county_name(county)
    p = BEAST_OUT_ROOT / county_norm / f"beast_seasonal_cuts_WY{wy}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p)
    if "parcel_id" in df.columns:
        df["UniqueID"] = df["parcel_id"].astype(str)
    elif "UniqueID" in df.columns:
        df["UniqueID"] = df["UniqueID"].astype(str)
    else:
        raise ValueError("Expected 'parcel_id' or 'UniqueID' in seasonal CSV.")
    if "season_cp_dates_iso" not in df.columns and "cp_dates_iso" in df.columns:
        df = df.rename(columns={"cp_dates_iso": "season_cp_dates_iso"})
    if "n_cp_season" not in df.columns and "n_change_points" in df.columns:
        df["n_cp_season"] = df["n_change_points"]
    return df


def _parse_cp_dates_iso(s: str) -> List[pd.Timestamp]:
    if pd.isna(s) or not str(s).strip():
        return []
    parts = [p.strip() for p in str(s).split(";") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(pd.to_datetime(p).normalize())
        except Exception:
            pass
    return out


def list_uids_for_county_wy(
    county: str,
    wy: int,
    n_cp_season_filter: Optional[int] = None,
) -> List[str]:
    """List all UniqueIDs in a county-year seasonal CSV.

    Args:
        county: County name.
        wy: Water year.
        n_cp_season_filter: If set, keep only rows where n_cp_season == this value.

    Returns:
        Sorted list of UniqueID strings.
    """
    df = _load_seasonal_csv(county, wy)
    if n_cp_season_filter is not None:
        m = pd.to_numeric(df["n_cp_season"], errors="coerce") == int(n_cp_season_filter)
        df = df.loc[m].copy()
    if config.min_cuttings > 0 and "n_cuttings" in df.columns:
        df = df[pd.to_numeric(df["n_cuttings"], errors="coerce") >= config.min_cuttings]
    return sorted(df["UniqueID"].astype(str).unique().tolist())


def get_harvest_dates_for_uid(
    county: str,
    wy: int,
    uid: str,
    n_cp_season_filter: Optional[int] = None,
) -> List[pd.Timestamp]:
    county_norm = _norm_county_name(county)
    uid_str = str(uid)
    start, end = water_year_bounds(wy)
    df = _load_seasonal_csv(county_norm, wy)
    if n_cp_season_filter is not None:
        mask_cp = pd.to_numeric(df["n_cp_season"], errors="coerce") == int(n_cp_season_filter)
        df = df.loc[mask_cp]
    df = df[df["UniqueID"].astype(str) == uid_str]
    if df.empty:
        raise ValueError(f"No BEAST row for UID={uid_str} in {county_norm}, WY{wy}")
    row = df.iloc[0]
    # Prefer matched_minima_iso (consensus-filtered) over raw season_cp_dates_iso
    dates_s = ""
    if "matched_minima_iso" in df.columns:
        dates_s = row.get("matched_minima_iso", "")
    if not dates_s or (isinstance(dates_s, float) and pd.isna(dates_s)):
        dates_s = row.get("season_cp_dates_iso", "") if "season_cp_dates_iso" in df.columns else ""
    dates = _parse_cp_dates_iso(dates_s)
    dates = [d for d in dates if (start <= d <= end)]
    return sorted(dates)


def _load_openet_api_for_wy(county: str, wy: int, uids: List[str]) -> Dict[str, pd.Series]:
    """Load daily ET from per-parcel JSON files (API download format).

    Each file is ``{uid}.json`` containing ``[{"time": "YYYY-MM-DD", "et": float}, ...]``
    spanning 2018-01-01 to 2024-12-31.
    """
    start, end = water_year_bounds(wy)
    county_norm = _norm_county_name(county)
    idx = pd.date_range(start, end, freq="D")
    api_dir = config.openet_api_daily_root

    out: Dict[str, pd.Series] = {}
    for uid in uids:
        uid_str = str(uid)
        jpath = api_dir / f"{uid_str}.json"
        if not jpath.exists():
            out[uid_str] = pd.Series(np.nan, index=idx, dtype=float)
            continue
        with open(jpath, "r") as fh:
            records = json.load(fh)
        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["et"] = pd.to_numeric(df["et"], errors="coerce")
        df = df.dropna(subset=["time"])
        df = df.set_index("time").sort_index()
        s = df["et"].reindex(idx)
        out[uid_str] = s

    # Validate daily ET ranges (ETFIX-03)
    for uid_key, series in out.items():
        validate_daily_et(series, uid=uid_key, context=f"WY{wy} {county_norm}")

    return out


def _load_openet_csv_for_wy(county: str, wy: int, uids: List[str]) -> Dict[str, pd.Series]:
    """Load daily ET from multi-parcel CSV files (old bulk export format)."""
    county_norm = _norm_county_name(county)
    start, end = water_year_bounds(wy)

    def _year_files(year: int) -> List[Path]:
        folder = OPENET_ROOT / str(year) / "OpenET Exports"
        if not folder.exists():
            return []
        return sorted(folder.glob("OpenET Exports_*.csv"))

    files = _year_files(wy - 1) + _year_files(wy)
    if not files:
        raise FileNotFoundError(f"No OpenET ETa CSVs for WY{wy} under {OPENET_ROOT}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        low = {c.lower(): c for c in df.columns}
        if "time" not in low or "uniqueid" not in low or "county" not in low:
            continue
        et_col = None
        for k in ["et", "eta", "et_mm", "et_actual"]:
            if k in low:
                et_col = low[k]
                break
        if et_col is None:
            continue
        df = df.rename(
            columns={
                low["time"]: "time",
                low["uniqueid"]: "UniqueID",
                low["county"]: "COUNTY",
                et_col: "et",
            }
        )
        df["UniqueID"] = df["UniqueID"].astype(str)
        df["COUNTY"] = df["COUNTY"].astype(str)
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["time"])
        mask = df["COUNTY"].str.strip().str.title().eq(county_norm)
        if uids:
            mask &= df["UniqueID"].isin([str(u) for u in uids])
        mask &= df["time"].between(start, end)
        frames.append(df.loc[mask, ["time", "UniqueID", "et"]])

    if not frames:
        raise ValueError(f"No ETa records for requested UIDs in WY{wy}")

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = raw["time"].dt.normalize()
    raw["et"] = pd.to_numeric(raw["et"], errors="coerce")
    grouped = raw.groupby(["UniqueID", "date"], as_index=False)["et"].sum()
    idx = pd.date_range(start, end, freq="D")

    out: Dict[str, pd.Series] = {}
    for uid, sub in grouped.groupby("UniqueID", sort=False):
        s = sub.set_index("date")["et"].astype(float).sort_index()
        out[str(uid)] = s.reindex(idx)
    for uid in uids:
        if str(uid) not in out:
            out[str(uid)] = pd.Series(np.nan, index=idx, dtype=float)

    # Validate daily ET ranges (ETFIX-03)
    for uid_key, series in out.items():
        validate_daily_et(series, uid=uid_key, context=f"WY{wy} {county_norm}")

    return out


def _load_openet_for_wy(county: str, wy: int, uids: List[str]) -> Dict[str, pd.Series]:
    """Load daily ET for a county-WY. Dispatches based on config.use_api_et."""
    if config.use_api_et:
        return _load_openet_api_for_wy(county, wy, uids)
    return _load_openet_csv_for_wy(county, wy, uids)


def _load_eto_or_etof_for_wy(
    county: str,
    wy: int,
    uids: List[str],
    kind: str = "eto",
) -> Dict[str, pd.Series]:
    assert kind in {"eto", "etof"}
    county_norm = _norm_county_name(county)
    start, end = water_year_bounds(wy)
    target = "eto" if kind == "eto" else "etof"

    frames = []
    for year in [wy - 1, wy]:
        year_dir = ETO_ETOF_ROOT / str(year)
        if not year_dir.exists():
            continue
        files = sorted(year_dir.glob(f"*_{target}_*.csv"))
        for f in files:
            df = pd.read_csv(f)
            low = {c.lower(): c for c in df.columns}
            if "time" not in low or "uniqueid" not in low or kind.lower() not in low:
                continue
            df = df.rename(
                columns={
                    low["time"]: "time",
                    low["uniqueid"]: "UniqueID",
                    low[kind]: kind,
                }
            )
            df["UniqueID"] = df["UniqueID"].astype(str)
            df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(None)
            df = df.dropna(subset=["time"])
            mask = (df["time"] >= start) & (df["time"] <= end)
            if uids:
                mask &= df["UniqueID"].isin([str(u) for u in uids])
            frames.append(df.loc[mask, ["time", "UniqueID", kind]])

    if not frames:
        return {}

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = raw["time"].dt.normalize()
    raw[kind] = pd.to_numeric(raw[kind], errors="coerce")
    grouped = raw.groupby(["UniqueID", "date"], as_index=False)[kind].mean()
    idx = pd.date_range(start, end, freq="D")

    out: Dict[str, pd.Series] = {}
    for uid, sub in grouped.groupby("UniqueID", sort=False):
        s = sub.set_index("date")[kind].astype(float).sort_index()
        out[str(uid)] = s.reindex(idx)
    for uid in uids:
        if str(uid) not in out:
            out[str(uid)] = pd.Series(np.nan, index=idx, dtype=float)
    return out


_LANDSAT_COUNTY_FALLBACK = {
    "Madera": "Fresno",       # same WRS paths 42-43, row 34
    "Kings": "Fresno",        # same WRS path 42, row 35 / path 43, row 34
    "Stanislaus": "San Joaquin",  # same WRS paths 43-44, row 33-34
}


def _load_landsat_passes(
    county: str,
    wy: int,
    cloud_cover_max: Optional[float] = None,
    min_spacing_days: int = 15,
    thin_track: bool = True,
) -> pd.DataFrame:
    county_norm = _norm_county_name(county)
    if not LANDSAT_META_CSV.exists():
        raise FileNotFoundError(f"Landsat metadata CSV not found: {LANDSAT_META_CSV}")

    df = pd.read_csv(LANDSAT_META_CSV, sep=None, engine="python")
    low = {c.lower(): c for c in df.columns}
    needed = {"date_only", "cloud_cover", "county"}
    if not needed.issubset(set(low)):
        raise ValueError(f"Landsat CSV missing: {needed}")

    df = df.rename(
        columns={
            low["date_only"]: "date_only",
            low["cloud_cover"]: "cloud_cover",
            low["county"]: "county",
        }
    )
    df["county"] = df["county"].astype(str).str.strip().str.title()
    df["date_only"] = pd.to_datetime(df["date_only"], errors="coerce").dt.normalize()
    df["cloud_cover"] = pd.to_numeric(df["cloud_cover"], errors="coerce")

    start, end = water_year_bounds(wy)
    df = df.loc[df["county"].eq(county_norm) & df["date_only"].between(start, end)].copy()

    # Fallback to same-track neighbor county when no passes found
    if df.empty and county_norm in _LANDSAT_COUNTY_FALLBACK:
        fallback = _LANDSAT_COUNTY_FALLBACK[county_norm]
        df_all = pd.read_csv(LANDSAT_META_CSV, sep=None, engine="python")
        df_all = df_all.rename(columns={low["date_only"]: "date_only", low["cloud_cover"]: "cloud_cover", low["county"]: "county"})
        df_all["county"] = df_all["county"].astype(str).str.strip().str.title()
        df_all["date_only"] = pd.to_datetime(df_all["date_only"], errors="coerce").dt.normalize()
        df_all["cloud_cover"] = pd.to_numeric(df_all["cloud_cover"], errors="coerce")
        df = df_all.loc[df_all["county"].eq(fallback) & df_all["date_only"].between(start, end)].copy()

    if df.empty:
        return df

    has_wrs = {"wrs_path", "wrs_row"}.issubset(df.columns)
    if has_wrs:
        df["wrs_path"] = pd.to_numeric(df["wrs_path"], errors="coerce")
        df["wrs_row"] = pd.to_numeric(df["wrs_row"], errors="coerce")
        df = df.dropna(subset=["wrs_path", "wrs_row"])
        if not df.empty:
            grp = (
                df.groupby(["wrs_path", "wrs_row"], as_index=False)
                .agg(n=("date_only", "nunique"), med_cc=("cloud_cover", "median"))
                .sort_values(["n", "med_cc"], ascending=[False, True])
            )
            best = grp.iloc[0]
            df = df[(df["wrs_path"] == best["wrs_path"]) & (df["wrs_row"] == best["wrs_row"])].copy()

    if cloud_cover_max is not None:
        df = df.loc[df["cloud_cover"] <= float(cloud_cover_max)].copy()
        if df.empty:
            return df

    if has_wrs and not df.empty:
        df = df.drop_duplicates(subset=["date_only", "wrs_path", "wrs_row"])

    df = df.sort_values(["date_only", "cloud_cover"]).drop_duplicates(subset=["date_only"], keep="first")

    if thin_track:
        selected = []
        last_date = None
        for _, row in df.iterrows():
            d = row["date_only"]
            if last_date is None or (d - last_date).days >= min_spacing_days:
                selected.append(row)
                last_date = d
        df = pd.DataFrame(selected).reset_index(drop=True) if selected else df.iloc[0:0].copy()

    keep_cols = ["date_only", "cloud_cover"]
    if has_wrs:
        keep_cols += ["wrs_path", "wrs_row"]
    return df[keep_cols].copy()


def _bootstrap_diff_fpre_fmin(
    pre_vals: np.ndarray,
    post_vals: np.ndarray,
    low_quantile: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pre_vals = pre_vals[np.isfinite(pre_vals)]
    post_vals = post_vals[np.isfinite(post_vals)]
    if pre_vals.size == 0 or post_vals.size == 0:
        return np.zeros(n_boot, dtype=float)

    diffs = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        a = rng.choice(pre_vals, size=pre_vals.size, replace=True)
        b = rng.choice(post_vals, size=post_vals.size, replace=True)
        f_pre = np.nanmedian(a)
        f_min = np.nanquantile(b, low_quantile)
        diffs[i] = max(float(f_pre - f_min), 0.0)
    return diffs


def compute_daily_and_monthly_for_uid(
    county: str,
    wy: int,
    uid: str,
    cloud_cover_max: float = 20.0,
    n_cp_season_filter: Optional[int] = None,
    r_days: int = 8,
    pre_window_days: int = 3,
    post_window_days: int = 3,
    low_quantile: float = 0.25,
    chosen_method: str = "A",
    ci_alpha: float = 0.10,
    n_boot: int = 400,
    inflate_by_cloud_gap: bool = True,
):
    county_norm = _norm_county_name(county)
    uid_str = str(uid)
    chosen_method = str(chosen_method).strip().upper()
    if chosen_method not in {"A", "B"}:
        raise ValueError("chosen_method must be 'A' or 'B'")

    start, end = water_year_bounds(wy)
    idx = pd.date_range(start, end, freq="D")

    et_dict = _load_openet_for_wy(county_norm, wy, [uid_str])
    ET_open = et_dict[uid_str].reindex(idx)

    eto_dict = _load_eto_or_etof_for_wy(county_norm, wy, [uid_str], kind="eto")
    ETo = eto_dict[uid_str].reindex(idx).interpolate(limit_direction="both")

    etof_dict = _load_eto_or_etof_for_wy(county_norm, wy, [uid_str], kind="etof")
    ETof = etof_dict[uid_str].reindex(idx).interpolate(limit_direction="both")

    harvest_dates = get_harvest_dates_for_uid(county_norm, wy, uid_str, n_cp_season_filter)
    if not harvest_dates:
        raise ValueError(f"No harvest dates for UID={uid_str} in WY{wy}")

    passes_df = _load_landsat_passes(county_norm, wy, cloud_cover_max=cloud_cover_max, thin_track=True)

    delta_corr = pd.Series(0.0, index=idx)
    month_period = idx.to_period("M")
    months = pd.Index(sorted(month_period.unique().to_timestamp(how="start")), name="month")
    rng = np.random.default_rng(12345)
    per_harvest = []

    def _first_clear_pass_after(h: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
        pass_dates_clear = pd.to_datetime(passes_df["date_only"].unique()) if not passes_df.empty else np.array([], dtype="datetime64[ns]")
        if pass_dates_clear.size == 0:
            return (end + pd.Timedelta(days=1), np.nan)
        after = pass_dates_clear[pass_dates_clear > h]
        if after.size == 0:
            return (end + pd.Timedelta(days=1), np.nan)
        p = pd.to_datetime(after.min()).normalize()
        cc = passes_df.loc[pd.to_datetime(passes_df["date_only"]).dt.normalize().eq(p), "cloud_cover"]
        return (p, float(cc.iloc[0]) if len(cc) else np.nan)

    passes_any_df = _load_landsat_passes(county_norm, wy, cloud_cover_max=None, thin_track=False)
    pass_dates_any = pd.to_datetime(passes_any_df["date_only"].unique()) if not passes_any_df.empty else np.array([], dtype="datetime64[ns]")

    def _first_any_pass_after(h: pd.Timestamp) -> pd.Timestamp:
        if pass_dates_any.size == 0:
            return (end + pd.Timedelta(days=1))
        after = pass_dates_any[pass_dates_any > h]
        if after.size == 0:
            return (end + pd.Timedelta(days=1))
        return pd.to_datetime(after.min()).normalize()

    for h in harvest_dates:
        h = pd.to_datetime(h).normalize()
        if h < start or h > end:
            continue

        p_clear, cc_clear = _first_clear_pass_after(h)
        if p_clear <= h:
            continue

        p_any = _first_any_pass_after(h)
        cloud_gap_days = int(max(0, (p_clear - p_any).days)) if np.isfinite(pd.to_datetime(p_any).value) else 0
        inflation = 1.0 + (cloud_gap_days / 16.0) if inflate_by_cloud_gap else 1.0

        pre_start = h - pd.Timedelta(days=pre_window_days)
        pre_end = h - pd.Timedelta(days=1)
        pre_vals = ETof.loc[(ETof.index >= pre_start) & (ETof.index <= pre_end)].to_numpy(dtype=float)

        post_end = h + pd.Timedelta(days=post_window_days)
        post_vals = ETof.loc[(ETof.index >= h) & (ETof.index <= post_end)].to_numpy(dtype=float)

        f_pre_hat = float(np.nanmedian(pre_vals)) if np.isfinite(pre_vals).any() else float(ETof.loc[ETof.index < h].iloc[-1])
        f_min_hat = float(np.nanquantile(post_vals[np.isfinite(post_vals)], low_quantile)) if np.isfinite(post_vals).any() else f_pre_hat
        diff_hat = max(f_pre_hat - f_min_hat, 0.0) * inflation

        off_dates = idx[(idx >= h) & (idx < p_clear)]
        if off_dates.empty:
            continue

        d_j = (p_clear - h).days
        d_j_star = min(int(d_j), int(r_days))

        weights_daily = pd.Series(0.0, index=off_dates)

        if chosen_method == "A":
            weights_daily.loc[off_dates] = ETo.loc[off_dates].astype(float).fillna(0.0).to_numpy()
        else:
            for t in off_dates:
                u = (t - h).days
                if u < d_j_star and r_days > 0:
                    w = (1.0 - (u / float(r_days)))
                    weights_daily.loc[t] = float(ETo.loc[t]) * w if np.isfinite(ETo.loc[t]) else 0.0

        delta_corr.loc[weights_daily.index] += diff_hat * weights_daily

        W_m = weights_daily.groupby(weights_daily.index.to_period("M")).sum()
        W_m.index = W_m.index.to_timestamp(how="start")
        W_m = W_m.reindex(months).fillna(0.0)

        diff_boot = _bootstrap_diff_fpre_fmin(pre_vals, post_vals, low_quantile, n_boot, rng) * inflation
        per_harvest.append({"W_m": W_m, "diff_boot": diff_boot})

    daily_df = pd.DataFrame({"ET_open": ET_open, "ETo": ETo, "ETof": ETof, "delta_corr": delta_corr})

    # Validate daily ET for this parcel (ETFIX-03)
    validate_daily_et(daily_df["ET_open"], uid=uid_str, context=f"WY{wy} {county_norm}")

    month_index = daily_df.index.to_period("M")
    monthly_open = daily_df["ET_open"].groupby(month_index).sum(min_count=1)
    monthly_delta = daily_df["delta_corr"].groupby(month_index).sum()
    monthly_df = pd.DataFrame({"ET_open": monthly_open, "delta_corr": monthly_delta})
    monthly_df.index = monthly_df.index.to_timestamp(how="start")
    monthly_df["ET_corr"] = (monthly_df["ET_open"] - monthly_df["delta_corr"]).clip(lower=0.0)

    monthly_df["ET_corr_ci_low"] = np.nan
    monthly_df["ET_corr_ci_high"] = np.nan

    if per_harvest:
        n_sim = int(max(50, min(2000, n_boot)))
        sim_delta = {m: np.zeros(n_sim, dtype=float) for m in monthly_df.index}
        for i in range(n_sim):
            acc = pd.Series(0.0, index=monthly_df.index)
            for ev in per_harvest:
                diff_i = float(ev["diff_boot"][i % len(ev["diff_boot"])])
                acc += ev["W_m"] * diff_i
            for m in monthly_df.index:
                sim_delta[m][i] = float(acc[m])

        ci_low = {}
        ci_high = {}
        for m in monthly_df.index:
            vals = sim_delta[m]
            ci_low[m] = np.quantile(vals, ci_alpha / 2.0)
            ci_high[m] = np.quantile(vals, 1.0 - ci_alpha / 2.0)

        monthly_df["ET_corr_ci_low"] = (monthly_df["ET_open"].sub(pd.Series(ci_high)).clip(lower=0.0))
        monthly_df["ET_corr_ci_high"] = (monthly_df["ET_open"].sub(pd.Series(ci_low)).clip(lower=0.0))

    return daily_df, monthly_df, harvest_dates, passes_df