"""Data provider for Phase 5 publication figures.

Computes the timing gap between detected harvest (cutting) events and
the nearest Landsat satellite overpass for each parcel-year.  This data
underpins FIG-03 (satellite pass timing gap histogram/heatmap).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import config
from .evi_provider import normalize_county_name, water_year_bounds
from .spatial_provider import COUNTY_ORDER


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_all_landsat_dates(
    county: str,
    wy: int,
    landsat_meta_csv: Optional[Path] = None,
) -> np.ndarray:
    """Load ALL unique Landsat pass dates for a county-WY (all tracks).

    Unlike ``LandsatDataProvider.load_landsat_passes`` which selects the
    dominant WRS path/row track, this returns every unique overpass date
    across all WRS tracks.  This gives a more accurate "nearest pass"
    estimate for timing gap computation.

    Returns:
        Sorted numpy array of datetime64[ns] values.
    """
    if landsat_meta_csv is None:
        landsat_meta_csv = config.landsat_meta_csv

    county_norm = normalize_county_name(county)
    df = pd.read_csv(landsat_meta_csv, sep=None, engine="python")

    # Normalize columns
    low = {c.lower(): c for c in df.columns}
    df = df.rename(columns={
        low["date_only"]: "date_only",
        low["county"]: "county",
    })
    df["county"] = df["county"].astype(str).str.strip().str.title()
    df["date_only"] = pd.to_datetime(df["date_only"], errors="coerce").dt.normalize()

    start, end = water_year_bounds(wy)
    mask = df["county"].eq(county_norm) & df["date_only"].between(start, end)
    dates = df.loc[mask, "date_only"].dropna().unique()

    return np.sort(dates)


def _nearest_gap(harvest_dates: np.ndarray, pass_dates: np.ndarray):
    """Vectorised nearest-pass lookup using searchsorted.

    For each harvest date, finds the nearest Landsat pass in either
    direction and the first pass strictly after.

    Args:
        harvest_dates: Array of datetime64[ns] harvest timestamps.
        pass_dates: Sorted array of datetime64[ns] Landsat pass timestamps.

    Returns:
        Tuple of (nearest_dates, gap_days, gap_days_after) arrays.
        - nearest_dates: datetime64[ns] of nearest pass
        - gap_days: int days to nearest pass (either direction)
        - gap_days_after: int days to first pass AFTER harvest (NaN if none)
    """
    n = len(harvest_dates)
    nearest_dates = np.empty(n, dtype="datetime64[ns]")
    gap_days = np.full(n, np.nan)
    gap_days_after = np.full(n, np.nan)

    if len(pass_dates) == 0:
        nearest_dates[:] = np.datetime64("NaT")
        return nearest_dates, gap_days, gap_days_after

    # Convert to int64 (nanoseconds) for searchsorted
    h_ns = harvest_dates.astype("int64")
    p_ns = pass_dates.astype("int64")

    idx = np.searchsorted(p_ns, h_ns, side="left")

    for i in range(n):
        pos = idx[i]

        # Candidates: pass at pos-1 (before or at) and pos (at or after)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(pass_dates):
            candidates.append(pos)

        if not candidates:
            nearest_dates[i] = np.datetime64("NaT")
            continue

        # Find nearest
        diffs = [abs((pass_dates[c] - harvest_dates[i]) / np.timedelta64(1, "D"))
                 for c in candidates]
        best_idx = candidates[int(np.argmin(diffs))]
        nearest_dates[i] = pass_dates[best_idx]
        gap_days[i] = min(diffs)

        # First pass AFTER harvest
        after_pos = pos
        # If pass_dates[pos] == harvest_dates[i], the "after" is the next one
        if after_pos < len(pass_dates) and pass_dates[after_pos] <= harvest_dates[i]:
            after_pos += 1
        if after_pos < len(pass_dates):
            gap_days_after[i] = (pass_dates[after_pos] - harvest_dates[i]) / np.timedelta64(1, "D")

    return nearest_dates, gap_days, gap_days_after


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_timing_gaps(
    counties: Optional[List[str]] = None,
    water_years: Optional[List[int]] = None,
    cloud_cover_max: Optional[float] = None,
) -> pd.DataFrame:
    """Compute timing gaps between harvest dates and Landsat overpasses.

    For every cutting event detected by BEAST, finds the nearest Landsat
    pass (across all WRS tracks) and computes the gap in days.  This
    powers the FIG-03 timing gap histogram.

    Args:
        counties: County names to include (default: COUNTY_ORDER, 10 counties).
        water_years: Water years to include (default: config.water_years).
        cloud_cover_max: Currently unused (all passes are included regardless
            of cloud cover to find true nearest overpass).  Reserved for
            future filtering.

    Returns:
        DataFrame with columns:
        - county (str): Normalized county name (Title Case)
        - WY (int): Water year
        - parcel_id (str): Parcel identifier
        - harvest_date (Timestamp): Date of cutting event
        - nearest_pass_date (Timestamp): Nearest Landsat overpass
        - gap_days (int): Absolute days to nearest pass
        - gap_days_after (float): Days to first pass AFTER cutting (NaN if none)
        - month (int): Month of cutting event (1-12)
    """
    if counties is None:
        counties = list(COUNTY_ORDER)
    if water_years is None:
        water_years = list(config.water_years)

    all_rows = []
    skipped = 0

    for county in counties:
        county_norm = normalize_county_name(county)

        for wy in water_years:
            # --- Load harvest dates from BEAST seasonal CSV ---
            csv_path = (
                config.beast_out_root_new
                / county_norm
                / f"beast_seasonal_cuts_WY{wy}.csv"
            )
            if not csv_path.exists():
                print(f"  [warn] BEAST CSV not found, skipping: {csv_path}")
                skipped += 1
                continue

            beast_df = pd.read_csv(csv_path)

            # Filter to OK rows with valid cutting dates
            if "ok" in beast_df.columns:
                beast_df = beast_df[beast_df["ok"].astype(str).str.lower() == "true"]

            dates_col = "season_cp_dates_iso"
            if dates_col not in beast_df.columns:
                print(f"  [warn] Column '{dates_col}' missing in {csv_path}")
                continue

            # Drop rows with NaN/empty dates
            mask = beast_df[dates_col].notna() & (beast_df[dates_col].astype(str).str.strip() != "")
            beast_df = beast_df[mask].copy()

            if beast_df.empty:
                continue

            # --- Load ALL Landsat pass dates for this county-WY ---
            pass_dates = _load_all_landsat_dates(county_norm, wy)

            if len(pass_dates) == 0:
                print(f"  [warn] No Landsat passes for {county_norm} WY{wy}")
                continue

            # --- Explode cutting dates and compute gaps ---
            for _, row in beast_df.iterrows():
                pid = str(row["parcel_id"])
                date_str = str(row[dates_col]).strip()

                if not date_str:
                    continue

                parts = date_str.split(";")
                harvest_dates_list = []
                for part in parts:
                    part = part.strip()
                    if part:
                        try:
                            harvest_dates_list.append(pd.Timestamp(part))
                        except (ValueError, TypeError):
                            continue

                if not harvest_dates_list:
                    continue

                h_arr = np.array(harvest_dates_list, dtype="datetime64[ns]")
                nearest, gaps, gaps_after = _nearest_gap(h_arr, pass_dates)

                for j in range(len(h_arr)):
                    all_rows.append({
                        "county": county_norm,
                        "WY": wy,
                        "parcel_id": pid,
                        "harvest_date": harvest_dates_list[j],
                        "nearest_pass_date": pd.Timestamp(nearest[j]),
                        "gap_days": int(gaps[j]) if not np.isnan(gaps[j]) else np.nan,
                        "gap_days_after": gaps_after[j],
                        "month": harvest_dates_list[j].month,
                    })

    if not all_rows:
        print("  [warn] No timing gap data produced")
        return pd.DataFrame(columns=[
            "county", "WY", "parcel_id", "harvest_date",
            "nearest_pass_date", "gap_days", "gap_days_after", "month",
        ])

    result = pd.DataFrame(all_rows)

    # Summary
    print(f"\nTiming gap summary:")
    print(f"  Total cutting events: {len(result):,}")
    print(f"  Median gap (nearest): {result['gap_days'].median():.1f} days")
    print(f"  Mean gap (nearest):   {result['gap_days'].mean():.1f} days")
    print(f"  Counties: {result['county'].nunique()}")
    print(f"  Water years: {sorted(result['WY'].unique())}")
    if skipped:
        print(f"  Skipped county-WY pairs (missing CSV): {skipped}")

    return result
