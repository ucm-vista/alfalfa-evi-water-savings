"""EVI data provider module.
Source lines: 4-510, 512-809
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from .config import config
from ..utils.gapfill import quartic_gapfill as _quartic_gapfill
from ..utils.smoothing import smooth_sg as _smooth_sg
from ..utils.whittaker import whittaker_smooth as _whittaker_smooth


def normalize_county_name(s: str) -> str:
    """Normalize county name: underscores -> spaces, Title Case.

    Args:
        s: County name string, may contain underscores

    Returns:
        Normalized county name with spaces and Title Case
    """
    s = str(s).replace("_", " ").strip()
    s = " ".join(s.split())
    return s.title()


def normalize_county_names(series: pd.Series) -> pd.Series:
    """Normalize county names in a pandas Series.

    Args:
        series: Series containing county names

    Returns:
        Series with normalized county names
    """
    s = series.astype(str).str.replace("_", " ").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.str.title()


def water_year_bounds(year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get start and end dates for a water year.

    Water Year N spans October 1 of year N-1 to September 30 of year N.

    Args:
        year: Water year (e.g., 2020)

    Returns:
        Tuple of (start_date, end_date)
    """
    start = pd.Timestamp(year=year - 1, month=10, day=1)
    end = pd.Timestamp(year=year, month=9, day=30)
    return start, end


def water_year_bounds_for_years(years: List[int]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get start and end dates covering multiple water years.

    Args:
        years: List of water years

    Returns:
        Tuple of (start_date, end_date) covering all water years
    """
    if not years:
        raise ValueError("years list cannot be empty")
    y0 = int(min(years))
    y1 = int(max(years))
    start = pd.Timestamp(year=y0 - 1, month=10, day=1)
    end = pd.Timestamp(year=y1, month=9, day=30)
    return start, end


def in_water_year(dates: pd.Series, year: int) -> pd.Series:
    """Check if dates fall within a water year.

    Args:
        dates: Series of dates
        year: Water year

    Returns:
        Boolean series indicating which dates are within the water year
    """
    start, end = water_year_bounds(year)
    return (dates >= start) & (dates <= end)


class EviDataProvider:
    """Provider for loading and processing EVI time-series data."""

    def __init__(self, csv_path: Optional[Path] = None):
        """Initialize EVI data provider.

        Args:
            csv_path: Path to EVI CSV file. If None, uses config.csv_path
        """
        self.csv_path = Path(csv_path) if csv_path else config.csv_path
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load EVI data from CSV and apply basic filtering.

        Returns:
            DataFrame with EVI data
        """
        df = pd.read_csv(self.csv_path, low_memory=False)

        # --- Column rename: emery_method_2 format → es_analysis format ---
        df = df.rename(columns={
            "evi_mean": "mean_evi",
            "evi_std": "std_evi",
            "evi_min": "min_evi",
            "evi_max": "max_evi",
        })

        # --- QA filter: drop scenes with bad qa_status ---
        if config.evi_filter_qa and "qa_status" in df.columns:
            df = df[df["qa_status"] == "ok"].copy()

        # --- Valid pixel fraction filter ---
        if "valid_pixels" in df.columns and "total_pixels" in df.columns:
            frac = df["valid_pixels"] / df["total_pixels"]
            df = df[frac.isna() | (frac >= config.evi_min_valid_pixel_fraction)].copy()

        # --- Scene-level cloud cover filter ---
        if "cloud_cover" in df.columns:
            df = df[df["cloud_cover"].isna() | (df["cloud_cover"] <= config.evi_cloud_cover_max)].copy()

        required_cols = {"date", "mean_evi", "county", "parcel_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        df["county_norm"] = normalize_county_names(df["county"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["date"]).copy()

        df["year"] = df["date"].dt.year

        self.data = df
        return df

    def filter_data(
        self,
        counties: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Filter loaded EVI data by counties and years.

        Args:
            counties: List of county names (normalized)
            years: List of calendar years

        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            self.load_data()

        df = self.data.copy()

        if counties:
            counties_norm = [normalize_county_name(c) for c in counties]
            df = df[df["county_norm"].isin(counties_norm)].copy()

        if years:
            df = df[df["year"].isin(years)].copy()

        return df

    def create_daily_timeseries(
        self,
        df: pd.DataFrame,
        county: str,
        parcel_id: str,
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Build daily EVI timeseries for a specific parcel.

        Args:
            df: DataFrame with EVI data
            county: County name
            parcel_id: Parcel identifier
            years: List of water years to include

        Returns:
            DataFrame with daily EVI values
        """
        if years is None and self.data is not None:
            years = sorted(self.data["year"].unique().tolist())
        elif years is None:
            years = config.water_years

        years = [int(y) for y in years]

        sub = df[
            (df["county_norm"] == normalize_county_name(county)) &
            (df["parcel_id"].astype(str) == str(parcel_id))
        ][["date", "mean_evi"]].sort_values("date")

        return daily_timeseries_water_year(sub, years)

    def process_parcel(
        self,
        county: str,
        parcel_id: str,
        years: List[int],
        interp_window: int = None,
        sg_window: int = None,
        sg_poly: int = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load, gap-fill, and smooth EVI for a single parcel.

        When config.evi_smoothing_method == "whittaker", produces a
        single-pass Whittaker smooth (returned as both filled and smoothed
        for backward compatibility).

        Args:
            county: County name
            parcel_id: Parcel identifier
            years: List of water years
            interp_window: Gap-fill window in days
            sg_window: Savitzky-Golay window size
            sg_poly: Savitzky-Golay polynomial order

        Returns:
            Tuple of (daily_df, gapfilled_series, smoothed_series)
        """
        if self.data is None:
            self.load_data()
        daily = self.create_daily_timeseries(self.data, county, parcel_id, years)

        if config.evi_smoothing_method == "whittaker":
            whit = self.smooth_whittaker(daily)
            return daily, whit, whit
        else:
            filled = self.quartic_gapfill(daily, interp_window)
            smoothed = self.smooth_sg(filled, sg_window, sg_poly)
            return daily, filled, smoothed

    @staticmethod
    def quartic_gapfill(daily_df: pd.DataFrame, window_days: int = None) -> pd.Series:
        """Fill gaps using local 4th-degree polynomial.

        Args:
            daily_df: DataFrame with date and mean_evi columns
            window_days: Window size in days for gap-filling

        Returns:
            Series with gap-filled values
        """
        if window_days is None:
            window_days = config.interp_window_days

        return _quartic_gapfill(daily_df, window_days)

    @staticmethod
    def smooth_sg(series: pd.Series, window: int = None, poly: int = None) -> pd.Series:
        """Apply Savitzky-Golay smoothing.

        Args:
            series: Series to smooth
            window: Window size (must be odd)
            poly: Polynomial order

        Returns:
            Smoothed series
        """
        if window is None:
            window = config.sg_window
        if poly is None:
            poly = config.sg_poly

        return _smooth_sg(series, window, poly)

    @staticmethod
    def smooth_whittaker(
        daily_df: pd.DataFrame,
        lmbda: float = None,
        order: int = None,
    ) -> pd.Series:
        """Whittaker smoother: single-pass gap-fill + smooth.

        Replaces the quartic gap-fill + SG smoothing two-step with a
        single penalized least-squares fit. NaN observations get weight 0
        and are smoothed through implicitly.

        Args:
            daily_df: DataFrame with date and mean_evi columns.
            lmbda: Smoothing parameter (None = config.whittaker_lambda).
            order: Difference order (None = config.whittaker_order).

        Returns:
            Series with smoothed values (same length as input, no NaNs).
        """
        if lmbda is None:
            lmbda = config.whittaker_lambda
        if order is None:
            order = config.whittaker_order

        y = daily_df["mean_evi"].to_numpy(dtype=float)
        z = _whittaker_smooth(y, weights=None, lmbda=lmbda, d=order)
        return pd.Series(z, index=daily_df.index, name="whittaker_mean_evi")


def daily_timeseries_water_year(
    df_sel: pd.DataFrame,
    years: List[int],
) -> pd.DataFrame:
    """Build a daily series over the water-year domain.

    Steps:
      1) Keep only rows within the water-year window
      2) Floor to day
      3) Aggregate duplicates per day by mean
      4) Reindex to full water-year grid (gaps = NaN)

    Args:
        df_sel: DataFrame with date and mean_evi columns
        years: List of water years

    Returns:
        DataFrame with columns [date, mean_evi]
    """
    start, end = water_year_bounds_for_years(years)
    full_index = pd.date_range(start, end, freq="D")

    if df_sel.empty:
        return pd.DataFrame({"date": full_index, "mean_evi": np.nan})

    dates = pd.to_datetime(df_sel["date"])
    mask = (dates >= start) & (dates <= end)
    dfw = df_sel.loc[mask, ["date", "mean_evi"]].copy()

    if dfw.empty:
        return pd.DataFrame({"date": full_index, "mean_evi": np.nan})

    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce").dt.tz_localize(None).dt.floor("D")

    daily_mean = (
        dfw.groupby("date", as_index=True)["mean_evi"]
           .mean()
           .sort_index()
    )

    s = daily_mean.reindex(full_index)
    s.name = "mean_evi"
    out = s.to_frame()
    out["date"] = out.index
    return out.reset_index(drop=True)


def build_daily_table(
    daily_df: pd.DataFrame,
    filled: pd.Series,
    smoothed: pd.Series,
) -> pd.DataFrame:
    """Combine Original, Gap-filled, and Smoothed into one daily table.

    When config.evi_smoothing_method == "whittaker", the gapfilled and
    smoothed columns both contain the Whittaker output, plus an explicit
    whittaker_mean_evi column is added.

    Args:
        daily_df: DataFrame with original daily data
        filled: Series with gap-filled values
        smoothed: Series with smoothed values

    Returns:
        Combined DataFrame
    """
    out = pd.DataFrame({
        "date": daily_df["date"],
        "original_mean_evi": daily_df["mean_evi"].to_numpy(),
        "gapfilled_mean_evi": np.asarray(filled, dtype=float),
        "smoothed_mean_evi": np.asarray(smoothed, dtype=float),
    })
    if config.evi_smoothing_method == "whittaker":
        out["whittaker_mean_evi"] = np.asarray(smoothed, dtype=float)
    return out