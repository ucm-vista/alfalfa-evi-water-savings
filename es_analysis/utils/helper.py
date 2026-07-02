"""Utility helper functions for EVI analysis."""

from typing import List, Tuple
import pandas as pd


def norm_name(s: pd.Series) -> pd.Series:
    """Normalize county names by replacing underscores with spaces, collapsing whitespace, and applying title case.

    Args:
        s: A pandas Series of string names to normalize.

    Returns:
        A normalized pandas Series with title cased names.
    """
    s = s.astype(str).str.replace("_", " ").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.str.title()


def water_year_bounds(wy: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Calculate start and end timestamps for a water year.

    Water Year 'wy' spans October 1st of the previous year (wy-1) through September 30th of the given year.

    Args:
        wy: The water year as an integer (e.g., 2020 for WY2020: Oct 1, 2019 to Sep 30, 2020).

    Returns:
        A tuple of (start_timestamp, end_timestamp) for the water year.
    """
    start = pd.Timestamp(year=wy - 1, month=10, day=1)
    end = pd.Timestamp(year=wy, month=9, day=30)
    return start, end


def in_water_year_domain(dt: pd.Series, years: List[int]) -> pd.Series:
    """Check which dates fall within the specified water year domain.

    Args:
        dt: A pandas Series of datetime values.
        years: A list of water years to check against.

    Returns:
        A boolean Series indicating whether each date is within the water year domain.
    """
    start, end = water_year_bounds_multi(years)
    return (dt >= start) & (dt <= end)


def nearest_odd(k: int) -> int:
    """Convert an integer to the nearest odd number.

    If the input is already odd, it returns the value unchanged.
    Otherwise, it returns k + 1.

    Args:
        k: An integer value.

    Returns:
        The nearest odd integer greater than or equal to k.
    """
    return k if k % 2 == 1 else k + 1


def norm_county_name(name: str) -> str:
    """Normalize a county name by replacing underscores with spaces and applying title case.

    Args:
        name: A county name string that may contain underscores.

    Returns:
        A normalized county name with spaces and title case (e.g., "san_joaquin" -> "San Joaquin").
    """
    s = str(name).replace("_", " ").strip()
    s = " ".join(s.split())
    return s.title()


def water_year_bounds_multi(years: List[int]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Calculate continuous timestamp bounds spanning multiple water years.

    For selected water years, build a continuous daily range from:
      start = Oct 1 of (min(years) - 1)
      end   = Sep 30 of max(years)

    Examples:
        years=[2020] -> Oct 1, 2019 ... Sep 30, 2020
        years=[2020, 2021] -> Oct 1, 2019 ... Sep 30, 2021

    Args:
        years: A list of water year integers.

    Returns:
        A tuple of (start_timestamp, end_timestamp) spanning all specified water years.
    """
    y0 = int(min(years))
    y1 = int(max(years))
    start = pd.Timestamp(year=y0 - 1, month=10, day=1)
    end = pd.Timestamp(year=y1, month=9, day=30)
    return start, end