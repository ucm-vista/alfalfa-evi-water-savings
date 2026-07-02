"""Water Year Type provider.

San Joaquin Valley Water Year Index (SJV WYI) classifications and
USDM peak drought categories for 10 study counties, WY2019-2024.

SJV WYI source: CA DWR Water Year Hydrologic Classification Indices
  https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST

USDM source: US Drought Monitor weekly reports, peak category per WY.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# SJV Water Year Index (Oct-Sep WY)
# Format: WY -> (index_value, type_label)
# ---------------------------------------------------------------------------
SJ_VALLEY_WY_INDEX: Dict[int, Tuple[float, str]] = {
    2019: (4.94, "Wet"),
    2020: (2.35, "Dry"),
    2021: (1.32, "Critical"),
    2022: (1.56, "Critical"),
    2023: (6.40, "Wet"),
    2024: (3.49, "Above Normal"),
}

# Canonical ordering of WY types (dry → wet)
WY_TYPE_ORDER = ["Critical", "Dry", "Below Normal", "Above Normal", "Wet"]

# Wong/Okabe-Ito colorblind-safe palette for WY types
WY_TYPE_COLORS: Dict[str, str] = {
    "Critical":     "#D55E00",  # vermillion
    "Dry":          "#E69F00",  # orange
    "Below Normal": "#F0E442",  # yellow
    "Above Normal": "#56B4E9",  # sky blue
    "Wet":          "#0072B2",  # blue
}

# ---------------------------------------------------------------------------
# USDM peak drought category per (county, WY)
# Categories: D0=Abnormally Dry, D1=Moderate, D2=Severe, D3=Extreme, D4=Exceptional
# Source: US Drought Monitor weekly composites, peak (worst) category during WY.
# ---------------------------------------------------------------------------
USDM_CATEGORIES_ORDER = ["None", "D0", "D1", "D2", "D3", "D4"]
USDM_COLORS: Dict[str, str] = {
    "None": "#FFFFFF",
    "D0":   "#FFFF00",  # abnormally dry
    "D1":   "#FCD37F",  # moderate drought
    "D2":   "#FFAA00",  # severe drought
    "D3":   "#E60000",  # extreme drought
    "D4":   "#730000",  # exceptional drought
}

USDM_PEAK: Dict[Tuple[str, int], str] = {
    # WY2019 — wet year, minimal drought
    ("Fresno", 2019): "None", ("San Joaquin", 2019): "None",
    ("Stanislaus", 2019): "None", ("Madera", 2019): "None",
    ("Kings", 2019): "None", ("Tulare", 2019): "None",
    ("Kern", 2019): "D0", ("Merced", 2019): "None",
    ("Imperial", 2019): "D1", ("Riverside", 2019): "D0",
    # WY2020 — dry year, moderate drought emerging
    ("Fresno", 2020): "D1", ("San Joaquin", 2020): "D0",
    ("Stanislaus", 2020): "D0", ("Madera", 2020): "D1",
    ("Kings", 2020): "D1", ("Tulare", 2020): "D1",
    ("Kern", 2020): "D2", ("Merced", 2020): "D0",
    ("Imperial", 2020): "D2", ("Riverside", 2020): "D2",
    # WY2021 — critical year, severe-extreme drought
    ("Fresno", 2021): "D3", ("San Joaquin", 2021): "D2",
    ("Stanislaus", 2021): "D2", ("Madera", 2021): "D3",
    ("Kings", 2021): "D3", ("Tulare", 2021): "D3",
    ("Kern", 2021): "D3", ("Merced", 2021): "D3",
    ("Imperial", 2021): "D3", ("Riverside", 2021): "D3",
    # WY2022 — critical year, exceptional drought in south
    ("Fresno", 2022): "D3", ("San Joaquin", 2022): "D2",
    ("Stanislaus", 2022): "D2", ("Madera", 2022): "D3",
    ("Kings", 2022): "D3", ("Tulare", 2022): "D3",
    ("Kern", 2022): "D4", ("Merced", 2022): "D3",
    ("Imperial", 2022): "D4", ("Riverside", 2022): "D4",
    # WY2023 — wet year, drought largely gone
    ("Fresno", 2023): "D0", ("San Joaquin", 2023): "None",
    ("Stanislaus", 2023): "None", ("Madera", 2023): "D0",
    ("Kings", 2023): "D0", ("Tulare", 2023): "D0",
    ("Kern", 2023): "D1", ("Merced", 2023): "None",
    ("Imperial", 2023): "D2", ("Riverside", 2023): "D1",
    # WY2024 — above normal, lingering spots
    ("Fresno", 2024): "None", ("San Joaquin", 2024): "None",
    ("Stanislaus", 2024): "None", ("Madera", 2024): "None",
    ("Kings", 2024): "D0", ("Tulare", 2024): "D0",
    ("Kern", 2024): "D0", ("Merced", 2024): "None",
    ("Imperial", 2024): "D1", ("Riverside", 2024): "D0",
}

# ---------------------------------------------------------------------------
# County region classification
# ---------------------------------------------------------------------------
SJ_VALLEY_COUNTIES = [
    "San Joaquin", "Stanislaus", "Merced", "Madera",
    "Fresno", "Tulare", "Kings", "Kern",
]
COLORADO_RIVER_COUNTIES = ["Imperial", "Riverside"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_wy_type(wy: int) -> str:
    """Return SJV WY type label for a given water year."""
    entry = SJ_VALLEY_WY_INDEX.get(wy)
    if entry is None:
        return "Unknown"
    return entry[1]


def get_wy_index(wy: int) -> Optional[float]:
    """Return SJV WY index value for a given water year."""
    entry = SJ_VALLEY_WY_INDEX.get(wy)
    return entry[0] if entry else None


def get_wy_color(wy: int) -> str:
    """Return color for a WY based on its type."""
    return WY_TYPE_COLORS.get(get_wy_type(wy), "#999999")


def get_wy_region(county: str) -> str:
    """Return region for a county: 'SJ Valley' or 'Colorado River'."""
    if county in COLORADO_RIVER_COUNTIES:
        return "Colorado River"
    return "SJ Valley"


def get_usdm_peak(county: str, wy: int) -> str:
    """Return peak USDM drought category for a county-WY pair."""
    return USDM_PEAK.get((county, wy), "Unknown")


def add_wy_type_columns(df: pd.DataFrame, wy_col: str = "WY") -> pd.DataFrame:
    """Add wy_type, wy_index, and wy_color columns to a DataFrame.

    Args:
        df: DataFrame with a water year column.
        wy_col: Name of the water year column.

    Returns:
        DataFrame with added columns (not a copy — modifies in place).
    """
    df["wy_type"] = df[wy_col].map(get_wy_type)
    df["wy_index"] = df[wy_col].map(get_wy_index)
    df["wy_color"] = df[wy_col].map(get_wy_color)
    return df


def add_usdm_columns(df: pd.DataFrame, county_col: str = "county",
                      wy_col: str = "WY") -> pd.DataFrame:
    """Add usdm_peak column to a DataFrame.

    Args:
        df: DataFrame with county and water year columns.
        county_col: Name of the county column.
        wy_col: Name of the water year column.

    Returns:
        DataFrame with added usdm_peak column (modifies in place).
    """
    df["usdm_peak"] = [
        get_usdm_peak(c, w) for c, w in zip(df[county_col], df[wy_col])
    ]
    return df


def get_wy_types_present(wys: List[int]) -> List[str]:
    """Return WY types present in a list of years, in canonical order."""
    types = {get_wy_type(wy) for wy in wys}
    return [t for t in WY_TYPE_ORDER if t in types]
