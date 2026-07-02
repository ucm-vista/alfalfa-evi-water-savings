"""
Utilities module
Shared utility functions for EVI analysis
"""

from .helper import (
    norm_name, water_year_bounds, in_water_year_domain, nearest_odd,
    norm_county_name, water_year_bounds_multi
)
from .gapfill import quartic_gapfill, quartic_gapfill_daily
from .smoothing import smooth_sg
from .whittaker import whittaker_smooth, whittaker_smooth_series
from .units import (
    M2_PER_ACRE, M3_PER_ACFT, MM_PER_FOOT,
    mm_to_acft_per_acre, mm_to_acft_total,
    acft_total_to_acft_per_acre, acft_per_acre_to_mm,
)

__all__ = [
    "norm_name",
    "water_year_bounds",
    "in_water_year_domain",
    "nearest_odd",
    "norm_county_name",
    "water_year_bounds_multi",
    "quartic_gapfill",
    "quartic_gapfill_daily",
    "smooth_sg",
    "whittaker_smooth",
    "whittaker_smooth_series",
    "M2_PER_ACRE",
    "M3_PER_ACFT",
    "MM_PER_FOOT",
    "mm_to_acft_per_acre",
    "mm_to_acft_total",
    "acft_total_to_acft_per_acre",
    "acft_per_acre_to_mm",
]