"""Unit conversion constants and functions for ET and area calculations.

Key identity: 1 ac-ft/acre = 1 ft of water depth = 304.8 mm
- mm and ac-ft/acre are BOTH depth units (water per unit area)
- ac-ft (without "per acre") is a VOLUME unit
- Converting mm -> ac-ft/acre requires NO area term (pure depth conversion)
- Converting mm -> ac-ft (volume) requires multiplying by area_acres
"""

import numpy as np

# Exact conversion constants
M2_PER_ACRE = 4046.8564224       # NIST exact
M3_PER_ACFT = 1233.48184         # standard
MM_PER_FOOT = 304.8              # exact (1 foot = 304.8 mm)


def mm_to_acft_per_acre(et_mm):
    """Convert ET depth from millimeters to ac-ft/acre.

    This is a pure depth conversion (no area involved).
    1 ac-ft/acre = 1 ft = 304.8 mm, so divide mm by 304.8.

    Args:
        et_mm: ET in millimeters (scalar, array, or Series).

    Returns:
        ET in ac-ft/acre (same type as input).
    """
    return et_mm / MM_PER_FOOT


def mm_to_acft_total(et_mm, area_acres):
    """Convert ET depth (mm) to total volume (ac-ft) for a given area.

    Converts depth to feet, then multiplies by area to get volume.
    volume_acft = (et_mm / 304.8) * area_acres

    Args:
        et_mm: ET in millimeters (scalar, array, or Series).
        area_acres: Parcel area in acres (scalar, array, or Series).

    Returns:
        Total ET volume in ac-ft.
    """
    return (et_mm / MM_PER_FOOT) * area_acres


def acft_total_to_acft_per_acre(acft_total, area_acres):
    """Convert total volume (ac-ft) to intensity (ac-ft/acre).

    Args:
        acft_total: Total volume in ac-ft (scalar, array, or Series).
        area_acres: Area in acres (scalar, array, or Series).

    Returns:
        Intensity in ac-ft/acre. NaN where area is zero or non-finite.
    """
    result = np.where(
        (np.asarray(area_acres, dtype=float) > 0) & np.isfinite(np.asarray(area_acres, dtype=float)),
        np.asarray(acft_total, dtype=float) / np.asarray(area_acres, dtype=float),
        np.nan,
    )
    # Return scalar if inputs were scalar
    if np.ndim(acft_total) == 0 and np.ndim(area_acres) == 0:
        return float(result)
    return result


def acft_per_acre_to_mm(et_acft_per_acre):
    """Convert ET intensity from ac-ft/acre back to millimeters.

    Inverse of mm_to_acft_per_acre.

    Args:
        et_acft_per_acre: ET in ac-ft/acre (scalar, array, or Series).

    Returns:
        ET in millimeters.
    """
    return et_acft_per_acre * MM_PER_FOOT
