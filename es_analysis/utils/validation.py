"""Validation checkpoints for ET pipeline stages.

Catches physically impossible values (hard errors) and unusual-but-possible
values (warnings) at key pipeline stages. All-NaN inputs pass silently.

Expected ranges for irrigated alfalfa in CA Central/Southern Valley:
  - Daily ET: 0-12 mm/day (typical), up to 30 mm/day (physical limit)
  - Cycle ET: 30-300 mm per cutting cycle
  - Annual ET: 800-1500 mm/year
  - Per-cutting: 0.1-1.0 ac-ft/acre per cutting
"""

import warnings
import numpy as np
import pandas as pd


class ETValidationWarning(UserWarning):
    """Warning for ET values outside expected ranges."""
    pass


MIN_CUMULATIVE_ET_MM = 30.0  # matches cycle ET lower bound for irrigated alfalfa


def validate_daily_et(et_series: pd.Series, uid: str = "", context: str = ""):
    """Validate daily ET values are in plausible range.

    Expected: 0-12 mm/day for irrigated alfalfa.
    Hard fail: negative values or > 30 mm/day (physically impossible).
    Soft warn: > 12 mm/day (unusual but possible in extreme heat).
    """
    if et_series.isna().all():
        return  # all-NaN is valid (no data)

    vals = et_series.dropna()
    if vals.empty:
        return

    min_val = float(vals.min())
    max_val = float(vals.max())

    if min_val < -0.1:
        raise ValueError(
            f"Negative daily ET detected ({min_val:.2f} mm/day). "
            f"UID={uid} {context}"
        )
    if max_val > 30.0:
        raise ValueError(
            f"Physically impossible daily ET ({max_val:.2f} mm/day > 30). "
            f"UID={uid} {context}"
        )
    if max_val > 12.0:
        warnings.warn(
            f"Unusually high daily ET ({max_val:.2f} mm/day > 12). "
            f"UID={uid} {context}",
            ETValidationWarning,
        )


def validate_cycle_et_mm(cycle_et_mm: float, uid: str = "", cut_index: int = 0):
    """Validate per-cycle ET total in mm.

    Expected: 30-300 mm per cutting cycle.
    Hard fail: negative or > 500 mm.
    Soft warn: < 30 mm (very short cycle or data gap) or > 300 mm.
    """
    if np.isnan(cycle_et_mm):
        return  # NaN is valid (missing data)

    if cycle_et_mm < 0:
        raise ValueError(
            f"Negative cycle ET ({cycle_et_mm:.1f} mm). "
            f"UID={uid} cut={cut_index}"
        )
    if cycle_et_mm > 800:
        raise ValueError(
            f"Physically impossible cycle ET ({cycle_et_mm:.1f} mm > 800). "
            f"UID={uid} cut={cut_index}"
        )
    if cycle_et_mm < 30:
        warnings.warn(
            f"Low cycle ET ({cycle_et_mm:.1f} mm < 30). "
            f"UID={uid} cut={cut_index}",
            ETValidationWarning,
        )
    if cycle_et_mm > 300:
        warnings.warn(
            f"High cycle ET ({cycle_et_mm:.1f} mm > 300). "
            f"UID={uid} cut={cut_index}",
            ETValidationWarning,
        )


def validate_annual_et_mm(annual_et_mm: float, uid: str = "", wy: int = 0):
    """Validate annual ET total in mm.

    Expected: 800-1500 mm for irrigated alfalfa in CA.
    Hard fail: < 0 or > 3000 mm.
    Soft warn: outside 800-1500 range.
    """
    if np.isnan(annual_et_mm):
        return

    if annual_et_mm < 0:
        raise ValueError(
            f"Negative annual ET ({annual_et_mm:.1f} mm). "
            f"UID={uid} WY={wy}"
        )
    if annual_et_mm > 3000:
        raise ValueError(
            f"Physically impossible annual ET ({annual_et_mm:.1f} mm > 3000). "
            f"UID={uid} WY={wy}"
        )
    if annual_et_mm < 800:
        warnings.warn(
            f"Low annual ET ({annual_et_mm:.1f} mm < 800). "
            f"UID={uid} WY={wy}",
            ETValidationWarning,
        )
    if annual_et_mm > 1500:
        warnings.warn(
            f"High annual ET ({annual_et_mm:.1f} mm > 1500). "
            f"UID={uid} WY={wy}",
            ETValidationWarning,
        )


def validate_per_cutting_acft_per_acre(value: float, uid: str = "", cut_index: int = 0):
    """Validate per-cutting ET in ac-ft/acre.

    Expected: 0.1-1.0 ac-ft/acre per cutting.
    Hard fail: negative or > 2.0.
    Soft warn: outside 0.1-1.0 range.
    """
    if np.isnan(value):
        return

    if value < 0:
        raise ValueError(
            f"Negative per-cutting ET ({value:.4f} ac-ft/acre). "
            f"UID={uid} cut={cut_index}"
        )
    if value > 2.0:
        raise ValueError(
            f"Physically impossible per-cutting ET ({value:.4f} ac-ft/acre > 2.0). "
            f"UID={uid} cut={cut_index}"
        )
    if value < 0.1:
        warnings.warn(
            f"Low per-cutting ET ({value:.4f} ac-ft/acre < 0.1). "
            f"UID={uid} cut={cut_index}",
            ETValidationWarning,
        )
    if value > 1.0:
        warnings.warn(
            f"High per-cutting ET ({value:.4f} ac-ft/acre > 1.0). "
            f"UID={uid} cut={cut_index}",
            ETValidationWarning,
        )
