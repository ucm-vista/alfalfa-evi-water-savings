"""Gap filling functions for EVI time series data."""

import numpy as np
import pandas as pd


def quartic_gapfill(daily_df: pd.DataFrame, window_days: int) -> pd.Series:
    """Fill gaps in EVI data using local 4th-degree polynomial interpolation.

    Fits a quartic polynomial within ±window_days around each gap point,
    but only fills gaps between the minimum and maximum observed dates
    (no edge extrapolation).

    Args:
        daily_df: A DataFrame with 'date' and 'mean_evi' columns containing daily EVI data.
        window_days: The window size in days for the local polynomial fit (±days around each gap).

    Returns:
        A pandas Series with the same index as the input, with gaps filled where possible.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)

    ser = daily_df.set_index("date")["mean_evi"].copy()
    ser.index = pd.to_datetime(ser.index)
    idx_sec = ser.index.astype("int64") // 10**9

    filled = ser.copy()
    valid_mask = ~ser.isna()
    if valid_mask.sum() < 5:
        return filled

    x_valid = idx_sec[valid_mask].astype(float).to_numpy()
    y_valid = ser[valid_mask].astype(float).to_numpy()
    lo, hi = x_valid.min(), x_valid.max()
    sec_window = int(window_days) * 24 * 3600

    for pos in np.where(~valid_mask)[0]:
        x0 = float(idx_sec[pos])
        if not (lo <= x0 <= hi):
            continue
        m = (x_valid >= x0 - sec_window) & (x_valid <= x0 + sec_window)
        xn, yn = x_valid[m], y_valid[m]
        if len(xn) >= 5:
            try:
                coeffs = np.polyfit(xn - x0, yn, deg=4)
                filled.iloc[pos] = np.polyval(coeffs, 0.0)
            except Exception:
                pass
    return filled


def quartic_gapfill_daily(series: pd.Series, window_days: int = 30) -> pd.Series:
    """Fill gaps in a daily EVI time series using local 4th-degree polynomial interpolation.

    Performs quartic gap-filling within ±window_days, ONLY between min/max observed dates.
    The series index must be a daily DatetimeIndex; NaN values mark gaps.

    Args:
        series: A pandas Series with a daily DatetimeIndex containing EVI values (NaNs mark gaps).
        window_days: The window size in days for the local polynomial fit (default: 30).

    Returns:
        A pandas Series with the same index, with gaps filled where possible.
    """
    if series.empty:
        return series.copy()
    s = series.copy()
    idx = pd.to_datetime(s.index)
    idx_sec = idx.astype("int64") // 10**9

    valid = ~s.isna()
    if valid.sum() < 5:
        return s

    x_valid = idx_sec[valid].astype(float).to_numpy()
    y_valid = s[valid].astype(float).to_numpy()
    lo, hi = x_valid.min(), x_valid.max()
    sec_window = int(window_days) * 86400

    for pos in np.where(~valid)[0]:
        x0 = float(idx_sec[pos])
        if not (lo <= x0 <= hi):
            continue
        m = (x_valid >= x0 - sec_window) & (x_valid <= x0 + sec_window)
        xn, yn = x_valid[m], y_valid[m]
        if len(xn) >= 5:
            try:
                coeffs = np.polyfit(xn - x0, yn, deg=4)
                s.iloc[pos] = np.polyval(coeffs, 0.0)
            except Exception:
                pass
    return s