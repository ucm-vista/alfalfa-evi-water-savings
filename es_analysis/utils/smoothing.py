"""Savitzky-Golay smoothing functions for EVI time series."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_sg(series: pd.Series, window: int = 15, poly: int = 3) -> pd.Series:
    """Apply Savitzky-Golay smoothing to non-NaN segments of a series.

    The function applies smoothing only to contiguous segments of non-NaN values
    in the input series. The window must be odd and >= poly+2; it is
    auto-adjusted if these constraints are not met.

    Args:
        series: Pandas Series or numpy array with numeric values (NaNs allowed).
        window: Window size for the Savitzky-Golay filter (must be odd). Auto-adjusted if even.
        poly: Polynomial order for the Savitzky-Golay filter.

    Returns:
        A pandas Series with smoothed values aligned to the input index.
    """
    from .helper import nearest_odd

    if isinstance(series, pd.Series):
        out = series.copy()
        y = series.to_numpy(dtype=float, copy=True)
    else:
        arr = np.asarray(series, dtype=float)
        out = pd.Series(arr, index=pd.RangeIndex(len(arr)))
        y = arr.copy()

    if len(y) == 0:
        return out

    w = int(nearest_odd(window))
    p = int(poly)
    if w <= p:
        w = p + 3
        if w % 2 == 0:
            w += 1

    isnan = np.isnan(y)
    n = len(y)
    start = 0
    while start < n:
        while start < n and isnan[start]:
            start += 1
        if start >= n:
            break
        end = start
        while end < n and not isnan[end]:
            end += 1
        seg = y[start:end]
        if len(seg) >= w:
            try:
                out.iloc[start:end] = savgol_filter(seg, window_length=w, polyorder=p, mode="interp")
            except Exception:
                pass
        start = end
    return out