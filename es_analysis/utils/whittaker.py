"""Whittaker smoother (penalized least squares) for EVI time-series.

Smooths irregularly-sampled data with per-observation quality weights.
Gaps (NaN) get weight 0 -- smoothed through based on surrounding signal,
not invented via polynomial interpolation.

Reference: Eilers, P.H.C. (2003) "A Perfect Smoother"
    Analytical Chemistry 75(14), 3631-3636.

Key advantage over gap-fill + Savitzky-Golay two-step:
  - Single-pass: no separate gap-fill stage that invents data
  - Weighted: real observations pull the curve proportionally to quality
  - Gaps are interpolated implicitly by the smoothness penalty
  - BEAST then sees a curve where low-weight regions are visibly smoother
    (lower curvature), providing a natural uncertainty signal
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def whittaker_smooth(y, weights=None, lmbda=1e4, d=2):
    """Whittaker smoother with optional per-observation weights.

    Solves: argmin_z  sum(w_i * (y_i - z_i)^2) + lmbda * sum((D^d z)^2)

    Args:
        y: Data values (1D array). NaN values get weight 0.
        weights: Per-observation weights in [0, 1]. None = 1.0 where finite.
        lmbda: Smoothing parameter. Larger = smoother. Typical range 1e2-1e6.
        d: Difference order (2 = penalize curvature, 3 = penalize jerk).

    Returns:
        Smoothed array (same length as y). Always finite.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n == 0:
        return y.copy()

    if weights is None:
        w = np.where(np.isfinite(y), 1.0, 0.0)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        w = np.where(np.isfinite(y), w, 0.0)

    # Replace NaN with 0 for the linear system (they have weight 0)
    y_clean = np.where(np.isfinite(y), y, 0.0)

    W = sparse.diags(w, 0, shape=(n, n), format="csc")
    D = _diff_matrix(n, d)
    A = W + lmbda * D.T.dot(D)
    b = W.dot(y_clean)
    z = spsolve(A, b)

    return np.asarray(z).ravel()


def _diff_matrix(n, d):
    """Build d-th order sparse difference matrix of size (n-d, n).

    First-order: D1[i,i] = -1, D1[i,i+1] = 1  (forward difference)
    Higher orders: apply first-order difference d times.
    """
    # Start with first-order difference: (n-1, n)
    diags = [-np.ones(n - 1), np.ones(n - 1)]
    D = sparse.diags(diags, [0, 1], shape=(n - 1, n), format="csc")
    for _ in range(d - 1):
        rows = D.shape[0]
        D1 = sparse.diags(
            [-np.ones(rows - 1), np.ones(rows - 1)],
            [0, 1], shape=(rows - 1, rows), format="csc",
        )
        D = D1.dot(D)
    return D


def whittaker_smooth_series(series, pixel_quality=None, lmbda=1e4, d=2):
    """Whittaker smooth a pandas Series with DatetimeIndex.

    Args:
        series: pd.Series with DatetimeIndex. NaN = missing data.
        pixel_quality: Optional pd.Series of quality weights (0-1),
            aligned to same index. E.g. valid_pixel_fraction * (1 - cloud_cover/100).
        lmbda: Smoothing parameter.
        d: Difference order.

    Returns:
        pd.Series with smoothed values, same index as input.
    """
    import pandas as pd

    y = series.to_numpy(dtype=float)

    if pixel_quality is not None:
        w = pixel_quality.reindex(series.index).to_numpy(dtype=float)
        w = np.where(np.isfinite(w), np.clip(w, 0.0, 1.0), 0.0)
    else:
        w = None

    z = whittaker_smooth(y, weights=w, lmbda=lmbda, d=d)

    return pd.Series(z, index=series.index, name=series.name)
