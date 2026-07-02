"""Sensitivity heatmaps for water savings analysis.

Figure 1 — ET sensitivity (2 rows x 4 cols):
  Row 0: Cap 0 savings — ET bins x cuttings, by WY type
  Row 1: Cap 1 savings — ET bins x cuttings, by WY type

Figure 2 — Cutoff-date sensitivity (3 rows x 4 cols):
  Row 0: Mean n_late_cuts — cutoff date x county (N→S), by WY type
  Row 1: Cap 0 savings   — cutoff date x county (N→S), by WY type
  Row 2: Cap 1 savings   — cutoff date x county (N→S), by WY type
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.data_providers.wy_type_provider import (
    WY_TYPE_COLORS,
    add_wy_type_columns,
)
from es_analysis.utils.units import mm_to_acft_per_acre
from es_analysis.utils.publication_style import (
    apply_style,
    save_pub_figure,
    add_panel_label,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET_BIN_EDGES = [0, 150, 300, 500, 700, 900, 1100, 1400, 1700, 2100, 2500]
ET_BIN_LABELS = [
    "0\u2013150", "150\u2013300", "300\u2013500", "500\u2013700",
    "700\u2013900", "900\u20131100", "1100\u20131400", "1400\u20131700",
    "1700\u20132100", "2100\u20132500",
]
ET_BIN_CENTERS = [75, 225, 400, 600, 800, 1000, 1250, 1550, 1900, 2300]
N_CUTTINGS_RANGE = list(range(1, 15))
WY_TYPES_DISPLAY = ["Critical", "Dry", "Above Normal", "Wet"]
MIN_EMPIRICAL_COUNT = 5

CUTOFF_DATES = [
    (7, 1), (7, 16), (8, 1), (8, 16), (9, 1), (9, 16), (10, 1),
]
CUTOFF_LABELS = [
    "Jul 1", "Jul 16", "Aug 1", "Aug 16", "Sep 1", "Sep 16", "Oct 1",
]

COUNTIES_NS = list(COUNTY_ORDER)

# Literature-informed annual cuttings per county
# (Putnam et al. 2008; Orloff et al. 2015; UC ANR alfalfa reports)
LITERATURE_ANNUAL_CUTS = {
    "San Joaquin": 8, "Stanislaus": 8, "Merced": 8,
    "Madera": 8, "Fresno": 9, "Tulare": 9,
    "Kings": 9, "Kern": 10, "Riverside": 12, "Imperial": 12,
}

# Observed per-late-cut ET by county (mm), from calibration data
COUNTY_PER_CUT_ET_MM = {
    "San Joaquin": 174, "Stanislaus": 164, "Merced": 161,
    "Madera": 176, "Fresno": 173, "Tulare": 167,
    "Kings": 172, "Kern": 159, "Riverside": 176, "Imperial": 175,
}

CMAP_BLUE_GREEN = LinearSegmentedColormap.from_list(
    "BlueGreen",
    ["#E0F7FA", "#80CBC4", "#388E3C", "#1B5E20"],
    N=256,
)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_date_list(s: str):
    if not isinstance(s, str) or s.strip() in ("[]", "", "nan"):
        return []
    return [pd.Timestamp(m) for m in re.findall(r"(\d{4}-\d{2}-\d{2})", s)]


def _parse_float_list(s: str):
    if not isinstance(s, str) or s.strip() in ("[]", "", "nan"):
        return []
    cleaned = re.sub(r"\bnan\b", 'float("nan")', s).replace("\n", ",")
    try:
        return [float(v) for v in ast.literal_eval(cleaned)]
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _fit_et_scaling_beta(df: pd.DataFrame, cal_partial: Dict) -> float:
    """Fit empirical β for the ET power scaling.

    Replaces the hard-coded sqrt (β=0.5).  Uses observed parcel-years with
    at least one late cut to fit:

        log(per_cut_obs / base_per_cut[nc]) = β · log(total_et / ref_et[nc])

    via least-squares with no intercept.  Returns the fitted β clipped to
    [0.1, 1.0] for stability.
    """
    sub = df[
        (df["n_late_cuts"] > 0)
        & (df["total_et_mm"] > 50)
        & (df["late_et_mm"] > 0)
    ].copy()
    if len(sub) < 50:
        return 0.5

    sub["per_cut_obs"] = sub["late_et_mm"] / sub["n_late_cuts"]
    sub["base_per_cut"] = sub["n_cuttings"].map(cal_partial["per_late_cut_et_by_ncuts"])
    sub["ref_et"] = sub["n_cuttings"].map(cal_partial["mean_et_by_ncuts"])
    sub = sub[(sub["base_per_cut"] > 0) & (sub["ref_et"] > 0)
              & (sub["per_cut_obs"] > 0)]
    if len(sub) < 50:
        return 0.5

    y = np.log(sub["per_cut_obs"] / sub["base_per_cut"]).values
    x = np.log(sub["total_et_mm"] / sub["ref_et"]).values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 50 or float(np.sum(x ** 2)) == 0.0:
        return 0.5

    beta = float(np.sum(x * y) / np.sum(x ** 2))
    return float(np.clip(beta, 0.1, 1.0))


def _fit_savings_regression(
    df: pd.DataFrame, cap_k: int,
) -> Optional[Dict]:
    """Fit a log-linear regression for direct savings prediction.

    Replaces the marginal-product assumption (`late_frac` averaging WY-type
    and n_cuttings marginals).  Model:

        log(saved_mm + 1) = β0
                          + β_nc · log(n_cuttings)
                          + β_et · log(total_et_mm)
                          + γ_{wy_type}     (one-hot, base level absorbed
                                              into the intercept)

    Fitted via OLS on observed parcel-years.  Returns a dict with the
    coefficients and an in-sample R², or None if the cap column is missing
    or there's not enough data.
    """
    col = f"saved_mm_cap{cap_k}"
    if col not in df.columns:
        return None

    sub = df[
        (df["n_cuttings"] > 0)
        & (df["total_et_mm"] > 50)
        & (df[col].notna())
        & (df[col] >= 0)
        & (df["wy_type"].notna())
        & (df["wy_type"] != "")
    ].copy()
    if len(sub) < 100:
        return None

    log_nc = np.log(sub["n_cuttings"].values.astype(float))
    log_et = np.log(sub["total_et_mm"].values.astype(float))
    y = np.log(sub[col].values.astype(float) + 1.0)

    wy_types_seen = sorted(sub["wy_type"].unique().tolist())
    base_wt = wy_types_seen[0]
    other_wts = wy_types_seen[1:]
    wy_arr = sub["wy_type"].values

    cols = [np.ones(len(sub)), log_nc, log_et]
    for wt in other_wts:
        cols.append((wy_arr == wt).astype(float))
    X = np.column_stack(cols)

    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rsq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "intercept": float(coefs[0]),
        "beta_nc": float(coefs[1]),
        "beta_et": float(coefs[2]),
        "wy_offsets": {wt: float(c) for wt, c in zip(other_wts, coefs[3:])},
        "base_wt": base_wt,
        "n_samples": int(len(sub)),
        "rsq": float(rsq),
    }


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute MAE / RMSE / NRMSE on observed (mm) scale."""
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean_y = float(np.mean(y_true))
    nrmse = float(rmse / mean_y) if mean_y > 0 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae_mm": mae,
        "rmse_mm": rmse,
        "nrmse": nrmse,
        "mean_y_mm": mean_y,
    }


def _format_diagnostics_footnote(df: pd.DataFrame) -> str:
    """Build a compact 3-line diagnostic + abbreviation note.

    Line 1 — regression model + generalization-gap formula.
    Line 2 — 5-fold CV diagnostics for Cap 0 and Cap 1 on a single line.
    Line 3 — abbreviation glossary.
    """
    res0 = evaluate_savings_regression(df, cap_k=0, n_folds=5, random_state=42)
    res1 = evaluate_savings_regression(df, cap_k=1, n_folds=5, random_state=42)

    def _fmt(res):
        if res is None:
            return "insufficient data"
        m_in = res["in_sample"]["overall"]
        m_cv = res["out_of_sample"]["overall"]
        gap = m_cv["rmse_mm"] / m_in["rmse_mm"] if m_in["rmse_mm"] > 0 else float("nan")
        return (
            f"N={res['n_samples']:,}, MAE={m_cv['mae_mm']:.0f}mm, "
            f"RMSE={m_cv['rmse_mm']:.0f}mm, NRMSE={m_cv['nrmse']:.2f}, "
            f"gap={gap:.3f}"
        )

    line1 = (
        "Model: log(saved + 1) = β₀ + β_nc·log(nc) + β_ET·log(total ET) "
        "+ γ_WY        |        Generalization gap = CV-RMSE / in-sample-RMSE"
    )
    line2 = (
        f"5-fold CV — Cap 0: {_fmt(res0)};   "
        f"Cap 1: {_fmt(res1)}"
    )
    line3 = (
        "Abbreviations:  nc = number of cuttings;  ET = evapotranspiration;  "
        "WY = water year;  ac-ft/acre = acre-feet per acre;  obs = observation(s);  "
        "MAE = mean absolute error;  RMSE = root-mean-square error;  "
        "NRMSE = RMSE / mean(observed savings);  CV = cross-validation."
    )
    return line1 + "\n" + line2 + "\n" + line3


def evaluate_savings_regression(
    df: pd.DataFrame, cap_k: int,
    n_folds: int = 5, random_state: int = 42,
) -> Optional[Dict]:
    """In-sample and k-fold-CV evaluation of the savings regression.

    Refits the same `_fit_savings_regression` model on each train fold and
    predicts on the held-out fold; aggregates predictions across folds for
    a clean out-of-sample read.  Reports MAE / RMSE / NRMSE in mm, both
    overall and per WY-type.
    """
    col = f"saved_mm_cap{cap_k}"
    if col not in df.columns:
        return None

    sub = df[
        (df["n_cuttings"] > 0)
        & (df["total_et_mm"] > 50)
        & (df[col].notna())
        & (df[col] >= 0)
        & (df["wy_type"].notna())
        & (df["wy_type"] != "")
    ].copy().reset_index(drop=True)
    if len(sub) < 100:
        return None

    log_nc = np.log(sub["n_cuttings"].values.astype(float))
    log_et = np.log(sub["total_et_mm"].values.astype(float))
    y_true_mm = sub[col].values.astype(float)
    y_log = np.log(y_true_mm + 1.0)

    wy_types_seen = sorted(sub["wy_type"].unique().tolist())
    base_wt = wy_types_seen[0]
    other_wts = wy_types_seen[1:]
    wy_arr = sub["wy_type"].values

    def build_X(idx: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(idx)), log_nc[idx], log_et[idx]]
        for wt in other_wts:
            cols.append((wy_arr[idx] == wt).astype(float))
        return np.column_stack(cols)

    def metrics_per_wy(y_true: np.ndarray, y_pred: np.ndarray,
                       wy: np.ndarray) -> Dict:
        out = {"overall": _mae_rmse(y_true, y_pred)}
        for wt in wy_types_seen:
            mask = wy == wt
            if mask.sum() > 0:
                out[wt] = _mae_rmse(y_true[mask], y_pred[mask])
        return out

    # ---- In-sample fit ----
    idx_full = np.arange(len(sub))
    X_full = build_X(idx_full)
    coefs, _, _, _ = np.linalg.lstsq(X_full, y_log, rcond=None)
    y_hat_log = X_full @ coefs
    y_hat_mm = np.maximum(0.0, np.exp(y_hat_log) - 1.0)
    in_sample = metrics_per_wy(y_true_mm, y_hat_mm, wy_arr)

    # ---- k-fold CV ----
    rng = np.random.default_rng(random_state)
    shuffled = np.arange(len(sub))
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)

    cv_y_true_parts, cv_y_pred_parts, cv_wy_parts = [], [], []
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        X_train = build_X(train_idx)
        coefs_k, _, _, _ = np.linalg.lstsq(X_train, y_log[train_idx], rcond=None)
        X_test = build_X(test_idx)
        y_pred_log = X_test @ coefs_k
        y_pred_mm = np.maximum(0.0, np.exp(y_pred_log) - 1.0)
        cv_y_true_parts.append(y_true_mm[test_idx])
        cv_y_pred_parts.append(y_pred_mm)
        cv_wy_parts.append(wy_arr[test_idx])

    cv_y_true = np.concatenate(cv_y_true_parts)
    cv_y_pred = np.concatenate(cv_y_pred_parts)
    cv_wy = np.concatenate(cv_wy_parts)
    out_of_sample = metrics_per_wy(cv_y_true, cv_y_pred, cv_wy)

    return {
        "cap_k": cap_k,
        "n_samples": int(len(sub)),
        "n_folds": int(n_folds),
        "wy_types": wy_types_seen,
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
    }


def calibrate_model(df: pd.DataFrame) -> Dict:
    """Derive late-fraction and per-cut-ET parameters from empirical data."""
    lc_wy, le_wy = {}, {}
    for wt in WY_TYPES_DISPLAY:
        sub = df[df["wy_type"] == wt]
        if len(sub) == 0:
            continue
        lc_wy[wt] = sub["n_late_cuts"].mean() / sub["n_cuttings"].mean()
        total_nz = sub["total_et_mm"].replace(0, np.nan)
        le_wy[wt] = sub["late_et_mm"].mean() / total_nz.mean()

    lc_nc, et_max_nc, per_cut_nc, mean_et_nc, late_frac_p95_nc = {}, {}, {}, {}, {}
    et_lo_nc, et_hi_nc, n_obs_nc, mean_n_late_nc = {}, {}, {}, {}
    for nc in N_CUTTINGS_RANGE:
        sub = df[df["n_cuttings"] == nc]
        sub_late = sub[sub["n_late_cuts"] > 0]
        sub_et = sub[(sub["total_et_mm"].notna()) & (sub["total_et_mm"] > 0)]
        n_obs_nc[nc] = len(sub_et)

        # Empirical mean late-cut count per n_cuttings — used to flag rows
        # where late cuts (after the cutoff date) are physically rare.
        if len(sub) >= 5:
            mean_n_late_nc[nc] = float(sub["n_late_cuts"].mean())
        elif len(sub) > 0:
            mean_n_late_nc[nc] = float(sub["n_late_cuts"].mean())
        else:
            # No data → use modeled late_frac × nc as a fallback
            mean_n_late_nc[nc] = nc * lc_wy.get("Critical", 0.33)
        if len(sub) >= 5:
            lc_nc[nc] = sub["n_late_cuts"].mean() / nc
            et_max_nc[nc] = float(sub["total_et_mm"].quantile(0.99) * 1.2)
        elif len(sub) > 0:
            lc_nc[nc] = sub["n_late_cuts"].mean() / nc
            et_max_nc[nc] = float(sub["total_et_mm"].max() * 1.3)
        else:
            lc_nc[nc] = lc_wy.get("Critical", 0.33)
            et_max_nc[nc] = nc * 250.0
        if len(sub_late) >= 3:
            per_cut_nc[nc] = float(
                (sub_late["late_et_mm"] / sub_late["n_late_cuts"]).mean()
            )
        else:
            per_cut_nc[nc] = 170.0
        mean_et_nc[nc] = float(sub["total_et_mm"].mean()) if len(sub) > 0 else nc * 130.0

        # Empirical 95th-percentile ceiling for late_et / total_et per
        # n_cuttings — replaces the previously hard-coded 0.55 cap with a
        # data-derived bound on modeled savings.
        sub_valid = sub[(sub["total_et_mm"] > 50) & (sub["late_et_mm"] >= 0)]
        if len(sub_valid) >= 5:
            ratio = (sub_valid["late_et_mm"] / sub_valid["total_et_mm"]).clip(0, 1)
            late_frac_p95_nc[nc] = float(np.nanpercentile(ratio, 95))
        else:
            late_frac_p95_nc[nc] = 0.6  # fallback when too few observations

        # Data-driven feasibility envelope on (n_cuttings × total_et).
        # ≥5 obs:   [observed_5th × 0.85, observed_95th × 1.15]  (modest buffer)
        # 1–4 obs:  [observed_min × 0.7,  observed_max × 1.3]    (looser buffer)
        # 0 obs:    [nc × 80, nc × 220]  (physics fallback: per-cut ET ∈ [80, 220] mm)
        if n_obs_nc[nc] >= 5:
            et_lo_nc[nc] = float(sub_et["total_et_mm"].quantile(0.05) * 0.85)
            et_hi_nc[nc] = float(sub_et["total_et_mm"].quantile(0.95) * 1.15)
        elif n_obs_nc[nc] > 0:
            et_lo_nc[nc] = float(sub_et["total_et_mm"].min() * 0.7)
            et_hi_nc[nc] = float(sub_et["total_et_mm"].max() * 1.3)
        else:
            et_lo_nc[nc] = nc * 80.0
            et_hi_nc[nc] = nc * 220.0

    cal: Dict = {
        "late_count_frac_by_wy": lc_wy,
        "late_et_frac_by_wy": le_wy,
        "late_count_frac_by_ncuts": lc_nc,
        "et_max_by_ncuts": et_max_nc,
        "per_late_cut_et_by_ncuts": per_cut_nc,
        "mean_et_by_ncuts": mean_et_nc,
        "late_et_frac_p95_by_ncuts": late_frac_p95_nc,
        "et_lo_by_ncuts": et_lo_nc,
        "et_hi_by_ncuts": et_hi_nc,
        "n_obs_by_ncuts": n_obs_nc,
        "mean_n_late_by_ncuts": mean_n_late_nc,
    }

    # Extension 2: empirical β for the per-cut-ET power scaling, fit from
    # observed parcel-years (replaces the hard-coded sqrt of β = 0.5).
    cal["et_scaling_beta"] = _fit_et_scaling_beta(df, cal)

    # Extension 1: log-linear regression on (n_cuttings, total_et, wy_type)
    # for direct savings prediction.  One model per cap strategy.
    # Replaces the marginal-product `late_frac` averaging.
    cal["savings_reg_cap0"] = _fit_savings_regression(df, cap_k=0)
    cal["savings_reg_cap1"] = _fit_savings_regression(df, cap_k=1)

    return cal


def _regression_predict_savings(
    cal: Dict, n_cuttings: float, total_et: float,
    wy_type: str, cap_k: int,
) -> Optional[float]:
    """Direct regression prediction for total Jul-1-baseline savings (mm).

    Returns the regression's saved_mm prediction for a (n_cuttings, total_et,
    wy_type, cap_k) point.  None if the regression for this cap was not fit.
    """
    reg = cal.get(f"savings_reg_cap{cap_k}")
    if reg is None or n_cuttings <= 0 or total_et <= 0:
        return None
    log_pred = (reg["intercept"]
                + reg["beta_nc"] * np.log(float(n_cuttings))
                + reg["beta_et"] * np.log(float(total_et)))
    if wy_type != reg["base_wt"]:
        log_pred += reg["wy_offsets"].get(wy_type, 0.0)
    return max(0.0, float(np.exp(log_pred) - 1.0))


def _model_savings_mm(
    n_cuttings: int, et_center: float, wy_type: str, cap_k: int, cal: Dict,
) -> float:
    """Predict savings (mm) for a (n_cuttings, ET, WY-type, cap) cell.

    Two-tier prediction:

    1. **Direct log-linear regression** (preferred when the fit succeeded):
        log(saved + 1) = β0
                       + β_nc · log(n_cuttings)
                       + β_et · log(total_et)
                       + γ_{wy_type}
       Replaces the marginal-product assumption that pooled the
       WY-type and n_cuttings late-fractions independently.

    2. **Marginal-product fallback** (only used when the regression isn't
       available — e.g. a cap_k whose column wasn't in the input data):
       same multiplicative model as before, but the ET power exponent is
       the empirically-fit β instead of a hard-coded sqrt.

    Both branches end with the same data-derived savings ceiling
    (95th-percentile of late_et / total_et per n_cuttings).
    """
    p95_frac = cal.get("late_et_frac_p95_by_ncuts", {}).get(n_cuttings, 0.6)
    et_ceiling = et_center * p95_frac

    reg = cal.get(f"savings_reg_cap{cap_k}")
    if reg is not None and n_cuttings > 0 and et_center > 0:
        log_pred = (reg["intercept"]
                    + reg["beta_nc"] * np.log(float(n_cuttings))
                    + reg["beta_et"] * np.log(float(et_center)))
        if wy_type != reg["base_wt"]:
            log_pred += reg["wy_offsets"].get(wy_type, 0.0)
        savings = max(0.0, float(np.exp(log_pred) - 1.0))
        return min(savings, et_ceiling)

    # Fallback: marginal-product model with empirically-fit β.
    lf_wy = cal["late_count_frac_by_wy"].get(wy_type, 0.33)
    lf_nc = cal["late_count_frac_by_ncuts"].get(n_cuttings, 0.33)
    late_frac = (lf_wy + lf_nc) / 2.0

    n_late = max(0, round(n_cuttings * late_frac))
    if n_late == 0:
        return 0.0

    base_per_cut = cal["per_late_cut_et_by_ncuts"].get(n_cuttings, 170.0)
    ref_et = cal["mean_et_by_ncuts"].get(n_cuttings, 900.0)
    beta = cal.get("et_scaling_beta", 0.5)
    scale = (et_center / max(ref_et, 100.0)) ** beta
    per_cut = base_per_cut * float(np.clip(scale, 0.4, 1.6))

    n_remove = n_late if cap_k == 0 else max(0, n_late - cap_k)
    raw_savings = n_remove * per_cut
    return min(raw_savings, et_ceiling)


# ---------------------------------------------------------------------------
# Grid — ET x cuttings
# ---------------------------------------------------------------------------
def build_sensitivity_grid(
    df: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    min_count: int = MIN_EMPIRICAL_COUNT,
) -> pd.DataFrame:
    cal = calibrate_model(df)
    df = df.copy()
    df["et_bin_idx"] = pd.cut(
        df["total_et_mm"], bins=ET_BIN_EDGES, labels=False, right=True,
    )

    rows = []
    for cap_k in cap_values:
        col = f"saved_mm_cap{cap_k}"
        if col not in df.columns:
            continue
        for wt in WY_TYPES_DISPLAY:
            for nc in N_CUTTINGS_RANGE:
                for bi in range(len(ET_BIN_LABELS)):
                    sub = df[
                        (df["wy_type"] == wt)
                        & (df["n_cuttings"] == nc)
                        & (df["et_bin_idx"] == bi)
                    ]
                    count = len(sub)
                    _lf = (cal["late_count_frac_by_wy"].get(wt, 0.33)
                           + cal["late_count_frac_by_ncuts"].get(nc, 0.33)) / 2

                    # Two-axis data-driven feasibility envelope:
                    #  (1) ET range: cells outside the observed
                    #      (n_cuttings × total_ET) envelope are gray.
                    #  (2) Late-cut feasibility: rows where the rounded
                    #      empirical mean late-cut count is < 2 are gray
                    #      because savings analysis is meaningless when
                    #      late cuts (after Jul 1) are essentially absent
                    #      — see empirical data: nc=1→2, nc=2→0, nc=3→1,
                    #      nc=4→1 rounded means.
                    et_center = ET_BIN_CENTERS[bi]
                    et_lo = cal["et_lo_by_ncuts"].get(nc, nc * 80.0)
                    et_hi = cal["et_hi_by_ncuts"].get(nc, nc * 220.0)
                    in_et_envelope = et_lo <= et_center <= et_hi

                    mean_late_emp = cal.get(
                        "mean_n_late_by_ncuts", {}
                    ).get(nc, nc * 0.33)
                    has_meaningful_late = round(mean_late_emp) >= 2

                    is_feasible = in_et_envelope and has_meaningful_late

                    if count >= min_count and has_meaningful_late:
                        # Observed cells render only if late cuts are
                        # physically meaningful for this n_cuttings.
                        mean_mm = float(sub[col].mean())
                        mean_n_late = round(sub["n_late_cuts"].mean())
                        is_model = False
                    elif is_feasible:
                        mean_mm = _model_savings_mm(
                            nc, et_center, wt, cap_k, cal,
                        )
                        mean_n_late = max(0, round(nc * _lf))
                        is_model = True
                    else:
                        # Outside one or both feasibility checks — gray cell.
                        mean_mm = np.nan
                        mean_n_late = np.nan
                        is_model = True

                    # Cell-type classification (drives the visual encoding):
                    if not np.isfinite(mean_mm):
                        cell_type = "infeasible"
                    elif count >= 5:
                        cell_type = "observed"
                    elif count > 0:
                        cell_type = "sparse"
                    else:
                        cell_type = "extrapolated"

                    rows.append({
                        "cap_k": cap_k, "wy_type": wt, "n_cuttings": nc,
                        "et_bin_idx": bi, "et_bin_label": ET_BIN_LABELS[bi],
                        "et_bin_center": ET_BIN_CENTERS[bi],
                        "mean_saved_mm": mean_mm,
                        "mean_saved_acft": float(mm_to_acft_per_acre(mean_mm))
                        if np.isfinite(mean_mm) else np.nan,
                        "mean_n_late": mean_n_late,
                        "count": count, "is_model": is_model,
                        "cell_type": cell_type,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grid — Cutoff date x County (with savings)
# ---------------------------------------------------------------------------
def build_cutoff_county_grid(
    df_base: pd.DataFrame,
    cal: Optional[Dict] = None,
    min_count: int = MIN_EMPIRICAL_COUNT,
) -> pd.DataFrame:
    """Build grid: county x cutoff_date x wy_type → n_late, savings_cap0/1.

    Hybrid approach:
      • **Empirical cut timing** — for each (county, WY type) we count how
        many of each parcel-year's late_cut_dates fall after each candidate
        cutoff date.  This is unchanged from the original observed-data path.
      • **Regression-based savings baseline** — total savings for the Jul-1
        baseline are predicted by the same log-linear regression used in
        the ET-cuttings sensitivity heatmap (`saved_mm = exp(β0 + β_nc·log(nc)
        + β_et·log(total_et) + γ_wy) − 1`).  Per-county/WY n_cuttings and
        total_et inputs come from observed parcel-year means.
      • **Cutoff scaling factor** — the regression baseline (Jul-1) is scaled
        to each cutoff date by:
          factor_cap0 = empirical_n_late_at_cutoff / empirical_n_late_at_jul1
          factor_cap1 = max(0, n_late_at_cutoff − 1) / max(0, n_late_jul1 − 1)
        so the cap-1 keep-one-cut asymmetry is preserved.

    A cell is sparse (NaN, gray) if the (county, WY) group has fewer than
    `min_count` parcel-years of cut-date data or the regression baseline
    is unavailable.
    """
    if cal is None:
        cal = calibrate_model(df_base)

    # -- Per-county / WY-type baseline inputs for the regression --
    baseline_inputs: Dict = {}
    for county in COUNTIES_NS:
        for wt in WY_TYPES_DISPLAY:
            sub = df_base[
                (df_base["county"] == county)
                & (df_base.get("wy_type") == wt)
            ]
            if len(sub) == 0:
                # Fallback: county-only mean (any WY)
                sub = df_base[df_base["county"] == county]
            if len(sub) > 0:
                nc_mean = float(sub["n_cuttings"].mean())
                et_med = float(sub["total_et_mm"].median())
            else:
                nc_mean = float(LITERATURE_ANNUAL_CUTS.get(county, 8))
                et_med = nc_mean * 150.0
            baseline_inputs[(county, wt)] = (nc_mean, et_med)

    # -- Pre-parse late_cut_dates and group by (county, WY) --
    groups: Dict = defaultdict(list)
    for _, row in df_base.iterrows():
        dates = _parse_date_list(str(row.get("late_cut_dates", "[]")))
        ets = _parse_float_list(str(row.get("late_cycle_et_mm_list", "[]")))
        county = row.get("county", "")
        wt = row.get("wy_type", "")
        wy = row.get("WY", 0)
        if not county or not wt or pd.isna(wy):
            continue
        if len(dates) != len(ets):
            continue
        groups[(county, wt)].append({
            "dates": dates, "ets": ets, "wy": int(wy),
        })

    # -- Pre-compute Jul-1 reference n_late per (county, WY) --
    jul1_n_late: Dict = {}
    for key, pys in groups.items():
        if len(pys) < min_count:
            continue
        n_lates = []
        for p in pys:
            jul1 = pd.Timestamp(year=p["wy"], month=7, day=1)
            n_lates.append(sum(1 for d in p["dates"] if d >= jul1))
        jul1_n_late[key] = float(np.mean(n_lates)) if n_lates else 0.0

    grid_rows = []
    for ci, (co_m, co_d) in enumerate(CUTOFF_DATES):
        for county_idx, county in enumerate(COUNTIES_NS):
            for wt in WY_TYPES_DISPLAY:
                key = (county, wt)
                pys = groups.get(key, [])

                # Regression baseline (Jul-1) for this county / WY
                nc_in, et_in = baseline_inputs[key]
                base_cap0 = _regression_predict_savings(
                    cal, nc_in, et_in, wt, cap_k=0,
                )
                base_cap1 = _regression_predict_savings(
                    cal, nc_in, et_in, wt, cap_k=1,
                )

                if (
                    len(pys) < min_count
                    or base_cap0 is None or base_cap1 is None
                    or key not in jul1_n_late
                ):
                    grid_rows.append({
                        "cutoff_idx": ci, "cutoff_label": CUTOFF_LABELS[ci],
                        "county": county, "county_idx": county_idx,
                        "wy_type": wt,
                        "mean_n_late": np.nan,
                        "mean_savings_cap0_acft": np.nan,
                        "mean_savings_cap1_acft": np.nan,
                        "is_sparse": True,
                    })
                    continue

                # Empirical n_late at this cutoff
                n_lates_at_cutoff = []
                for p in pys:
                    cutoff = pd.Timestamp(
                        year=p["wy"], month=co_m, day=co_d,
                    )
                    n_lates_at_cutoff.append(
                        sum(1 for d in p["dates"] if d >= cutoff)
                    )
                emp_n_late = float(np.mean(n_lates_at_cutoff))
                ref_jul1 = jul1_n_late[key]

                # Cutoff scaling factors (preserve cap-1 asymmetry)
                factor_cap0 = emp_n_late / max(ref_jul1, 0.01)
                if ref_jul1 > 1.0:
                    factor_cap1 = max(0.0, emp_n_late - 1.0) / (ref_jul1 - 1.0)
                else:
                    factor_cap1 = 0.0

                sav0 = base_cap0 * factor_cap0
                sav1 = base_cap1 * factor_cap1

                grid_rows.append({
                    "cutoff_idx": ci, "cutoff_label": CUTOFF_LABELS[ci],
                    "county": county, "county_idx": county_idx,
                    "wy_type": wt,
                    "mean_n_late": int(round(emp_n_late)),
                    "mean_savings_cap0_acft": float(
                        mm_to_acft_per_acre(sav0)
                    ),
                    "mean_savings_cap1_acft": float(
                        mm_to_acft_per_acre(sav1)
                    ),
                    "is_sparse": False,
                })

    return pd.DataFrame(grid_rows)


# ---------------------------------------------------------------------------
# Generic heatmap drawer
# ---------------------------------------------------------------------------
def _draw_heatmap_panel(
    ax, data, is_model, vmin, vmax,
    y_tick_labels, x_tick_labels,
    cmap=None,
    show_ylabel=True, show_xlabel=True,
    ylabel="", xlabel="",
    annotate=True, aux_data=None,
    fmt=".2f", x_rotation=45,
    x_fontsize=7, y_fontsize=7, cell_fontsize=6.0,
    cell_types=None,
    text_stroke=True,
):
    if cmap is None:
        cmap = CMAP_BLUE_GREEN
    n_rows, n_cols = data.shape

    # Make NaN (infeasible) cells render as a clean gray that's clearly
    # distinguishable from any of the cmap's normal values.
    cmap_local = cmap.copy() if hasattr(cmap, "copy") else cmap
    cmap_local.set_bad("#E8E8E8")

    ax.set_facecolor("#E8E8E8")
    im = ax.imshow(
        data, cmap=cmap_local, aspect="auto",
        vmin=vmin, vmax=vmax, origin="lower",
        interpolation="nearest",
    )
    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.4)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.4)

    has_aux = aux_data is not None
    # Single-color text scheme.  When `text_stroke=True` (default), a thin
    # dark stroke gives legibility against any background (used by the
    # cutoff figures).  When `text_stroke=False`, the text color flips from
    # pure white on darker cells to a light grey on the lightest cmap
    # region so the number stays readable without an outline.
    white_stroke = (
        [patheffects.withStroke(linewidth=1.4, foreground="#222222")]
        if text_stroke else []
    )
    aux_stroke = (
        [patheffects.withStroke(linewidth=1.0, foreground="#222222")]
        if text_stroke else []
    )
    span = max(vmax - vmin, 1e-9)

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                if not np.isfinite(v):
                    continue

                # Style: italic when displayed value is from the model,
                # bold roman when directly observed.  This is independent
                # of the underlying observation count anchoring.
                if is_model[i, j]:
                    sty, weight = "italic", "normal"
                else:
                    sty, weight = "normal", "bold"

                # Text color: white on darker cells; light-grey "off-white"
                # tone on the very light end of the cmap so numbers remain
                # readable without a path stroke.  When `text_stroke=True`
                # the dark stroke handles legibility everywhere → white.
                if text_stroke:
                    text_color = "white"
                else:
                    norm_v = (v - vmin) / span
                    text_color = "#A8A8A8" if norm_v < 0.30 else "white"

                # Border: cell_types tells us about underlying observation
                # anchoring (independent of which value is displayed).
                ct = cell_types[i, j] if cell_types is not None else None
                if ct == "sparse":
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="#555555",
                        linestyle=":", linewidth=0.7, zorder=3,
                    ))
                elif ct == "extrapolated":
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="#222222",
                        linestyle="--", linewidth=0.95, zorder=3,
                    ))

                if has_aux and np.isfinite(aux_data[i, j]):
                    t1 = ax.text(
                        j, i + 0.18, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=cell_fontsize, color=text_color,
                        fontstyle=sty, fontweight=weight,
                    )
                    t1.set_path_effects(white_stroke)
                    t2 = ax.text(
                        j, i - 0.22, f"({aux_data[i, j]:.0f})",
                        ha="center", va="center",
                        fontsize=cell_fontsize - 1.2,
                        color=text_color, alpha=0.85,
                    )
                    t2.set_path_effects(aux_stroke)
                else:
                    t = ax.text(
                        j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=cell_fontsize, color=text_color,
                        fontstyle=sty, fontweight=weight,
                    )
                    t.set_path_effects(white_stroke)

    ax.set_yticks(range(n_rows))
    if show_ylabel:
        ax.set_yticklabels(y_tick_labels, fontsize=y_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
    else:
        ax.set_yticklabels([])

    ax.set_xticks(range(n_cols))
    if show_xlabel:
        ax.set_xticklabels(x_tick_labels, fontsize=x_fontsize,
                           rotation=x_rotation, ha="right")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis="both", length=2.5, width=0.5, pad=2)
    _style_ax(ax)
    return im


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------
def _et_grid_to_arrays(grid_panel):
    nr, nc = len(N_CUTTINGS_RANGE), len(ET_BIN_LABELS)
    data = np.full((nr, nc), np.nan)
    is_m = np.zeros((nr, nc), dtype=bool)
    nlate = np.full((nr, nc), np.nan)
    cell_types = np.full((nr, nc), "observed", dtype=object)
    for _, r in grid_panel.iterrows():
        ri, ci = int(r["n_cuttings"]) - 1, int(r["et_bin_idx"])
        if 0 <= ri < nr and 0 <= ci < nc:
            data[ri, ci] = r["mean_saved_acft"]
            is_m[ri, ci] = r["is_model"]
            if "mean_n_late" in r.index:
                nlate[ri, ci] = r["mean_n_late"]
            if "cell_type" in r.index:
                cell_types[ri, ci] = r["cell_type"]
    return data, is_m, nlate, cell_types


def _cutoff_grid_to_arrays(grid_panel, value_col="mean_n_late"):
    nr, nc = len(COUNTIES_NS), len(CUTOFF_DATES)
    data = np.full((nr, nc), np.nan)
    is_sp = np.zeros((nr, nc), dtype=bool)
    for _, r in grid_panel.iterrows():
        ri, ci = int(r["county_idx"]), int(r["cutoff_idx"])
        if 0 <= ri < nr and 0 <= ci < nc:
            data[ri, ci] = r[value_col]
            is_sp[ri, ci] = r["is_sparse"]
    return data, is_sp


# ---------------------------------------------------------------------------
# Figure 1: ET sensitivity (panels a–h)
# ---------------------------------------------------------------------------
def sensitivity_heatmap(
    df_savings, df_base,
    cap_values=(0, 1),
    min_count=MIN_EMPIRICAL_COUNT,
    out_dir=None,
):
    """ET x cuttings sensitivity heatmap (2 rows x 4 cols)."""
    apply_style()

    merge_cols = ["UniqueID", "WY"]
    sav_cols = merge_cols + [c for c in df_savings.columns
                             if c.startswith("saved_mm_cap")]
    df = df_base.merge(
        df_savings[sav_cols].drop_duplicates(subset=merge_cols),
        on=merge_cols, how="inner",
    )
    add_wy_type_columns(df)

    wy_types = [wt for wt in WY_TYPES_DISPLAY if wt in df["wy_type"].unique()]
    n_wy = len(wy_types)
    n_caps = len(cap_values)

    grid = build_sensitivity_grid(df, cap_values=cap_values, min_count=min_count)

    fig = plt.figure(figsize=(18, 10.8))
    gs = GridSpec(
        n_caps, n_wy + 1,
        width_ratios=[1] * n_wy + [0.035],
        wspace=0.10, hspace=0.28,
        left=0.065, right=0.97, top=0.92, bottom=0.13,
    )

    axes = np.empty((n_caps, n_wy), dtype=object)
    cap_row_labels = {
        0: "Cap 0 \u2014 remove all late cuts",
        1: "Cap 1 \u2014 keep 1 late cut",
    }
    panel_idx = 0

    # Shared color scale across BOTH cap rows so cells of equal NEB look
    # equally green regardless of whether they live in the Cap 0 or Cap 1
    # row.  This makes Cap 1 panels look paler \u2014 that's correct, Cap 1
    # genuinely saves less than Cap 0 \u2014 and makes cross-row comparison valid.
    vmin = 0.0
    vmax = max(grid["mean_saved_acft"].max(), 0.01)

    last_im = None
    for ri, cap_k in enumerate(cap_values):
        cap_grid = grid[grid["cap_k"] == cap_k]

        for ci, wt in enumerate(wy_types):
            ax = fig.add_subplot(gs[ri, ci])
            axes[ri, ci] = ax
            panel_grid = cap_grid[cap_grid["wy_type"] == wt]
            data, is_mod, nlate_arr, ct_arr = _et_grid_to_arrays(panel_grid)

            last_im = _draw_heatmap_panel(
                ax, data, is_mod, vmin, vmax,
                y_tick_labels=N_CUTTINGS_RANGE,
                x_tick_labels=[f"{c:d}" for c in ET_BIN_CENTERS],
                show_ylabel=(ci == 0),
                show_xlabel=True,
                ylabel="Number of cuttings",
                xlabel="Segment ET \u2014 bin center (mm)",
                aux_data=nlate_arr,
                x_rotation=0,
                x_fontsize=7,
                cell_types=ct_arr,
                text_stroke=False,
            )

            if ri == 0:
                color = WY_TYPE_COLORS.get(wt, "black")
                ax.set_title(wt, fontsize=11, color=color,
                             fontweight="bold", pad=8)

            if ci == 0:
                ax.annotate(
                    cap_row_labels.get(cap_k, f"Cap {cap_k}"),
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-0.35, 0.5), textcoords="axes fraction",
                    fontsize=9, ha="center", va="center", rotation=90,
                    fontstyle="italic", color="#444444",
                )

            add_panel_label(ax, chr(ord("a") + panel_idx), x=-0.06, y=1.06)
            panel_idx += 1

    # Single colorbar spanning both cap rows (shared vmax/vmin).
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Late-cut savings (ac-ft/acre)", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.5)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        "Water Savings Sensitivity \u2014 Cutoff: July 1\n"
        "Savings ac-ft/acre  (n late cuts in parentheses)  |  "
        "bold roman = observed (\u22655 obs),  italic + dotted = sparse (1\u20134 obs),  "
        "italic + dashed = extrapolated (0 obs),  gray = outside observed (n_cuttings \u00d7 ET) envelope",
        fontsize=12, y=0.985, linespacing=1.5, fontweight="bold",
    )

    fig.text(
        0.5, 0.022, _format_diagnostics_footnote(df),
        ha="center", va="bottom", fontsize=8.5, color="#222222",
        family="monospace", linespacing=1.4,
    )

    n_empirical = int((~grid["is_model"]).sum())
    n_nan = int(grid["mean_saved_acft"].isna().sum())
    summary = {
        "chart": "sensitivity_heatmap",
        "n_cells": len(grid),
        "n_empirical": n_empirical,
        "n_model_filled": int(grid["is_model"].sum()) - n_nan,
        "n_masked": n_nan,
        "cap_values": list(cap_values),
        "wy_types": wy_types,
    }
    for cap_k in cap_values:
        cg = grid[grid["cap_k"] == cap_k]
        summary[f"global_min_cap{cap_k}_acft"] = float(
            cg["mean_saved_acft"].min())
        summary[f"global_max_cap{cap_k}_acft"] = float(
            cg["mean_saved_acft"].max())

    if out_dir is not None:
        save_pub_figure(fig, "sensitivity_heatmap", out_dir)
        csv_path = Path(out_dir) / "sensitivity_grid.csv"
        grid.to_csv(csv_path, index=False)
        print(f"  Grid CSV: {csv_path}")
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# Figure 2: Cutoff-date sensitivity (panels a–l)
# ---------------------------------------------------------------------------
def cutoff_sensitivity_figure(
    df_base,
    df_savings=None,
    min_count=MIN_EMPIRICAL_COUNT,
    out_dir=None,
):
    """Cutoff-date x county sensitivity figure (3 rows x 4 cols).

    Row 0: mean n_late_cuts (rounded to integer)
    Row 1: cap 0 savings (ac-ft/acre)
    Row 2: cap 1 savings (ac-ft/acre)

    `df_savings` is required so the savings regression can fit on
    `saved_mm_cap0` / `saved_mm_cap1` columns.  If omitted, savings
    cells will be sparse (gray) since the regression baseline can't
    be computed.
    """
    apply_style()

    df_base_wt = df_base.copy()
    if df_savings is not None and "saved_mm_cap0" not in df_base_wt.columns:
        merge_cols = ["UniqueID", "WY"]
        sav_cols = merge_cols + [
            c for c in df_savings.columns if c.startswith("saved_mm_cap")
        ]
        df_base_wt = df_base_wt.merge(
            df_savings[sav_cols].drop_duplicates(subset=merge_cols),
            on=merge_cols, how="inner",
        )
    add_wy_type_columns(df_base_wt)

    wy_types = [wt for wt in WY_TYPES_DISPLAY
                if wt in df_base_wt["wy_type"].unique()]
    n_wy = len(wy_types)

    cal = calibrate_model(df_base_wt)
    cutoff_grid = build_cutoff_county_grid(
        df_base_wt, cal=cal, min_count=min_count,
    )
    if cutoff_grid.empty:
        raise ValueError("Cutoff grid is empty — check input data.")

    county_labels = [
        c.replace("San Joaquin", "S. Joaquin") for c in COUNTIES_NS
    ]

    ROW_CFG = [
        ("mean_n_late", "Mean late cuts (n)", ".0f", None),
        ("mean_savings_cap0_acft", "Cap 0 savings (ac-ft/acre)", ".2f", None),
        ("mean_savings_cap1_acft", "Cap 1 savings (ac-ft/acre)", ".2f", None),
    ]
    n_fig_rows = len(ROW_CFG)
    row_labels = [
        "Late cuts \u2014 count",
        "Cap 0 \u2014 remove all late cuts",
        "Cap 1 \u2014 keep 1 late cut",
    ]

    fig = plt.figure(figsize=(18, 13.6))
    gs = GridSpec(
        n_fig_rows, n_wy + 1,
        width_ratios=[1] * n_wy + [0.035],
        height_ratios=[1, 1, 1],
        wspace=0.10, hspace=0.30,
        left=0.065, right=0.97, top=0.93, bottom=0.11,
    )

    axes = np.empty((n_fig_rows, n_wy), dtype=object)
    panel_idx = 0

    for ri, (val_col, cbar_label, fmt, _) in enumerate(ROW_CFG):
        # Compute vmin/vmax across all WY types for this metric
        valid = cutoff_grid[val_col].dropna()
        vmin = 0.0
        vmax = max(valid.max(), 0.01) if len(valid) > 0 else 1.0

        # Row 0 (n_late) is the empirical count \u2192 bold roman.
        # Rows 1, 2 (cap0/cap1 savings) come from the regression baseline
        # \u00d7 cutoff factor, so they should render italic.
        row_is_model = (ri > 0)

        for ci, wt in enumerate(wy_types):
            ax = fig.add_subplot(gs[ri, ci])
            axes[ri, ci] = ax
            panel_g = cutoff_grid[cutoff_grid["wy_type"] == wt]
            data, is_sp = _cutoff_grid_to_arrays(panel_g, value_col=val_col)

            # Drive style by whether the displayed value is observed/modeled
            is_model_arr = np.full_like(is_sp, row_is_model, dtype=bool)

            im = _draw_heatmap_panel(
                ax, data, is_model_arr, vmin, vmax,
                y_tick_labels=county_labels,
                x_tick_labels=CUTOFF_LABELS,
                show_ylabel=(ci == 0),
                show_xlabel=(ri == n_fig_rows - 1),
                ylabel="County (N \u2192 S)" if ci == 0 else "",
                xlabel="Cutoff date",
                fmt=fmt,
                x_fontsize=7, y_fontsize=7, cell_fontsize=6.0,
                text_stroke=False,
            )

            if ri == 0:
                color = WY_TYPE_COLORS.get(wt, "black")
                ax.set_title(wt, fontsize=11, color=color,
                             fontweight="bold", pad=8)

            if ci == 0:
                ax.annotate(
                    row_labels[ri],
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-0.35, 0.5), textcoords="axes fraction",
                    fontsize=9, ha="center", va="center", rotation=90,
                    fontstyle="italic", color="#444444",
                )

            add_panel_label(ax, chr(ord("a") + panel_idx), x=-0.06, y=1.08)
            panel_idx += 1

        cbar_ax = fig.add_subplot(gs[ri, -1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)
        cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        "Cutoff-Date Sensitivity: Late Cuts & Water Savings by County (N \u2192 S)\n"
        "Cutoff date Jul 1 \u2192 Oct 1  |  Savings = log-linear regression baseline "
        "\u00d7 empirical cutoff-scaling factor  |  "
        f"bold roman = empirical (\u2265 {min_count} obs),  gray = sparse",
        fontsize=12, y=0.985, linespacing=1.5, fontweight="bold",
    )

    fig.text(
        0.5, 0.022, _format_diagnostics_footnote(df_base_wt),
        ha="center", va="bottom", fontsize=8.5, color="#222222",
        family="monospace", linespacing=1.4,
    )

    summary = {
        "chart": "cutoff_sensitivity",
        "n_cells": len(cutoff_grid),
        "n_non_sparse": int((~cutoff_grid["is_sparse"]).sum()),
        "cutoff_dates": CUTOFF_LABELS,
        "counties_ns": COUNTIES_NS,
        "wy_types": wy_types,
    }

    if out_dir is not None:
        save_pub_figure(fig, "cutoff_sensitivity", out_dir)
        csv_path = Path(out_dir) / "cutoff_county_grid.csv"
        cutoff_grid.to_csv(csv_path, index=False)
        print(f"  Cutoff grid CSV: {csv_path}")
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# Figure 3: Model-only ET sensitivity
# ---------------------------------------------------------------------------
def sensitivity_heatmap_modeled(
    df_savings, df_base,
    cap_values=(0, 1),
    out_dir=None,
):
    """Model-only ET sensitivity heatmap — all cells use calibrated model."""
    apply_style()

    merge_cols = ["UniqueID", "WY"]
    sav_cols = merge_cols + [c for c in df_savings.columns
                             if c.startswith("saved_mm_cap")]
    df = df_base.merge(
        df_savings[sav_cols].drop_duplicates(subset=merge_cols),
        on=merge_cols, how="inner",
    )
    add_wy_type_columns(df)

    wy_types = [wt for wt in WY_TYPES_DISPLAY if wt in df["wy_type"].unique()]
    n_wy = len(wy_types)
    n_caps = len(cap_values)

    # Force all cell *values* to model by requiring impossible observation
    # count.  cell_type still reflects the underlying observation count so
    # the reader can see which cells had real data behind the modeled value.
    grid = build_sensitivity_grid(df, cap_values=cap_values, min_count=999999)

    fig = plt.figure(figsize=(18, 10.8))
    gs = GridSpec(
        n_caps, n_wy + 1,
        width_ratios=[1] * n_wy + [0.035],
        wspace=0.10, hspace=0.28,
        left=0.065, right=0.97, top=0.92, bottom=0.13,
    )

    axes = np.empty((n_caps, n_wy), dtype=object)
    cap_row_labels = {
        0: "Cap 0 \u2014 remove all late cuts",
        1: "Cap 1 \u2014 keep 1 late cut",
    }
    panel_idx = 0

    vmin = 0.0
    vmax = max(grid["mean_saved_acft"].max(), 0.01)

    last_im = None
    for ri, cap_k in enumerate(cap_values):
        cap_grid = grid[grid["cap_k"] == cap_k]

        for ci, wt in enumerate(wy_types):
            ax = fig.add_subplot(gs[ri, ci])
            axes[ri, ci] = ax
            panel_grid = cap_grid[cap_grid["wy_type"] == wt]
            data, is_mod, nlate_arr, ct_arr = _et_grid_to_arrays(panel_grid)

            last_im = _draw_heatmap_panel(
                ax, data, is_mod, vmin, vmax,
                y_tick_labels=N_CUTTINGS_RANGE,
                x_tick_labels=[f"{c:d}" for c in ET_BIN_CENTERS],
                show_ylabel=(ci == 0),
                show_xlabel=True,
                ylabel="Number of cuttings",
                xlabel="Segment ET \u2014 bin center (mm)",
                aux_data=nlate_arr,
                x_rotation=0,
                x_fontsize=7,
                cell_types=ct_arr,
                text_stroke=False,
            )
            if ri == 0:
                color = WY_TYPE_COLORS.get(wt, "black")
                ax.set_title(wt, fontsize=11, color=color,
                             fontweight="bold", pad=8)
            if ci == 0:
                ax.annotate(
                    cap_row_labels.get(cap_k, f"Cap {cap_k}"),
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-0.35, 0.5), textcoords="axes fraction",
                    fontsize=9, ha="center", va="center", rotation=90,
                    fontstyle="italic", color="#444444",
                )
            add_panel_label(ax, chr(ord("a") + panel_idx), x=-0.06, y=1.06)
            panel_idx += 1

    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Late-cut savings (ac-ft/acre)", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.5)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        "Water Savings Sensitivity \u2014 Modeled (Cutoff: July 1)\n"
        "All cell values from calibrated model.  Border indicates underlying "
        "observation count: none = \u22655 obs, dotted = 1\u20134 obs, dashed = 0 obs.  "
        "Gray = outside observed (n_cuttings \u00d7 ET) envelope.",
        fontsize=12, y=0.985, linespacing=1.5, fontweight="bold",
    )

    fig.text(
        0.5, 0.022, _format_diagnostics_footnote(df),
        ha="center", va="bottom", fontsize=8.5, color="#222222",
        family="monospace", linespacing=1.4,
    )

    n_nan = int(grid["mean_saved_acft"].isna().sum())
    summary = {
        "chart": "sensitivity_heatmap_modeled",
        "n_cells": len(grid),
        "n_filled": len(grid) - n_nan,
        "n_masked": n_nan,
        "cap_values": list(cap_values),
    }

    if out_dir is not None:
        save_pub_figure(fig, "sensitivity_heatmap_modeled", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# Figure 4: Model-only cutoff sensitivity
# ---------------------------------------------------------------------------
def _model_cut_dates(n_cuts, wy=2022):
    """Generate evenly-spaced cut dates within a WY (Oct 1 → Sep 30)."""
    start = pd.Timestamp(wy - 1, 10, 1)
    total_days = 365
    interval = total_days / n_cuts
    # Center cuts within each interval
    return [start + pd.Timedelta(days=interval * (i + 0.5))
            for i in range(n_cuts)]


def build_cutoff_county_grid_modeled(
    cal: Dict, df_base: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Fully modeled cutoff grid using regression baseline + literature timing.

    For each (county, WY type), the regression `_regression_predict_savings`
    gives the Jul-1 baseline savings using:
      • n_cuttings = LITERATURE_ANNUAL_CUTS[county]   (8 → 12 N → S)
      • total_et   = empirical median total ET observed for that county
                     (falls back to nc × 150 mm if df_base is not supplied)

    Each cutoff date is then scaled from the baseline by counting evenly-spaced
    literature cut dates that fall after the cutoff:
      factor_cap0 = n_cuts_after_cutoff / n_cuts_after_jul1
      factor_cap1 = max(0, n_after_cutoff − 1) / max(0, n_after_jul1 − 1)
    """
    # Per-county empirical median total ET (or fallback)
    county_total_et: Dict[str, float] = {}
    for county in COUNTIES_NS:
        if df_base is not None:
            sub = df_base[df_base["county"] == county]["total_et_mm"]
            sub = sub[(sub.notna()) & (sub > 0)]
            if len(sub) > 0:
                county_total_et[county] = float(sub.median())
                continue
        # Fallback: nc × 150 mm/cut
        nc = LITERATURE_ANNUAL_CUTS.get(county, 8)
        county_total_et[county] = float(nc * 150.0)

    grid_rows = []
    for county_idx, county in enumerate(COUNTIES_NS):
        n_cuts = LITERATURE_ANNUAL_CUTS.get(county, 8)
        total_et = county_total_et[county]
        modeled_dates = _model_cut_dates(n_cuts)

        # Reference n_cuts after Jul 1 (for scaling factor normalization)
        jul1 = pd.Timestamp(2022, 7, 1)
        n_after_jul1 = sum(1 for d in modeled_dates if d >= jul1)

        for ci, (co_m, co_d) in enumerate(CUTOFF_DATES):
            cutoff = pd.Timestamp(2022, co_m, co_d)
            n_late_at_cutoff = sum(1 for d in modeled_dates if d >= cutoff)

            factor_cap0 = (
                n_late_at_cutoff / max(n_after_jul1, 1)
            )
            if n_after_jul1 > 1:
                factor_cap1 = max(0, n_late_at_cutoff - 1) / (n_after_jul1 - 1)
            else:
                factor_cap1 = 0.0

            for wt in WY_TYPES_DISPLAY:
                base_cap0 = _regression_predict_savings(
                    cal, n_cuts, total_et, wt, cap_k=0,
                ) or 0.0
                base_cap1 = _regression_predict_savings(
                    cal, n_cuts, total_et, wt, cap_k=1,
                ) or 0.0

                sav0 = base_cap0 * factor_cap0
                sav1 = base_cap1 * factor_cap1

                grid_rows.append({
                    "cutoff_idx": ci, "cutoff_label": CUTOFF_LABELS[ci],
                    "county": county, "county_idx": county_idx,
                    "wy_type": wt,
                    "mean_n_late": int(n_late_at_cutoff),
                    "mean_savings_cap0_acft": float(
                        mm_to_acft_per_acre(sav0)),
                    "mean_savings_cap1_acft": float(
                        mm_to_acft_per_acre(sav1)),
                    "is_sparse": False,
                })

    return pd.DataFrame(grid_rows)


def cutoff_sensitivity_modeled(
    df_base,
    df_savings=None,
    out_dir=None,
):
    """Model-only cutoff sensitivity figure (3 rows x 4 cols).

    Uses LITERATURE_ANNUAL_CUTS per county and the calibrated savings
    regression for the Jul-1 baseline, then scales each cutoff date by
    the count of evenly-spaced literature cuts that fall after it.

    `df_savings` is required so the savings regression can fit on the
    `saved_mm_cap*` columns.  If omitted, the regression coefficients
    will be missing and predicted savings will fall through to zero.
    """
    apply_style()

    df_cal = df_base.copy()
    if df_savings is not None and "saved_mm_cap0" not in df_cal.columns:
        merge_cols = ["UniqueID", "WY"]
        sav_cols = merge_cols + [
            c for c in df_savings.columns if c.startswith("saved_mm_cap")
        ]
        df_cal = df_cal.merge(
            df_savings[sav_cols].drop_duplicates(subset=merge_cols),
            on=merge_cols, how="inner",
        )
    add_wy_type_columns(df_cal)
    cal = calibrate_model(df_cal)

    wy_types = [wt for wt in WY_TYPES_DISPLAY
                if wt in df_cal["wy_type"].unique()]
    n_wy = len(wy_types)

    cutoff_grid = build_cutoff_county_grid_modeled(cal, df_base=df_cal)

    county_labels = [
        c.replace("San Joaquin", "S. Joaquin") for c in COUNTIES_NS
    ]

    ROW_CFG = [
        ("mean_n_late", "Mean late cuts (n)", ".0f"),
        ("mean_savings_cap0_acft", "Cap 0 savings (ac-ft/acre)", ".2f"),
        ("mean_savings_cap1_acft", "Cap 1 savings (ac-ft/acre)", ".2f"),
    ]
    n_fig_rows = len(ROW_CFG)
    row_labels = [
        "Late cuts \u2014 count",
        "Cap 0 \u2014 remove all late cuts",
        "Cap 1 \u2014 keep 1 late cut",
    ]

    fig = plt.figure(figsize=(18, 13.6))
    gs = GridSpec(
        n_fig_rows, n_wy + 1,
        width_ratios=[1] * n_wy + [0.035],
        height_ratios=[1, 1, 1],
        wspace=0.10, hspace=0.30,
        left=0.065, right=0.97, top=0.93, bottom=0.11,
    )

    axes = np.empty((n_fig_rows, n_wy), dtype=object)
    panel_idx = 0

    for ri, (val_col, cbar_label, fmt) in enumerate(ROW_CFG):
        valid = cutoff_grid[val_col].dropna()
        vmin = 0.0
        vmax = max(valid.max(), 0.01) if len(valid) > 0 else 1.0

        for ci, wt in enumerate(wy_types):
            ax = fig.add_subplot(gs[ri, ci])
            axes[ri, ci] = ax
            panel_g = cutoff_grid[cutoff_grid["wy_type"] == wt]
            data, is_sp = _cutoff_grid_to_arrays(panel_g, value_col=val_col)

            # All cells in the modeled cutoff figure are model-derived
            # (literature counts \u00d7 regression baseline) \u2192 italic, no border.
            is_model_arr = np.ones_like(is_sp, dtype=bool)

            im = _draw_heatmap_panel(
                ax, data, is_model_arr, vmin, vmax,
                y_tick_labels=county_labels,
                x_tick_labels=CUTOFF_LABELS,
                show_ylabel=(ci == 0),
                show_xlabel=(ri == n_fig_rows - 1),
                ylabel="County (N \u2192 S)" if ci == 0 else "",
                xlabel="Cutoff date",
                fmt=fmt,
                x_fontsize=7, y_fontsize=7, cell_fontsize=6.0,
                text_stroke=False,
            )

            if ri == 0:
                color = WY_TYPE_COLORS.get(wt, "black")
                ax.set_title(wt, fontsize=11, color=color,
                             fontweight="bold", pad=8)
            if ci == 0:
                ax.annotate(
                    row_labels[ri],
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-0.35, 0.5), textcoords="axes fraction",
                    fontsize=9, ha="center", va="center", rotation=90,
                    fontstyle="italic", color="#444444",
                )
            add_panel_label(ax, chr(ord("a") + panel_idx), x=-0.06, y=1.08)
            panel_idx += 1

        cbar_ax = fig.add_subplot(gs[ri, -1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.5)
        cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        "Cutoff-Date Sensitivity \u2014 Modeled\n"
        "Savings = log-linear regression baseline (LITERATURE_ANNUAL_CUTS, "
        "median total ET, WY type) \u00d7 literature cut-timing scaling factor.\n"
        "Cutoff date Jul 1 \u2192 Oct 1  |  italic = all values from calibrated model",
        fontsize=12, y=0.985, linespacing=1.5, fontweight="bold",
    )

    fig.text(
        0.5, 0.022, _format_diagnostics_footnote(df_cal),
        ha="center", va="bottom", fontsize=8.5, color="#222222",
        family="monospace", linespacing=1.4,
    )

    summary = {
        "chart": "cutoff_sensitivity_modeled",
        "n_cells": len(cutoff_grid),
        "literature_cuts": LITERATURE_ANNUAL_CUTS,
        "county_per_cut_et_mm": COUNTY_PER_CUT_ET_MM,
    }

    if out_dir is not None:
        save_pub_figure(fig, "cutoff_sensitivity_modeled", out_dir)
        csv_path = Path(out_dir) / "cutoff_county_grid_modeled.csv"
        cutoff_grid.to_csv(csv_path, index=False)
        print(f"  Modeled cutoff grid CSV: {csv_path}")
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    from es_analysis.data_providers.config import config

    parser = argparse.ArgumentParser(description="Sensitivity heatmaps")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name:
        from es_analysis.utils.run_output import get_run_root
        run_root = get_run_root(args.run_name)
        data_dir = Path(args.data_dir) if args.data_dir else run_root / "water_savings"
    else:
        data_dir = Path(args.data_dir) if args.data_dir else config.water_saving_out_dir

    out = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).parent.parent.parent
        / "output" / "figures" / "water_savings_sim_2"
    )

    df_sav = pd.read_csv(data_dir / "late_cut_savings_parcel_year_mm.csv")
    df_base = pd.read_csv(data_dir / "late_cut_base_parcel_year.csv")

    # Figure 1: ET sensitivity (observed + model)
    print("=== Figure 1: ET Sensitivity ===")
    _, _, s1 = sensitivity_heatmap(df_sav, df_base, out_dir=out)
    print(f"  Cells: {s1['n_cells']} (empirical: {s1['n_empirical']}, "
          f"model: {s1['n_model_filled']}, masked: {s1['n_masked']})")

    # Figure 2: Cutoff sensitivity (observed + literature-adjusted)
    print("\n=== Figure 2: Cutoff Sensitivity ===")
    _, _, s2 = cutoff_sensitivity_figure(df_base, out_dir=out)
    print(f"  Cells: {s2['n_cells']} (non-sparse: {s2['n_non_sparse']})")

    # Figure 3: ET sensitivity (model-only)
    print("\n=== Figure 3: ET Sensitivity (Modeled) ===")
    _, _, s3 = sensitivity_heatmap_modeled(df_sav, df_base, out_dir=out)
    print(f"  Cells: {s3['n_cells']} (filled: {s3['n_filled']}, "
          f"masked: {s3['n_masked']})")

    # Figure 4: Cutoff sensitivity (model-only)
    print("\n=== Figure 4: Cutoff Sensitivity (Modeled) ===")
    _, _, s4 = cutoff_sensitivity_modeled(df_base, out_dir=out)
    print(f"  Cells: {s4['n_cells']}")
