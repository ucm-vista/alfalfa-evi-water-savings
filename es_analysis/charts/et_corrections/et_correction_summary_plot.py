"""Aggregate ET correction summary plots.

Three-panel figure showing ET correction results at county and WY level:
  (a) By county: actual vs corrected ET total (ac-ft/acre) with error bars
  (b) By water year: actual vs corrected ET total (ac-ft/acre)
  (c) Correction percentage by county x WY (grouped bar or table)

Uses pre-computed ET correction CSVs from et_stats_provider.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.utils.units import mm_to_acft_per_acre
from es_analysis.utils.publication_style import (
    WONG_PALETTE,
    DOUBLE_COL_WIDTH,
    apply_style,
    save_pub_figure,
    add_panel_label,
)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)


def _find_csv(out_dir: Path, pattern: str) -> Optional[Path]:
    """Find the most recent CSV matching pattern."""
    matches = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_parcel_year(et_corr_dir: Path) -> Optional[pd.DataFrame]:
    """Load parcel-year ET correction CSV."""
    p = _find_csv(et_corr_dir, "offphase_parcel_year_*.csv")
    if p is None:
        return None
    return pd.read_csv(p)


def _load_county_wy(et_corr_dir: Path) -> Optional[pd.DataFrame]:
    """Load county-WY ET correction CSV."""
    p = _find_csv(et_corr_dir, "offphase_county_WY_annual_totals_*.csv")
    if p is None:
        return None
    return pd.read_csv(p)


def _ci_arms_acft(center_mm: float, lo_mm: float, hi_mm: float) -> Tuple[float, float]:
    """Asymmetric 90% bootstrap CI arm lengths in ac-ft/acre.

    ``center_mm`` is the mean corrected ET; ``lo_mm``/``hi_mm`` are the mean
    per-parcel CI bounds. Returns (lower_arm, upper_arm) for use as matplotlib
    ``yerr``. NaN bounds collapse to zero-length arms (no visible bar).
    """
    if any(np.isnan(v) for v in (center_mm, lo_mm, hi_mm)):
        return 0.0, 0.0
    arm_lo = float(mm_to_acft_per_acre(max(center_mm - lo_mm, 0.0)))
    arm_hi = float(mm_to_acft_per_acre(max(hi_mm - center_mm, 0.0)))
    return arm_lo, arm_hi


def et_correction_summary(
    counties: Optional[List[str]] = None,
    wy_start: int = 2019,
    wy_end: int = 2024,
    et_corr_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Generate aggregate ET correction summary plots.

    Produces a multi-panel figure:
      (a) Per-county mean actual vs corrected ET (ac-ft/acre) with correction %
      (b) Per-WY mean actual vs corrected ET (ac-ft/acre)

    Args:
        counties: Counties to include (defaults to all in data).
        wy_start: First water year.
        wy_end: Last water year.
        et_corr_dir: Directory with ET correction CSVs.
        out_dir: Output directory for figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    apply_style()

    if et_corr_dir is None:
        et_corr_dir = Path(__file__).parent.parent.parent / "output" / "et_correction"
    if out_dir is None:
        out_dir = Path(__file__).parent.parent.parent / "output" / "figures" / "et_corrections"

    # Try parcel-year data first (most flexible)
    df_py = _load_parcel_year(et_corr_dir)
    df_cw = _load_county_wy(et_corr_dir)

    if df_py is None and df_cw is None:
        raise FileNotFoundError(
            f"No ET correction CSVs found in {et_corr_dir}.\n"
            "Run ET stats first: python alfalfa_et_gdd5_pipeline.py --action et_stats"
        )

    # Determine available counties
    if df_cw is not None:
        all_counties = df_cw["county"].unique().tolist()
    elif df_py is not None:
        all_counties = df_py["county"].unique().tolist()
    else:
        all_counties = []

    if counties is not None:
        plot_counties = [c for c in COUNTY_ORDER if c in counties and c in all_counties]
    else:
        plot_counties = [c for c in COUNTY_ORDER if c in all_counties]

    n_panels = 2
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(DOUBLE_COL_WIDTH * 1.4, DOUBLE_COL_WIDTH * 0.5),
    )

    color_actual = WONG_PALETTE[1]   # orange
    color_corr = WONG_PALETTE[3]     # bluish green
    bar_width = 0.35

    # ---- Panel (a): Per-county mean per-parcel ET ----
    ax_county = axes[0]

    if df_py is not None and len(plot_counties) > 0:
        # Compute mean per-parcel annual ET by county
        x = np.arange(len(plot_counties))
        means_actual = []
        means_corr = []
        pct_reductions = []

        et_open_col = [c for c in df_py.columns if "ET_open" in c and "annual" in c.lower()]
        et_corr_col_name = [c for c in df_py.columns if "ET_corr" in c and "annual" in c.lower() and "ci" not in c.lower()]

        # Fallback: look for common column names
        if not et_open_col:
            et_open_col = [c for c in df_py.columns if c.startswith("ET_open_annual")]
        if not et_corr_col_name:
            et_corr_col_name = [c for c in df_py.columns if c.startswith("ET_corr_annual") and "ci" not in c]

        ci_lo_col = "ET_corr_ci_low_annual_mm"
        ci_hi_col = "ET_corr_ci_high_annual_mm"
        has_ci = ci_lo_col in df_py.columns and ci_hi_col in df_py.columns

        if et_open_col and et_corr_col_name:
            oc = et_open_col[0]
            cc = et_corr_col_name[0]
            err_lo: List[float] = []
            err_hi: List[float] = []
            for county in plot_counties:
                sub = df_py[df_py["county"] == county]
                m_a = float(sub[oc].mean())
                m_c = float(sub[cc].mean())
                means_actual.append(float(mm_to_acft_per_acre(m_a)))
                means_corr.append(float(mm_to_acft_per_acre(m_c)))
                pct = 100 * (m_a - m_c) / m_a if m_a > 0 else 0
                pct_reductions.append(pct)
                if has_ci:
                    a_lo, a_hi = _ci_arms_acft(
                        m_c, float(sub[ci_lo_col].mean()), float(sub[ci_hi_col].mean()))
                    err_lo.append(a_lo)
                    err_hi.append(a_hi)

            ax_county.bar(x - bar_width / 2, means_actual, bar_width,
                          color=color_actual, edgecolor="black", linewidth=0.5,
                          label="Actual ET")
            ax_county.bar(x + bar_width / 2, means_corr, bar_width,
                          color=color_corr, edgecolor="black", linewidth=0.5,
                          label="Corrected ET",
                          yerr=([err_lo, err_hi] if has_ci else None), capsize=2,
                          error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="black"))

            # Annotate % reduction (above the error-bar cap)
            for i, pct in enumerate(pct_reductions):
                cap = err_hi[i] if has_ci else 0.0
                y_top = max(means_actual[i], means_corr[i] + cap)
                ax_county.text(x[i], y_top + 0.02, f"-{pct:.1f}%",
                               ha="center", va="bottom", fontsize=7, color="red")

            ax_county.set_xticks(x)
            ax_county.set_xticklabels(plot_counties, rotation=35, ha="right", fontsize=8)
    elif df_cw is not None:
        # Fallback: use county-WY totals, compute mean per-parcel
        x = np.arange(len(plot_counties))
        means_actual = []
        means_corr = []
        pct_reductions = []

        ci_lo_col = "ET_corr_ci_low_total_mm"
        ci_hi_col = "ET_corr_ci_high_total_mm"
        has_ci = ci_lo_col in df_cw.columns and ci_hi_col in df_cw.columns
        err_lo = []
        err_hi = []

        for county in plot_counties:
            sub = df_cw[df_cw["county"] == county]
            total_open = sub["ET_open_total_mm"].sum()
            total_corr = sub["ET_corr_total_mm"].sum()
            n_py = sub["parcel_years"].sum()
            mean_a = total_open / n_py if n_py > 0 else 0
            mean_c = total_corr / n_py if n_py > 0 else 0
            means_actual.append(float(mm_to_acft_per_acre(mean_a)))
            means_corr.append(float(mm_to_acft_per_acre(mean_c)))
            pct = 100 * (mean_a - mean_c) / mean_a if mean_a > 0 else 0
            pct_reductions.append(pct)
            if has_ci:
                if n_py > 0:
                    a_lo, a_hi = _ci_arms_acft(
                        mean_c, sub[ci_lo_col].sum() / n_py, sub[ci_hi_col].sum() / n_py)
                else:
                    a_lo, a_hi = 0.0, 0.0
                err_lo.append(a_lo)
                err_hi.append(a_hi)

        ax_county.bar(x - bar_width / 2, means_actual, bar_width,
                      color=color_actual, edgecolor="black", linewidth=0.5,
                      label="Actual ET")
        ax_county.bar(x + bar_width / 2, means_corr, bar_width,
                      color=color_corr, edgecolor="black", linewidth=0.5,
                      label="Corrected ET",
                      yerr=([err_lo, err_hi] if has_ci else None), capsize=2,
                      error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="black"))

        for i, pct in enumerate(pct_reductions):
            cap = err_hi[i] if has_ci else 0.0
            y_top = max(means_actual[i], means_corr[i] + cap)
            ax_county.text(x[i], y_top + 0.02, f"-{pct:.1f}%",
                           ha="center", va="bottom", fontsize=7, color="red")

        ax_county.set_xticks(x)
        ax_county.set_xticklabels(plot_counties, rotation=35, ha="right", fontsize=8)

    ax_county.set_ylabel("Mean annual ET (ac-ft/acre)", fontsize=9)
    ax_county.set_title("Mean annual ET by county (all WYs pooled)", fontsize=10, pad=18)
    ax_county.set_ylim(0, 5)
    ax_county.tick_params(axis="y", labelsize=8)
    ax_county.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.002), frameon=False, borderaxespad=0.0, columnspacing=1.0, handlelength=1.7)
    _style_ax(ax_county)
    add_panel_label(ax_county, "a")

    # ---- Panel (b): Per-WY mean ET ----
    ax_wy = axes[1]

    wys = list(range(wy_start, wy_end + 1))
    x_wy = np.arange(len(wys))
    wy_actual = []
    wy_corr = []
    wy_pct = []
    wy_err_lo: List[float] = []
    wy_err_hi: List[float] = []
    wy_has_ci = False

    if df_cw is not None:
        cw_lo_col = "ET_corr_ci_low_total_mm"
        cw_hi_col = "ET_corr_ci_high_total_mm"
        wy_has_ci = cw_lo_col in df_cw.columns and cw_hi_col in df_cw.columns
        for wy in wys:
            sub = df_cw[df_cw["WY"] == wy]
            total_open = sub["ET_open_total_mm"].sum()
            total_corr = sub["ET_corr_total_mm"].sum()
            n_py = sub["parcel_years"].sum()
            mean_a = total_open / n_py if n_py > 0 else 0
            mean_c = total_corr / n_py if n_py > 0 else 0
            wy_actual.append(float(mm_to_acft_per_acre(mean_a)))
            wy_corr.append(float(mm_to_acft_per_acre(mean_c)))
            pct = 100 * (mean_a - mean_c) / mean_a if mean_a > 0 else 0
            wy_pct.append(pct)
            if wy_has_ci:
                if n_py > 0:
                    a_lo, a_hi = _ci_arms_acft(
                        mean_c, sub[cw_lo_col].sum() / n_py, sub[cw_hi_col].sum() / n_py)
                else:
                    a_lo, a_hi = 0.0, 0.0
                wy_err_lo.append(a_lo)
                wy_err_hi.append(a_hi)
    elif df_py is not None and et_open_col and et_corr_col_name:
        oc = et_open_col[0]
        cc = et_corr_col_name[0]
        py_lo_col = "ET_corr_ci_low_annual_mm"
        py_hi_col = "ET_corr_ci_high_annual_mm"
        wy_has_ci = py_lo_col in df_py.columns and py_hi_col in df_py.columns
        for wy in wys:
            sub = df_py[df_py["WY"] == wy]
            m_a = float(sub[oc].mean()) if len(sub) > 0 else 0
            m_c = float(sub[cc].mean()) if len(sub) > 0 else 0
            wy_actual.append(float(mm_to_acft_per_acre(m_a)))
            wy_corr.append(float(mm_to_acft_per_acre(m_c)))
            pct = 100 * (m_a - m_c) / m_a if m_a > 0 else 0
            wy_pct.append(pct)
            if wy_has_ci:
                if len(sub) > 0:
                    a_lo, a_hi = _ci_arms_acft(
                        m_c, float(sub[py_lo_col].mean()), float(sub[py_hi_col].mean()))
                else:
                    a_lo, a_hi = 0.0, 0.0
                wy_err_lo.append(a_lo)
                wy_err_hi.append(a_hi)

    if wy_actual:
        ax_wy.bar(x_wy - bar_width / 2, wy_actual, bar_width,
                   color=color_actual, edgecolor="black", linewidth=0.5,
                   label="Actual ET")
        ax_wy.bar(x_wy + bar_width / 2, wy_corr, bar_width,
                   color=color_corr, edgecolor="black", linewidth=0.5,
                   label="Corrected ET",
                   yerr=([wy_err_lo, wy_err_hi] if wy_has_ci else None), capsize=2,
                   error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="black"))

        for i, pct in enumerate(wy_pct):
            cap = wy_err_hi[i] if wy_has_ci else 0.0
            y_top = max(wy_actual[i], wy_corr[i] + cap)
            ax_wy.text(x_wy[i], y_top + 0.02, f"-{pct:.1f}%",
                       ha="center", va="bottom", fontsize=7, color="red")

        ax_wy.set_xticks(x_wy)
        ax_wy.set_xticklabels([f"WY{wy}" for wy in wys], rotation=35, ha="right", fontsize=8)

    ax_wy.set_ylabel("Mean annual ET (ac-ft/acre)", fontsize=9)
    ax_wy.set_title("Mean annual ET by water year (all counties pooled)", fontsize=10, pad=18)
    ax_wy.set_ylim(0, 5)
    ax_wy.tick_params(axis="y", labelsize=8)
    ax_wy.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.002), frameon=False, borderaxespad=0.0, columnspacing=1.0, handlelength=1.7)
    _style_ax(ax_wy)
    add_panel_label(ax_wy, "b")

    fig.suptitle(
        f"ET Correction Summary: Actual vs. Corrected Annual ET "
        f"({len(plot_counties)} Counties, WY{wy_start}\u2013{wy_end})",
        fontsize=11, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Summary
    summary = {
        "n_counties": len(plot_counties),
        "counties": plot_counties,
        "wy_range": f"{wy_start}-{wy_end}",
    }
    if pct_reductions:
        summary["mean_pct_reduction_county"] = float(np.mean(pct_reductions))
    if wy_pct:
        summary["mean_pct_reduction_wy"] = float(np.mean(wy_pct))

    if out_dir is not None:
        save_pub_figure(fig, "et_correction_summary", out_dir)
        summary["out_dir"] = str(out_dir)

    print(f"  ET correction summary: {len(plot_counties)} counties, WY{wy_start}-{wy_end}")
    if pct_reductions:
        print(f"  Mean reduction by county: {[f'{p:.1f}%' for p in pct_reductions]}")
    if wy_pct:
        print(f"  Mean reduction by WY: {[f'{p:.1f}%' for p in wy_pct]}")

    return fig, axes, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    fig, axes, summary = et_correction_summary()
    print(f"Summary: {summary}")
