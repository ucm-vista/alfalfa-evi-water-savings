"""Cuttings analysis plots using matched_minima_iso segments.

Generates three plot types per county (and an all-counties aggregate):
  1. scatter_all_years   - All WYs in one panel, colored by year
  2. scatter_by_year     - Faceted per-WY scatter
  3. boxplot_year_colored - Box plots with year-colored overlay points

Supports ``et_mode`` parameter:
  - "actual"    : plot actual OpenET ET (default, blue dots)
  - "corrected" : plot corrected ET (green dots)
  - "both"      : overlay actual (lightgrey) and corrected (green)

Uses matched_minima_iso (physical cutting dates) for segment construction
so that ET and GDD5 are correctly aligned with n_cuttings.

Directory structure:
  output/figures/cuttings_analysis/{County}/{plot_type}.png
  output/figures/cuttings_analysis/all_counties/{plot_type}.png
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.parcel_summary_provider import (
    build_parcel_summary_matched,
    build_multicounty_matched,
)
from es_analysis.data_providers.evi_provider import normalize_county_name
from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.utils.publication_style import (
    WONG_PALETTE,
    DOUBLE_COL_WIDTH,
    apply_style,
    add_panel_label,
)

# Year colors using Wong palette
YEAR_COLORS = {
    2019: WONG_PALETTE[1],  # orange
    2020: WONG_PALETTE[2],  # sky blue
    2021: WONG_PALETTE[3],  # bluish green
    2022: WONG_PALETTE[5],  # blue
    2023: WONG_PALETTE[6],  # vermillion
    2024: WONG_PALETTE[7],  # reddish purple
}


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out_dir / name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Scatter: all years in one panel, colored by WY
# ---------------------------------------------------------------------------

def scatter_all_years(
    df: pd.DataFrame,
    county_label: str,
    out_dir: Path,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    gdd_col: str = "gdd5_mean",
    cut_col: str = "n_cuttings",
    et_mode: str = "actual",
) -> Dict:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45), sharey=True)

    summary = {"county": county_label, "n": len(df), "et_mode": et_mode}

    corr_col = "et_cum_corrected_mm"
    has_corr = corr_col in df.columns and et_mode in ("corrected", "both")

    for wy, color in YEAR_COLORS.items():
        sub = df[df["WY"] == wy]
        if sub.empty:
            continue

        if et_mode == "both":
            # Actual in lightgrey behind
            axes[0].scatter(sub[et_col], sub[cut_col], s=12, alpha=0.3,
                            edgecolor="none", color="lightgrey")
            if has_corr:
                axes[0].scatter(sub[corr_col], sub[cut_col], s=12, alpha=0.5,
                                edgecolor="none", color=color, label=str(wy))
        elif et_mode == "corrected" and has_corr:
            axes[0].scatter(sub[corr_col], sub[cut_col], s=12, alpha=0.5,
                            edgecolor="none", color=color, label=str(wy))
        else:
            axes[0].scatter(sub[et_col], sub[cut_col], s=12, alpha=0.5,
                            edgecolor="none", color=color, label=str(wy))

        axes[1].scatter(sub[gdd_col], sub[cut_col], s=12, alpha=0.5,
                        edgecolor="none", color=color, label=str(wy))

    et_label = {"actual": "Cumulative segment ET (actual, mm)",
                "corrected": "Cumulative segment ET (corrected, Method A, mm)",
                "both": "Segment ET (mm) [grey=actual, color=corrected]"}[et_mode]
    axes[0].set_xlabel(et_label)
    axes[0].set_ylabel("Number of cuttings")
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _style_ax(axes[0])
    add_panel_label(axes[0], "a")
    axes[0].legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.7)

    axes[1].set_xlabel("Cumulative GDD5 (\u00b0C\u00b7day)")
    _style_ax(axes[1])
    add_panel_label(axes[1], "b")
    axes[1].legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.7)

    n_parcels = len(df)
    n_wys = df["WY"].nunique() if "WY" in df.columns else 0
    wy_range = f"WY{int(df['WY'].min())}\u2013{int(df['WY'].max())}" if n_wys > 0 else ""
    fig.suptitle(f"{county_label} \u2014 {n_parcels:,} parcel-years, {wy_range}", fontsize=9, y=1.02)
    fig.tight_layout()
    suffix = f"_et_{et_mode}" if et_mode != "actual" else ""
    _save(fig, out_dir, f"scatter_all_years{suffix}")
    return summary


# ---------------------------------------------------------------------------
# 2. Scatter: faceted per year
# ---------------------------------------------------------------------------

def scatter_by_year(
    df: pd.DataFrame,
    county_label: str,
    out_dir: Path,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    gdd_col: str = "gdd5_mean",
    cut_col: str = "n_cuttings",
    et_mode: str = "actual",
) -> Dict:
    apply_style()
    wys = sorted(df["WY"].unique())
    n_wys = len(wys)
    if n_wys == 0:
        return {"county": county_label, "n": 0}

    corr_col = "et_cum_corrected_mm"
    has_corr = corr_col in df.columns and et_mode in ("corrected", "both")

    fig, axes = plt.subplots(n_wys, 2, figsize=(DOUBLE_COL_WIDTH, 2.2 * n_wys), squeeze=False)

    for i, wy in enumerate(wys):
        sub = df[df["WY"] == wy]
        color = YEAR_COLORS.get(wy, WONG_PALETTE[0])

        if et_mode == "both":
            axes[i, 0].scatter(sub[et_col], sub[cut_col], s=12, alpha=0.4,
                               edgecolor="none", color="lightgrey")
            if has_corr:
                axes[i, 0].scatter(sub[corr_col], sub[cut_col], s=12, alpha=0.5,
                                   edgecolor="none", color="#6BCB77")
        elif et_mode == "corrected" and has_corr:
            axes[i, 0].scatter(sub[corr_col], sub[cut_col], s=12, alpha=0.5,
                               edgecolor="none", color="#6BCB77")
        else:
            axes[i, 0].scatter(sub[et_col], sub[cut_col], s=12, alpha=0.5,
                               edgecolor="none", color=color)

        axes[i, 0].set_ylabel("Cuttings")
        axes[i, 0].set_title(f"WY {wy} (n={len(sub)})", fontsize=8)
        axes[i, 0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _style_ax(axes[i, 0])

        axes[i, 1].scatter(sub[gdd_col], sub[cut_col], s=12, alpha=0.5,
                           edgecolor="none", color=color)
        axes[i, 1].set_title(f"WY {wy} (n={len(sub)})", fontsize=8)
        axes[i, 1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _style_ax(axes[i, 1])

    et_label = {"actual": "Cumulative segment ET (actual, mm)",
                "corrected": "Cumulative segment ET (corrected, Method A, mm)",
                "both": "Segment ET (mm)"}[et_mode]
    axes[-1, 0].set_xlabel(et_label)
    axes[-1, 1].set_xlabel("Cumulative GDD5 (\u00b0C\u00b7day)")
    fig.suptitle(f"{county_label} - Year by Year", fontsize=10, y=1.01)
    fig.tight_layout()
    suffix = f"_et_{et_mode}" if et_mode != "actual" else ""
    _save(fig, out_dir, f"scatter_by_year{suffix}")
    return {"county": county_label, "n_wys": n_wys}


# ---------------------------------------------------------------------------
# 3. Box plots with year-colored overlay points
# ---------------------------------------------------------------------------

def boxplot_year_colored(
    df: pd.DataFrame,
    county_label: str,
    out_dir: Path,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    gdd_col: str = "gdd5_mean",
    cut_col: str = "n_cuttings",
    et_mode: str = "actual",
) -> Dict:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.5))

    corr_col = "et_cum_corrected_mm"
    has_corr = corr_col in df.columns and et_mode in ("corrected", "both")

    # Choose which ET column to use for the left panel
    use_et_col = corr_col if (et_mode == "corrected" and has_corr) else et_col
    et_label = {"actual": "Cumulative segment ET (actual, mm)",
                "corrected": "Cumulative segment ET (corrected, Method A, mm)",
                "both": "Segment ET (mm) [grey=actual, color=corrected]"}[et_mode]

    cut_vals = sorted(df[cut_col].dropna().unique())
    cut_vals = [int(c) for c in cut_vals]

    for col_idx, (val_col, xlabel) in enumerate([
        (use_et_col, et_label),
        (gdd_col, "Cumulative GDD5 (\u00b0C\u00b7day)"),
    ]):
        ax = axes[col_idx]
        box_data = [df[df[cut_col] == c][val_col].dropna().values for c in cut_vals]
        # Data points behind, box-and-whiskers on top
        for wy, color in YEAR_COLORS.items():
            sub = df[df["WY"] == wy]
            if sub.empty:
                continue
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(sub))

            if col_idx == 0 and et_mode == "both" and has_corr:
                ax.scatter(sub[et_col], sub[cut_col].values + jitter,
                           s=6, alpha=0.2, color="lightgrey", edgecolor="none",
                           zorder=1)
                ax.scatter(sub[corr_col], sub[cut_col].values + jitter,
                           s=6, alpha=0.35, color=color, edgecolor="none",
                           label=str(wy), zorder=2)
            else:
                ax.scatter(sub[val_col], sub[cut_col].values + jitter,
                           s=6, alpha=0.35, color=color, edgecolor="none",
                           label=str(wy), zorder=2)

        bp = ax.boxplot(box_data, positions=cut_vals, widths=0.6, vert=False,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8, alpha=0.85),
                        medianprops=dict(color="black", linewidth=1.2),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        zorder=5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of cuttings")
        ax.set_yticks(cut_vals)
        _style_ax(ax)
        add_panel_label(ax, "ab"[col_idx])

    axes[0].legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.7)
    axes[1].legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.7)
    n_parcels = len(df)
    n_wys = df["WY"].nunique() if "WY" in df.columns else 0
    wy_range = f"WY{int(df['WY'].min())}\u2013{int(df['WY'].max())}" if n_wys > 0 else ""
    fig.suptitle(f"{county_label} \u2014 {n_parcels:,} parcel-years, {wy_range}", fontsize=9, y=1.02)
    fig.tight_layout()
    suffix = f"_et_{et_mode}" if et_mode != "actual" else ""
    _save(fig, out_dir, f"boxplot_year_colored{suffix}")
    return {"county": county_label, "n_cut_groups": len(cut_vals)}


# ---------------------------------------------------------------------------
# 4. CDF by water year × county (faceted columns by WY)
# ---------------------------------------------------------------------------

def _empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted x and y = i/n for an empirical CDF."""
    sorted_x = np.sort(values)
    cdf_y = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    return sorted_x, cdf_y


def _county_palette(counties: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """Pick 10 distinct colors (tab10) keyed by county name in N→S order."""
    cmap = plt.cm.tab10
    return {c: cmap(i % 10) for i, c in enumerate(counties)}


def cdf_by_year_county(
    df: pd.DataFrame,
    out_dir: Path,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    cut_col: str = "n_cuttings",
) -> Dict:
    """2 rows × N WY-columns figure of CDFs colored by county.

    Row 0: cumulative segment ET CDF per WY (one curve per county).
    Row 1: number-of-cuttings CDF per WY (step, one curve per county).
    """
    apply_style()
    wys = sorted(int(w) for w in df["WY"].dropna().unique())
    counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    n_wy = len(wys)
    if n_wy == 0 or len(counties) == 0:
        print("  cdf_by_year_county: no WY/county data, skipping")
        return {"n_wys": 0, "n_counties": 0}

    county_colors = _county_palette(counties)

    fig, axes = plt.subplots(
        2, n_wy,
        figsize=(2.9 * n_wy + 3.2, 7.4),
        sharey=True,
    )
    if n_wy == 1:
        axes = axes.reshape(2, 1)

    panel_idx = 0
    for ci, wy in enumerate(wys):
        ax_et = axes[0, ci]
        ax_ct = axes[1, ci]
        sub_wy = df[df["WY"] == wy]

        for county in counties:
            sub_cy = sub_wy[sub_wy["county"] == county]
            et_vals = sub_cy[et_col].dropna()
            et_vals = et_vals[et_vals > 0].values
            ct_vals = sub_cy[cut_col].dropna().values

            if len(et_vals) > 0:
                x, y = _empirical_cdf(et_vals)
                ax_et.plot(
                    x, y, color=county_colors[county], linewidth=1.3,
                    label=county if ci == 0 else None,
                )
            if len(ct_vals) > 0:
                x, y = _empirical_cdf(ct_vals)
                ax_ct.step(
                    x, y, where="post",
                    color=county_colors[county], linewidth=1.3,
                )

        ax_et.set_title(f"WY{wy}", fontsize=11, fontweight="bold", pad=4)
        ax_et.set_ylim(0, 1)
        ax_ct.set_ylim(0, 1)
        ax_et.set_xlim(left=0)
        ax_ct.set_xlim(0, 13)

        if ci == 0:
            ax_et.set_ylabel("CDF — cumulative segment ET", fontsize=9)
            ax_ct.set_ylabel("CDF — number of cuttings", fontsize=9)
        ax_ct.set_xlabel("Number of cuttings", fontsize=8)
        if ci == n_wy // 2:
            ax_et.set_xlabel("")
        ax_et.set_xlabel("Cumulative segment ET (mm)", fontsize=8)

        ax_et.tick_params(axis="both", labelsize=7.5)
        ax_ct.tick_params(axis="both", labelsize=7.5)
        ax_et.grid(alpha=0.25, linewidth=0.4)
        ax_ct.grid(alpha=0.25, linewidth=0.4)
        _style_ax(ax_et)
        _style_ax(ax_ct)

        add_panel_label(ax_et, chr(ord("a") + panel_idx))
        panel_idx += 1
    for ci in range(n_wy):
        add_panel_label(axes[1, ci], chr(ord("a") + panel_idx))
        panel_idx += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right", bbox_to_anchor=(0.995, 0.5),
        ncol=1, fontsize=8.5, frameon=True, framealpha=0.95,
        title="County (N → S)", title_fontsize=9,
    )

    n_parcel_years = len(df)
    fig.suptitle(
        f"Empirical CDFs by Water Year — {n_parcel_years:,} parcel-years, "
        "10 counties",
        fontsize=12, y=0.995, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    _save(fig, out_dir, "cdf_by_year_county")
    return {"n_wys": n_wy, "n_counties": len(counties),
            "n_parcel_years": n_parcel_years}


# ---------------------------------------------------------------------------
# 5. Summary CDF: WY-averaged per parcel, colored by county
# ---------------------------------------------------------------------------

def cdf_summary_by_county(
    df: pd.DataFrame,
    out_dir: Path,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    cut_col: str = "n_cuttings",
) -> Dict:
    """Summary CDF: each parcel's WY-averaged ET and cuttings, colored by county.

    For each parcel (`UniqueID`) we compute mean ET and mean n_cuttings
    across all its water years, then plot the empirical CDF of those
    per-parcel averages within each county.  This gives one CDF per county
    representing the typical multi-year behavior of its parcels.
    """
    apply_style()
    counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    if len(counties) == 0:
        print("  cdf_summary_by_county: no county data, skipping")
        return {"n_counties": 0, "n_parcels": 0}

    parcel_id = "UniqueID" if "UniqueID" in df.columns else None
    if parcel_id is None:
        # Fallback: aggregate at the parcel-year level
        agg_df = df[[et_col, cut_col, "county"]].rename(
            columns={et_col: "et_avg", cut_col: "cut_avg"}
        )
    else:
        valid = df[(df[et_col].notna()) & (df[et_col] > 0)]
        agg_df = valid.groupby([parcel_id, "county"]).agg(
            et_avg=(et_col, "mean"),
            cut_avg=(cut_col, "mean"),
        ).reset_index()

    county_colors = _county_palette(counties)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax = axes[0]
    for county in counties:
        vals = agg_df[(agg_df["county"] == county)]["et_avg"].dropna()
        vals = vals[vals > 0].values
        if len(vals) == 0:
            continue
        x, y = _empirical_cdf(vals)
        ax.plot(
            x, y, color=county_colors[county], linewidth=2.0,
            label=f"{county} (n={len(vals)})",
        )
    ax.set_xlabel("Cumulative segment ET (mm) — WY-averaged per parcel",
                  fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.30, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=10)
    add_panel_label(ax, "a")
    _style_ax(ax)

    ax = axes[1]
    for county in counties:
        vals = agg_df[(agg_df["county"] == county)]["cut_avg"].dropna().values
        if len(vals) == 0:
            continue
        x, y = _empirical_cdf(vals)
        ax.step(
            x, y, where="post", color=county_colors[county], linewidth=2.0,
            label=f"{county} (n={len(vals)})",
        )
    ax.set_xlabel("Number of cuttings — WY-averaged per parcel", fontsize=11)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.30, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=10)
    add_panel_label(ax, "b")
    _style_ax(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right", bbox_to_anchor=(0.998, 0.5),
        ncol=1, fontsize=9.5, frameon=True, framealpha=0.95,
        title="County (N → S)", title_fontsize=10,
    )

    n_parcels_total = len(agg_df)
    fig.suptitle(
        "Summary CDF — WY-averaged per parcel, colored by county   "
        f"(N = {n_parcels_total:,} parcels across "
        f"{df['WY'].nunique() if 'WY' in df.columns else 0} water years)",
        fontsize=13, y=0.995, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.88, 0.97])
    _save(fig, out_dir, "cdf_summary_by_county")
    return {"n_counties": len(counties), "n_parcels": n_parcels_total}


# ---------------------------------------------------------------------------
# 6. Joint plot — central scatter + cuttings CDF (top) + ET CDF (right)
# ---------------------------------------------------------------------------

def cdf_joint_summary(
    df: pd.DataFrame,
    out_dir: Path,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    cut_col: str = "n_cuttings",
) -> Dict:
    """Joint plot: ET vs cuttings scatter + cuttings CDF (top) + ET CDF (right).

    Combines the joint distribution of (cuttings, cumulative ET) with the two
    marginal CDFs in a single figure, so the coupling between cuttings count
    and total ET (per-cut ET ≈ constant across counties) is visible alongside
    the marginal distributions.  WY-averaged per parcel — same data
    aggregation as `cdf_summary_by_county`, same county color palette.
    """
    apply_style()
    counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    if len(counties) == 0:
        print("  cdf_joint_summary: no county data, skipping")
        return {"n_counties": 0, "n_parcels": 0}

    parcel_id = "UniqueID" if "UniqueID" in df.columns else None
    if parcel_id is None:
        agg_df = df[[et_col, cut_col, "county"]].rename(
            columns={et_col: "et_avg", cut_col: "cut_avg"}
        )
    else:
        valid = df[(df[et_col].notna()) & (df[et_col] > 0)]
        agg_df = valid.groupby([parcel_id, "county"]).agg(
            et_avg=(et_col, "mean"),
            cut_avg=(cut_col, "mean"),
        ).reset_index()

    county_colors = _county_palette(counties)

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = GridSpec(
        2, 2,
        width_ratios=[3.5, 1],
        height_ratios=[1, 3.5],
        wspace=0.05, hspace=0.05,
        left=0.075, right=0.84, top=0.93, bottom=0.07,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # ----- Central scatter -----
    for county in counties:
        sub = agg_df[agg_df["county"] == county]
        if len(sub) == 0:
            continue
        ax_main.scatter(
            sub["cut_avg"], sub["et_avg"],
            s=11, alpha=0.32, color=county_colors[county],
            edgecolor="none", zorder=2,
        )
        # Per-county median marker (large diamond, edged)
        med_x = float(sub["cut_avg"].median())
        med_y = float(sub["et_avg"].median())
        ax_main.scatter(
            med_x, med_y,
            s=140, marker="D", color=county_colors[county],
            edgecolor="black", linewidth=1.2, zorder=5,
            label=f"{county} (n={len(sub)})",
        )

    # Cross-county per-cut ET trend line
    ratio = (agg_df["et_avg"] / agg_df["cut_avg"].replace(0, np.nan)).dropna()
    avg_per_cut = float(ratio.median())
    cuts_x = np.linspace(0.5, 13, 50)
    ax_main.plot(
        cuts_x, avg_per_cut * cuts_x,
        color="#444444", linestyle="--", linewidth=1.4, alpha=0.75,
        zorder=1, label=f"Trend: ET ≈ {avg_per_cut:.0f} mm/cut × cuttings",
    )

    ax_main.set_xlabel("Number of cuttings (WY-averaged per parcel)",
                       fontsize=11)
    ax_main.set_ylabel("Cumulative segment ET (mm, WY-averaged per parcel)",
                       fontsize=11)
    ax_main.set_xlim(0, 13)
    ax_main.set_ylim(0, float(agg_df["et_avg"].max()) * 1.05)
    ax_main.grid(alpha=0.30, linewidth=0.5)
    ax_main.tick_params(axis="both", labelsize=10)
    _style_ax(ax_main)

    # ----- Top marginal: cuttings CDF (step) -----
    for county in counties:
        vals = agg_df[agg_df["county"] == county]["cut_avg"].dropna().values
        if len(vals) == 0:
            continue
        x, y = _empirical_cdf(vals)
        ax_top.step(
            x, y, where="post", color=county_colors[county], linewidth=1.5,
        )
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Cuttings CDF", fontsize=10)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=9)
    ax_top.grid(alpha=0.30, linewidth=0.5)
    _style_ax(ax_top)

    # ----- Right marginal: ET CDF (rotated — ET on Y, CDF on X) -----
    for county in counties:
        vals = agg_df[agg_df["county"] == county]["et_avg"].dropna()
        vals = vals[vals > 0].values
        if len(vals) == 0:
            continue
        x, y = _empirical_cdf(vals)
        ax_right.plot(
            y, x, color=county_colors[county], linewidth=1.5,
        )
    ax_right.set_xlim(0, 1)
    ax_right.set_xlabel("ET CDF", fontsize=10)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=9)
    ax_right.grid(alpha=0.30, linewidth=0.5)
    _style_ax(ax_right)

    # ----- Legend in the right gutter -----
    handles, labels = ax_main.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(0.85, 0.5),
        ncol=1, fontsize=9, frameon=True, framealpha=0.95,
        title="County (N → S)\n+ trend line", title_fontsize=10,
    )

    n_parcels_total = len(agg_df)
    n_wys = df["WY"].nunique() if "WY" in df.columns else 0
    fig.suptitle(
        "Joint Distribution of Cuttings × Cumulative ET — WY-averaged per parcel\n"
        f"N = {n_parcels_total:,} parcels across {n_wys} water years   |   "
        f"diamonds = per-county medians   |   dashed line = ~{avg_per_cut:.0f} mm/cut trend",
        fontsize=12, y=0.985, fontweight="bold",
    )

    _save(fig, out_dir, "cdf_joint_summary")
    return {
        "n_counties": len(counties),
        "n_parcels": n_parcels_total,
        "avg_per_cut_mm": avg_per_cut,
    }


# ---------------------------------------------------------------------------
# 7. Per-county CDFs in a horizontal small-multiples row
# ---------------------------------------------------------------------------

def cdf_per_county_horizontal(
    df: pd.DataFrame,
    out_dir: Path,
    et_col: str = "et_cum_minET_to_last_cut_mm",
    cut_col: str = "n_cuttings",
) -> Dict:
    """Compact horizontal small-multiples CDF — one panel per county.

    Uses raw parcel-year data so the cuttings step CDF shows its true integer
    staircase.  Each panel:
      • Solid line — Cumulative segment ET CDF on the bottom x-axis
      • Dashed step — N cuttings CDF on the twin top x-axis
      • Both curves drawn in the county's color
    Layout: 10 narrow panels packed tightly (no left/right spines), shared
    y-axis ticks on the leftmost panel only, single bottom legend that
    identifies the county palette and the two line styles.
    """
    apply_style()
    counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    if len(counties) == 0:
        print("  cdf_per_county_horizontal: no county data, skipping")
        return {"n_counties": 0, "n_parcels": 0}

    # Use raw parcel-year data (not WY-averaged) so the cuttings staircase
    # is real and the ET CDFs show their natural shape.
    raw_df = df[
        (df[et_col].notna())
        & (df[et_col] > 0)
        & (df[cut_col].notna())
        & (df["county"].isin(counties))
    ][[et_col, cut_col, "county"]].rename(
        columns={et_col: "et_val", cut_col: "cut_val"}
    )

    # Tight x-axis ranges from data quantiles so the CDF "S-curve" is visible
    # rather than a near-vertical line at the data mass.
    et_lo = float(raw_df["et_val"].quantile(0.01))
    et_hi = float(raw_df["et_val"].quantile(0.99)) * 1.02
    cut_lo = max(0.5, float(raw_df["cut_val"].quantile(0.01)) - 0.5)
    cut_hi = float(raw_df["cut_val"].quantile(0.99)) + 0.5

    county_colors = _county_palette(counties)

    n_cty = len(counties)
    n_per_row = 5
    n_rows_grid = 2
    fig, axes = plt.subplots(
        n_rows_grid, n_per_row,
        figsize=(1.95 * n_per_row + 1.2, 7.0),
        sharey=False,  # manual ylim per panel — sharey would otherwise
                       # cause set_yticklabels on one row to overwrite the
                       # other (yaxis state is shared under sharey=True).
        gridspec_kw={"wspace": 0.10, "hspace": 0.0},
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.085)

    spine_grey = "#808080"

    for row_idx in range(n_rows_grid):
        row_counties = counties[
            row_idx * n_per_row : (row_idx + 1) * n_per_row
        ]
        is_top_row = (row_idx == 0)
        is_bot_row = (row_idx == n_rows_grid - 1)

        for ci, county in enumerate(row_counties):
            ax = axes[row_idx, ci]
            sub = raw_df[raw_df["county"] == county]
            color = county_colors[county]

            # ET CDF (solid, bottom axis)
            et_vals = sub["et_val"].values
            if len(et_vals) > 0:
                x_et, y_et = _empirical_cdf(et_vals)
                ax.plot(x_et, y_et, color=color, linewidth=1.7,
                        linestyle="-", zorder=3)
            ax.set_xlim(et_lo, max(1550.0, et_hi))
            ax.set_ylim(0, 1)

            # Cuttings CDF (dashed step, top twin axis)
            ax_cut = ax.twiny()
            cut_vals = sub["cut_val"].values
            if len(cut_vals) > 0:
                x_cut, y_cut = _empirical_cdf(cut_vals)
                ax_cut.step(x_cut, y_cut, where="post", color=color,
                            linewidth=1.7, linestyle="--", zorder=4)
            ax_cut.set_xlim(cut_lo, max(12.5, cut_hi))
            ax_cut.set_ylim(0, 1)

            # Drop left/right spines on both axes
            for sp in ("left", "right"):
                ax.spines[sp].set_visible(False)
                ax_cut.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(spine_grey)
            ax.spines["bottom"].set_linewidth(0.7)
            ax_cut.spines["top"].set_color(spine_grey)
            ax_cut.spines["top"].set_linewidth(0.7)

            # Merge the two near-overlapping spines between the two rows
            # into a single line: hide the top-row's bottom spine so only
            # the bottom-row's top spine remains as the divider.
            if is_top_row:
                ax.spines["bottom"].set_visible(False)
            if is_bot_row:
                ax_cut.spines["top"].set_visible(True)

            # Per-row tick visibility: cuttings axis only on TOP row,
            # ET axis only on BOTTOM row, so the figure has exactly one
            # cuttings axis and one ET axis as requested.
            if is_top_row:
                ax_cut.tick_params(axis="x", labelsize=7,
                                   length=2.5, width=0.5,
                                   top=True, labeltop=True,
                                   bottom=False, labelbottom=False,
                                   color=spine_grey, labelcolor="#444")
                ax.tick_params(axis="x", which="both",
                               bottom=False, labelbottom=False,
                               top=False, labeltop=False)
            if is_bot_row:
                ax.tick_params(axis="x", labelsize=7,
                               length=2.5, width=0.5,
                               bottom=True, labelbottom=True,
                               top=False, labeltop=False,
                               color=spine_grey, labelcolor="#444")
                ax_cut.tick_params(axis="x", which="both",
                                   top=False, labeltop=False,
                                   bottom=False, labelbottom=False)

            # Y-axis ticks only on leftmost panel of each row.  Tick mark
            # and label colors match the grey used on the cuttings/ET axes
            # so the figure has a single visual treatment for all axes.
            if ci == 0:
                ax.tick_params(axis="y", left=True, right=False,
                               labelsize=8.5,
                               color=spine_grey, labelcolor="#444")
            else:
                ax.tick_params(axis="y", left=False, right=False,
                               labelleft=False)

            ax.set_xticks([500, 1000, 1500])
            ax_cut.set_xticks([3, 6, 9, 12])
            ax.grid(axis="y", alpha=0.20, linewidth=0.4)

            # County name in the top-left corner of each panel (in the
            # county's color), with a small offset from the panel edges.
            # This replaces the bottom legend so each curve is identified
            # in-place.
            ax.text(
                0.05, 0.95, county,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9, fontweight="bold",
                color=color, zorder=10,
            )

    # Y-labels on leftmost of each row (matching the grey of the ET and
    # cuttings axis labels for visual consistency).
    axes[0, 0].set_ylabel("CDF", fontsize=10, color="#444")
    axes[1, 0].set_ylabel("CDF", fontsize=10, color="#444")

    # The two rows touch at y=0/y=1 where the top-row's "0.0" label and
    # the bottom-row's "1.0" label otherwise stack on top of each other.
    # Hide the boundary labels on the leftmost panels and add a single
    # merged "0/1" annotation in their place.
    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axes[0, 0].set_yticks(yticks)
    axes[0, 0].set_yticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
    axes[1, 0].set_yticks(yticks)
    axes[1, 0].set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", ""])

    # Place the merged "0/1" label at the row boundary, just left of the
    # leftmost panel column, in figure coordinates.
    bbox_top = axes[0, 0].get_position()
    bbox_bot = axes[1, 0].get_position()
    boundary_y = (bbox_top.y0 + bbox_bot.y1) / 2.0
    boundary_x = bbox_bot.x0 - 0.008
    fig.text(
        boundary_x, boundary_y, "0/1",
        ha="right", va="center", fontsize=8.5, color="#444",
    )

    # Shared axis labels — exactly one cuttings axis (top) and one ET axis
    # (bottom), positioned outside the panel band.
    fig.text(0.5, 0.025, "Cumulative segment ET (mm) — bottom axis (solid)",
             ha="center", va="center", fontsize=9.5, color="#333")
    fig.text(0.5, 0.935, "Number of cuttings — top axis (dashed step)",
             ha="center", va="center", fontsize=9.5, color="#333")

    # Per-panel county labels are placed inside each panel (top-left
    # corner) so no figure-level legend is needed.

    n_obs = len(raw_df)
    n_wys = df["WY"].nunique() if "WY" in df.columns else 0
    fig.suptitle(
        f"Per-county CDFs (Cumulative ET + N Cuttings) — raw parcel-years   "
        f"|   {n_obs:,} parcel-years across {n_wys} water years",
        fontsize=11, y=0.985, fontweight="bold",
    )

    _save(fig, out_dir, "cdf_per_county_horizontal")
    return {"n_counties": n_cty, "n_parcel_years": n_obs}


# ---------------------------------------------------------------------------
# Generate all plots for one county
# ---------------------------------------------------------------------------

def generate_county_plots(
    county: str,
    base_out: Path,
    wy_start: int = 2019,
    wy_end: int = 2024,
    min_et_mm: float = 0.0,
    et_mode: str = "actual",
    method: str = "A",
) -> pd.DataFrame:
    """Build matched data and generate all 3 plot types for a county."""
    county_norm = normalize_county_name(county)
    county_dir = county_norm.replace(" ", "_")
    out_dir = base_out / county_dir

    print(f"\n--- {county_norm} (et_mode={et_mode}, method={method}) ---")
    frames = []
    for wy in range(wy_start, wy_end + 1):
        try:
            df = build_parcel_summary_matched(
                county=county_norm, wy=wy,
                et_mode=et_mode, method=method,
            )
            frames.append(df)
            print(f"  WY{wy}: {len(df)} parcels, "
                  f"cuts={df['n_cuttings'].mean():.1f}, "
                  f"ET range={df['et_cum_minET_to_last_cut_mm'].min():.0f}-{df['et_cum_minET_to_last_cut_mm'].max():.0f} mm")
        except (FileNotFoundError, ValueError) as e:
            print(f"  WY{wy}: skipped ({e})")

    if not frames:
        print(f"  No data for {county_norm}, skipping plots.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    n_before = len(df_all)

    # Filter implausible parcels with very low cumulative ET
    low_et = df_all["et_cum_minET_to_last_cut_mm"] < min_et_mm
    n_filtered = int(low_et.sum())
    if n_filtered > 0:
        print(f"  Filtered {n_filtered}/{n_before} parcels with ET < {min_et_mm} mm "
              f"({100 * n_filtered / n_before:.1f}%)")
        if n_filtered <= 20:
            filtered_rows = df_all[low_et][["UniqueID", "WY", "n_cuttings", "et_cum_minET_to_last_cut_mm"]]
            for _, r in filtered_rows.iterrows():
                print(f"    {r['UniqueID']} WY{r['WY']}: {r['n_cuttings']} cuts, {r['et_cum_minET_to_last_cut_mm']:.1f} mm")
        df_all = df_all[~low_et].copy()

    print(f"  Total: {len(df_all)} parcel-years (after filtering)")

    scatter_all_years(df_all, county_norm, out_dir, et_mode=et_mode)
    scatter_by_year(df_all, county_norm, out_dir, et_mode=et_mode)
    boxplot_year_colored(df_all, county_norm, out_dir, et_mode=et_mode)

    return df_all


# ---------------------------------------------------------------------------
# Main entry: generate for specified counties
# ---------------------------------------------------------------------------

def main(
    counties: Optional[List[str]] = None,
    et_mode: str = "actual",
    method: str = "A",
) -> None:
    """Generate all cuttings analysis plots for the given counties."""
    import matplotlib
    matplotlib.use("Agg")

    base_out = Path(__file__).parent.parent.parent / "output" / "figures" / "cuttings_analysis"

    if counties is None:
        counties = list(COUNTY_ORDER)

    all_frames = []
    for county in counties:
        df = generate_county_plots(
            county, base_out, et_mode=et_mode, method=method,
        )
        if not df.empty:
            all_frames.append(df)

    if len(all_frames) > 1:
        df_combined = pd.concat(all_frames, ignore_index=True)
        agg_dir = base_out / "all_counties"
        label = f"All {len(counties)} Counties"
        print(f"\n--- Aggregate ({len(df_combined)} parcel-years) ---")
        scatter_all_years(df_combined, label, agg_dir, et_mode=et_mode)
        scatter_by_year(df_combined, label, agg_dir, et_mode=et_mode)
        boxplot_year_colored(df_combined, label, agg_dir, et_mode=et_mode)
        # New: CDF figures (additive — do not alter existing plots).
        cdf_by_year_county(df_combined, agg_dir)
        cdf_summary_by_county(df_combined, agg_dir)


def _load_combined_cached_or_build(
    run_name: str = "alfalfa_run_6",
    et_mode: str = "actual",
    method: str = "A",
) -> pd.DataFrame:
    """Load multicounty_matched parquet from the run's data/ cache, falling
    back to a full rebuild only when the cache is missing.  Reading the
    parquet takes a fraction of a second; a rebuild can take ~50 minutes,
    so always prefer the cache when present.
    """
    from es_analysis.utils.run_output import load_dataframe
    df = load_dataframe("multicounty_matched", run_name)
    if df is not None:
        print(f"[cache] Loaded multicounty_matched from {run_name}/data/ "
              f"({len(df):,} rows)")
        return df
    print(f"[cache miss] No cached parquet under {run_name}/data/, "
          f"rebuilding via build_multicounty_matched(et_mode={et_mode!r}) "
          f"— this may take a while...")
    return build_multicounty_matched(et_mode=et_mode, method=method)


def generate_cdf_plots(
    run_name: str = "alfalfa_run_6",
    et_mode: str = "actual",
    method: str = "A",
) -> None:
    """Cache-aware standalone entry-point for the two new CDF figures.

    Loads `multicounty_matched.parquet` from the run's data/ directory if
    it exists; otherwise rebuilds via `build_multicounty_matched`.  Then
    writes `cdf_by_year_county.{png,pdf}` and `cdf_summary_by_county.{png,pdf}`
    to `<run>/cuttings_analysis/all_counties/`.
    """
    import matplotlib
    matplotlib.use("Agg")

    df = _load_combined_cached_or_build(run_name, et_mode=et_mode, method=method)
    base_out = (
        Path(__file__).parent.parent.parent
        / "output" / "figures" / run_name / "cuttings_analysis"
    )
    agg_dir = base_out / "all_counties"
    print(f"--- CDF figures → {agg_dir} ---")
    print(f"  {cdf_by_year_county(df, agg_dir)}")
    print(f"  {cdf_summary_by_county(df, agg_dir)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--counties", nargs="+", default=None)
    parser.add_argument("--et-mode", choices=["actual", "corrected", "both"],
                        default="actual", help="ET mode: actual, corrected, or both")
    parser.add_argument("--method", choices=["A", "B"], default="A",
                        help="ET correction method A or B")
    parser.add_argument("--cdf-only", action="store_true",
                        help="Skip per-county and aggregate plots; "
                             "just regenerate the two CDF figures using the "
                             "cached multicounty_matched.parquet (fast path).")
    parser.add_argument("--run-name", default="alfalfa_run_6",
                        help="Run name for cache lookup (default: alfalfa_run_6)")
    args = parser.parse_args()
    if args.cdf_only:
        generate_cdf_plots(
            run_name=args.run_name, et_mode=args.et_mode, method=args.method,
        )
    else:
        main(args.counties, et_mode=args.et_mode, method=args.method)
