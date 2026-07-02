"""Water Year Type analysis charts.

Seven charts examining cuttings, ET, corrections, and savings
stratified by DWR Water Year Type (SJV index) with optional USDM
drought overlay.

All functions follow the (fig, axes, summary) return convention and
use publication styling from utils/publication_style.py.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from es_analysis.utils.publication_style import (
    apply_style,
    save_pub_figure,
    add_panel_label,
    DOUBLE_COL_WIDTH,
    SINGLE_COL_WIDTH,
    WONG_PALETTE,
)
from es_analysis.data_providers.wy_type_provider import (
    WY_TYPE_ORDER,
    WY_TYPE_COLORS,
    SJ_VALLEY_WY_INDEX,
    get_wy_type,
    get_wy_color,
    add_wy_type_columns,
    get_wy_types_present,
)
from es_analysis.data_providers.statistics_provider import COUNTY_ORDER
from es_analysis.utils.units import mm_to_acft_per_acre


def _wy_labels_for_type(wy_type: str, wys: List[int]) -> str:
    """Build an annotation string listing WYs belonging to a type."""
    matching = [str(wy) for wy in sorted(wys) if get_wy_type(wy) == wy_type]
    return ", ".join(matching)


def _wy_type_tick_labels(types_present: List[str], wys: List[int]) -> List[str]:
    """Build clean two-line x-tick labels: 'Critical\n(2021, 2022)'."""
    labels = []
    for wt in types_present:
        wy_str = _wy_labels_for_type(wt, wys)
        labels.append(f"{wt}\n({wy_str})")
    return labels


# -----------------------------------------------------------------------
# Chart 1: Cuttings by WY type × county
# -----------------------------------------------------------------------
def wy_type_cuttings_bar(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Grouped bar: WY type × county, mean n_cuttings + SEM.

    Args:
        df: DataFrame with columns county, WY, n_cuttings.
        output_dir: Where to save figure (None = don't save).

    Returns:
        (fig, ax, summary_dict)
    """
    apply_style()
    df = add_wy_type_columns(df.copy())
    wys = sorted(df["WY"].unique())
    types_present = get_wy_types_present(wys)

    counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    n_types = len(types_present)
    n_counties = len(counties)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.5))
    bar_width = 0.8 / n_types
    x = np.arange(n_counties)

    summary_data = []
    for j, wt in enumerate(types_present):
        means, sds, maxes = [], [], []
        for county in counties:
            sub = df[(df["county"] == county) & (df["wy_type"] == wt)]
            vals = sub["n_cuttings"].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            sds.append(vals.std() if len(vals) > 1 else 0)
            maxes.append(float(vals.max()) if len(vals) > 0 else 0)
        offset = (j - (n_types - 1) / 2) * bar_width
        label_str = f"{wt} ({_wy_labels_for_type(wt, wys)})"

        # Translucent bar to max height
        ax.bar(
            x + offset, maxes, bar_width * 0.9,
            color=WY_TYPE_COLORS[wt], alpha=0.2, edgecolor="black",
            linewidth=0.3, linestyle="--",
            label=f"Max" if j == 0 else None,
        )
        # Solid bar for mean
        ax.bar(
            x + offset, means, bar_width * 0.9,
            yerr=sds, capsize=2, label=label_str,
            color=WY_TYPE_COLORS[wt], edgecolor="black", linewidth=0.4,
        )
        # Black diamond on mean
        ax.scatter(
            x + offset, means, marker="D", color="black",
            s=14, zorder=5, linewidths=0.3, edgecolors="white",
        )
        for i, county in enumerate(counties):
            summary_data.append({
                "county": county, "wy_type": wt,
                "mean_cuttings": means[i], "max_cuttings": maxes[i],
                "sd": sds[i],
            })

    ax.set_xticks(x)
    ax.set_xticklabels(counties, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of cuttings", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Alfalfa Cuttings by Water Year Type (all counties, WY2019\u20132024)", pad=18, fontsize=11)
    # Re-order legend: WY types first, then max
    handles, labels = ax.get_legend_handles_labels()
    # Move "Max" entry (first handle) to end
    handles = handles[1:] + handles[:1]
    labels = labels[1:] + labels[:1]
    ax.legend(
        handles,
        labels,
        fontsize=7,
        ncol=n_types + 1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.002),
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.0,
        handlelength=1.8,
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    summary = {"chart": "wy_type_cuttings_bar", "n_counties": n_counties,
               "types": types_present, "data": summary_data}
    if output_dir:
        save_pub_figure(fig, "wy_type_cuttings_bar", output_dir)
        pd.DataFrame(summary_data).to_csv(
            Path(output_dir).parent / "WY_stats" / "wy_type_cuttings.csv",
            index=False,
        )
    return fig, ax, summary


# -----------------------------------------------------------------------
# Chart 2: Annual ET bar colored by WY type
# -----------------------------------------------------------------------
def wy_type_annual_et_bar(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    et_col: str = "total_et_mm",
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Bar per WY colored by type, mean ET (mm).

    Args:
        df: DataFrame with columns WY, and et_col.
        output_dir: Where to save figure.
        et_col: Column name for ET in mm.

    Returns:
        (fig, ax, summary_dict)
    """
    apply_style()
    df = add_wy_type_columns(df.copy())
    wys = sorted(df["WY"].unique())

    agg = df.groupby("WY").agg(
        mean_et=(et_col, "mean"),
        sd_et=(et_col, "std"),
        n=(et_col, "count"),
    ).reindex(wys)

    fig, ax = plt.subplots(figsize=(4.58, 3.55))
    colors = [get_wy_color(wy) for wy in wys]
    x = np.arange(len(wys))

    ax.bar(
        x, agg["mean_et"], yerr=agg["sd_et"], capsize=3,
        color=colors, edgecolor="black", linewidth=0.5,
    )

    ax.set_xticks(x)
    labels = [f"{wy}\n({get_wy_type(wy)})" for wy in wys]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(x[0] - 0.32, x[-1] + 0.32)
    ax.set_ylabel("Mean segment ET (mm)")
    ax.set_title("Cumulative Segment ET by Water Year (all counties pooled)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    summary_data = []
    for wy in wys:
        row = agg.loc[wy]
        summary_data.append({
            "WY": wy, "wy_type": get_wy_type(wy),
            "mean_et_mm": row["mean_et"], "sd": row["sd_et"], "n": int(row["n"]),
        })

    summary = {"chart": "wy_type_annual_et_bar", "data": summary_data}
    if output_dir:
        save_pub_figure(fig, "wy_type_annual_et_bar", output_dir)
        pd.DataFrame(summary_data).to_csv(
            Path(output_dir).parent / "WY_stats" / "wy_type_annual_et.csv",
            index=False,
        )
    return fig, ax, summary


# -----------------------------------------------------------------------
# Chart 3: Heatmap — county × WY, mean ET
# -----------------------------------------------------------------------
def wy_type_county_et_heatmap(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    et_col: str = "total_et_mm",
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Heatmap: county × WY, mean ET (mm) annotated.

    Args:
        df: DataFrame with columns county, WY, and et_col.
        output_dir: Where to save figure.
        et_col: Column name for ET in mm.

    Returns:
        (fig, ax, summary_dict)
    """
    apply_style()
    counties = [c for c in COUNTY_ORDER if c in df["county"].unique()]
    wys = sorted(df["WY"].unique())

    pivot = df.groupby(["county", "WY"])[et_col].mean().unstack(fill_value=np.nan)
    pivot = pivot.reindex(index=counties, columns=wys)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    # Annotate cells — show value or "N/A" for missing
    for i in range(len(counties)):
        for j in range(len(wys)):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                txt_color = "white" if val > pivot.values[np.isfinite(pivot.values)].mean() else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color=txt_color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=5, color="grey", fontstyle="italic")

    ax.set_xticks(range(len(wys)))
    wy_labels = [f"{wy}\n({get_wy_type(wy)[0]})" for wy in wys]  # first letter
    ax.set_xticklabels(wy_labels, fontsize=6)
    ax.set_yticks(range(len(counties)))
    ax.set_yticklabels(counties, fontsize=7)
    ax.set_title("Mean Cumulative Segment ET (mm) by County and Water Year")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("ET (mm)", fontsize=7)
    fig.tight_layout()

    summary = {"chart": "wy_type_county_et_heatmap",
               "n_counties": len(counties), "n_wys": len(wys)}
    if output_dir:
        save_pub_figure(fig, "wy_type_county_et_heatmap", output_dir)
        pivot.to_csv(
            Path(output_dir).parent / "WY_stats" / "wy_type_county_et_heatmap.csv",
        )
    return fig, ax, summary


# -----------------------------------------------------------------------
# Chart 4: Actual vs corrected ET by WY type
# -----------------------------------------------------------------------
def wy_type_et_correction_bar(
    df_et: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Grouped bar: WY type, actual vs corrected ET (mm).

    Args:
        df_et: DataFrame with columns WY, ET_open_annual_mm, ET_corr_annual_mm.
        output_dir: Where to save figure.

    Returns:
        (fig, ax, summary_dict)
    """
    apply_style()
    df_et = add_wy_type_columns(df_et.copy())
    wys = sorted(df_et["WY"].unique())
    types_present = get_wy_types_present(wys)

    agg = df_et.groupby("wy_type").agg(
        mean_open=("ET_open_annual_mm", "mean"),
        sd_open=("ET_open_annual_mm", "std"),
        mean_corr=("ET_corr_annual_mm", "mean"),
        sd_corr=("ET_corr_annual_mm", "std"),
        n=("ET_open_annual_mm", "count"),
    ).reindex(types_present)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 3.2))
    x = np.arange(len(types_present)) * 0.78
    w = 0.35

    ax.bar(x - w / 2, agg["mean_open"], w, yerr=agg["sd_open"], capsize=3,
           label="OpenET (actual)", color=WONG_PALETTE[5], edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, agg["mean_corr"], w, yerr=agg["sd_corr"], capsize=3,
           label="Corrected", color=WONG_PALETTE[3], edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    labels = []
    for wt in types_present:
        wy_str = _wy_labels_for_type(wt, wys)
        labels.append(f"{wt}\n({wy_str})")
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(x[0] - 0.32, x[-1] + 0.32)
    ax.set_ylabel("Mean annual ET (mm)")
    ax.set_title("ET Correction by Water Year Type (Method A, all counties)")
    ax.legend(fontsize=6)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    summary_data = []
    for wt in types_present:
        row = agg.loc[wt]
        summary_data.append({
            "wy_type": wt,
            "mean_open_mm": row["mean_open"], "mean_corr_mm": row["mean_corr"],
            "delta_mm": row["mean_open"] - row["mean_corr"],
            "n": int(row["n"]),
        })

    summary = {"chart": "wy_type_et_correction_bar", "data": summary_data}
    if output_dir:
        save_pub_figure(fig, "wy_type_et_correction_bar", output_dir)
        pd.DataFrame(summary_data).to_csv(
            Path(output_dir).parent / "WY_stats" / "wy_type_et_correction.csv",
            index=False,
        )
    return fig, ax, summary


# -----------------------------------------------------------------------
# Chart 5: Savings by WY type × cap
# -----------------------------------------------------------------------
def wy_type_savings_bar(
    df_sav: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Grouped bar: WY type × cap, mean savings (ac-ft/acre).

    Args:
        df_sav: DataFrame with columns WY, saved_mm_cap0, saved_mm_cap1, etc.
        cap_values: Cap levels to plot.
        output_dir: Where to save figure.

    Returns:
        (fig, ax, summary_dict)
    """
    apply_style()
    df_sav = add_wy_type_columns(df_sav.copy())
    wys = sorted(df_sav["WY"].unique())
    types_present = get_wy_types_present(wys)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 3.2))
    n_caps = len(cap_values)
    bar_width = 0.7 / n_caps
    x = np.arange(len(types_present))

    cap_colors = [WONG_PALETTE[i + 1] for i in range(n_caps)]
    summary_data = []

    for j, cap_k in enumerate(cap_values):
        col = f"saved_mm_cap{cap_k}"
        if col not in df_sav.columns:
            continue
        means, sds = [], []
        for wt in types_present:
            sub = df_sav[df_sav["wy_type"] == wt][col].dropna()
            acft = mm_to_acft_per_acre(sub)
            means.append(acft.mean() if len(acft) > 0 else 0)
            sds.append(acft.std() if len(acft) > 1 else 0)
        offset = (j - (n_caps - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width * 0.9,
            yerr=sds, capsize=2,
            label=f"Cap {cap_k} late cuts",
            color=cap_colors[j], edgecolor="black", linewidth=0.4,
        )
        for i, wt in enumerate(types_present):
            summary_data.append({
                "wy_type": wt, "cap": cap_k,
                "mean_savings_acft_per_acre": means[i], "sd": sds[i],
            })

    ax.set_xticks(x)
    labels = [f"{wt}\n({_wy_labels_for_type(wt, wys)})" for wt in types_present]
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Mean savings (ac-ft/acre)")
    ax.set_title("Water Savings by WY Type (actual ET, all counties)")
    ax.legend(fontsize=6)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    summary = {"chart": "wy_type_savings_bar", "data": summary_data}
    if output_dir:
        save_pub_figure(fig, "wy_type_savings_bar", output_dir)
        pd.DataFrame(summary_data).to_csv(
            Path(output_dir).parent / "WY_stats" / "wy_type_savings.csv",
            index=False,
        )
    return fig, ax, summary


# -----------------------------------------------------------------------
# Chart 6: Late-cut prevalence by WY type
# -----------------------------------------------------------------------
def wy_type_late_cut_pct_bar(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Stacked bar: WY type, percentage with 0 / 1 / 2+ late cuts.

    Args:
        df: DataFrame with columns WY, n_late_cuts.
        output_dir: Where to save figure.

    Returns:
        (fig, ax, summary_dict)
    """
    apply_style()
    df = add_wy_type_columns(df.copy())
    wys = sorted(df["WY"].unique())
    types_present = get_wy_types_present(wys)

    # Bin late cuts: 0, 1, 2+
    df["late_bin"] = df["n_late_cuts"].clip(upper=2).astype(int)
    bin_labels = {0: "0 late cuts", 1: "1 late cut", 2: "2+ late cuts"}
    bin_colors = [WONG_PALETTE[3], WONG_PALETTE[1], WONG_PALETTE[6]]

    fig, ax = plt.subplots(figsize=(4.58, 3.55))
    x = np.arange(len(types_present)) * 0.78

    summary_data = []
    bottoms = np.zeros(len(types_present))
    for b_val in [0, 1, 2]:
        pcts = []
        for wt in types_present:
            sub = df[df["wy_type"] == wt]
            total = len(sub)
            count = (sub["late_bin"] == b_val).sum()
            pct = 100.0 * count / total if total > 0 else 0
            pcts.append(pct)
            summary_data.append({
                "wy_type": wt, "late_bin": bin_labels[b_val],
                "pct": pct, "count": int(count), "total": total,
            })
        ax.bar(x, pcts, 0.42, bottom=bottoms, label=bin_labels[b_val],
               color=bin_colors[b_val], edgecolor="black", linewidth=0.4)
        bottoms += np.array(pcts)

    ax.set_xticks(x)
    labels = [f"{wt}\n({_wy_labels_for_type(wt, wys)})" for wt in types_present]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Percentage of parcel-years")
    ax.set_title("Late-Season Cutting Prevalence by WY Type (cutoff Jul 1)", pad=18)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.002),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    ax.set_ylim(0, 105)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    summary = {"chart": "wy_type_late_cut_pct_bar", "data": summary_data}
    if output_dir:
        save_pub_figure(fig, "wy_type_late_cut_pct_bar", output_dir)
        pd.DataFrame(summary_data).to_csv(
            Path(output_dir).parent / "WY_stats" / "wy_type_late_cut_pct.csv",
            index=False,
        )
    return fig, ax, summary


# -----------------------------------------------------------------------
# Chart 7: 2×2 multipanel summary
# -----------------------------------------------------------------------
def wy_type_summary_multipanel(
    df_base: pd.DataFrame,
    df_et: pd.DataFrame,
    df_sav: pd.DataFrame,
    cap_values: Tuple[int, ...] = (0, 1),
    output_dir: Optional[Path] = None,
    et_col: str = "total_et_mm",
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """2×2 panel: (a) cuttings, (b) annual ET, (c) correction, (d) savings.

    Args:
        df_base: DataFrame for cuttings (county, WY, n_cuttings).
        df_et: DataFrame for ET correction (WY, ET_open_annual_mm, ET_corr_annual_mm).
        df_sav: DataFrame for savings (WY, saved_mm_cap0, ...).
        cap_values: Cap levels for savings panel.
        output_dir: Where to save figure.
        et_col: Column for total ET in df_base.

    Returns:
        (fig, axes_array, summary_dict)
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.5))

    df_base_c = add_wy_type_columns(df_base.copy())
    df_et_c = add_wy_type_columns(df_et.copy())
    df_sav_c = add_wy_type_columns(df_sav.copy())

    wys = sorted(set(df_base_c["WY"].unique()) | set(df_et_c["WY"].unique()))
    types_present = get_wy_types_present(wys)

    # --- Panel (a): Cuttings by WY type ---
    ax_a = axes[0, 0]
    agg_cuts = df_base_c.groupby("wy_type").agg(
        mean_cuts=("n_cuttings", "mean"),
        sd_cuts=("n_cuttings", "std"),
    ).reindex(types_present)
    x = np.arange(len(types_present))
    colors = [WY_TYPE_COLORS[wt] for wt in types_present]
    ax_a.bar(x, agg_cuts["mean_cuts"], 0.6, yerr=agg_cuts["sd_cuts"], capsize=3,
             color=colors, edgecolor="black", linewidth=0.4)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(_wy_type_tick_labels(types_present, wys), fontsize=5)
    ax_a.set_ylabel("Mean cuttings")
    ax_a.set_ylim(bottom=0)
    add_panel_label(ax_a, "a")

    # --- Panel (b): Annual ET by WY ---
    ax_b = axes[0, 1]
    wy_sorted = sorted(df_base_c["WY"].unique())
    agg_et = df_base_c.groupby("WY").agg(
        mean_et=(et_col, "mean"),
        sd_et=(et_col, "std"),
    ).reindex(wy_sorted)
    x_b = np.arange(len(wy_sorted))
    colors_b = [get_wy_color(wy) for wy in wy_sorted]
    ax_b.bar(x_b, agg_et["mean_et"], 0.6, yerr=agg_et["sd_et"], capsize=3,
             color=colors_b, edgecolor="black", linewidth=0.4)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([str(wy) for wy in wy_sorted], fontsize=6)
    ax_b.set_ylabel("Mean ET (mm)")
    ax_b.set_ylim(bottom=0)
    add_panel_label(ax_b, "b")

    # --- Panel (c): Actual vs corrected ET ---
    ax_c = axes[1, 0]
    agg_corr = df_et_c.groupby("wy_type").agg(
        mean_open=("ET_open_annual_mm", "mean"),
        sd_open=("ET_open_annual_mm", "std"),
        mean_corr=("ET_corr_annual_mm", "mean"),
        sd_corr=("ET_corr_annual_mm", "std"),
    ).reindex(types_present)
    x_c = np.arange(len(types_present))
    w = 0.3
    ax_c.bar(x_c - w / 2, agg_corr["mean_open"], w, yerr=agg_corr["sd_open"],
             capsize=2, label="OpenET", color=WONG_PALETTE[5], edgecolor="black", linewidth=0.4)
    ax_c.bar(x_c + w / 2, agg_corr["mean_corr"], w, yerr=agg_corr["sd_corr"],
             capsize=2, label="Corrected", color=WONG_PALETTE[3], edgecolor="black", linewidth=0.4)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(_wy_type_tick_labels(types_present, wys), fontsize=5)
    ax_c.set_ylabel("Mean ET (mm)")
    ax_c.legend(fontsize=5, loc="upper right")
    ax_c.set_ylim(bottom=0)
    add_panel_label(ax_c, "c")

    # --- Panel (d): Savings by WY type × cap ---
    ax_d = axes[1, 1]
    n_caps = len(cap_values)
    bw = 0.6 / max(n_caps, 1)
    x_d = np.arange(len(types_present))
    cap_colors = [WONG_PALETTE[i + 1] for i in range(n_caps)]
    for j, cap_k in enumerate(cap_values):
        col = f"saved_mm_cap{cap_k}"
        if col not in df_sav_c.columns:
            continue
        means, sds = [], []
        for wt in types_present:
            sub = df_sav_c[df_sav_c["wy_type"] == wt][col].dropna()
            acft = mm_to_acft_per_acre(sub)
            means.append(acft.mean() if len(acft) > 0 else 0)
            sds.append(acft.std() if len(acft) > 1 else 0)
        offset = (j - (n_caps - 1) / 2) * bw
        ax_d.bar(x_d + offset, means, bw * 0.9, yerr=sds, capsize=2,
                 label=f"Cap {cap_k}", color=cap_colors[j],
                 edgecolor="black", linewidth=0.4)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(_wy_type_tick_labels(types_present, wys), fontsize=5)
    ax_d.set_ylabel("Savings (ac-ft/acre)")
    ax_d.legend(fontsize=5, loc="upper right")
    ax_d.set_ylim(bottom=0)
    add_panel_label(ax_d, "d")

    fig.suptitle("Water Year Type Analysis Summary", fontsize=10, y=1.01)
    fig.tight_layout()

    summary = {"chart": "wy_type_summary_multipanel",
               "panels": ["cuttings", "annual_et", "correction", "savings"]}
    if output_dir:
        save_pub_figure(fig, "wy_type_summary_multipanel", output_dir)
    return fig, axes, summary
