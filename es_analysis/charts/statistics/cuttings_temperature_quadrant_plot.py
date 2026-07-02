"""Cuttings vs. temperature quadrant scatter (standalone figure).

Companion to ``spatial_ridgeline_plots.py`` (which is left untouched). Shows how the
number of cuttings varies along a temperature gradient across the study counties as a
1x2 quadrant scatter (one panel for mean air temperature, one for max air temperature).

To avoid the heavy overplotting that occurs when every county-WY point collapses onto
an integer cuttings row, each county is summarized by a single marker at its
multi-year mean, with whiskers showing the water-year range (min-max) in both
temperature (horizontal) and number of cuttings (vertical). Dashed crosshairs at the
median temperature and the median (whole-number) cuttings split each panel into four
labeled quadrants.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.spatial_provider import COUNTY_ORDER
from es_analysis.utils.publication_style import (
    DOUBLE_COL_WIDTH,
    apply_style,
    save_pub_figure,
    add_panel_label,
)

# (metric prefix, panel title, x-axis label) for the two panels
_COL_CFG = [
    ("tmean", "Mean air temperature", "Mean air temperature (°C)"),
    ("tmax", "Max air temperature", "Max air temperature (°C)"),
]


def _style_axes_black_border(ax: plt.Axes) -> None:
    """Black full border, no grid (matches the other statistics scatter charts)."""
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.8)
    ax.tick_params(colors="black")


def _corr_text(x: np.ndarray, y: np.ndarray) -> str:
    """Compact n / slope / r annotation (kept for reuse by companion charts)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 2:
        return f"n = {n}"
    r = (
        float(np.corrcoef(x, y)[0, 1])
        if (np.std(x) > 0 and np.std(y) > 0)
        else np.nan
    )
    slope = float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else np.nan
    return f"n = {n}\nslope = {slope:.2f}\nr = {r:.2f}"


def _build_or_load_temp_df(run_name: str = "alfalfa_run_6") -> pd.DataFrame:
    """Load the cached per-parcel-year cuttings+temperature df, or build & cache it.

    Builds ``tmax_mean`` and ``tmin_mean`` (mean daily Daymet temperature over each
    parcel's cut-cycle segments, same windowing used for ``gdd5_mean``), then derives
    ``tmean_mean = (tmax_mean + tmin_mean) / 2``. Result is cached as
    ``data/cuttings_temperature.parquet`` for fast re-renders.
    """
    from es_analysis.utils.run_output import load_dataframe, save_dataframe

    cached = load_dataframe("cuttings_temperature", run_name)
    if cached is not None:
        print(f"[load] cuttings_temperature cache: {len(cached)} rows")
        return cached

    from es_analysis.data_providers.parcel_summary_provider import (
        build_multicounty_df_parcel_year,
    )

    print("[build] computing tmax_mean per parcel-year (all counties x WYs)...")
    dtmax = build_multicounty_df_parcel_year(
        daymet_var="tmax", month_start=10, month_end=9, wy_start=2019, wy_end=2024,
    )
    print("[build] computing tmin_mean per parcel-year (all counties x WYs)...")
    dtmin = build_multicounty_df_parcel_year(
        daymet_var="tmin", month_start=10, month_end=9, wy_start=2019, wy_end=2024,
    )

    keys = ["UniqueID", "county", "WY"]
    merged = dtmax.merge(dtmin[keys + ["tmin_mean"]], on=keys, how="inner")
    merged["tmean_mean"] = (merged["tmax_mean"] + merged["tmin_mean"]) / 2.0
    out = merged[keys + ["n_cuttings", "tmax_mean", "tmin_mean", "tmean_mean"]].copy()

    save_dataframe(out, "cuttings_temperature", run_name)
    print(f"[build] cached cuttings_temperature ({len(out)} rows)")
    return out


# Short county labels for the dense Central-Valley cluster.
_COUNTY_ABBR = {
    "San Joaquin": "SJ", "Stanislaus": "Stan", "Merced": "Mer", "Madera": "Mad",
    "Fresno": "Fre", "Tulare": "Tul", "Kings": "Kin", "Kern": "Ker",
    "Riverside": "Riv", "Imperial": "Imp",
}


def cuttings_temperature_quadrant(
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
    wy_start: int = 2019,
    wy_end: int = 2024,
    show_labels: bool = False,
) -> Tuple[plt.Figure, Dict]:
    """1x2 quadrant scatter of county-mean cuttings vs. temperature, with WY whiskers.

    Args:
        df: Per-parcel-year frame with columns ``county``, ``WY``, ``n_cuttings``,
            ``tmax_mean``, ``tmean_mean`` (from :func:`_build_or_load_temp_df`).
        out_dir: Output directory (``png/`` + ``pdf/`` subdirs are created).
        wy_start: First water year (inclusive).
        wy_end: Last water year (inclusive).
        show_labels: Annotate each marker with a short county abbreviation.

    Returns:
        Tuple of (figure, summary_dict).
    """
    apply_style()

    d = df[df["WY"].between(wy_start, wy_end)].copy()
    d = d.dropna(subset=["county", "WY", "n_cuttings", "tmax_mean", "tmean_mean"])

    counties = [c for c in COUNTY_ORDER if c in d["county"].unique()]
    cmap = mpl.colormaps.get_cmap("tab10")
    county_color = {c: cmap(i % 10) for i, c in enumerate(counties)}

    # Per county-WY means; cuttings rounded to whole numbers (no fractional cuttings).
    g = d.groupby(["county", "WY"], as_index=False).agg(
        n_cuttings=("n_cuttings", "mean"),
        tmean=("tmean_mean", "mean"),
        tmax=("tmax_mean", "mean"),
    )
    g["n_cuttings"] = g["n_cuttings"].round().astype(int)

    # Per-county summary: center = mean over WYs rounded to a whole number of cuttings,
    # whiskers = WY range (min..max). Cuttings stay whole integers everywhere.
    csum = (
        g.groupby("county")
        .agg(
            tmean_c=("tmean", "mean"), tmean_lo=("tmean", "min"), tmean_hi=("tmean", "max"),
            tmax_c=("tmax", "mean"), tmax_lo=("tmax", "min"), tmax_hi=("tmax", "max"),
            cut_c=("n_cuttings", "mean"), cut_lo=("n_cuttings", "min"), cut_hi=("n_cuttings", "max"),
        )
        .reindex(counties)
        .reset_index()
    )
    csum["cut_c"] = csum["cut_c"].round().astype(int)

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH * 1.3, DOUBLE_COL_WIDTH * 0.64), sharey=True,
    )

    cut_axis_lo = int(g["n_cuttings"].min())
    cut_axis_hi = int(g["n_cuttings"].max())
    summary: Dict = {"counties": counties, "n_counties": len(counties), "corr": {}}

    for col, (mkey, title, xlabel) in enumerate(_COL_CFG):
        ax = axes[col]
        cc, lo, hi = f"{mkey}_c", f"{mkey}_lo", f"{mkey}_hi"

        # Whiskers first (semi-transparent), so the opaque markers sit clearly on top.
        for _, r in csum.iterrows():
            c = r["county"]
            xe_lo, xe_hi = max(r[cc] - r[lo], 0.0), max(r[hi] - r[cc], 0.0)
            ye_lo = max(r["cut_c"] - r["cut_lo"], 0.0)
            ye_hi = max(r["cut_hi"] - r["cut_c"], 0.0)
            ax.errorbar(
                r[cc], r["cut_c"],
                xerr=[[xe_lo], [xe_hi]], yerr=[[ye_lo], [ye_hi]],
                fmt="none", ecolor=county_color[c], elinewidth=1.0,
                capsize=2.5, capthick=0.9, alpha=0.5, zorder=3,
            )
        # County mean markers on top.
        ax.scatter(
            csum[cc].to_numpy(float), csum["cut_c"].to_numpy(float),
            s=80, c=[county_color[c] for c in csum["county"]],
            edgecolor="black", linewidth=0.8, zorder=5,
        )
        if show_labels:
            mx = float(np.nanmedian(csum[cc].to_numpy(float)))
            my = float(np.nanmedian(csum["cut_c"].to_numpy(float)))
            for _, r in csum.iterrows():
                ox = 6 if r[cc] >= mx else -6
                oy = 6 if r["cut_c"] >= my else -8
                ax.annotate(
                    _COUNTY_ABBR.get(r["county"], r["county"]),
                    (r[cc], r["cut_c"]), textcoords="offset points", xytext=(ox, oy),
                    ha="left" if ox > 0 else "right",
                    va="bottom" if oy > 0 else "top",
                    fontsize=6, color="0.15", zorder=6,
                )

        xv = csum[cc].to_numpy(float)
        yv = csum["cut_c"].to_numpy(float)

        # Quadrant crosshairs: median temperature, median cuttings (whole number).
        x_med = float(np.nanmedian(xv))
        y_med = float(np.round(np.nanmedian(yv)))
        ax.axvline(x_med, ls="--", color="gray", lw=0.9, alpha=0.6, zorder=0)
        ax.axhline(y_med, ls="--", color="gray", lw=0.9, alpha=0.6, zorder=0)

        # Whole-number cuttings ticks, with headroom for the corner labels.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(cut_axis_lo - 0.8, cut_axis_hi + 0.9)

        # Corner quadrant labels.
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        dx = 0.015 * (xhi - xlo)
        dy = 0.02 * (yhi - ylo)
        ax.text(xhi - dx, yhi - dy, "Hot · Many cuts", ha="right", va="top",
                fontsize=6.5, color="0.5", zorder=2)
        ax.text(xlo + dx, yhi - dy, "Cool · Many cuts", ha="left", va="top",
                fontsize=6.5, color="0.5", zorder=2)
        ax.text(xhi - dx, ylo + dy, "Hot · Few cuts", ha="right", va="bottom",
                fontsize=6.5, color="0.5", zorder=2)
        ax.text(xlo + dx, ylo + dy, "Cool · Few cuts", ha="left", va="bottom",
                fontsize=6.5, color="0.5", zorder=2)

        _style_axes_black_border(ax)
        add_panel_label(ax, "ab"[col])
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        if col == 0:
            ax.set_ylabel("Mean cuttings (count)", fontsize=9)

        corr = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) > 1 else float("nan")
        summary["corr"]["ab"[col]] = corr

    n_wy = wy_end - wy_start + 1
    fig.suptitle(
        f"Cuttings along the temperature gradient by county "
        f"({len(counties)} counties, {n_wy} water years: WY{wy_start}–{wy_end})",
        fontsize=11, fontweight="bold", y=1.02,
    )

    # County legend on the right, with a compact whisker-meaning note beneath it.
    county_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=county_color[c], markeredgecolor="black",
               markeredgewidth=0.6, label=c)
        for c in counties
    ]
    fig.legend(handles=county_handles, title="County", frameon=False,
               loc="upper left", bbox_to_anchor=(0.865, 0.92), fontsize=8,
               title_fontsize=9, labelspacing=0.55, handletextpad=0.5)
    fig.text(
        0.868, 0.26,
        "Marker = WY mean\nWhiskers = WY range\n(min–max)",
        ha="left", va="top", fontsize=7, style="italic", color="0.3",
    )

    fig.tight_layout(rect=(0, 0.02, 0.84, 0.95))

    if out_dir is not None:
        save_pub_figure(fig, "cuttings_temperature_quadrant", out_dir)
        summary["out_dir"] = str(out_dir)

    print(f"  cuttings_temperature_quadrant: {len(csum)} county markers; "
          f"corr={summary['corr']}")
    return fig, summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    df = _build_or_load_temp_df("alfalfa_run_6")
    out = (
        Path(__file__).parent.parent.parent
        / "output" / "figures" / "alfalfa_run_6"
        / "cuttings_analysis" / "all_counties"
    )
    fig, summary = cuttings_temperature_quadrant(df, out_dir=out)
    print(f"Summary: {summary}")
