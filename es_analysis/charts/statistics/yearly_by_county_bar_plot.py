"""Yearly boxplot aggregated by county.

Source: alfalfa_evi_jovyan.py line 6658
Imports data from statistics_provider module.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from es_analysis.utils.helper import norm_county_name
from es_analysis.data_providers.config import BEAST_OUT_ROOT

COUNTY_ORDER = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]


def _norm_county_name(name: str) -> str:
    """Normalize county name."""
    s = str(name).replace("_", " ").strip()
    return " ".join(s.split()).title()


def _counties_on_disk() -> List[str]:
    """List counties with data on disk."""
    have = {p.name for p in BEAST_OUT_ROOT.glob("*") if p.is_dir() and list(p.glob("beast_seasonal_cuts_WY*.csv"))}
    return [c for c in COUNTY_ORDER if _norm_county_name(c) in have]


def _detect_years_for_county(county: str) -> List[int]:
    """Find years available for a county."""
    years = []
    for f in (BEAST_OUT_ROOT / _norm_county_name(county)).glob("beast_seasonal_cuts_WY*.csv"):
        tail = f.stem.replace("beast_seasonal_cuts_WY", "")
        yr = "".join(ch for ch in tail if ch.isdigit())
        if len(yr) >= 4:
            years.append(int(yr[:4]))
    return sorted(set(years))


def _load_metric(county: str, year: int, metric: str) -> np.ndarray:
    """Load a column array for county and year."""
    f = BEAST_OUT_ROOT / _norm_county_name(county) / f"beast_seasonal_cuts_WY{year}.csv"
    if not f.exists():
        return np.array([], dtype=float)
    df = pd.read_csv(f)
    col = metric
    if metric == "n_cp_season" and col not in df.columns:
        col = "n_change_points" if "n_change_points" in df.columns else None
    if not col or col not in df.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()


def yearly_by_county_bar_plot(
    metric: str = "n_cuttings",
    years: Optional[Iterable[int]] = None,
    title_prefix: str = "",
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create boxplot by county pooling all years.

    Args:
        metric: Metric to plot (e.g., "n_cuttings", "n_cp_season").
        years: Years to include. If None, auto-detects from disk.
        title_prefix: Prefix for plot title.
        outfile: Output path for saving figure. If None, auto-generates.

    Returns:
        Tuple of (figure, axes, summary_dict).

    Summary dict contains:
        - counties: List of counties included.
        - years: List of years included.
        - metric: Metric plotted.
        - num_parcels: Total number of parcels across all data.
        - outfile: Path where figure was saved.
    """
    counties = _counties_on_disk()
    if not counties:
        print("No counties found in", BEAST_OUT_ROOT)
        return None, None, {}

    if years is None:
        years_set = set()
        for c in counties:
            years_set |= set(_detect_years_for_county(c))
        years = sorted(years_set)
    else:
        years = sorted({int(y) for y in years})
    if not years:
        print("No years found.")
        return None, None, {}

    data_by_county = {}
    total_parcels = 0
    for c in counties:
        arrs = []
        for y in years:
            arr = _load_metric(c, y, metric)
            if arr.size:
                arrs.append(arr)
        if arrs:
            data_by_county[c] = np.concatenate(arrs)
            total_parcels += len(data_by_county[c])

    if not data_by_county:
        print("No data available.")
        return None, None, {}

    labels = list(data_by_county.keys())
    values = list(data_by_county.values())

    fig, ax = plt.subplots(figsize=(11, 5))

    cmap = plt.cm.tab20.colors
    color_map = {c: cmap[i % len(cmap)] for i, c in enumerate(COUNTY_ORDER)}

    for i, (label, val) in enumerate(zip(labels, values)):
        col = color_map.get(label, "#777777")
        bp = ax.boxplot(
            [val],
            positions=[i + 1],
            vert=True,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            widths=0.6,
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=col, alpha=0.30, edgecolor=col)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=col)
        mu = float(np.mean(val))
        ax.scatter([i + 1], [mu], s=28, c=[col], zorder=5)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right")

    legend_handles = [
        Patch(
            facecolor=color_map[l],
            edgecolor=color_map[l],
            alpha=0.30,
            label=l,
        )
        for l in labels
    ]
    ax.legend(
        handles=legend_handles,
        title="County",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
    )

    ylabel_text = "Cuttings" if metric == "n_cuttings" else "Seasonal change points"
    ax.set_ylabel(ylabel_text)
    ttl = title_prefix or "All years pooled"
    ax.set_title(f"{ttl}: by county — {metric}")
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    if outfile is None:
        fname = f"box_by_county_{metric}.png"
        outfile = BEAST_OUT_ROOT / fname
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.show()
    plt.close(fig)

    summary = {
        "counties": counties,
        "years": years,
        "metric": metric,
        "num_parcels": total_parcels,
        "outfile": str(outfile),
    }
    print("Saved:", outfile)
    return fig, ax, summary