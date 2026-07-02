"""Month by year boxplots for all counties.

Source: alfalfa_evi_jovyan.py line 6555
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


def monthly_by_county_bar_plot(
    metric: str = "n_cuttings",
    years: Optional[Iterable[int]] = None,
    title_prefix: str = "",
    outfile: Optional[Path] = None,
    box_width: float = 0.10,
    county_gap: float = 0.06,
    year_spacing: float = 1.30,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create grouped boxplots by year with counties as groups.

    Args:
        metric: Metric to plot (e.g., "n_cuttings", "n_cp_season").
        years: Years to include. If None, auto-detects from disk.
        title_prefix: Prefix for plot title.
        outfile: Output path for saving figure. If None, auto-generates.
        box_width: Width of each county box in axis units.
        county_gap: Space between counties inside a year.
        year_spacing: Distance between successive year centers.

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

    data_by_year = {y: {} for y in years}
    total_parcels = 0
    for c in counties:
        for y in years:
            arr = _load_metric(c, y, metric)
            if arr.size:
                data_by_year[y][c] = arr
                total_parcels += len(arr)

    nC = len(counties)
    group_width = nC * box_width + (nC - 1) * county_gap
    x_centers = [1 + i * year_spacing for i in range(len(years))]

    fig_w = max(11, 1.1 * len(years) * year_spacing + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    cmap = plt.cm.tab20.colors
    color_map = {c: cmap[i % len(cmap)] for i, c in enumerate(COUNTY_ORDER)}

    for xc, y in zip(x_centers, years):
        left_edge = xc - group_width / 2.0
        for ci, c in enumerate(counties):
            vals = data_by_year[y].get(c)
            if vals is None or not len(vals):
                continue
            xpos = left_edge + ci * (box_width + county_gap) + box_width / 2.0
            bp = ax.boxplot(
                [vals],
                positions=[xpos],
                widths=box_width,
                vert=True,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
            )
            col = color_map.get(c, "#777777")
            for patch in bp["boxes"]:
                patch.set(facecolor=col, alpha=0.30, edgecolor=col)
            for w in bp["whiskers"] + bp["caps"]:
                w.set(color=col)
            mu = float(np.mean(vals))
            ax.scatter([xpos], [mu], s=22, c=[col], zorder=5)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(y) for y in years])

    handles = [
        Patch(
            facecolor=color_map.get(c, "#777777"),
            edgecolor=color_map.get(c, "#777777"),
            alpha=0.30,
            label=c,
        )
        for c in counties
    ]
    ax.legend(
        handles,
        [c for c in counties],
        title="County (N→S)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )

    ax.set_xlabel("Year")
    ylabel_text = "Cuttings" if metric == "n_cuttings" else "Seasonal change points"
    ax.set_ylabel(ylabel_text)
    ttl = title_prefix or "All counties (north → south)"
    ax.set_title(f"{ttl}: grouped by year — {metric}")
    ax.grid(axis="y", alpha=0.15)

    pad = max(0.4, group_width / 2 + 0.2)
    ax.set_xlim(x_centers[0] - pad, x_centers[-1] + pad)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    if outfile is None:
        fname = f"box_by_year_{metric}.png"
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