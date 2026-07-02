"""Scatter plot of cut dates.

Source:alfalfa_evi_jovyan.py line 6961 (simplified scatter version).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from datetime import timedelta

from es_analysis.utils.helper import norm_county_name, water_year_bounds
from es_analysis.data_providers.config import BEAST_OUT_ROOT

COUNTY_ORDER = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]


def _norm_county_name(name: str) -> str:
    """Normalize county name."""
    s = str(name).replace("_", " ").strip()
    return " ".join(s.split()).title()


def _parse_cp_dates_iso(iso_str: str) -> List[pd.Timestamp]:
    """Parse ISO format dates string into list of timestamps."""
    if pd.isna(iso_str) or not iso_str:
        return []
    dates = []
    for part in iso_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            dt = pd.to_datetime(part, utc=True)
            if dt is not pd.NaT:
                dates.append(dt.tz_localize(None))
        except Exception:
            continue
    return sorted(dates)


def _load_seasonal_csv(county: str, wy: int) -> pd.DataFrame:
    """Load seasonal BEAST cuts CSV for county and water year."""
    p = BEAST_OUT_ROOT / _norm_county_name(county) / f"beast_seasonal_cuts_WY{wy}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "parcel_id" in df.columns and "UniqueID" not in df.columns:
        df["UniqueID"] = df["parcel_id"].astype(str)
    return df


def cut_dates_scatter_plot(
    county: str,
    wy: int,
    filter_n_cp_season: Optional[int] = None,
    unique_ids: Optional[List[str]] = None,
    title_prefix: str = "",
    figsize: Tuple[float, float] = (12, 5),
    marker_size: float = 60.0,
    alpha: float = 0.70,
    jitter_y: float = 0.05,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create scatter plot of cut dates for a county and water year.

    Args:
        county: County name.
        wy: Water year.
        filter_n_cp_season: Filter parcels by specific n_cp_season value.
        unique_ids: List of specific unique IDs to include.
        title_prefix: Prefix for plot title.
        figsize: Figure size (width, height).
        marker_size: Size of markers.
        alpha: Transparency of markers.
        jitter_y: Random vertical jitter (fraction of y-axis).
        outfile: Output path for saving figure. If None, figure is not saved.

    Returns:
        Tuple of (figure, axes, summary_dict).

    Summary dict contains:
        - county: County name.
        - wy: Water year.
        - filter_n_cp_season: Filter applied (None if not filtered).
        - num_parcels: Number of parcels included.
        - total_cut_dates: Total number of cut dates plotted.
        - min_cut_date: Earliest cut date.
        - max_cut_date: Latest cut date.
        - outfile: Path where figure was saved (None if not saved).
    """
    county_norm = _norm_county_name(county)
    start, end = water_year_bounds(wy)

    df = _load_seasonal_csv(county_norm, wy)

    if df.empty:
        print(f"No data for {county_norm} WY{wy}")
        return None, None, {}

    if filter_n_cp_season is not None:
        m = (
            pd.to_numeric(df["n_cp_season"], errors="coerce")
            == int(filter_n_cp_season)
        )
        df = df.loc[m].copy()

    if unique_ids:
        want = set(map(str, unique_ids))
        df = df[df["UniqueID"].astype(str).isin(want)].copy()

    if df.empty:
        print("No parcels match the selection.")
        return None, None, {}

    num_parcels = len(df)

    all_cut_dates = []
    for uid, sub in df.groupby("UniqueID", sort=False):
        iso_dates = (
            sub["season_cp_dates_iso"].iloc[0]
            if "season_cp_dates_iso" in sub
            else ""
        )
        lst = _parse_cp_dates_iso(iso_dates)
        lst = [d for d in lst if (d >= start) and (d <= end)]
        if lst:
            all_cut_dates.extend([(d, uid) for d in lst])

    if not all_cut_dates:
        print("No cut dates within the water year.")
        return None, None, {}

    cut_dates, uids = zip(*all_cut_dates)
    cut_dates = pd.to_datetime(cut_dates, errors="coerce")

    min_cut_date = cut_dates.min()
    max_cut_date = cut_dates.max()

    fig, ax = plt.subplots(figsize=figsize)

    np.random.seed(42)
    jitter = np.random.uniform(-jitter_y, jitter_y, size=len(cut_dates))
    y_values = 0.5 + jitter

    cmap = plt.cm.tab20.colors
    uid_to_idx = {uid: i % len(cmap) for i, uid in enumerate(sorted(set(uids)))}
    colors = [cmap[uid_to_idx[uid]] for uid in uids]

    ax.scatter(cut_dates, y_values, s=marker_size, c=colors, alpha=alpha, edgecolors="0.25", linewidth=0.5)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")
    
    ax.set_xlim(start - timedelta(days=5), end + timedelta(days=5))
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%b\n%Y"))
    
    ax.set_xlabel(f"Water Year {wy} ({start.date()} → {end.date()})")
    
    title_parts = [county_norm, f"WY {wy}: seasonal cut dates"]
    if filter_n_cp_season is not None:
        title_parts.append(f"(n_cp_season={filter_n_cp_season})")
    ax.set_title(f"{title_prefix} {' | '.join(title_parts)}")
    
    ax.grid(axis="x", alpha=0.15)
    
    fig.tight_layout()

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
        plt.show()
        plt.close(fig)

    summary = {
        "county": county_norm,
        "wy": wy,
        "filter_n_cp_season": filter_n_cp_season,
        "num_parcels": num_parcels,
        "total_cut_dates": len(all_cut_dates),
        "min_cut_date": min_cut_date.isoformat() if pd.notna(min_cut_date) else None,
        "max_cut_date": max_cut_date.isoformat() if pd.notna(max_cut_date) else None,
        "outfile": str(outfile) if outfile else None,
    }
    print(f"Cut dates scatter: {len(all_cut_dates)} cut dates from {num_parcels} parcels")
    return fig, ax, summary