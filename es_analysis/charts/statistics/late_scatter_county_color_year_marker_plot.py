"""Scatter plot: late cuttings vs late-season ET.

Source: alfalfa_evi_jovyan.py line 17415
Colored by county, marked by water year.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from es_analysis.utils.helper import norm_county_name

COUNTY_ORDER = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]

_YEAR_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h"]


def _norm_county_name(name: str) -> str:
    """Normalize county name."""
    s = str(name).replace("_", " ").strip()
    return " ".join(s.split()).title()


def _style_axes_black_border(ax: plt.Axes):
    """Style axes with black borders."""
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1.0)
    ax.tick_params(colors="black")


def _regression_stats_text(x: np.ndarray, y: np.ndarray) -> str:
    """Get regression statistics for text display."""
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 2:
        return f"n = {n}\nr = nan\nR² = nan"
    r = (
        float(np.corrcoef(x, y)[0, 1])
        if (np.std(x) > 0 and np.std(y) > 0)
        else np.nan
    )
    r2 = float(r * r) if np.isfinite(r) else np.nan
    slope = np.nan
    if np.std(x) > 0:
        slope = float(np.polyfit(x, y, 1)[0])
    return f"n = {n}\nslope = {slope:.3g}\nr = {r:.3f}\nR² = {r2:.3f}"


def late_scatter_county_color_year_marker_plot(
    df: pd.DataFrame,
    wy_start: int,
    wy_end: int,
    counties: Optional[Iterable[str]] = None,
    y_col: str = "late_et_mm",
    x_col: str = "n_late_cuts",
    title: str = "Late cuttings vs late-season ET",
    legend_fontsize: int = 9,
    marker_size: float = 45.0,
    alpha: float = 0.78,
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create scatter plot of late cuttings vs late-season ET.

    Points are colored by county and marked by water year.

    Args:
        df: Input DataFrame with columns x_col, y_col, "county", and "WY".
        wy_start: Start water year.
        wy_end: End water year.
        counties: List of counties to include. If None, uses COUNTY_ORDER.
        y_col: Column name for Y-axis values.
        x_col: Column name for X-axis values.
        title: Plot title.
        legend_fontsize: Font size for legend.
        marker_size: Size of markers.
        alpha: Transparency of markers.
        outfile: Output path for saving figure. If None, figure is not saved.

    Returns:
        Tuple of (figure, axes, summary_dict).

    Summary dict contains:
        - wy_start: Start water year.
        - wy_end: End water year.
        - x_col: X column.
        - y_col: Y column.
        - counties: List of counties included.
        - years: List of years included.
        - num_points: Total number of points plotted.
        - correlation: Correlation coefficient between x and y.
        - outfile: Path where figure was saved (None if not saved).
    """
    d = df[df["WY"].between(wy_start, wy_end)].copy()
    
    num_total = len(d)
    num_missing_x = d[x_col].isna().sum()
    num_missing_y = d[y_col].isna().sum()
    
    d = d.dropna(subset=[x_col, y_col, "county", "WY"])
    
    num_valid = len(d)

    if counties is None:
        counties_use = COUNTY_ORDER
    else:
        counties_use = [_norm_county_name(c) for c in counties]
    d = d[d["county"].isin(counties_use)].copy()

    if d.empty:
        print("No data to plot after filtering.")
        return None, None, {}

    unique_counties = [c for c in counties_use if c in d["county"].unique()]
    base_cmap = mpl.colormaps.get_cmap("tab10")
    county_color = {
        c: base_cmap(i / max(1, len(unique_counties) - 1))
        for i, c in enumerate(unique_counties)
    }

    years = sorted(d["WY"].unique().tolist())
    year_marker = {
        wy: _YEAR_MARKERS[i % len(_YEAR_MARKERS)] for i, wy in enumerate(years)
    }

    fig, ax = plt.subplots(figsize=(10.5, 6))

    for wy in years:
        suby = d[d["WY"] == wy]
        for c in unique_counties:
            sub = suby[suby["county"] == c]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col].to_numpy(dtype=float),
                sub[y_col].to_numpy(dtype=float),
                s=marker_size,
                marker=year_marker[wy],
                facecolor=county_color[c],
                edgecolor="0.25",
                linewidth=0.8,
                alpha=alpha,
            )

    ax.set_xlabel("Late cuttings after cutoff (count)")
    ax.set_ylabel("Late-season ET (mm)")
    ax.set_title(title)

    x_vals = d[x_col].to_numpy(dtype=float)
    y_vals = d[y_col].to_numpy(dtype=float)
    correlation = float(np.corrcoef(x_vals, y_vals)[0, 1]) if len(x_vals) > 1 else np.nan

    stats_txt = _regression_stats_text(x_vals, y_vals)
    ax.text(
        0.02,
        0.98,
        stats_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.55, edgecolor="none"),
    )

    county_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=county_color[c],
            markeredgecolor="0.25",
            label=c,
        )
        for c in unique_counties
    ]
    leg1 = ax.legend(
        handles=county_handles,
        title="County",
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        labelspacing=0.35,
        handletextpad=0.4,
    )
    ax.add_artist(leg1)

    year_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=year_marker[wy],
            linestyle="",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="0.25",
            label=f"WY {wy}",
        )
        for wy in years
    ]
    ax.legend(
        handles=year_handles,
        title="Water year",
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 1,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
        labelspacing=0.35,
        handletextpad=0.4,
    )

    _style_axes_black_border(ax)
    fig.tight_layout()

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
        plt.show()
        plt.close(fig)

    summary = {
        "wy_start": wy_start,
        "wy_end": wy_end,
        "x_col": x_col,
        "y_col": y_col,
        "counties": unique_counties,
        "years": years,
        "num_total": num_total,
        "num_missing_x": num_missing_x,
        "num_missing_y": num_missing_y,
        "num_valid": num_valid,
        "num_points": len(d),
        "correlation": correlation,
        "outfile": str(outfile) if outfile else None,
    }
    print(f"Late scatter plot: {len(d)} points, correlation={correlation:.3f}")
    return fig, ax, summary