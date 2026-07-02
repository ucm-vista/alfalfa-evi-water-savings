"""Debug parcel EVI/ET plot with segment shading.

Two-panel plot showing EVI (top) and ET (bottom) with kept segment
windows shaded and cut dates marked as dashed vertical lines.

Source: alfalfa_evi_jovyan.py lines 10639-10665
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_axes_full_border(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


# ---------------------------------------------------------------------------
# Main chart function
# ---------------------------------------------------------------------------

def debug_parcel_plot(
    evi: pd.Series,
    et: pd.Series,
    cut_dates: List[pd.Timestamp],
    segs_kept: List[Tuple[pd.Timestamp, pd.Timestamp]],
    county: str,
    wy: int,
    unique_id: str,
    evi_mode: str = "smoothed",
    outfile: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Create a debug plot for a single parcel.

    Top panel:    EVI time series with kept segments shaded
    Bottom panel: ET time series with kept segments shaded
    Both panels:  Cut dates shown as dashed vertical lines

    Args:
        evi: EVI time series (pd.Series with DatetimeIndex).
        et: ET time series (pd.Series with DatetimeIndex).
        cut_dates: List of cut date timestamps.
        segs_kept: List of (start, end) segment tuples.
        county: County name.
        wy: Water year.
        unique_id: Parcel unique ID.
        evi_mode: EVI variant label ("smoothed" or "gapfilled").
        outfile: Optional output path for saving figure.

    Returns:
        Tuple of (figure, axes, summary_dict).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # EVI panel
    if evi is not None and not evi.empty:
        axes[0].plot(evi.index, evi.values, linewidth=1.2)
    axes[0].set_ylabel(f"EVI ({evi_mode})")
    axes[0].set_title(
        f"{county} WY{wy} — {unique_id}\n"
        "EVI with segment windows (kept shaded)"
    )

    # ET panel
    if et is not None and not et.empty:
        axes[1].plot(et.index, et.values, linewidth=1.2)
    axes[1].set_ylabel("OpenET ET (mm/day)")
    axes[1].set_title("ET with segment windows (kept shaded)")

    # Shade kept segments
    for s, e in segs_kept:
        axes[0].axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.15)
        axes[1].axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.15)

    # Mark cut dates
    for c in cut_dates:
        axes[0].axvline(
            pd.to_datetime(c), linestyle="--", linewidth=1.0, color="gray"
        )
        axes[1].axvline(
            pd.to_datetime(c), linestyle="--", linewidth=1.0, color="gray"
        )

    _style_axes_full_border(axes[0])
    _style_axes_full_border(axes[1])
    fig.tight_layout()

    summary = {
        "county": county,
        "wy": wy,
        "unique_id": unique_id,
        "evi_mode": evi_mode,
        "n_cut_dates": len(cut_dates),
        "n_segments_kept": len(segs_kept),
    }

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        summary["outfile"] = str(outfile)
        print(f"Debug parcel plot saved: {outfile}")

    return fig, axes, summary
