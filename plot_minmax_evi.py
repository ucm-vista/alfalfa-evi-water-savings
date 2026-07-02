"""
Multi-signal EVI diagnostic plot for min/max parcels.

Shows 4 distinct EVI traces alongside BEAST cutting annotations:
1. Raw observations (HLS satellite) — grey scatter
2. Gap-filled (quartic) from CSV — blue dashed
3. SG smoothed from CSV — green solid (**BEAST input**)
4. Whittaker (lambda=100) live — orange solid (interval estimation)

Plus:
- Red vertical lines: matched cutting dates
- Pink dashed vertical lines: BEAST change-points

Title includes base/union minima counts for Tier 3 auditing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "es_analysis"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from es_analysis.data_providers import EviDataProvider, BEASTDataProvider
from es_analysis.data_providers.config import config
from es_analysis.utils import water_year_bounds

BEAST_DIR = Path(config.beast_out_root_new)
OUT_DIR = Path("es_analysis/output/figures/alfalfa_run_4/test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 12 parcels: (county, type, parcel_id, wy)
PARCELS = [
    ("Fresno",   "min", "1032881",  2023),
    ("Fresno",   "max", "1036762",  2020),
    ("Madera",   "min", "2004722",  2019),
    ("Madera",   "max", "2009550",  2021),
    ("Kings",    "min", "1602724",  2024),
    ("Kings",    "max", "1604472",  2022),
    ("Tulare",   "min", "5421929",  2024),
    ("Tulare",   "max", "5408069",  2020),
    ("Merced",   "min", "2403078",  2024),
    ("Merced",   "max", "2405110",  2020),
    ("Imperial", "min", "1306541",  2019),
    ("Imperial", "max", "1304567",  2022),
]


def load_cutting_info(county: str, wy: int, parcel_id: str):
    """Load cutting dates, CPs, and metadata from BEAST CSV."""
    county_dir = county.replace(" ", "_")
    csv_path = BEAST_DIR / county_dir / f"beast_seasonal_cuts_WY{wy}.csv"
    df = pd.read_csv(csv_path)
    df["parcel_id"] = df["parcel_id"].astype(str)
    row = df[df["parcel_id"] == parcel_id]
    if row.empty:
        return [], [], 0, -1

    row = row.iloc[0]
    n_cuts = int(row.get("n_cuttings", 0))
    fallback = int(row.get("fallback_used", -1))

    # Parse matched minima (cutting dates)
    minima_str = str(row.get("matched_minima_iso", ""))
    cut_dates = []
    if minima_str and minima_str != "nan":
        for p in minima_str.split(";"):
            p = p.strip()
            if p:
                try:
                    cut_dates.append(pd.to_datetime(p))
                except Exception:
                    pass

    # Parse BEAST change-point dates
    cp_str = str(row.get("season_cp_dates_iso", ""))
    cp_dates = []
    if cp_str and cp_str != "nan":
        for p in cp_str.split(";"):
            p = p.strip()
            if p:
                try:
                    cp_dates.append(pd.to_datetime(p))
                except Exception:
                    pass

    return cut_dates, cp_dates, n_cuts, fallback


def compute_minima_counts(bp: BEASTDataProvider, sg_series: pd.Series):
    """Re-derive base and union minima counts from SG series."""
    result = bp.build_inclusive_minima(sg_series)
    return len(result["base"]), len(result["union"])


def plot_parcel(county, ptype, parcel_id, wy):
    """Plot 4-signal EVI with cutting dates for a single parcel-year."""
    bp = BEASTDataProvider()
    ep = EviDataProvider()

    # -- Load CSV signals (raw, gap-filled, SG smoothed) --
    try:
        csv_df = bp.load_county_year_csv(county, wy)
    except FileNotFoundError as e:
        print(f"  SKIP {county} {ptype} {parcel_id} WY{wy}: {e}")
        return None

    parcel_df = csv_df[csv_df["parcel_id"].astype(str) == str(parcel_id)].copy()
    if parcel_df.empty:
        print(f"  SKIP {county} {ptype} {parcel_id} WY{wy}: parcel not in CSV")
        return None

    parcel_df = parcel_df.sort_values("date").reset_index(drop=True)
    dates = parcel_df["date"]
    raw_evi = parcel_df["original_mean_evi"]
    gapfilled_evi = parcel_df["gapfilled_mean_evi"]
    sg_evi = parcel_df["smoothed_mean_evi"]

    # -- Compute Whittaker (lambda=100) live --
    # Build a daily_df compatible with smooth_whittaker (needs "mean_evi" column)
    daily_for_whit = pd.DataFrame({"mean_evi": raw_evi.values}, index=dates)
    whit_series = ep.smooth_whittaker(daily_for_whit, lmbda=1e2)
    whit_evi = whit_series.values

    # -- Compute minima counts from SG series --
    sg_series = pd.Series(sg_evi.values, index=dates, dtype=float, name="smoothed_mean_evi")
    # Reindex to daily for build_inclusive_minima
    full_idx = pd.date_range(sg_series.index.min(), sg_series.index.max(), freq="D")
    sg_daily = sg_series.reindex(full_idx)
    n_base, n_union = compute_minima_counts(bp, sg_daily)

    # -- Load cutting info --
    cut_dates, cp_dates, n_cuts, fallback = load_cutting_info(county, wy, parcel_id)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(14, 5.5))

    # 1. Raw observations — grey scatter
    raw_mask = raw_evi.notna()
    ax.scatter(dates[raw_mask], raw_evi[raw_mask],
               s=14, alpha=0.5, color="#999999", zorder=2,
               label="Raw obs (HLS)")

    # 2. Gap-filled (quartic) — blue dashed
    gf_mask = gapfilled_evi.notna()
    ax.plot(dates[gf_mask], gapfilled_evi[gf_mask],
            linewidth=1.2, alpha=0.7, color="#0072B2", linestyle="--", zorder=3,
            label="Gap-filled (quartic) \u2014 CSV")

    # 3. SG smoothed — green solid (BEAST input)
    sg_mask = sg_evi.notna()
    ax.plot(dates[sg_mask], sg_evi[sg_mask],
            linewidth=2.0, alpha=0.9, color="#009E73", zorder=4,
            label="SG smoothed \u2014 \u0042EAST input")

    # 4. Whittaker — orange solid (interval estimation)
    ax.plot(dates, whit_evi,
            linewidth=1.5, alpha=0.85, color="#D55E00", zorder=3,
            label="Whittaker \u03bb=100 \u2014 interval est.")

    # Cutting dates — solid red vertical lines
    for i, cd in enumerate(cut_dates):
        label = f"Cutting dates (n={len(cut_dates)}) [tier {fallback}]" if i == 0 else None
        ax.axvline(cd, color="#CC0000", linewidth=1.8, alpha=0.8,
                   linestyle="-", zorder=5, label=label)

    # BEAST change-points — dashed pink vertical lines
    for i, cpd in enumerate(cp_dates):
        label = f"BEAST CPs (n={len(cp_dates)})" if i == 0 else None
        ax.axvline(cpd, color="#CC79A7", linewidth=1.2, alpha=0.6,
                   linestyle="--", zorder=4, label=label)

    # WY bounds
    start, end = water_year_bounds(wy)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")

    tag = "MIN" if ptype == "min" else "MAX"
    title = (f"{county} | {parcel_id} | WY{wy} | {tag} | "
             f"n_cuts={n_cuts} | fallback={fallback} | "
             f"base={n_base} union={n_union}")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # Y-axis: accommodate all signals
    all_vals = pd.concat([raw_evi, gapfilled_evi, sg_evi], ignore_index=True).dropna()
    ymax = max(0.7, float(all_vals.max()) * 1.15) if len(all_vals) > 0 else 0.7
    ax.set_ylim(-0.05, ymax)

    fig.tight_layout()
    fname = f"{county}_{ptype}_{parcel_id}_WY{wy}_evi_cuttings.png"
    outpath = OUT_DIR / fname
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    for county, ptype, pid, wy in PARCELS:
        print(f"\n{county} {ptype.upper()}: parcel={pid} WY{wy}")
        plot_parcel(county, ptype, pid, wy)
    print("\nDone!")
