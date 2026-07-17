#!/usr/bin/env python3
"""Auto-select validation parcels and their cut dates.

Picks 2 alfalfa parcels each in San Joaquin (north), Kern (central), and
Imperial (south) for WY2022, favouring clean, well-defined cut cycles, and
resolves 2 growing-season cut dates per parcel (best for cloud-free PlanetScope
coverage and a crisp EVI trough). Writes ``parcels.csv``.

Run from the repo root (evi_analysis/):
    python -m es_analysis.planet_validation.select_parcels
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]          # evi_analysis/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from es_analysis.data_providers.config import config
from es_analysis.data_providers.spatial_provider import load_parcels_for_county
from es_analysis.data_providers.et_provider import _load_seasonal_csv, _parse_cp_dates_iso
from es_analysis.data_providers.evi_provider import water_year_bounds

# region label -> county (north / central / south of the study transect)
REGION_COUNTY = {"north": "San Joaquin", "central": "Kern", "south": "Imperial"}
WY = 2022
N_PER_COUNTY = 2
GROW_MONTHS = (4, 5, 6, 7, 8, 9)     # Apr-Sep: cloud-free + strong EVI cycles
MIN_CUT_GAP_DAYS = 30                 # spacing between the 2 chosen cut dates

OUT_DIR = REPO / "es_analysis" / "planet_validation"
# Candidate parcels come from the cached matched dataset behind the
# cuttings_analysis boxplots (same BEAST-16 + three-tier pipeline). For the
# PlanetScope trough validation we then use the TROUGH dates (matched_minima_iso),
# not the peak change-points.
MATCHED_PARQUET = (REPO / "es_analysis" / "output" / "figures" / "alfalfa_run_6"
                   / "data" / "multicounty_matched.parquet")


def _candidate_table() -> pd.DataFrame:
    """Rank candidate parcel-years (WY2022) by cut-cycle cleanliness, using the
    same matched dataset that feeds the cuttings_analysis boxplot figures."""
    base = pd.read_parquet(MATCHED_PARQUET)
    base = base[(base["WY"] == WY)
                & (base["county"].isin(REGION_COUNTY.values()))].copy()
    base["UniqueID"] = base["UniqueID"].astype(str)
    if "cut_match_ratio" not in base.columns:
        base["cut_match_ratio"] = np.nan
    return base[base["n_cuttings"].between(4, 8)]


def _pick_two_cut_dates(cut_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    """Pick 2 well-separated cut dates, preferring the Apr-Sep window."""
    cds = sorted(pd.Timestamp(d).normalize() for d in cut_dates)
    grow = [d for d in cds if d.month in GROW_MONTHS]
    pool = grow if len(grow) >= 2 else cds
    if len(pool) < 2:
        return pool
    # greedy: first, then the farthest date that is >= MIN_CUT_GAP_DAYS later
    first = pool[0]
    later = [d for d in pool if (d - first).days >= MIN_CUT_GAP_DAYS]
    second = later[len(later) // 2] if later else pool[-1]
    return [first, second]


def select_parcels(write: bool = True) -> pd.DataFrame:
    cand = _candidate_table()
    wy_start, wy_end = water_year_bounds(WY)
    rows = []

    for region, county in REGION_COUNTY.items():
        gdf = load_parcels_for_county(county)          # EPSG:4326, [UniqueID, geometry]
        geom_by_uid = dict(zip(gdf["UniqueID"].astype(str), gdf.geometry))
        beast = _load_seasonal_csv(county, WY)

        sub = cand[cand["county"] == county].copy()
        # cleanest first (high match ratio), then more cuts
        sub = sub.sort_values(["cut_match_ratio", "n_cuttings"],
                              ascending=[False, False])

        picked = 0
        for _, r in sub.iterrows():
            uid = str(r["UniqueID"])
            geom = geom_by_uid.get(uid)
            if geom is None or geom.is_empty:
                continue
            brow = beast[beast["UniqueID"] == uid]
            if brow.empty:
                continue
            # TROUGHS only: matched_minima_iso are the physical EVI minima (cut
            # signatures). BEAST change-points also include PEAKS, which we skip.
            troughs = _parse_cp_dates_iso(brow.iloc[0].get("matched_minima_iso"))
            cut_dates = sorted(d.normalize() for d in troughs
                               if wy_start <= d.normalize() <= wy_end)
            two = _pick_two_cut_dates(cut_dates)
            if len(two) < 2:
                continue

            c = geom.centroid
            minx, miny, maxx, maxy = geom.bounds
            rows.append({
                "region": region, "county": county, "UniqueID": uid, "WY": WY,
                "lat": round(c.y, 6), "lon": round(c.x, 6),
                "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
                "n_cuttings": int(r["n_cuttings"]),
                "cut_match_ratio": r.get("cut_match_ratio", np.nan),
                "cut_dates": ";".join(d.strftime("%Y-%m-%d") for d in cut_dates),
                "cut1_date": two[0].strftime("%Y-%m-%d"),
                "cut2_date": two[1].strftime("%Y-%m-%d"),
                "geometry_wkt": geom.wkt,
            })
            picked += 1
            if picked >= N_PER_COUNTY:
                break

        if picked < N_PER_COUNTY:
            print(f"[warn] only {picked}/{N_PER_COUNTY} parcels found for {county}")

    df = pd.DataFrame(rows)
    if write and not df.empty:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "parcels.csv"
        df.to_csv(out, index=False)
        print(f"[saved] {out}  ({len(df)} parcels)")
    return df


if __name__ == "__main__":
    df = select_parcels(write=True)
    cols = ["region", "county", "UniqueID", "lat", "lon",
            "n_cuttings", "cut_match_ratio", "cut1_date", "cut2_date"]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df[cols].to_string(index=False))
