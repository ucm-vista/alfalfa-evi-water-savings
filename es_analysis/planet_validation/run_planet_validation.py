#!/usr/bin/env python3
"""Orchestrate PlanetScope EVI validation of BEAST cut (trough) dates.

Our cut dates are the BEAST **troughs** (matched_minima_iso = the bare-field EVI
minimum right after a harvest). Validation = does independent PlanetScope EVI
show the same trough at the same time? Primary metric: PlanetScope trough date
minus BEAST trough date (≈0 confirms the cut timing).

Quota safe: metadata search is free; only the parcel window is ever read.

Run from repo root (evi_analysis/):
    python -m es_analysis.planet_validation.run_planet_validation --smoke
    python -m es_analysis.planet_validation.run_planet_validation
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from es_analysis.planet_validation import planet_client as pc
from es_analysis.planet_validation.evi_overlay_plot import overlay_figure
from es_analysis.planet_validation.select_parcels import select_parcels

OUT_DIR = REPO / "es_analysis" / "planet_validation"
CYE_ROOT = REPO / "county_year_exports_new"
WY = 2022
TROUGH_HALFWIN = 18          # search PlanetScope/HLS min within +/- this of the BEAST trough
MIN_PREFLIGHT = 4            # min clear scenes before spending any quota


def load_hls(county: str, uid: str, wy: int = WY) -> pd.DataFrame:
    f = CYE_ROOT / county / f"WY{wy}.csv"
    d = pd.read_csv(f, parse_dates=["date"])
    d = d[d["parcel_id"].astype(str) == str(uid)]
    return d[["date", "original_mean_evi", "smoothed_mean_evi"]].sort_values("date")


def trough_near(df: pd.DataFrame, val_col: str, cut_date: pd.Timestamp,
                despike: bool = False):
    """Date/value/depth of the EVI minimum within +/-TROUGH_HALFWIN of cut_date.

    despike=True locates the minimum on a 3-pt rolling median (robust to a single
    cloud-contaminated PlanetScope scene) but reports the observed EVI there.
    """
    w = df[(df["date"] >= cut_date - pd.Timedelta(days=TROUGH_HALFWIN))
           & (df["date"] <= cut_date + pd.Timedelta(days=TROUGH_HALFWIN))]
    w = w.dropna(subset=[val_col]).sort_values("date")
    if w.empty:
        return None, np.nan, np.nan
    vals = w[val_col].to_numpy()
    locator = (pd.Series(vals).rolling(3, center=True, min_periods=1).median().to_numpy()
               if (despike and len(vals) >= 3) else vals)
    i = int(np.argmin(locator))
    tdate = pd.Timestamp(w["date"].values[i])
    drop = float(np.nanmax(vals) - vals[i])
    return tdate, float(vals[i]), drop


def main():
    ap = argparse.ArgumentParser(description="PlanetScope EVI validation of cut (trough) dates.")
    ap.add_argument("--window-days", type=int, default=30)   # +/- around the trough
    ap.add_argument("--cloud-max", type=float, default=0.5)  # scene search (HLS evi_cloud_cover_max=50%)
    # per-parcel clear gate: stricter than HLS 0.5 because PlanetScope UDM2 is coarser
    ap.add_argument("--min-clear-frac", type=float, default=0.85)
    ap.add_argument("--max-scenes", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="1 parcel / 1 cut only")
    args = ap.parse_args()

    pc.api_key()                       # fail fast if no key
    sess = pc._session()

    parcels_csv = OUT_DIR / "parcels.csv"
    parcels = (pd.read_csv(parcels_csv) if parcels_csv.exists()
               else select_parcels(write=True))
    if args.smoke:
        parcels = parcels.head(1)

    all_points, summary = [], []
    fig_dir = OUT_DIR / "figures"

    for _, p in parcels.iterrows():
        county, uid, region = p["county"], str(p["UniqueID"]), p["region"]
        hls = load_hls(county, uid)
        cut_list = [p["cut1_date"]] if args.smoke else [p["cut1_date"], p["cut2_date"]]

        for n, cd in enumerate(cut_list, start=1):
            cut_date = pd.Timestamp(cd)         # BEAST trough (EVI minimum)
            print(f"\n=== {region}/{county} {uid} cut{n} trough {cut_date.date()} ===")

            feats = pc.preflight_count(p["geometry_wkt"], cut_date,
                                       window_days=args.window_days,
                                       cloud_max=args.cloud_max, sess=sess)
            print(f"  pre-flight (free): {len(feats)} clear PSScenes in window")
            if len(feats) < MIN_PREFLIGHT:
                print(f"  [skip] < {MIN_PREFLIGHT} scenes")
                summary.append({"region": region, "county": county, "UniqueID": uid,
                                "cut_n": n, "beast_trough_date": cut_date.date(),
                                "status": "insufficient_coverage", "n_planet_dates": len(feats)})
                continue

            planet = pc.collect_window_evi(
                p["geometry_wkt"], cut_date, window_days=args.window_days,
                cloud_max=args.cloud_max, max_scenes=args.max_scenes,
                min_clear_frac=args.min_clear_frac, features=feats, sess=sess)
            print(f"  read {len(planet)} clean PlanetScope parcel-EVI points")
            for _, r in planet.iterrows():
                all_points.append({"region": region, "county": county, "UniqueID": uid,
                                   "cut_n": n, **r.to_dict()})

            # No despike: the clear_frac>=0.85 gate already drops contaminated
            # scenes, and a real cut trough is a sharp V we must NOT smooth away.
            p_tr, _, p_drop = (trough_near(planet.rename(columns={"evi_median": "v"}),
                                           "v", cut_date)
                               if not planet.empty else (None, np.nan, np.nan))
            h_tr, _, _ = trough_near(hls, "smoothed_mean_evi", cut_date)

            offset = None if p_tr is None else int((p_tr - cut_date).days)      # PRIMARY
            hls_off = None if h_tr is None else int((h_tr - cut_date).days)

            name = f"{region}_{county.replace(' ', '_')}_{uid}_cut{n}"
            meta = {"region": region, "county": county, "UniqueID": uid, "cut_n": n}
            overlay_figure(hls, planet, cut_date, p_tr, h_tr, offset, meta, fig_dir, name)

            summary.append({
                "region": region, "county": county, "UniqueID": uid, "cut_n": n,
                "beast_trough_date": cut_date.date(),
                "planet_trough_date": (None if p_tr is None else p_tr.date()),
                "offset_days": offset,
                "hls_check_offset_days": hls_off,
                "planet_evi_drop": (None if not np.isfinite(p_drop) else round(p_drop, 3)),
                "n_planet_dates": len(planet), "status": "ok"})
            print(f"  PlanetScope trough {p_tr.date() if p_tr is not None else None} | "
                  f"offset vs BEAST {offset} d | HLS-check {hls_off} d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if all_points:
        pd.DataFrame(all_points).to_csv(OUT_DIR / "planet_evi_points.csv", index=False)
        print(f"\n[saved] {OUT_DIR / 'planet_evi_points.csv'}")
    sdf = pd.DataFrame(summary)
    if not sdf.empty:
        ok = sdf[(sdf["status"] == "ok") & sdf["offset_days"].notna()].copy()
        if not ok.empty:
            ab = ok["offset_days"].abs()
            sdf = pd.concat([sdf, pd.DataFrame([{
                "region": "ALL", "status": "summary",
                "n_planet_dates": int(ok["n_planet_dates"].sum()),
                "offset_days": round(ok["offset_days"].mean(), 1),
                "beast_trough_date": f"mean|offset|={ab.mean():.1f}d",
                "planet_trough_date": f"median|offset|={ab.median():.0f}d",
                "hls_check_offset_days": f"within±5d={100*(ab <= 5).mean():.0f}%",
            }])], ignore_index=True)
        sdf.to_csv(OUT_DIR / "validation_summary.csv", index=False)
        print(f"[saved] {OUT_DIR / 'validation_summary.csv'}")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print("\n" + sdf.to_string(index=False))


if __name__ == "__main__":
    main()
