#!/usr/bin/env python3
"""Cloud-native PlanetScope access for parcel EVI (no full-scene downloads).

Uses the Planet Data API v1 for FREE metadata search, then activates only the
needed Surface-Reflectance + UDM2 assets and reads ONLY the parcel window via a
GDAL ``/vsicurl`` windowed COG read. Per-parcel median EVI is computed with the
project's EVI formula, masking clouds/shadows with UDM2.

Auth: ``PL_API_KEY`` environment variable (never hard-coded / logged).

EVI (matches reporting/process_vi_turbo.py):
    EVI = 2.5*(NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1)   on reflectance (DN*1e-4)
Band layout (auto-detected by band count):
    4-band ortho_analytic_4b_sr : Blue=1, Green=2, Red=3, NIR=4
    8-band ortho_analytic_8b_sr : Blue=2, Red=6, NIR=8
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
import shapely.wkt
from shapely import force_2d
from shapely.geometry import mapping


def _load_geom(geom_wkt: str):
    """Parse WKT and drop any Z coordinate (Planet + masking need 2D)."""
    return force_2d(shapely.wkt.loads(geom_wkt))

SEARCH_URL = "https://api.planet.com/data/v1/quick-search"
ITEM_TYPE = "PSScene"
SR_ASSETS = ["ortho_analytic_4b_sr", "ortho_analytic_8b_sr"]
UDM_ASSET = "ortho_udm2"
SR_SCALE = 1e-4                       # SR DN -> reflectance
BAND_MAP = {                          # 1-based band indices: (blue, red, nir)
    "ortho_analytic_4b_sr": (1, 3, 4),
    "ortho_analytic_8b_sr": (2, 6, 8),
}
_GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIPLEX="YES",
    CPL_VSIL_CURL_USE_HEAD="NO",
    VSI_CACHE="TRUE",
)


def api_key() -> str:
    """Planet key from PL_API_KEY env, else a gitignored local .pl_api_key file."""
    k = os.environ.get("PL_API_KEY", "").strip()
    if not k:
        kf = Path(__file__).with_name(".pl_api_key")
        if kf.exists():
            k = kf.read_text().strip()
    if not k:
        raise RuntimeError(
            "No Planet key: set PL_API_KEY or create planet_validation/.pl_api_key")
    return k


def _session() -> requests.Session:
    s = requests.Session()
    s.auth = (api_key(), "")
    return s


# --------------------------------------------------------------------------
# FREE metadata search
# --------------------------------------------------------------------------
def search_scenes(geom_geojson: dict, start: str, end: str,
                  cloud_max: float = 0.5, max_pages: int = 4,
                  sess: Optional[requests.Session] = None) -> List[dict]:
    """Return PSScene features intersecting geom in [start,end] with cloud<=max.

    Dates are 'YYYY-MM-DD'. This is metadata-only and consumes NO download quota.
    """
    sess = sess or _session()
    body = {
        "item_types": [ITEM_TYPE],
        "filter": {"type": "AndFilter", "config": [
            {"type": "GeometryFilter", "field_name": "geometry", "config": geom_geojson},
            {"type": "DateRangeFilter", "field_name": "acquired",
             "config": {"gte": f"{start}T00:00:00Z", "lte": f"{end}T23:59:59Z"}},
            {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": cloud_max}},
        ]},
    }
    feats, url, page = [], SEARCH_URL, 0
    r = sess.post(url, json=body, timeout=60)
    r.raise_for_status()
    while True:
        js = r.json()
        feats.extend(js.get("features", []))
        nxt = js.get("_links", {}).get("_next")
        page += 1
        if not nxt or page >= max_pages:
            break
        r = sess.get(nxt, timeout=60)
        r.raise_for_status()
    return feats


def choose_scenes(features: List[dict], cut_date: pd.Timestamp,
                  max_scenes: int = 14) -> List[dict]:
    """Sample the cut CYCLE densely: alfalfa is cut at the EVI peak, so the
    post-cut trough sits ~2-3 weeks LATER. Prioritise the [cut-10, cut+40] core
    (where the decline + trough live) evenly, then add a couple of context
    scenes at the window edges for the pre-cut baseline / regrowth."""
    if not features:
        return []
    feats = sorted(features, key=lambda f: f["properties"]["acquired"])
    dates = np.array([pd.Timestamp(f["properties"]["acquired"]).tz_localize(None)
                      for f in feats])
    lo = cut_date - pd.Timedelta(days=10)
    hi = cut_date + pd.Timedelta(days=40)
    core = [i for i, d in enumerate(dates) if lo <= d <= hi]
    ctx = [i for i, d in enumerate(dates) if d < lo or d > hi]

    keep = set()
    n_core = min(len(core), max(max_scenes - 2, 1))
    if core:
        sel = np.linspace(0, len(core) - 1, n_core).round().astype(int)
        keep.update(core[j] for j in sel)
    if ctx:                                   # earliest + latest for baseline
        keep.add(ctx[0]); keep.add(ctx[-1])
    return [feats[i] for i in sorted(keep)][:max_scenes]


# --------------------------------------------------------------------------
# Activation + windowed COG read (parcel only)
# --------------------------------------------------------------------------
def _assets(feature: dict, sess: requests.Session) -> dict:
    return sess.get(feature["_links"]["assets"], timeout=60).json()


def _trigger(assets: dict, asset_type: str, sess: requests.Session) -> None:
    """Fire (non-blocking) activation for an asset if it isn't active yet."""
    a = assets.get(asset_type)
    if a is not None and a.get("status") != "active":
        try:
            sess.get(a["_links"]["activate"], timeout=60)
        except Exception:
            pass


def _wait_active(feature: dict, asset_type: str, sess: requests.Session,
                 timeout_s: int = 600) -> Optional[str]:
    """Poll until an asset is active; return its signed location URL."""
    t0 = time.time()
    while True:
        cur = _assets(feature, sess).get(asset_type, {})
        if cur.get("status") == "active":
            return cur.get("location")
        if time.time() - t0 >= timeout_s:
            return None
        time.sleep(6)


def _read_window(url: str, geom_wgs84) -> "tuple[np.ndarray, int]":
    """Windowed COG read clipped to geom; returns (masked bands array, nbands)."""
    with rasterio.Env(**_GDAL_ENV):
        with rasterio.open(f"/vsicurl/{url}") as ds:
            g = transform_geom("EPSG:4326", ds.crs, mapping(geom_wgs84))
            arr, _ = rio_mask(ds, [g], crop=True, filled=False)   # (bands,h,w) masked
            return arr, ds.count


def parcel_median_evi(sr_url: str, udm_url: str, geom_wgs84, sr_asset: str) -> dict:
    """Median EVI over clear parcel pixels from a single scene."""
    sr, nb = _read_window(sr_url, geom_wgs84)
    b_blue, b_red, b_nir = BAND_MAP[sr_asset]
    if nb < max(b_blue, b_red, b_nir):
        return {"evi_median": np.nan, "clear_frac": 0.0, "band_count": nb}

    blue = sr[b_blue - 1].astype("float64") * SR_SCALE
    red = sr[b_red - 1].astype("float64") * SR_SCALE
    nir = sr[b_nir - 1].astype("float64") * SR_SCALE
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        evi = 2.5 * (nir - red) / denom

    # parcel footprint = pixels not masked out by the polygon clip
    inpoly = ~np.ma.getmaskarray(sr[b_red - 1])
    clear = inpoly.copy()
    try:
        udm, _ = _read_window(udm_url, geom_wgs84)
        clear_band = udm[0].astype("float64")           # UDM2 band 1 = clear (1)
        clm = ~np.ma.getmaskarray(udm[0])
        if clear_band.shape == inpoly.shape:
            clear = inpoly & clm & (np.ma.getdata(clear_band) == 1)
    except Exception:
        pass                                            # fall back to polygon-only

    evi = np.ma.getdata(evi)
    valid = clear & np.isfinite(evi) & (evi > -1.0) & (evi < 1.5)
    n_poly = int(inpoly.sum())
    n_clear = int(valid.sum())
    return {
        "evi_median": float(np.median(evi[valid])) if n_clear else np.nan,
        "clear_frac": (n_clear / n_poly) if n_poly else 0.0,
        "band_count": nb,
        "n_pixels": n_poly,
    }


def scene_evi(feature: dict, sr_asset: str, geom_wgs84, sess: requests.Session,
              min_clear_frac: float = 0.5, timeout_s: int = 600) -> Optional[dict]:
    """Wait for SR+UDM2 (already triggered) and return the parcel median EVI row."""
    sr_url = _wait_active(feature, sr_asset, sess, timeout_s=timeout_s)
    if sr_url is None:
        return None
    udm_url = _wait_active(feature, UDM_ASSET, sess, timeout_s=120)
    try:
        res = parcel_median_evi(sr_url, udm_url or "", geom_wgs84, sr_asset)
    except Exception as e:
        print(f"    [skip] read failed {feature['id']}: {e}")
        return None
    if not np.isfinite(res["evi_median"]) or res["clear_frac"] < min_clear_frac:
        return None
    acquired = pd.Timestamp(feature["properties"]["acquired"]).tz_localize(None)
    return {"date": acquired.normalize(), "scene_id": feature["id"],
            "cloud_cover": feature["properties"].get("cloud_cover"),
            "sr_asset": sr_asset, **res}


# --------------------------------------------------------------------------
# High-level: one cut window -> DataFrame of PlanetScope EVI points
# --------------------------------------------------------------------------
def preflight_count(geom_wkt: str, cut_date: pd.Timestamp, window_days: int = 40,
                    cloud_max: float = 0.5,
                    sess: Optional[requests.Session] = None) -> List[dict]:
    """FREE search: return candidate features around a cut date (no quota)."""
    sess = sess or _session()
    geom = _load_geom(geom_wkt)
    gj = mapping(geom)
    start = (cut_date - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (cut_date + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    return search_scenes(gj, start, end, cloud_max=cloud_max, sess=sess)


def collect_window_evi(geom_wkt: str, cut_date: pd.Timestamp, window_days: int = 40,
                       cloud_max: float = 0.5, max_scenes: int = 14,
                       min_clear_frac: float = 0.5,
                       features: Optional[List[dict]] = None,
                       sess: Optional[requests.Session] = None) -> pd.DataFrame:
    """Return a DataFrame of PlanetScope median-EVI points around one cut date."""
    sess = sess or _session()
    geom = _load_geom(geom_wkt)
    if features is None:
        features = preflight_count(geom_wkt, cut_date, window_days, cloud_max, sess)
    chosen = choose_scenes(features, cut_date, max_scenes=max_scenes)

    # Phase 1 — fire all activations up front (they run in parallel on Planet)
    prepared = []
    for f in chosen:
        assets = _assets(f, sess)
        sr_asset = next((a for a in SR_ASSETS if a in assets), None)
        if sr_asset is None:
            continue
        _trigger(assets, sr_asset, sess)
        _trigger(assets, UDM_ASSET, sess)
        prepared.append((f, sr_asset))

    # Phase 2 — wait for each (mostly already active by now) and read the window
    rows = []
    for f, sr_asset in prepared:
        row = scene_evi(f, sr_asset, geom, sess, min_clear_frac=min_clear_frac)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df
