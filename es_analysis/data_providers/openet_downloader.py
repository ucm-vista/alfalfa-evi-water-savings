"""OpenET REST API downloader for parcel-level ET timeseries.

Downloads ET data one parcel at a time using the /raster/timeseries/polygon
endpoint. Supports daily and monthly intervals, resumable via raw JSON file
existence checks (parallel-safe), and assembles raw JSON into year-partitioned
CSVs matching existing format.
"""

import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from .config import config
from .evi_provider import normalize_county_name
from .spatial_provider import COUNTY_ORDER

logger = logging.getLogger(__name__)

API_URL = "https://openet-api.org/raster/timeseries/polygon"
DATE_START = "2018-01-01"
DATE_END = "2024-12-31"
EXPECTED_DAILY_RECORDS = 2557   # 2018-01-01 to 2024-12-31
EXPECTED_MONTHLY_RECORDS = 84   # 7 years x 12 months

# Retry backoff seconds for 5xx / timeout errors
RETRY_BACKOFFS = [60, 120, 180]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _geometry_to_flat_coords(geom) -> List[float]:
    """Convert a shapely Polygon to flat [lon, lat, ...] for the API.

    Drops Z values and interior rings (holes).
    """
    coords = list(geom.exterior.coords)
    flat = []
    for c in coords:
        flat.append(float(c[0]))  # lon
        flat.append(float(c[1]))  # lat
    return flat


# ---------------------------------------------------------------------------
# Manifest (progress tracking)
# ---------------------------------------------------------------------------

def _manifest_path(download_root: Path) -> Path:
    return download_root / "manifest.json"


def _load_manifest(download_root: Path) -> Dict[str, Any]:
    """Load manifest.json or return empty structure."""
    p = _manifest_path(download_root)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def _save_manifest(download_root: Path, manifest: Dict[str, Any]) -> None:
    """Atomic save of manifest.json via temp file rename."""
    p = _manifest_path(download_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, str(p))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Single-parcel download
# ---------------------------------------------------------------------------

def _download_single_parcel(
    uid: str,
    county: str,
    flat_coords: List[float],
    download_root: Path,
    api_key: str,
    interval: str = "monthly",
    variable: str = "et",
    model: str = "ensemble",
    units: str = "mm",
) -> Tuple[str, Dict[str, Any]]:
    """Download ET timeseries for one parcel.

    Returns:
        (status, info) where status is one of:
        "success", "quota_exceeded", "error"
    """
    raw_dir = download_root / "raw" / interval
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{uid}.json"

    payload = {
        "date_range": [DATE_START, DATE_END],
        "geometry": flat_coords,
        "interval": interval,
        "model": model,
        "variable": variable,
        "reference_et": "gridMET",
        "reducer": "mean",
        "file_format": "JSON",
        "units": units,
    }
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt, backoff in enumerate(RETRY_BACKOFFS + [0]):
        try:
            logger.debug("Request %s (attempt %d)", uid, attempt + 1)
            resp = requests.post(
                API_URL, json=payload, headers=headers, timeout=600,
            )

            if resp.status_code == 200:
                data = resp.json()
                with open(out_path, "w") as f:
                    json.dump(data, f)
                n_records = len(data) if isinstance(data, list) else 0
                return ("success", {
                    "uid": uid, "county": county,
                    "n_records": n_records, "path": str(out_path),
                })

            if resp.status_code == 429:
                logger.warning("HTTP 429 (rate limited) on UID %s", uid)
                return ("quota_exceeded", {
                    "uid": uid, "county": county,
                    "status_code": 429,
                    "detail": resp.text[:500],
                })

            if resp.status_code == 404:
                # Parcel not found — save empty, mark completed
                with open(out_path, "w") as f:
                    json.dump([], f)
                return ("success", {
                    "uid": uid, "county": county,
                    "n_records": 0, "path": str(out_path),
                    "note": "404 — empty result",
                })

            if resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if backoff > 0:
                    logger.warning(
                        "HTTP %d on UID %s, retrying in %ds",
                        resp.status_code, uid, backoff,
                    )
                    time.sleep(backoff)
                    continue
                # Final attempt failed
                return ("error", {
                    "uid": uid, "county": county,
                    "error": last_err,
                })

            # Other 4xx
            return ("error", {
                "uid": uid, "county": county,
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
            })

        except requests.exceptions.Timeout:
            last_err = "Request timeout"
            if backoff > 0:
                logger.warning("Timeout on UID %s, retrying in %ds", uid, backoff)
                time.sleep(backoff)
                continue
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            if backoff > 0:
                logger.warning("Error on UID %s: %s, retrying in %ds", uid, e, backoff)
                time.sleep(backoff)
                continue

    return ("error", {"uid": uid, "county": county, "error": last_err})


# ---------------------------------------------------------------------------
# Parcel queue
# ---------------------------------------------------------------------------

def build_parcel_queue(
    counties: List[str],
    parcel_shp: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Load shapefile, reproject to WGS84, return queue GeoDataFrame.

    Returns:
        GeoDataFrame with columns: UniqueID, COUNTY, geometry, CLASS2, ACRES, MAIN_CROP
    """
    if parcel_shp is None:
        parcel_shp = config.parcel_shp

    gdf = gpd.read_file(parcel_shp)

    if gdf.crs is None:
        gdf.set_crs(epsg=4269, inplace=True)
    gdf = gdf.to_crs(epsg=4326)

    gdf["COUNTY_norm"] = (
        gdf["COUNTY"].astype(str).str.replace("_", " ").str.strip().str.title()
    )
    county_set = {normalize_county_name(c) for c in counties}
    gdf = gdf[gdf["COUNTY_norm"].isin(county_set)].copy()

    gdf["UniqueID"] = gdf["UniqueID"].astype(str)

    # Deduplicate parcels (same UID can appear in multiple years)
    gdf = gdf.drop_duplicates(subset=["UniqueID"]).copy()

    keep = ["UniqueID", "COUNTY_norm", "geometry"]
    for col in ["CLASS2", "ACRES", "MAIN_CROP"]:
        if col in gdf.columns:
            keep.append(col)
    gdf = gdf[keep].copy()
    gdf = gdf.rename(columns={"COUNTY_norm": "COUNTY"})

    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------

def _already_downloaded(download_root: Path, interval: str, uid: str) -> bool:
    """Check if a raw JSON file already exists for this UID (parallel-safe)."""
    jf = download_root / "raw" / interval / f"{uid}.json"
    return jf.exists()


def run_download(
    counties: Optional[List[str]] = None,
    download_root: Optional[Path] = None,
    api_key: Optional[str] = None,
    api_keys: Optional[List[str]] = None,
    max_workers: int = 1,
    max_requests: Optional[int] = None,
    delay: float = 3.0,
    dry_run: bool = False,
    interval: str = "daily",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Download ET for all parcels in given counties.

    Args:
        counties: County names (default: COUNTY_ORDER).
        download_root: Where to store raw JSON + manifest.
        api_key: Single OpenET API key.
        api_keys: Multiple API keys — rotates on 429.
        max_workers: Concurrent download threads (default: 1).
            Each 7-year daily request takes ~66s, and the rate limit is
            20 req/min per key, so you can safely use ~15 workers per key.
        max_requests: Stop after this many API calls.
        delay: Seconds between request submissions per worker.
        dry_run: If True, just report queue size.
        interval: "daily" or "monthly".
        verbose: Extra logging.

    Returns:
        Summary dict with counts.
    """
    if counties is None:
        counties = list(COUNTY_ORDER)
    if download_root is None:
        download_root = config.openet_download_root
    download_root = Path(download_root)

    # Build key list: explicit list > single key > config
    keys: List[str] = []
    if api_keys:
        keys = list(api_keys)
    elif api_key:
        keys = [api_key]
    else:
        keys = [config.openet_api_key]

    queue = build_parcel_queue(counties)

    # File-based skip: check for existing raw JSON (parallel-safe)
    raw_dir = download_root / "raw" / interval
    raw_dir.mkdir(parents=True, exist_ok=True)
    done_uids = {p.stem for p in raw_dir.glob("*.json")}
    todo = queue[~queue["UniqueID"].isin(done_uids)].copy()

    logger.info(
        "Queue: %d total parcels, %d already downloaded, %d to do",
        len(queue), len(done_uids), len(todo),
    )

    if dry_run:
        by_county = todo.groupby("COUNTY").size().to_dict()
        print(f"\n[dry-run] {len(todo)} parcels to download ({interval} interval)")
        print(f"[dry-run] {len(done_uids)} already downloaded (raw JSON exists)")
        print(f"[dry-run] Workers: {max_workers}, Keys: {len(keys)}")
        for c, n in sorted(by_county.items()):
            print(f"  {c}: {n}")
        est_sec = len(todo) / max(max_workers, 1) * 69
        print(f"[dry-run] Estimated time: {est_sec / 3600:.1f} hours")
        return {
            "total": len(queue),
            "completed": len(done_uids),
            "todo": len(todo),
            "by_county": by_county,
        }

    if len(todo) == 0:
        print("All parcels already downloaded.")
        return {"n_success": 0, "n_error": 0, "n_calls": 0, "n_skipped": 0, "stopped": None}

    # Prepare work items: list of (uid, county, flat_coords, assigned_key)
    work_items = []
    for i, (_, row) in enumerate(todo.iterrows()):
        uid = row["UniqueID"]
        county = row["COUNTY"]
        try:
            flat_coords = _geometry_to_flat_coords(row.geometry)
        except Exception as e:
            logger.error("Bad geometry for UID %s: %s", uid, e)
            continue
        # Round-robin key assignment across workers
        assigned_key = keys[i % len(keys)]
        work_items.append((uid, county, flat_coords, assigned_key))

    if max_requests is not None:
        work_items = work_items[:max_requests]

    # Thread-safe counters
    lock = threading.Lock()
    counters = {"success": 0, "error": 0, "skipped": 0, "calls": 0}
    stop_event = threading.Event()
    exhausted_keys = set()

    def _worker(uid, county, flat_coords, assigned_key):
        if stop_event.is_set():
            return

        # Double-check file (another thread may have finished it)
        if _already_downloaded(download_root, interval, uid):
            with lock:
                counters["skipped"] += 1
            return

        status, info = _download_single_parcel(
            uid=uid, county=county, flat_coords=flat_coords,
            download_root=download_root, api_key=assigned_key,
            interval=interval,
        )

        with lock:
            counters["calls"] += 1

            if status == "success":
                counters["success"] += 1
                if verbose:
                    logger.info(
                        "[%d done] %s (%s): %d records",
                        counters["success"], uid, county,
                        info.get("n_records", 0),
                    )
            elif status == "quota_exceeded":
                exhausted_keys.add(assigned_key)
                if len(exhausted_keys) >= len(keys):
                    logger.warning("All API keys exhausted (429). Stopping.")
                    stop_event.set()
                else:
                    # Try with a different key
                    alt_key = [k for k in keys if k not in exhausted_keys][0]
                    logger.warning(
                        "Key exhausted for UID %s, retrying with alternate key.", uid,
                    )
                    s2, i2 = _download_single_parcel(
                        uid=uid, county=county, flat_coords=flat_coords,
                        download_root=download_root, api_key=alt_key,
                        interval=interval,
                    )
                    counters["calls"] += 1
                    if s2 == "success":
                        counters["success"] += 1
                    elif s2 == "quota_exceeded":
                        exhausted_keys.add(alt_key)
                        if len(exhausted_keys) >= len(keys):
                            logger.warning("All API keys exhausted. Stopping.")
                            stop_event.set()
                    else:
                        counters["error"] += 1
            else:
                counters["error"] += 1
                logger.warning("Error on UID %s: %s", uid, info.get("error"))

    # Sequential mode
    if max_workers <= 1:
        for uid, county, flat_coords, assigned_key in work_items:
            if stop_event.is_set():
                break
            _worker(uid, county, flat_coords, assigned_key)
            if delay > 0 and not stop_event.is_set():
                time.sleep(delay)
    else:
        # Threaded mode: submit work with controlled pacing
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for uid, county, flat_coords, assigned_key in work_items:
                if stop_event.is_set():
                    break
                f = pool.submit(_worker, uid, county, flat_coords, assigned_key)
                futures.append(f)
                # Small stagger to avoid burst at start
                time.sleep(0.2)

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error("Worker exception: %s", e)

    stopped = None
    if stop_event.is_set():
        stopped = "all_keys_quota_exceeded" if len(exhausted_keys) >= len(keys) else "quota_exceeded"

    print(
        f"\nDone. {counters['success']} success, {counters['error']} errors, "
        f"{counters['skipped']} skipped, {counters['calls']} API calls."
    )
    if stopped:
        print(f"Stopped early: {stopped}")

    return {
        "n_success": counters["success"],
        "n_error": counters["error"],
        "n_calls": counters["calls"],
        "n_skipped": counters["skipped"],
        "stopped": stopped,
    }


# ---------------------------------------------------------------------------
# Assemble CSVs from raw JSON
# ---------------------------------------------------------------------------

def assemble_csvs(
    download_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    interval: str = "monthly",
) -> Dict[int, Path]:
    """Read raw JSON files and write year-partitioned CSVs.

    Output CSVs have columns: time, UniqueID, CLASS2, ACRES, MAIN_CROP, COUNTY, et
    matching the format expected by _load_openet_for_wy().

    Returns:
        Dict mapping year -> CSV path.
    """
    if download_root is None:
        download_root = config.openet_download_root
    if output_root is None:
        output_root = download_root
    download_root = Path(download_root)
    output_root = Path(output_root)

    raw_dir = download_root / "raw" / interval
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw data at {raw_dir}")

    manifest = _load_manifest(download_root)
    completed_key = f"completed_{interval}"
    completed = manifest.get(completed_key, manifest.get("completed", {}))

    # Load parcel metadata for CLASS2, ACRES, MAIN_CROP
    queue = build_parcel_queue(list(COUNTY_ORDER))
    meta = queue.set_index("UniqueID")[
        [c for c in ["COUNTY", "CLASS2", "ACRES", "MAIN_CROP"] if c in queue.columns]
    ]

    all_records = []
    json_files = sorted(raw_dir.glob("*.json"))
    logger.info("Reading %d JSON files from %s", len(json_files), raw_dir)

    for jf in json_files:
        uid = jf.stem
        try:
            with open(jf) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Bad JSON: %s", jf)
            continue

        if not isinstance(data, list) or len(data) == 0:
            continue

        county = completed.get(uid, {}).get("county", "")
        if not county and uid in meta.index:
            county = meta.loc[uid, "COUNTY"]

        # Get metadata
        class2 = ""
        acres = ""
        main_crop = ""
        if uid in meta.index:
            row = meta.loc[uid]
            class2 = row.get("CLASS2", "")
            acres = row.get("ACRES", "")
            main_crop = row.get("MAIN_CROP", "")

        for rec in data:
            t = rec.get("time", "")
            et_val = rec.get("et", np.nan)
            all_records.append({
                "time": t,
                "UniqueID": uid,
                "CLASS2": class2,
                "ACRES": acres,
                "MAIN_CROP": main_crop,
                "COUNTY": county,
                "et": et_val,
            })

    if not all_records:
        print("No records found.")
        return {}

    df = pd.DataFrame(all_records)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["year"] = df["time"].dt.year

    # If monthly interval, convert monthly ET (mm/month) values as-is
    # (the downstream loader sums daily values, for monthly we keep them directly)

    paths = {}
    for year, ydf in df.groupby("year"):
        year = int(year)
        out_dir = output_root / str(year) / "OpenET Exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "OpenET Exports_api_download.csv"

        # Format time as date string
        ydf = ydf.copy()
        ydf["time"] = ydf["time"].dt.strftime("%Y-%m-%d")
        ydf = ydf.drop(columns=["year"])
        ydf.to_csv(out_path, index=False)
        paths[year] = out_path
        logger.info("Wrote %s (%d rows)", out_path, len(ydf))

    print(f"\nAssembled {len(df)} records into {len(paths)} year CSVs.")
    return paths


# ---------------------------------------------------------------------------
# Validate download completeness
# ---------------------------------------------------------------------------

def validate_download(
    download_root: Optional[Path] = None,
    counties: Optional[List[str]] = None,
    interval: str = "monthly",
) -> pd.DataFrame:
    """Check completeness of downloaded data.

    Returns:
        DataFrame with per-county completeness stats.
    """
    if download_root is None:
        download_root = config.openet_download_root
    if counties is None:
        counties = list(COUNTY_ORDER)
    download_root = Path(download_root)

    expected = EXPECTED_MONTHLY_RECORDS if interval == "monthly" else EXPECTED_DAILY_RECORDS
    raw_dir = download_root / "raw" / interval

    queue = build_parcel_queue(counties)
    manifest = _load_manifest(download_root)
    completed_key = f"completed_{interval}"
    completed = manifest.get(completed_key, manifest.get("completed", {}))

    results = []
    for county in counties:
        county_norm = normalize_county_name(county)
        county_uids = queue[queue["COUNTY"] == county_norm]["UniqueID"].tolist()
        n_total = len(county_uids)
        n_completed = 0
        n_full = 0
        n_partial = 0
        n_empty = 0
        n_missing = 0

        for uid in county_uids:
            jf = raw_dir / f"{uid}.json"
            if jf.exists():
                n_completed += 1
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    n_rec = len(data) if isinstance(data, list) else 0
                    if n_rec >= expected:
                        n_full += 1
                    elif n_rec > 0:
                        n_partial += 1
                    else:
                        n_empty += 1
                except json.JSONDecodeError:
                    n_empty += 1
            else:
                n_missing += 1

        results.append({
            "county": county_norm,
            "total_parcels": n_total,
            "completed": n_completed,
            "full": n_full,
            "partial": n_partial,
            "empty": n_empty,
            "missing": n_missing,
            "pct_complete": round(100 * n_completed / n_total, 1) if n_total > 0 else 0,
        })

    df = pd.DataFrame(results)
    return df


def get_status(download_root: Optional[Path] = None) -> Dict[str, Any]:
    """Quick status summary from manifest."""
    if download_root is None:
        download_root = config.openet_download_root
    download_root = Path(download_root)

    manifest = _load_manifest(download_root)

    status = {}
    for interval in ["monthly", "daily"]:
        ck = f"completed_{interval}"
        fk = f"failed_{interval}"
        n_done = len(manifest.get(ck, {}))
        n_fail = len(manifest.get(fk, {}))
        raw_dir = download_root / "raw" / interval
        n_files = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
        status[interval] = {
            "completed": n_done,
            "failed": n_fail,
            "raw_files": n_files,
        }

    # Legacy keys
    n_legacy = len(manifest.get("completed", {}))
    if n_legacy > 0:
        status["legacy_completed"] = n_legacy

    return status
