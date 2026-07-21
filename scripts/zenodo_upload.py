#!/usr/bin/env python3
"""Create a Zenodo DRAFT deposition and upload the data archives.

Reads ZENODO_TOKEN from the environment or the repo-root .env (never printed).
Idempotent: reuses a prior draft recorded in release_assets/.zenodo_draft.json
and skips files already uploaded with a matching checksum. Does NOT publish.

Usage:
    python scripts/zenodo_upload.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

API = "https://zenodo.org/api"
REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "release_assets" / ".zenodo_draft.json"

FILES = [
    REPO / "release_assets" / "intermediate_data_v1.tar.gz",
    REPO / "release_assets" / "analysis_full_v1.tar.gz",
]

METADATA = {
    "upload_type": "software",
    "title": ("Detecting alfalfa cutting timing from satellite to verify "
              "late-season water-saving opportunities in California"),
    "creators": [
        {"name": "Sarwar, Abid", "affiliation": "University of California, Merced"},
        {"name": "Silberman, Emery", "affiliation": "University of California, Merced"},
    ],
    "description": (
        "<p>Software and reproduction data for the alfalfa EVI cutting-detection and "
        "irrigation water-savings analysis over the California Central Valley (10 counties, "
        "water years 2019-2024). Development platform: "
        "<a href=\"https://github.com/ucm-vista/alfalfa-evi-water-savings\">"
        "github.com/ucm-vista/alfalfa-evi-water-savings</a> (MIT license).</p>"
        "<p>Files:</p><ul>"
        "<li><b>alfalfa-evi-water-savings-1.1.0.zip</b> &mdash; the full source repository "
        "(the es_analysis package, BEAST + pipeline drivers, and the two selected result "
        "runs), archived byte-for-byte from the GitHub v1.1.0 release.</li>"
        "<li><b>intermediate_data_v1.tar.gz</b> (~207 MB unpacked) &mdash; BEAST "
        "cutting-detection outputs, per-county/water-year EVI exports, and statistics "
        "tables. The minimum needed to reproduce every downstream figure/table "
        "without the raw rebuild or the R/BEAST stage.</li>"
        "<li><b>analysis_full_v1.tar.gz</b> (~4.2 GB unpacked) &mdash; the full "
        "processed per-parcel analysis tree (EVI time-series CSVs, per-parcel "
        "GeoTIFFs and PNG renders, JSON metadata) plus aggregate all-parcel EVI "
        "time-series and QC report.</li></ul>"
        "<p>Unpack the tarballs with <code>tar -xzf &lt;file&gt; -C &lt;repo root&gt;</code>. "
        "Raw satellite inputs (HLS EVI, Daymet, OpenET) are re-downloadable from source "
        "and are not included. Cutting events are detected with BEAST (Rbeast) on HLS "
        "EVI; ET is from OpenET.</p>"
    ),
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": "1.1.0",
    "language": "eng",
    "keywords": ["alfalfa", "EVI", "BEAST change-point detection", "evapotranspiration",
                 "OpenET", "water savings", "remote sensing", "agriculture",
                 "California Central Valley"],
    "related_identifiers": [
        {"identifier": "https://github.com/ucm-vista/alfalfa-evi-water-savings",
         "relation": "isSupplementTo", "scheme": "url"},
    ],
}


def load_token():
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ZENODO_TOKEN="):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not tok:
        sys.exit("ERROR: ZENODO_TOKEN not found in environment or .env")
    return tok


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tok = load_token()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {tok}"

    # reuse or create the draft
    dep = None
    if STATE.exists():
        dep_id = json.loads(STATE.read_text()).get("id")
        r = s.get(f"{API}/deposit/depositions/{dep_id}")
        if r.status_code == 200 and not r.json().get("submitted", False):
            dep = r.json()
            print(f"Reusing existing draft {dep_id}")
    if dep is None:
        r = s.post(f"{API}/deposit/depositions", json={})
        r.raise_for_status()
        dep = r.json()
        print(f"Created draft {dep['id']}")

    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    STATE.write_text(json.dumps({"id": dep_id, "bucket": bucket}, indent=2))

    # existing files on the deposition (name -> checksum "md5:...")
    existing = {f["filename"]: f.get("checksum", "") for f in dep.get("files", [])}

    for path in FILES:
        name = path.name
        local = md5(path)
        if existing.get(name, "").split(":")[-1] == local:
            print(f"  skip {name} (already uploaded, checksum matches)")
            continue
        size = path.stat().st_size
        print(f"  uploading {name} ({size/1e9:.2f} GB) ...", flush=True)
        with open(path, "rb") as fh:
            r = s.put(f"{bucket}/{name}", data=fh)
        r.raise_for_status()
        remote = r.json().get("checksum", "").split(":")[-1]
        ok = "OK" if remote == local else f"MISMATCH (remote {remote} != local {local})"
        print(f"    uploaded; checksum {ok}")
        if remote != local:
            sys.exit("ERROR: checksum mismatch after upload")

    # metadata
    r = s.put(f"{API}/deposit/depositions/{dep_id}", json={"metadata": METADATA})
    r.raise_for_status()
    dep = r.json()

    doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    html = dep.get("links", {}).get("html", "")
    result = {
        "id": dep_id,
        "reserved_doi": doi,
        "draft_url": html,
        "record_id": str(dep_id),
        "files": [{"name": f["filename"], "size": f["filesize"],
                   "checksum": f.get("checksum", "")} for f in dep.get("files", [])],
        "submitted": dep.get("submitted", False),
        "state": dep.get("state", ""),
    }
    (REPO / "release_assets" / ".zenodo_result.json").write_text(json.dumps(result, indent=2))
    print("\n=== DRAFT READY (not published) ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
