#!/usr/bin/env bash
# Download and unpack the full processed-analysis archive (~2.7 GB compressed,
# ~4.2 GB unpacked): the complete per-parcel processed tree — EVI time-series CSVs,
# per-parcel GeoTIFFs + PNG renders, JSON metadata — plus the aggregate all-parcel
# EVI time-series and the QC failure report. Optional; not needed for the core
# reproduction (use fetch_intermediate.sh for that).
#
# Zenodo record: https://zenodo.org/records/21420387  (DOI 10.5281/zenodo.21420387)
set -euo pipefail

ROOT="${WORK_ROOT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
URL="${ALFALFA_FULL_CSV_URL:-https://zenodo.org/records/21420387/files/analysis_full_v1.tar.gz?download=1}"
ARCHIVE="$ROOT/release_assets/analysis_full_v1.tar.gz"

mkdir -p "$ROOT/release_assets"
echo "Downloading full analysis archive (~2.7 GB) -> $ARCHIVE"
curl -L --fail -o "$ARCHIVE" "$URL"

# Integrity check:
echo "299f69d72fbd238d8da81ad50ea7f1604ff4bdc2d49770e611d34e522ac0cf8a  $ARCHIVE" | sha256sum -c -

echo "Extracting into $ROOT ..."
# Paths inside the tarball are relative to the repo root.
tar -xzf "$ARCHIVE" -C "$ROOT"
rm -f "$ARCHIVE"
echo "Done. Restored processed_data/, reports/, and root EVI CSVs."
