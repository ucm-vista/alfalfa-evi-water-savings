#!/usr/bin/env bash
# Download and unpack the intermediate-data bundle (~207 MB): BEAST outputs,
# county-year EVI exports, and stat CSVs needed to rerun the downstream analysis
# without the raw rebuild or the R/BEAST stage.
#
# Zenodo record: https://zenodo.org/records/21420387  (DOI 10.5281/zenodo.21420387)
set -euo pipefail

ROOT="${WORK_ROOT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
URL="${ALFALFA_INTERMEDIATE_URL:-https://zenodo.org/records/21420387/files/intermediate_data_v1.tar.gz?download=1}"
ARCHIVE="$ROOT/release_assets/intermediate_data_v1.tar.gz"

if [[ "$URL" == *XXXXXXX* ]]; then
  echo "ERROR: set the download URL first."
  echo "  Either edit URL in this script, or run:  ALFALFA_INTERMEDIATE_URL=<url> bash scripts/fetch_intermediate.sh"
  exit 1
fi

mkdir -p "$ROOT/release_assets"
echo "Downloading intermediate data (~207 MB) -> $ARCHIVE"
curl -L --fail -o "$ARCHIVE" "$URL"

# Integrity check:
echo "13c72e7deac4947d6e45f394ceb0c15768147275371e88083e6055454bb5665e  $ARCHIVE" | sha256sum -c -

echo "Extracting into $ROOT ..."
# Paths inside the tarball are relative to the repo root.
tar -xzf "$ARCHIVE" -C "$ROOT"
rm -f "$ARCHIVE"
echo "Done. Intermediates restored (beast_outputs_new/, county_year_exports_new/, stat CSVs)."
