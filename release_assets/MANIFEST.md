# Release assets — upload-later bundle

Git-ignored (only this manifest is tracked). Upload to Zenodo when ready, then paste the
download URL into `scripts/fetch_intermediate.sh` and record the SHA256 in its check.

## Intermediate-data bundle (BUILT, staged here)

| Field | Value |
|-------|-------|
| File | `intermediate_data_v1.tar.gz` |
| Size | 114 MB compressed (~207 MB uncompressed) |
| SHA256 | `13c72e7deac4947d6e45f394ceb0c15768147275371e88083e6055454bb5665e` |
| Unpack | `tar -xzf intermediate_data_v1.tar.gz -C <repo root>` (== `$DATA_ROOT`, default repo root) |
| Fetch helper | `scripts/fetch_intermediate.sh` |

### Contents (top-level dirs, relative to repo root)

| Dir | Size | Role |
|-----|------|------|
| `beast_outputs_new/` | 15 MB | BEAST cutting-detection results + per-parcel debug JSON |
| `county_year_exports_new/` | 167 MB | per-county/water-year parcel EVI export CSVs (BEAST inputs) |
| `statistics_exports/` | 16 MB | precomputed statistics tables |
| `water_saving_scenarios_stats/` | 4.9 MB | late-cut savings tables (pipeline inputs) |
| `Cutting_weather_stats/` | 3.7 MB | merged cut + weather stats |

`config.py` resolves all five under `$DATA_ROOT` (default = repo root), so unpacking here
makes the downstream analysis runnable without the multi-GB raw rebuild or the R/BEAST stage.

## Not included (raw inputs — documented only)

See `data/README.md`: EVI (1.7 GB), OpenET, Daymet (11 GB), DWR crop-mapping / parcel
shapefiles. Place them under `$WORK_ROOT` (default `/home/jovyan/work`).

## Suggested Zenodo record

One record: the source snapshot (GitHub archive) + `intermediate_data_v1.tar.gz`. Mint a DOI
and fill it into `CITATION.cff` and the README.
