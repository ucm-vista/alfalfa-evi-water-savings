# Release assets — Zenodo data record

The archives below (git-ignored; only this manifest is tracked) are published on **Zenodo**:

- **Record:** https://zenodo.org/records/21420387
- **DOI:** [10.5281/zenodo.21420387](https://doi.org/10.5281/zenodo.21420387)
- **Status:** draft (download links go live once the record is published on zenodo.org)

Fetch helpers: `scripts/fetch_intermediate.sh` (bundle 1) and `scripts/fetch_full_csv.sh` (bundle 2).

## Bundle 1 — Intermediate-data bundle

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

## Bundle 2 — Full processed-analysis archive

| Field | Value |
|-------|-------|
| File | `analysis_full_v1.tar.gz` |
| Size | 2.7 GB compressed (~4.2 GB uncompressed) |
| SHA256 | `299f69d72fbd238d8da81ad50ea7f1604ff4bdc2d49770e611d34e522ac0cf8a` |
| Unpack | `tar -xzf analysis_full_v1.tar.gz -C <repo root>` |
| Fetch helper | `scripts/fetch_full_csv.sh` |

### Contents (top-level, relative to repo root)

| Item | Size | Role |
|------|------|------|
| `processed_data/` | 3.5 GB | per-parcel processed tree: EVI time-series CSVs (2,036), GeoTIFFs (2,036), PNG renders (6,103), JSON metadata (2,036), across 36 counties |
| `reports/` | 175 MB | aggregate all-parcel EVI time-series CSV, per-county xlsx, charts PDF |
| `all_parcels_evi_timeseries.csv` | 142 MB | aggregate all-parcel EVI time-series |
| `combined_evi_timeseries.csv` | 134 MB | combined EVI time-series |
| `failure_report.csv` | 244 MB | EVI-extraction QC / failure report |

`.ipynb_checkpoints/` are excluded. Optional — not needed for the core reproduction.

## Not included (raw inputs — documented only)

See `data/README.md`: EVI (1.7 GB), OpenET, Daymet (11 GB), DWR crop-mapping / parcel
shapefiles. Place them under `$WORK_ROOT` (default `/home/jovyan/work`).
