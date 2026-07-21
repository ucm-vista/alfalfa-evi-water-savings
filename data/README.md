# Data directory

Large inputs are not committed. Everything resolves from `$WORK_ROOT` (default
`/home/jovyan/work`), which must contain `emery_method/`, `emery_method_2/`, and
`shapefile/`. Set `WORK_ROOT` in `.env` if your layout differs.

**Data license.** The Zenodo-archived data products (`intermediate_data_v1.tar.gz`,
`analysis_full_v1.tar.gz`) are released under **CC-BY 4.0**. The analysis *code* is licensed
separately under MIT (see the repository [`LICENSE`](../LICENSE)).

## Two tiers

### 1. Intermediate data (~207 MB) — Zenodo bundle (recommended)

Regenerates the downstream analysis without the raw rebuild or the R/BEAST stage.

```bash
bash scripts/fetch_intermediate.sh
```

Unpacks into the repo root:

| Component | Size | Role |
|-----------|------|------|
| `beast_outputs_new/` | 15 MB | BEAST cutting-detection results (`beast_seasonal_cuts_WY####.csv` + per-parcel debug JSON) |
| `county_year_exports_new/` | 167 MB | per-county/water-year parcel EVI export CSVs (BEAST inputs) |
| `water_saving_scenarios_stats/` | 4.9 MB | late-cut savings tables (pipeline inputs) |
| `Cutting_weather_stats/` | 4.2 MB | merged cut + weather stats |
| `statistics_exports/` | 16 MB | precomputed statistics tables |

### 2. Raw inputs (multi-GB) — obtain from source, place under `$WORK_ROOT`

| Data | Expected path (under `$WORK_ROOT`) | Size | Source / how to get |
|------|-----------------------------------|------|---------------------|
| Primary EVI time series | `emery_method_2/earthdata_downloader/vi_results/combined_evi_timeseries.csv` | 1.7 GB | HLS EVI extraction (NASA Earthdata / `earthdata_downloader`); or override with `EVI_CSV` |
| OpenET actual ET | `emery_method/reporting/actual_ET_new/` and `actual_ET_api/` | ~175 MB | OpenET API via `es_analysis/runners/run_openet_download.py` (needs `OPENET_API_KEY_1`) |
| ETo/ETof Landsat passes | `emery_method/reporting/ETo_ETof_LandSat_passes/` | 162 MB | OpenET/Landsat overlap export |
| Daymet weather (GDD) | `emery_method/reporting/daymet_alfalfa/` | 11 GB | Daymet (ORNL DAAC) for the parcel footprints |
| Parcel + county shapefiles | `emery_method/shapefile/` | 35 MB | derived parcel/county boundaries |
| DWR i15 crop mapping (alfalfa) | bundled in this repo at `shapefile/` | 7.4 MB | CA DWR i15 Crop Mapping (public) |

The bundled `shapefile/` (i15 crop-mapping alfalfa parcels, `*_34ac.*`) is public DWR data
included for convenience. The larger `emery_method/shapefile/` parcel/county boundaries are
external.
