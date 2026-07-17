# PlanetScope EVI Validation of Alfalfa Cut Dates

Independent validation of BEAST-detected cutting dates (from HLS EVI) against
higher-cadence **PlanetScope** (~3 m) EVI. For each parcel/cut we sample
PlanetScope median EVI over `cut_date ± 40 d`, locate the EVI trough, and report
the day-offset vs our detected cut date. We validate the **timing** of the
post-cut EVI minimum, not absolute EVI magnitude (PlanetScope and HLS differ
spectrally).

## Cloud-native & quota-safe
- Metadata search (Planet Data API v1 `quick-search`) is **free** — used for all discovery.
- Only the **parcel window** is ever read: activated SR/UDM2 COGs are read via GDAL
  `/vsicurl` clipped to the parcel polygon. No full scenes are downloaded.
- Cloud filter matches the HLS **EVI** pipeline: scene cloud ≤ 50%, parcel clear-pixel ≥ 50%.

## Parcels & cut dates
Auto-selected: 2 parcels each in **San Joaquin** (north), **Kern** (central),
**Imperial** (south) for **WY2022**, cleanest cut cycles (`cut_match_ratio=1.0`,
`n_cuttings` 4–8). Cut dates come from the **same `recover_cut_dates()` pipeline**
that feeds `output/figures/alfalfa_run_6/cuttings_analysis/*/boxplot_year_colored.png`
(sourced from `data/multicounty_matched.parquet`), so they equal the figure's dates.

## Run (from repo root `evi_analysis/`)
```bash
# 1. Select parcels (no API key; writes parcels.csv)
python -m es_analysis.planet_validation.select_parcels

# 2. Smoke test — 1 parcel / 1 cut
PL_API_KEY=<your_key> python -m es_analysis.planet_validation.run_planet_validation --smoke

# 3. Full run — 6 parcels × 2 cuts (12 windows)
PL_API_KEY=<your_key> python -m es_analysis.planet_validation.run_planet_validation
```
The API key is read only from `PL_API_KEY` (never written to disk or logs).

## Outputs
- `parcels.csv` — selected parcels, centroids, AOI, 2 cut dates each.
- `planet_evi_points.csv` — every PlanetScope parcel-EVI point (date, evi_median, clear_frac, scene_id, band_count).
- `figures/png|pdf/<region>_<county>_<UniqueID>_cut<n>.png` — HLS EVI + BEAST cut date vs PlanetScope EVI + trough.
- `validation_summary.csv` — per cycle: our_cut_date, planet_trough_date, offset_days, evi_drop, n_planet_dates; plus an overall row (mean offset, median |offset|, % within ±5 d).

## Modules
- `select_parcels.py` — parcel + cut-date selection.
- `planet_client.py` — Planet Data API search/activate + windowed COG EVI (matches project EVI formula).
- `evi_overlay_plot.py` — per-cycle overlay figure.
- `run_planet_validation.py` — orchestrator (free pre-flight → sample → trough → summary).
