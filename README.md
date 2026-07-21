# Detecting alfalfa cutting timing from satellite to verify late-season water-saving opportunities in California

[![Software DOI](https://img.shields.io/badge/Software%20DOI-10.5281%2Fzenodo.21480209-blue.svg)](https://doi.org/10.5281/zenodo.21480209)
[![Data DOI](https://img.shields.io/badge/Data%20DOI-10.5281%2Fzenodo.21420386-blue.svg)](https://doi.org/10.5281/zenodo.21420386)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible code and final results for a remote-sensing analysis of alfalfa in the
California Central Valley (10 counties, water years 2019–2024). The workflow:

1. **Cutting detection** — detect alfalfa harvest events from HLS **EVI** time series
   using **BEAST** (Bayesian Estimator of Abrupt change, Seasonality and Trend; the
   `Rbeast` package) in a consensus/ensemble configuration.
2. **ET correction** — correct **OpenET** actual-ET for off-phase (post-cutting) periods.
3. **Water-savings simulation** — estimate irrigation water saved by delaying the last
   summer cutting, per parcel-year, with sensitivity and strategy simulations.

Core savings model (see `es_analysis/output/figures/water_savings_sim_2/model_specification.md`):
`S = min(n_remove × e_cut, 0.55 × ET)`, cutoff date July 1; ~10,220 parcel-years,
1,791 parcels, 10 counties.

## Final selected outputs (in this repo)

| Run | What it is |
|-----|-----------|
| `es_analysis/output/figures/alfalfa_run_6/` | Selected full analysis run — cuttings, ET-correction, water-savings tables + figures, WY-type stats, and 18 machine-readable `data/*.parquet`. |
| `es_analysis/output/figures/water_savings_sim_2/` | Selected strategy/sensitivity simulation figures (Fig 5–9) + companion CSVs + 3 markdown method specs. |

## Repository layout

```
es_analysis/            Python package
  data_providers/       data loading/transform (config.py, beast_provider.py, et_provider.py, …)
  charts/               ~70 plotting scripts (evi, beast, et_corrections, statistics, water_savings)
  runners/              entrypoints (run_alfalfa_run_2.py = versioned master, run_water_savings_sim.py, …)
  utils/                shared helpers (publication_style, units, run_output, …)
  planet_validation/    PlanetScope cross-sensor validation of cut dates (+ bundled figures/CSVs)
  output/figures/       the two selected runs (everything else here is git-ignored)
alfalfa_et_gdd5_pipeline.py   post-BEAST orchestrator (CLI: --action …)
run_beast_parallel.py         BEAST ensemble runner (produces beast_outputs_new/)
run_beast_remaining.py, repair_beast_alignment.py, compare_alignment_fix.py, plot_minmax_evi.py
shapefile/              DWR i15 crop-mapping alfalfa parcels (public, bundled)
docs/                   figure_documentation.md
data/                   where external inputs go (see data/README.md)
release_assets/         (git-ignored) staged intermediate-data bundle for Zenodo
```

## Install

```bash
conda env create -f environment.yml    # geospatial stack + pip deps
conda activate alfalfa-evi
# or: pip install -r requirements.txt   (needs system GDAL/PROJ)

cp .env.example .env                    # then edit:
#   OPENET_API_KEY_1=...   (only needed to re-download ET)
#   WORK_ROOT=/home/jovyan/work         (root holding emery_method/, emery_method_2/, shapefile/)
```

All external paths resolve from `$WORK_ROOT` (default `/home/jovyan/work`) — no absolute
paths are baked into the code, and no API key is stored in source.

## Data

The data and the software are archived as **two separate, cross-linked Zenodo records**:

- **Data** (CC-BY-4.0): **[10.5281/zenodo.21420386](https://doi.org/10.5281/zenodo.21420386)** — the two data archives below.
- **Software** (MIT): **[10.5281/zenodo.21480209](https://doi.org/10.5281/zenodo.21480209)** — this repository, archived as a `.zip`.

Both concept DOIs always resolve to their latest version, and each record links to the other and to this GitHub repository.

- **Fast path (recommended)** — fetch the ~207 MB intermediate bundle (BEAST outputs +
  county-year EVI exports + stat CSVs) from Zenodo:
  ```bash
  bash scripts/fetch_intermediate.sh
  ```
  This lets you rerun the downstream analysis without the raw rebuild or the R/BEAST stage.
- **Full processed data (optional, ~2.7 GB)** — the complete per-parcel tree (EVI
  time-series CSVs, per-parcel GeoTIFFs + PNGs, JSON metadata) plus aggregate all-parcel
  EVI time-series and the QC report:
  ```bash
  bash scripts/fetch_full_csv.sh
  ```
- **Repo-only path** — the `water_savings_sim_2` figures regenerate from the committed
  `alfalfa_run_6` outputs alone (no external data needed):
  ```bash
  python -m es_analysis.charts.water_savings.strategy_simulation
  ```
- **Full raw rebuild** — see [`data/README.md`](data/README.md) for provenance and how to
  obtain EVI (1.7 GB), OpenET, Daymet (11 GB), and the DWR crop-mapping / parcel shapefiles.

## Reproduce (three stages)

```bash
# 1. BEAST cutting detection  (slow; needs raw EVI + Rbeast). Skip if you fetched intermediates.
python run_beast_parallel.py --max-concurrent 4 --n-jobs 32
#    -> writes beast_outputs_new/<county>/beast_seasonal_cuts_WY####.csv

# 2. Full analysis run  -> es_analysis/output/figures/alfalfa_run_6/
python alfalfa_et_gdd5_pipeline.py --action run_versioned --run-name alfalfa_run_6 \
       --et-mode actual --class-mode sji

# 3. Simulation figures -> es_analysis/output/figures/water_savings_sim_2/
python -m es_analysis.charts.water_savings.strategy_simulation
python -m es_analysis.charts.water_savings.sensitivity_heatmap
```

## PlanetScope cut-date validation

Independent cross-sensor check of the BEAST cut dates against **PlanetScope** (~3 m, near-daily)
EVI — code in [`es_analysis/planet_validation/`](es_analysis/planet_validation/). Needs a Planet
**Education & Research** API key (`PL_API_KEY`; free for academics). Cloud-native and quota-safe:
free metadata search + windowed cloud-optimised-GeoTIFF reads of **only each parcel** (no full
scenes are downloaded).

```bash
export PL_API_KEY=...     # or add to .env, or es_analysis/planet_validation/.pl_api_key (git-ignored)

# select 6 parcels (2 per county) and sample PlanetScope around each cut trough
python -m es_analysis.planet_validation.select_parcels
python -m es_analysis.planet_validation.run_planet_validation

# the figures + CSVs are bundled; regenerate figures WITHOUT the API from saved data:
python -m es_analysis.planet_validation.replot --all
python -m es_analysis.planet_validation.overview_chart
```

Cut dates are the BEAST **troughs** (`matched_minima_iso`), matching the cuttings analysis. Result
(WY2022, 6 parcels across San Joaquin / Kern / Imperial, 12 cut cycles): PlanetScope reproduces the
cut troughs within **±5 days** (mean |offset| 2.4 d, median 2 d, bias −0.6 d, n=12). See the bundled
`es_analysis/planet_validation/` outputs and Part 5 of
`es_analysis/output/figures/alfalfa_run_6/revisions.md`.

## Notes

- **BEAST/Rbeast**: `Rbeast`'s pip wheel bundles a compiled backend, so no separate R
  install is required. `run_beast_parallel.py` runs BEAST per `(county, water-year)` with
  auto-retry/backoff; it defaults to the `subproc` call mode to avoid native segfaults.
- **Security**: the OpenET and Planet (`PL_API_KEY`) API keys are read from `.env` / the
  environment / a git-ignored key file only — never stored in source. If a key was ever shared,
  **rotate it**.
- **Third-party material**: `Montazar et al. (2026)` is cited in the methods, not
  redistributed.

## Open Research

**Software.** The analysis software developed for this study — *Detecting alfalfa cutting timing
from satellite to verify late-season water-saving opportunities in California* (Version 1.1.0) —
is openly developed on GitHub (https://github.com/ucm-vista/alfalfa-evi-water-savings) under the
**MIT** license and is archived on Zenodo at **https://doi.org/10.5281/zenodo.21480209**
(Sarwar & Silberman, 2026).

**Data.** The reproduction datasets (BEAST cutting-detection outputs, county/water-year EVI
exports, statistics tables, and the full processed per-parcel analysis tree) are archived as a
**separate** Zenodo record under **CC-BY 4.0** at **https://doi.org/10.5281/zenodo.21420386**
(`intermediate_data_v1.tar.gz` and `analysis_full_v1.tar.gz`). Raw satellite inputs (HLS EVI,
Daymet, OpenET) are publicly re-downloadable from their original providers; see
[`data/README.md`](data/README.md) for provenance and access.

The two Zenodo records and this GitHub repository are mutually cross-linked (each record's
related-identifiers point to the other record's DOI and to the repository).

**Licensing.** The **source code** (this repository / the archived `.zip`) is released under the
**MIT license** ([`LICENSE`](LICENSE)); the **data archives** are released under **CC-BY 4.0**.
Software and data are licensed and deposited separately by design, following AGU / Earth's Future
open-research guidance.

## Citation & license

If you use this work, please cite the **software** and the **data** separately (APA):

**Software:**
> Sarwar, A., & Silberman, E. (2026). *Detecting alfalfa cutting timing from satellite to verify late-season water-saving opportunities in California* (Version 1.1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21480209

**Data:**
> Sarwar, A., & Silberman, E. (2026). *Detecting alfalfa cutting timing from satellite to verify late-season water-saving opportunities in California — reproduction data* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21420386

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Code is MIT ([`LICENSE`](LICENSE)).
