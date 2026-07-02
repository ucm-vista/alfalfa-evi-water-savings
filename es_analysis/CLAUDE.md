# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic analysis pipeline for ~1,791 alfalfa parcels across CA Central/Southern Valley (2019-2024). Two paper arguments: (1) satellite pass timing causes OpenET ET misestimation, (2) shoulder-season cuttings aren't worth the water cost. Pipeline: HLS EVI → BEAST cutting detection → OpenET ET correction → water savings scenarios.

Refactored from a monolithic notebook (`alfalfa_evi_jovyan.py`, ~21K lines) into modular `es_analysis/` package. Hardcoded paths are intentional (JupyterHub environment, not a distributable package).

## Running the Pipeline

Runners are invoked as modules from the repo root (`evi_analysis/`):

```bash
# ET correction statistics (main computation pipeline)
python -m es_analysis.runners.run_et_stats --counties Fresno Kern --wy-start 2019 --wy-end 2024 --n-boot 500 --workers 4

# Cutting weather statistics
python -m es_analysis.runners.run_cutting_weather_stats

# Statistical tests
python -m es_analysis.runners.run_statistical_tests

# Method validation
python -m es_analysis.runners.run_method_validation

# Publication figures
python -m es_analysis.runners.run_publication_figures

# All plot stages in dependency order (EVI → BEAST → ET → Statistics → Multi-panel)
python -m es_analysis.runners.run_all_plots --county Fresno --water-year 2022

# Statistics plots
python -m es_analysis.runners.run_statistics_plots

# Late-water savings workflow
python -m es_analysis.runners.run_late_water_workflow

# OpenET data download (requires API key in config.py)
python -m es_analysis.runners.run_openet_download

# Individual chart scripts can also run standalone
python es_analysis/charts/evi/evi_raw_plot.py
```

There is no test suite, linter, or build system. This is a research analysis codebase.

## Architecture

### Three-layer pattern: Providers → Charts → Runners

- **`data_providers/`**: Load, transform, and compute data. Each provider is a module with functions (not classes, except `EviDataProvider`, `BEASTDataProvider`, `LandsatDataProvider`). All providers are re-exported through `data_providers/__init__.py`.
- **`charts/`**: Each file produces one figure type. Charts import from providers and `utils/`. Organized into subdirectories: `evi/`, `beast/`, `et_corrections/`, `statistics/`, `multi_panel/`.
- **`runners/`**: Orchestrate groups of charts or computation pipelines. CLI entry points with `argparse`. Import providers lazily inside `main()`.

### Key providers and what they do

| Provider | Role |
|---|---|
| `config.py` | All paths, parameters, county lists. Singleton `config` instance. |
| `evi_provider.py` | Load raw EVI CSV, gap-fill (quartic), smooth (Savitzky-Golay). `EviDataProvider` class. |
| `beast_provider.py` | Load BEAST change-point outputs (seasonal CSVs per county/WY). |
| `et_provider.py` | Load OpenET daily ETa, compute off-phase ET corrections (Method A/B), bootstrap CIs. |
| `et_stats_provider.py` | Bulk ET correction across all counties/WYs. Main computation entry: `run_et_correction_stats()`. |
| `late_water_provider.py` | Water savings from capping late-season cuttings. Full workflow: `run_late_water_saving_workflow()`. |
| `evi_cut_window_provider.py` | Find pre-cut EVI minima, compute cut-cycle time segments. |
| `cutting_stats_provider.py` | Per-cutting weather statistics (GDD, temperature, precipitation via Daymet). |
| `spatial_provider.py` | Parcel shapefiles, area calculations, county boundaries. |
| `daymet_provider.py` | Daymet climate data (GDD5, temperature, precipitation). |
| `publication_figures_provider.py` | Data prep for Phase 5 publication figures. |

### Utils

- **`units.py`**: Unit conversions. mm → ac-ft/acre = divide by 304.8 (pure depth, no area). mm → ac-ft volume = (mm/304.8) × area_acres. Use named functions (`mm_to_acft_per_acre`, `mm_to_acft_total`) not raw constants.
- **`validation.py`**: Range checks at pipeline stages. Daily ET: 0-12 mm/day. Cycle ET: 30-300 mm. Per-cutting: 0.1-1.0 ac-ft/acre. Hard fails on physically impossible values.
- **`publication_style.py`**: Wong/Okabe-Ito colorblind-safe palette, journal sizing constants, `apply_style()` / `save_pub_figure()`.
- **`gapfill.py`**: Quartic interpolation for EVI gap-filling.
- **`smoothing.py`**: Savitzky-Golay smoothing.

## Data Layout

All data paths are configured in `data_providers/config.py`. Key locations:

- **Raw EVI**: `emery_method_2/earthdata_downloader/vi_results/combined_evi_timeseries.csv`
- **BEAST outputs**: `beast_outputs_new/{County}/beast_seasonal_cuts_WY{year}.csv`
- **OpenET daily**: `emery_method/reporting/actual_ET_new/`
- **ETo/ETof**: `emery_method/reporting/ETo_ETof_LandSat_passes/`
- **Daymet**: `emery_method/reporting/daymet_alfalfa/`
- **Shapefiles**: `emery_method/shapefile/`
- **Pipeline outputs**: `es_analysis/output/` (figures, data, logs), `statistics_exports/`, `water_saving_scenarios_stats/`

Water years span Oct 1 (WY-1) through Sep 30 (WY). The 10 study counties: Fresno, San Joaquin, Stanislaus, Madera, Kings, Tulare, Kern, Imperial, Riverside, Merced.

## Important Conventions

- County names use spaces in data (`"San Joaquin"`) but underscores in filenames (`San_Joaquin/`). Use `normalize_county_name()` from `evi_provider.py` for normalization.
- Chart scripts add the parent-of-parent to `sys.path` so they can run standalone: `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`.
- ET correction has two methods: Method A (`f_pre - f_min`, default `config.chosen_method = "A"`) and Method B (triangular decay).
- Bootstrap confidence intervals use `config.n_boot` (4000 for final, 500 for bulk via `n_boot_bulk`).
- The `COUNTY_ORDER` list in `spatial_provider.py` / `statistics_provider.py` defines canonical county ordering for plots.

## Known Data Issues

- **Imperial County**: Severe OpenET data coverage gap — median ET 137 mm (expected 1200+), with 772/979 late-cycle parcel-years showing zero ET. Treat Imperial County results with caution.
- **OpenET API key**: Hardcoded in `data_providers/config.py` line 26 (`openet_api_key`). Do not commit changes that expose this key in logs or outputs.

## Key Statistical Results (boot500)

These are the final pipeline results (N=500 bootstrap replicates):
- Valid N: 9,513 parcel-year-cutting observations (10,220 minus 707 NaN)
- Wilcoxon signed-rank: T=0, p≈0, r=1.0 (all corrections positive)
- Cohen's d: 0.56 (medium effect), mean correction 47.28 mm = 0.155 ac-ft/acre = 5.81% of ET
- Kruskal-Wallis: H=2700, p≈0, eta²=0.283 (large county variation)
- Compact letter display: 9 letter groups across 10 counties, 42/45 pairs significant

## Planning and State

All 6 phases are complete (17/17 plans executed). The `.planning/` directory tracks project roadmap and phase execution:
- `ROADMAP.md`: 6-phase plan (ET fix → Pipeline verify → Stats → Validation → Figures → Cleanup)
- `STATE.md`: Current phase progress
- `phases/NN-*/`: Research, plans, summaries, and verification per phase
