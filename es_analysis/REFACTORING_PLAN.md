# EVI Analysis Repository Refactoring Plan

**Date:** 2026-02-02
**Source:** `alfalfa_evi_jovyan.py` (21,184 lines)
**Goal:** Reorganize monolithic script into modular, maintainable structure

---

## Directory Structure

```
es_analysis/
├── data_providers/              # Centralized data loading and transformation
│   ├── __init__.py
│   ├── evi_provider.py          # All EVI-related data operations
│   ├── beast_provider.py        # BEAST cutting detection results
│   ├── et_provider.py           # OpenET, ETo, ETof data loading
│   ├── landsat_provider.py      # Landsat metadata and passes
│   ├── statistics_provider.py   # Aggregated statistics
│   └── config.py                # Shared configuration (paths, parameters)
│
├── charts/                      # Individual chart scripts (one-per-file)
│   ├── evi/
│   ├── beast/
│   ├── et_corrections/
│   ├── statistics/
│   └── multi_panel/
│
├── runners/                     # Scripts to run related chart groups
│   ├── run_evi_plots.py
│   ├── run_beast_plots.py
│   ├── run_et_plots.py
│   ├── run_statistics_plots.py
│   └── run_all_plots.py
│
├── utils/                       # Shared utility functions
│   ├── __init__.py
│   ├── helpers.py                # Water year tools, normalizations
│   ├── gapfill.py                # Quartic gap-fill
│   ├── smoothing.py              # Savitzky-Golay smoothing
│   └── plotting.py               # Common plotting utilities
│
├── output/                       # Generated outputs
│   ├── figures/
│   ├── data/
│   └── logs/
│
├── config/                       # Configuration files
│   ├── paths.yaml
│   └── parameters.yaml
│
└── REFACTORING_PLAN.md          # This file
```

---

## Complete Chart Inventory with Source References

### Category 1: EVI Time-Series Charts (4 charts)

| Chart File | Source Line(s) | Type | Description |
|-----------|----------------|------|-------------|
| `charts/evi/evi_raw_plot.py` | 222-259 | Line + Scatter | Original EVI with NaN gaps + blue markers for observed days |
| `charts/evi/evi_gapfilled_plot.py` | 243 | Line plot | Gap-filled EVI using local 4th-degree polynomial interpolation |
| `charts/evi/evi_smoothed_plot.py` | 244 | Line plot | Smoothed EVI time series using Savitzky-Golay |
| `charts/evi/evi_combined_plot.py` | 222-259 | Multi-line | Combined plot showing original, gap-filled, and smoothed EVI |

**Data Provider:** `data_providers/evi_provider.py`
**Key Functions in Original:**
- `daily_timeseries_water_year()` - line 92
- `quartic_gapfill()` - line 128
- `smooth_sg()` - line 165
- `build_daily_table()` - line 213
- `plot_series()` - line 222

---

### Category 2: County/Year EVI Pipeline Charts (4 charts)

| Chart File | Source Line(s) | Type | Description |
|-----------|----------------|------|-------------|
| `charts/evi/evi_county_year_original_plot.py` | 743 | Line plot | Original (raw) EVI from loaded county/year CSV |
| `charts/evi/evi_county_year_observations_plot.py` | 745-746 | Scatter plot | Raw observation points on EVI line |
| `charts/evi/evi_county_year_gapfilled_plot.py` | 747 | Line plot | Gap-filled (quartic) EVI |
| `charts/evi/evi_county_year_smoothed_plot.py` | 748 | Line plot | Smoothed (SG) EVI |

**Data Provider:** `data_providers/evi_provider.py`
**Key Functions in Original:**
- `process_county_year()` - line 646
- `plot_from_saved()` - line 706
- `aggregate_daily()` - line 624

---

### Category 3: BEAST Basic Cutting Detection Charts (6 charts)

| Chart File | Source Line(s) | Type | Description |
|-----------|----------------|------|-------------|
| `charts/beast/beast_decomposition_plot.py` | 972 | BEAST decomposition | Full BEAST seasonal/trend decomposition with change points |
| `charts/beast/beast_evi_with_cps_original_plot.py` | 989 | Line plot | Original EVI time series |
| `charts/beast/beast_evi_with_cps_observations_plot.py` | 991 | Scatter plot | Raw EVI observation points |
| `charts/beast/beast_evi_with_cps_gapfilled_plot.py` | 992 | Line plot | Gap-filled EVI |
| `charts/beast/beast_evi_with_cps_smoothed_plot.py` | 993 | Line plot | Smoothed EVI |
| `charts/beast/beast_evi_with_cps_combined_plot.py` | 983-1004 | Multi-line + vertical lines | EVI with change point (CP) locations marked as vertical dashed lines |

**Data Provider:** `data_providers/beast_provider.py`
**Key Functions in Original:**
- `_plot_decomposition()` - line 969
- `_plot_evi_with_cps()` - line 983
- `plot_beast_evi()` - line 1009
- `_run_beast()` - line 927
- `_extract_cp_from_component()` - line 894

---

### Category 4: BEAST Improved Cutting Detection Charts (14 charts)

| Chart File | Source Line(s) | Type | Description |
|-----------|----------------|------|-------------|
| `charts/beast/beast_n_cuttings_boxplot.py` | 6421 | Horizontal boxplot | Distribution of cuttings per parcel across years with mean markers |
| `charts/beast/beast_n_cp_season_boxplot.py` | 6437 | Horizontal boxplot | Distribution of seasonal change points per parcel across years |
| `charts/statistics/monthly_by_county_bar_plot.py` | 6555 | Grouped bar chart | County comparisons of monthly metrics (ET, reductions, etc.) |
| `charts/statistics/yearly_by_county_bar_plot.py` | 6658 | Grouped bar chart | Yearly comparisons of monthly metrics by county |
| `charts/et_corrections/et_cut_dates_and_et_plot.py` | 6961 | Bar chart | Monthly ET (OpenET) side-by-side for water years |
| `charts/et_corrections/et_cut_dates_and_et_corrected_plot.py` | 6961 | Bar chart | Monthly corrected ET |
| `charts/beast/beast_trend_cp_medians_yearly.py` | 7855 | Line plot | Median trend change points plotted across years |
| `charts/beast/beast_trend_cp_medians_by_order_histogram.py` | 7890 | Bar plot | Frequency of trend CP order (1st, 2nd, 3rd cuts) |
| `charts/beast/beast_trend_cps_allpoints.py` | 8013 | Scatter plot | All trend CP dates with county colors |
| `charts/beast/beast_trend_cp_medians_by_order_with_errors.py` | 8054 | Bar plot with error bars | Median trend CP dates with variability |
| `charts/beast/beast_county_year_boxplots.py` | 8678 | Boxplot | Distribution of cuttings by county and year |
| `charts/beast/beast_all_counties_boxplot.py` | 8721 | Boxplot | Distribution across all counties for comparison |
| `charts/beast/beast_evi_with_cps_and_mins_plot.py` | 8496 | Line + vertical lines | EVI with BEAST CPs (red) and detected minima (green) |

**Data Provider:** `data_providers/beast_provider.py`
**Key Functions in Original:**
- `run_seasonal_for_year()` - line 1772
- `run_trend_for_range()` - line 1832
- `plot_n_cuttings_boxplots()` - line 6421
- `plot_n_cp_season_boxplots()` - line 6437
- `plot_trend_cp_medians_yearly()` - line 7855
- `plot_county_year_boxplots()` - line 8678

---

### Category 5: ET Corrections Charts (9 charts)

| Chart File | Source Line(s) | Type | Description |
|-----------|----------------|------|-------------|
| `charts/et_corrections/et_monthly_openet_bar_plot.py` | 2522 | Bar chart (light grey) | Monthly OpenET totals |
| `charts/et_corrections/et_monthly_corrected_bar_plot.py` | 2582 | Bar chart (light green) | Monthly corrected ET (method A or B) |
| `charts/et_corrections/et_monthly_ci_errorbars_plot.py` | 2602-2621 | Error bars | Confidence interval around corrected ET bars |
| `charts/et_corrections/et_harvest_dates_scatter_plot.py` | 2626 | Scatter (X marker) | Harvest/cut dates displayed at top of the plot |
| `charts/et_corrections/et_landsat_passes_scatter_plot.py` | 2638 | Scatter (diamond) | Landsat observation passes |
| `charts/et_corrections/et_daily_scatter_plot.py` | 2677-2687 | Scatter plot | Daily ET (OpenET) on left secondary axis |
| `charts/et_corrections/etof_daily_line_plot.py` | 2707-2716 | Line plot (dashed) | ETof (fraction of ETo) on right secondary axis |
| `charts/et_corrections/et_full_two_panel_plot.py` | 4765-5160 | Two-panel stacked | TOP: Daily ET + ETof, BOTTOM: Monthly ET bars with CI |
| `charts/et_corrections/et_full_multiwy_plot.py` | 3654-3835 | Multi-panel multi-year | Same as two-panel but spanning multiple water years |

**Data Provider:** `data_providers/et_provider.py`
**Key Functions in Original:**
- `plot_et_corrections_full()` - line 2522
- `plot_et_corrections_full_multiwy()` - line 3654
- `compute_daily_and_monthly_for_uid()` - line 1238 (multiple implementations)
- `_bootstrap_diff_fpre_fmin()` - line 1248

---

### Category 6: Multi-Panel County Charts (6 charts)

| Chart File | Source Line(s) | Type | Description |
|-----------|----------------|------|-------------|
| `charts/multi_panel/county_points_two_panel_v1_plot.py` | 12324 | Two-panel scatter | Left: monthly trends, Right: annual summaries |
| `charts/multi_panel/county_points_two_panel_v2_plot.py` | 13797 | Two-panel | Similar to v1 with different metric combinations |
| `charts/multi_panel/county_points_four_panel_plot.py` | 13936 | Four-panel grid | Comprehensive county-level: annual totals, percent reduction, uncertainties, cut dates |
| `charts/multi_panel/county_map_and_ridgelines_plot.py` | 15927 | Map + Ridgeline | Spatial distribution of parcels with temporal ridgeline overlay |

**Data Provider:** `data_providers/statistics_provider.py`
**Key Functions in Original:**
- `plot_county_points_two_panel()` - line 12324
- `plot_county_points_four_panel()` - line 13936
- `plot_county_map_and_ridgelines()` - line 15927

---

### Category 7: Summary Charts (3 charts)

| Chart File | Source Line(s) | Type | Description |
|-----------|----------------|------|-------------|
| `charts/statistics/bar_by_year_plot.py` | 17319 | Bar chart | Total ET vs corrected ET by year across counties |
| `charts/statistics/bar_by_county_plot.py` | 17357 | Bar chart | County comparisons of ET totals and reductions |
| `charts/statistics/late_scatter_county_color_year_marker_plot.py` | 17415 | Scatter plot | Cut dates vs ET reductions with county colors and year markers |

**Note:** Lines 18583-19975 contain near-identical versions of these charts (likely duplicates)

**Data Provider:** `data_providers/statistics_provider.py`
**Key Functions in Original:**
- `plot_bar_by_year()` - line 17319
- `plot_bar_by_county()` - line 17357
- `plot_late_scatter_county_color_year_marker()` - line 17415

---

## Data Provider Specifications

### data_providers/config.py
**Source: Lines 25-34, 532-547, 6421-6466**

Configuration for paths, parameters, and constants.

**Key Settings:**
- Input CSV paths
- Output directories
- Selected counties list
- Water years range
- Processing parameters (gap-fill window, smoothing parameters, BEAST settings)

---

### data_providers/evi_provider.py
**Source: Lines 4-510, 512-809**

Handles all EVI-related data operations.

**Key Classes:**
- `EviDataProvider`: Main interface for EVI data

**Key Methods:**
- `load_raw_evi(county, parcel_id, water_year)` - from line 37
- `normalize_county_names()` - from line 44
- `build_daily_timeseries()` - from line 92
- `quartic_gapfill()` - from line 128
- `smooth_sg()` - from line 165
- `build_daily_table()` - from line 213
- `process_county_year()` - from line 646

**Input Files:**
- `all_parcels_evi_timeseries.csv` (line 26)
- `combined_evi_timeseries.csv` (line 533)

**Output Files:**
- `county_year_exports_new/{County}/WY{wy}.csv` (line 702)

---

### data_providers/beast_provider.py
**Source: Lines 811-1178, 1180-1880**

Handles BEAST cutting detection and results.

**Key Classes:**
- `BeastDataProvider`: Main interface for BEAST results
- `SeasonalCutsDetector`: Seasonal cutting detection
- `TrendCpDetector`: Trend change point detection

**Key Methods:**
- `load_county_year_csv(county, wy)` - from line 878
- `load_seasonal_cuts_csv(county, wy)` - from line 513
- `run_beast(series, cp_component)` - from line 927
- `extract_change_points(result, component)` - from line 894
- `run_seasonal_for_county(county, wy)` - from line 1772
- `run_trend_for_range(county, wy_range)` - from line 1832

**Input Files:**
- `county_year_exports_new/{County}/WY{wy}.csv`

**Output Files:**
- `beast_outputs_new/{County}/beast_seasonal_cuts_WY{wy}.csv`

**BEAST Parameters:**
- Seasonal sweep: sseg_minlength [20-45], sseg_leftmargin [30-45], scp_minmax (6,12)
- Trend sweep: tseg_minlength [90-180], tseg_leftmargin [90-180]
- CP sources: "season" or "trend"

---

### data_providers/et_provider.py
**Source: Lines 518-1880 (distributed), 2522-5160**

Handles OpenET, ETo, ETof data and ET corrections.

**Key Classes:**
- `ETDataProvider`: Main interface for ET data
- `ETCorrectionCalculator`: Compute daily/monthly corrections

**Key Methods:**
- `load_openet_for_wy(county, wy, uid)` - from line 569
- `load_eto_for_wy(wy)` - from line 647
- `load_etof_for_wy(wy)` - from line 180
- `compute_daily_and_monthly_for_uid(uid, county, wy)` - from line 1238
- `bootstrap_diff_fpre_fmin(...)` - from line 1248

**Input Files:**
- OpenET: `{OPENET_ROOT}/{year}/OpenET Exports/*.csv`
- ETo: `{ETO_ETOF_ROOT}/{year}/ETo_*.csv`
- ETof: `{ETO_ETOF_ROOT}/{year}/ETof_*.csv`
- BEAST outputs: `beast_outputs_new/{County}/beast_seasonal_cuts_WY{wy}.csv`

**Correction Methods:**
- Method A: Constant ETo weight
- Method B: Triangular ramp

**Parameters:**
- Cloud cover max: 20%
- R days: 8
- Pre/post windows: 3 days each
- Low quantile: 0.25
- CI alpha: 0.10

---

### data_providers/landsat_provider.py
**Source: Lines 50-54, 720, 2232, 3139, 4429**

Handles Landsat metadata and pass information.

**Key Classes:**
- `LandsatDataProvider`: Main interface for Landsat data

**Key Methods:**
- `load_landsat_passes(county, date_range, cloud_max=20)` - from line 720
- `get_dominant_wrs_track(passes_df)` - from line 2232

**Input Files:**
- Landsat metadata: `/ETo_ETof_LandSat_passes/LandSat_Meta_data/landsat89_pass...csv`

**Columns:**
- date_only, cloud_cover, county, wrs_path, wrs_row

---

### data_providers/statistics_provider.py
**Source: Lines 5219-5600+

Handles aggregated statistics and summaries.

**Key Classes:**
- `StatisticsDataProvider`: Main interface for statistics

**Key Methods:**
- `get_monthly_statistics(county, wy_range)` - from line 5336
- `get_parcel_year_statistics(county, wy_range)` - from line 5467
- `get_county_wy_aggregations()` - from line 5496
- `get_county_pooled_statistics()` - from line 5512
- `get_wy_pooled_statistics()` - from line 5529
- `get_monthly_seasonality()` - from line 5546

**Output DataFrames:**
- `df_monthly`: Parcel-month level statistics
- `df_parcel_year`: Parcel-year aggregated statistics
- `df_fail`: Failed computation records

---

## Utility Modules

### utils/helpers.py
**Source: Lines 44, 72-90, 550, 555, 561, 870, 875, 1270-1278**

Common helper functions.

**Functions:**
- `normalize_county_name()` - line 44/870
- `water_year_bounds(year)` - line 73/875
- `in_water_year_domain(dt, years)` - line 87
- `nearest_odd(k)` - line 561
- `build_wy_grid(start, end)` - line 643

---

### utils/gapfill.py
**Source: Lines 128-163**

Quartic gap-fill implementation.

**Functions:**
- `quartic_gapfill(daily_df, window_days)` - line 128
- `quartic_gapfill_daily(series, window_days)` - line 590

**Parameters:**
- INTERP_WINDOW_DAYS: 30 (default)
- No edge extrapolation

---

### utils/smoothing.py
**Source: Lines 165-211, 564-588**

Savitzky-Golay smoothing implementation.

**Functions:**
- `smooth_sg(series, window, poly)` - line 165

**Parameters:**
- SG_WINDOW: 15 (must be odd)
- SG_POLY: 3

---

### utils/plotting.py
**Source: Multiple plot functions**

Common plotting utilities.

**Functions:**
- `setup_mpl_params()` - line 860
- `save_figure(fig, path, dpi=150)`
- Water year date axis helpers

---

## Runner Scripts

### runners/run_evi_plots.py
Runs all EVI-related charts.

**Charts:**
1. `charts/evi/evi_raw_plot.py`
2. `charts/evi/evi_gapfilled_plot.py`
3. `charts/evi/evi_smoothed_plot.py`
4. `charts/evi/evi_combined_plot.py`
5. `charts/evi/evi_county_year_original_plot.py`
6. `charts/evi/evi_county_year_observations_plot.py`
7. `charts/evi/evi_county_year_gapfilled_plot.py`
8. `charts/evi/evi_county_year_smoothed_plot.py`

**Execution Order:** Dependent on data from `evi_provider.py`

---

### runners/run_beast_plots.py
Runs all BEAST-related charts.

**Charts:**
1. `charts/beast/beast_decomposition_plot.py`
2. `charts/beast/beast_evi_with_cps_*_plot.py` (5 variants)
3. `charts/beast/beast_n_cuttings_boxplot.py`
4. `charts/beast/beast_n_cp_season_boxplot.py`
5. `charts/beast/beast_trend_cp_medians_yearly.py`
6. `charts/beast/beast_trend_cp_medians_by_order_histogram.py`
7. `charts/beast/beast_trend_cps_allpoints.py`
8. `charts/beast/beast_trend_cp_medians_by_order_with_errors.py`
9. `charts/beast/beast_county_year_boxplots.py`
10. `charts/beast/beast_all_counties_boxplot.py`
11. `charts/beast/beast_evi_with_cps_and_mins_plot.py`

**Execution Order:** Dependent on data from `beast_provider.py` and `evi_provider.py`

---

### runners/run_et_plots.py
Runs all ET correction charts.

**Charts:**
1. `charts/et_corrections/et_monthly_openet_bar_plot.py`
2. `charts/et_corrections/et_monthly_corrected_bar_plot.py`
3. `charts/et_corrections/et_monthly_ci_errorbars_plot.py`
4. `charts/et_corrections/et_harvest_dates_scatter_plot.py`
5. `charts/et_corrections/et_landsat_passes_scatter_plot.py`
6. `charts/et_corrections/et_daily_scatter_plot.py`
7. `charts/et_corrections/etof_daily_line_plot.py`
8. `charts/et_corrections/et_full_two_panel_plot.py`
9. `charts/et_corrections/et_full_multiwy_plot.py`
10. `charts/et_corrections/et_cut_dates_and_et_plot.py`
11. `charts/et_corrections/et_cut_dates_and_et_corrected_plot.py`

**Execution Order:** Dependent on data from `et_provider.py`, `beast_provider.py`, `landsat_provider.py`

---

### runners/run_statistics_plots.py
Runs all statistics and summary charts.

**Charts:**
1. `charts/statistics/monthly_by_county_bar_plot.py`
2. `charts/statistics/yearly_by_county_bar_plot.py`
3. `charts/statistics/bar_by_year_plot.py`
4. `charts/statistics/bar_by_county_plot.py`
5. `charts/statistics/late_scatter_county_color_year_marker_plot.py`
6. `charts/statistics/cut_dates_scatter_plot.py`
7. `charts/statistics/n_cuttings_boxplot.py`

**Execution Order:** Dependent on data from `statistics_provider.py`

---

### runners/run_all_plots.py
Runs all charts in dependency order.

**Execution Order:**
1. EVI plots
2. BEAST plots
3. ET correction plots
4. Statistics plots
5. Multi-panel charts

---

## Migration Phases

### Phase 1: Setup Foundation (Week 1)
- [ ] Create directory structure ✓
- [ ] Create `data_providers/config.py` with all paths and parameters
- [ ] Create `utils/` module with shared functions
- [ ] Create empty `data_providers/` with stub classes

### Phase 2: Build Data Providers (Week 2)
- [ ] Implement `evi_provider.py`
- [ ] Implement `beast_provider.py`
- [ ] Implement `et_provider.py`
- [ ] Implement `landsat_provider.py`
- [ ] Implement `statistics_provider.py`
- [ ] Implement `config.py`

### Phase 3: Create Chart Modules (Week 3-4)
- [ ] EVI charts (4 files)
- [ ] BEAST charts (14 files)
- [ ] ET correction charts (11 files)
- [ ] Statistics charts (7 files)
- [ ] Multi-panel charts (4 files)

### Phase 4: Create Runners (Week 5)
- [ ] `run_evi_plots.py`
- [ ] `run_beast_plots.py`
- [ ] `run_et_plots.py`
- [ ] `run_statistics_plots.py`
- [ ] `run_all_plots.py`

### Phase 5: Testing & Validation (Week 6)
- [ ] Test each chart script independently
- [ ] Test each runner
- [ ] Compare outputs with original script
- [ ] Document any differences
- [ ] Create README with usage instructions

---

## Benefits of This Structure

1. **Maintainability:** Issues with data sources isolated to provider files
2. **Testability:** Each component can be unit tested independently
3. **Scalability:** Add new charts without modifying existing code
4. **Reusability:** Data providers can serve multiple charts
5. **Traceability:** Clear data lineage from source → provider → chart
6. **Flexibility:** Easy to experiment with parameters in `config.py`
7. **Parallel Execution:** Independent charts can run in parallel
8. **Single Responsibility:** Each file has one clear purpose

---

## Configuration Parameters

### Water Year Definition
```
Water Year = Oct 1 (year-1) to Sep 30 (year)
Source: Lines 73-85, 555, 875, 1277
```

### Gap-fill Parameters
```
INTERP_WINDOW_DAYS: 30
Source: Lines 26, 545, 848
```

### Smoothing Parameters
```
SG_WINDOW: 15 (must be odd)
SG_POLY: 3
Source: Lines 26, 546, 849
```

### BEAST Parameters
**Seasonal Sweep:**
- sseg_minlength: [20, 27, 35, 45]
- sseg_leftmargin: [30, 35, 40, 45]
- scp_minmax: (6, 12)
**Source:** Lines 1241-1246, 249-254

**Trend Sweep:**
- tseg_minlength: [90, 120, 180]
- tseg_leftmargin: [90, 120, 180]
**Source:** Lines 1248-1253

**CP Sources:**
- "season" (default for cut detection)
- "trend" (for trend breaks)

### ET Correction Parameters
```
Cloud cover max: 20%
R days: 8
Pre/post windows: 3 days each
Low quantile: 0.25
Method: A (constant ETo weight) or B (triangular ramp)
CI alpha: 0.10
Bootstrap samples: 4000 (default), 500 for bulk
Source: Lines 2343, 2360, 2366, 2404, 2428, 2450, 2455
```

### Selected Counties
```
Fresno, San Joaquin, Stanislaus, Madera, Kings, Tulare,
Kern, Los Angeles, San Bernardino, Imperial, Riverside, Merced
Source: Lines 30-33, 537-541
```

### Water Years
```
2019–2023 (with extensions to 2024 in some sections)
Source: Lines 34, 542
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ INPUT DATA:                                                       │
│  • all_parcels_evi_timeseries.csv                                 │
│  • combined_evi_timeseries.csv                                   │
│  • OpenET ETa CSVs                                                │
│  • ETo/ETof CSVs                                                  │
│  • Landsat metadata CSV                                          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ CATEGORY 1: EVI TIME-SERIES PROCESSING                            │
│  • Normalize county names                                         │
│  • Build daily water-year series                                  │
│  • Gap-fill (quartic)                                             │
│  • Smooth (Savitzky-Golay)                                        │
│  • Plot EVI sequences                                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ OUTPUT: county_year_exports_new/{County}/WY{wy}.csv              │
│   (original, gapfilled, smoothed EVI)                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────────────────────┐
                              ▼                                         ▼
┌─────────────────────────────────┐   ┌────────────────────────────────┐
│ CATEGORY 2: CSV EXPORT          │   │ CATEGORY 3: BEAST (BASIC)       │
│  • Aggregate daily → WY grid    │   │  • Run BEAST season/trend       │
│  • Save county/year CSVs        │   │  • Extract CPs                 │
│  • Plot from saved CSVs         │   │  • Plot decomposition           │
└─────────────────────────────────┘   └────────────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────────┐
                              │ CATEGORY 4: BEAST (IMPROVED)   │
                              │  • Inclusive minima detection  │
                              │  • CP-gated cutting detection  │
                              │  • Seasonal & trend CPs       │
                              │  • Parallel processing         │
                              └────────────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────────┐
                              │ OUTPUT: beast_outputs_new/      │
                              │   {County}/beast_seasonal_      │
                              │   cuts_WY{wy}.csv               │
                              └────────────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────────┐
                              │ CATEGORY 5: ET CORRECTIONS     │
                              │  • Load OpenET, ETo, ETof       │
                              │  • Load BEAST harvest dates     │
                              │  • Load Landsat passes          │
                              │  • Compute daily corrections    │
                              │  • Compute monthly + CI         │
                              │  • Plot dual-subplot            │
                              └────────────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────────┐
                              │ CATEGORY 6: STATS & EXPORT     │
                              │  • Process all parcels         │
                              │  • Monthly/annual aggregations │
                              │  • County/WY trends            │
                              │  • Seasonal patterns           │
                              │  • Export statistics CSVs      │
                              └────────────────────────────────┘
```

---

## Appendix: Source Function Index

### Data Processing Functions
- `_norm_name()` - line 44, 550
- `daily_timeseries_water_year()` - line 92
- `quartic_gapfill()` - line 128, 590
- `smooth_sg()` - line 165, 564
- `build_daily_table()` - line 213
- `aggregate_daily()` - line 624
- `process_county_year()` - line 646
- `water_year_bounds()` - line 73, 555, 875, 1277
- `nearest_odd()` - line 561

### BEAST Functions
- `_load_county_year_csv()` - line 878, 1280
- `_series_for_beast()` - line 889, 1291
- `_extract_cp_from_component()` - line 894
- `_run_beast()` - line 927
- `_plot_decomposition()` - line 969
- `_plot_evi_with_cps()` - line 983
- `plot_beast_evi()` - line 1009
- `batch_beast_for_county()` - line 1072
- `run_seasonal_for_year()` - line 1772
- `run_trend_for_range()` - line 1832

### ET Correction Functions
- `_load_seasonal_csv()` - line 513, 2031, 2928, 4228
- `_parse_cp_dates_iso()` - line 519, 2049, 2936, 4246
- `get_harvest_dates_for_uid()` - line 530, 2061, 2948, 4258
- `_load_openet_for_wy()` - line 569, 2099, 3006, 4296
- `_load_eto_or_etof_for_wy()` - line 647, 2167, 3074, 4364
- `_load_evi_for_uid_wy()` - line 127, 4065
- `_load_landsat_passes()` - line 720, 2232, 3139, 4429
- `_bootstrap_diff_fpre_fmin()` - line 1248, 2307, 3214, 4504
- `compute_daily_and_monthly_for_uid()` - line 1238, 2338, 4528
- `plot_et_corrections_full()` - line 2522, 3429, 4765
- `plot_et_corrections_full_multiwy()` - line 3929, 4939

### Statistics Functions
- `_safe_div()` - line 5299
- `_desc()` - line 5307
- `_month_to_wy_month()` - line 5325
- `_wy_month_label()` - line 5329
- `plot_n_cuttings_boxplots()` - line 6421
- `plot_n_cp_season_boxplots()` - line 6437
- `plot_allcounties_by_year_grouped()` - line 6555
- `plot_allyears_by_county()` - line 6658
- `plot_cut_dates_and_et()` - line 6961
- `plot_trend_cp_medians_yearly()` - line 7855
- `plot_trend_cp_medians_by_order()` - line 7890, 8054
- `plot_trend_cps_allpoints()` - line 8013
- `plot_county_year_boxplots()` - line 8678
- `plot_all_counties_box()` - line 8721

### Plotting Functions
- `plot_series()` - line 222
- `plot_from_saved()` - line 706
- `plot_county_points_two_panel()` - line 12324, 13797
- `plot_county_points_four_panel()` - line 13936
- `plot_county_map_and_ridgelines()` - line 15927
- `plot_bar_by_year()` - line 17319
- `plot_bar_by_county()` - line 17357
- `plot_late_scatter_county_color_year_marker()` - line 17415

---

**Total Charts:** 40 unique visualizations
**Total Charts (including duplicates):** 39+ variants across sections
**Source Files:** 1 monolithic script (21,184 lines)
**Target Structure:** ~60+ modular files organized by responsibility