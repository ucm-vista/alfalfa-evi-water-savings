# Alfalfa EVI Analysis - Figure Documentation

This document provides comprehensive documentation of all figures generated in `alfalfa_evi_jovyan.py`, including:
- The code that generates each figure
- Explanation of what the figure shows
- Logic and methodology
- Data sources used
- Critical assessment of appropriateness and data usage

---

## Table of Contents

1. [EVI Time Series Plotting](#1-evi-time-series-plotting)
2. [BEAST Change Point Detection](#2-beast-change-point-detection)
3. [Off-Phase ET Corrections](#3-off-phase-et-corrections)
4. [Cutting Count Distributions](#4-cutting-count-distributions)
5. [Cut Dates with ET Visualization](#5-cut-dates-with-et-visualization)

---

## 1. EVI Time Series Plotting

### 1.1 Interactive Widget-based Plotting with `process_selection()`

**Location in code:** Lines 261-278, with UI at lines 280-378

**What it generates:**
- Interactive time series plot showing:
  - Original raw EVI data (blue line with points)
  - Gap-filled EVI using quartic interpolation (green line)
  - Smoothed EVI using Savitzky-Golay filter (orange line)

**Code:**
```python
def plot_series(daily_df, filled, smoothed, county, parcel, years, win_days, sg_w, sg_p):
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    
    # Original (raw) as line-with-gaps + points
    ax.plot(daily_df["date"], daily_df["mean_evi"], label="Original (raw)", linewidth=1.2)
    raw_mask = ~daily_df["mean_evi"].isna()
    ax.scatter(daily_df.loc[raw_mask, "date"], daily_df.loc[raw_mask, "mean_evi"],
               s=12, alpha=0.9, label="Raw obs (points)")
    
    # Gap-filled and smoothed
    ax.plot(daily_df["date"], filled,   label="Gap-filled (quartic)", linewidth=1.4, alpha=0.9)
    ax.plot(daily_df["date"], smoothed, label="Smoothed (SG)", linewidth=1.6, alpha=0.95)
    
    # Force x-limits to water-year domain
    start, end = water_year_bounds(years)
    ax.set_xlim(start, end)
```

**Logic:**
1. Raw data is loaded from CSV and filtered by county, parcel, and year
2. Data is normalized to water year domain (Oct 1 - Sep 30)
3. Any duplicate dates are aggregated by mean
4. Gaps are interpolated using quartic polynomial interpolation within ±30 days
5. Savitzky-Golay smoothing is applied to the gap-filled series

**Data used:**
- CSV_PATH: `/home/jovyan/work/emery_method/reporting/evi_analysis/all_parcels_evi_timeseries.csv`
- Required columns: `date`, `mean_evi`, `county`, `parcel_id`
- Water year range: 2019-2023

**Assessment:** ✅ **APPROPRIATE**
- Clear visualization of data processing pipeline
- Shows all three stages: raw → gap-filled → smoothed
- X-axis limited to water year prevents confusion
- Points distinguish actual observations from interpolated values

---

### 1.2 Plot from Saved CSVs (`plot_from_saved`)

**Location in code:** Lines 706-757

**What it generates:**
- Similar to above but loads pre-processed data from saved CSVs
- Shows Original, Gap-filled, and Smoothed EVI lines

**Logic:**
- Loads `OUTPUT_ROOT/<County>/WY{wy}.csv`
- Plots for single parcel or ALL parcels in that county/year
- Saves plots to `PLOTS_ROOT/<County>/WY{wy}_parcel_{pid}.png`

**Data used:**
- Pre-processed county/year CSVs with columns:
  - `original_mean_evi`, `gapfilled_mean_evi`, `smoothed_mean_evi`

**Assessment:** ✅ **APPROPRIATE**
- Useful for batch processing and reviewing stored results
- Consistent visualization approach

---

## 2. BEAST Change Point Detection

### 2.1 BEAST Decomposition Plot (`_plot_decomposition`)

**Location in code:** Lines 969-981

**What it generates:**
- Full BEAST decomposition showing:
  - Original time series
  - Trend component
  - Seasonal component  
  - Residuals
  - Change point probabilities

**Code:**
```python
def _plot_decomposition(result, county, wy, parcel):
    plt.figure(figsize=(12, 10))
    rb.plot(result, ncpStat='median')
    plt.suptitle(f"BEAST EVI Decomposition\nCounty={county}, WY={wy}, Parcel={parcel}",
                 y=0.98, fontsize=14)
    plt.subplots_adjust(top=0.90, hspace=0.0, wspace=0.0)
    plt.show()
```

**Logic:**
1. BEAST algorithm decomposes EVI into trend + seasonal + error
2. Identifies change points where the seasonal pattern shifts
3. `ncpStat='median'` shows median number of change points
4. Outputs change point dates and probabilities

**Data used:**
- Smoothed EVI time series
- BEAST parameters: period=365.0, season="harmonic"
- Scp_minmax=(0, 10) constrains number of seasonal change points

**Assessment:** ✅ **APPROPRIATE**
- BEAST is well-established for time series decomposition
- Visual inspection allows quality control
- Harmonic seasonality appropriate for annual vegetation cycles

---

### 2.2 EVI with Change Points Overlay (`_plot_evi_with_cps`)

**Location in code:** Lines 983-1004

**What it generates:**
- EVI time series with vertical lines marking detected change points
- Shows raw, gap-filled, and smoothed series
- Visual representation of when cuts occur

**Code:**
```python
def _plot_evi_with_cps(df_parcel, cp_dates, county, wy, parcel, out_dir, label_suffix=""):
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.plot(df_parcel["date"], df_parcel["original_mean_evi"], label="Original (raw)", linewidth=1.0)
    ax.plot(df_parcel["date"], df_parcel["gapfilled_mean_evi"], label="Gap-filled (quartic)", linewidth=1.2, alpha=0.9)
    ax.plot(df_parcel["date"], df_parcel["smoothed_mean_evi"], label="Smoothed (SG)", linewidth=1.5, alpha=0.95)
    
    # Vertical lines at change points
    for d in cp_dates:
        ax.axvline(d, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8)
```

**Logic:**
1. Run BEAST on smoothed EVI to detect change points
2. Extract change point dates from "season" component
3. Plot vertical lines at detected dates
4. Number of cuts estimated as n_CP / 2.0 (assuming each growing cycle has 2 change points)

**Data used:**
- County/year CSVs with processed EVI
- BEAST output change points

**Assessment:** ⚠️ **REQUIRES VALIDATION**
- **Potential issue:** Assumption that n_cuts = n_CP / 2.0 may not hold for all parcels
- Change points represent seasonal transitions, not necessarily harvest events
- Need ground truth validation to confirm this relationship
- **Recommendation:** Add validation against field data or high-res imagery

---

### 2.3 Cutting Count Boxplots by Year (`plot_county_year_boxplots`)

**Location in code:** Lines 738-767

**What it generates:**
- Horizontal boxplot showing distribution of cutting counts per water year
- One box per year
- Shows min, max, quartiles, and mean

**Logic:**
- Loads `beast_seasonal_cuts_WY*.csv` for each year
- Extracts `n_cuttings` column
- Creates horizontal boxplot
- Mean shown as black dot

**Assessment:** ✅ **APPROPRIATE**
- Good for understanding temporal patterns
- Boxplots appropriate for showing distributions

---

## 3. Off-Phase ET Corrections

### 3.1 Main Off-Phase Correction Plot (`plot_et_corrections_full`)

**Location in code:** Lines 2522-2739 (revised version around lines 4765-4934)

**What it generates:**
- Two-panel subplot:
  - **TOP:** Daily ET (red scatter) + ETof (black dashed line)
  - **BOTTOM:** Monthly ET bars (OpenET vs Corrected) with confidence intervals
- Harvest dates marked with '×' symbols
- Landsat passes marked with diamond symbols

**Code structure (revised version):**
```python
def plot_et_corrections_full(
    daily_df, monthly_df, harvest_dates, passes_df,
    county, wy, uid, cloud_cover_max, chosen_method="A", ci_alpha=0.10
):
    fig, (ax_top, ax_month) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4]}
    )
    
    # TOP: Daily ET + ETof
    ax_top.scatter(daily_df.index, daily_df["ET_open"], ...)
    ax_top2.plot(daily_df.index, daily_df["ETof"], linestyle="--", ...)
    
    # BOTTOM: Monthly bars
    ax_month.bar(x_actual, monthly_df["ET_open"], color="lightgrey", ...)
    ax_month.bar(x_corr, monthly_df["ET_corr"], color="#6BCB77", ...)
    ax_month.errorbar(x_corr, y_mid, yerr=[err_lower, err_upper], ...)
```

**Logic:**
1. **Harvest Detection:** BEAST identifies change points in EVI as harvest dates
2. **Off-Phase Window:** From harvest date to first clear Landsat pass after harvest
3. **Correction Methods:**
   - **Method A:** `delta = (f_pre - f_min) × ETo`
   - **Method B:** `delta = (f_pre - f_t) × ETo` with triangular decay
4. **Uncertainty:** Bootstrap confidence intervals (4000 samples)
5. **Cloud Gap Inflation:** If clouds delay observation, increase correction by 1 + (gap_days/16)

**Data used:**
- **OpenET data:** Daily ETa from `/home/jovyan/work/emery_method/reporting/actual_ET_new/`
- **Reference ET (ETo):** From `OpenET_ETo_ETof_overlap/`
- **ET fraction (ETof):** ET/ETo ratio
- **Landsat metadata:** Cloud cover, pass dates
- **Harvest dates:** From BEAST change points

**Assessment:** ⚠️ **MIXED - REQUIRES SCRUTINY**

**Strengths:**
- ✅ Dual-panel design clearly separates daily dynamics from monthly totals
- ✅ Confidence intervals provide uncertainty quantification
- ✅ Cloud gap inflation accounts for observation delays

**Potential Issues:**
1. **Method A vs B choice:** No clear guidance on which to use
2. **f_pre and f_min windows:** Fixed 5-day windows may not capture field-specific recovery patterns
3. **Triangular decay (Method B):** Assumes linear recovery - may not match actual alfalfa regrowth
4. **Change point → harvest assumption:** Needs validation
5. **Over-correction risk:** If change points are false positives, correction removes valid ET

**Recommendations:**
- Add sensitivity analysis showing correction magnitude vs window size
- Validate against eddy covariance or lysimeter data
- Compare Method A vs B results statistically

---

### 3.2 Multi-Year Off-Phase Plot (`plot_et_corrections_full_multiwy`)

**Location in code:** Lines 3654-3834

**What it generates:**
- Same two-panel layout as above but spanning multiple water years
- X-axis shows continuous timeline from WY_START to WY_END
- Automatic tick spacing adjustment (monthly if ≤18 months, quarterly if longer)

**Logic:**
- Runs `compute_daily_and_monthly_for_uid()` for each water year
- Concatenates daily and monthly DataFrames
- Combines harvest dates across all years
- Adjusts month locator interval based on timespan

**Assessment:** ✅ **APPROPRIATE**
- Useful for multi-year trend analysis
- Consistent visualization with single-year plots
- Smart tick spacing prevents overcrowding

---

## 4. Cutting Count Distributions

### 4.1 Boxplots by County and Year (`plot_allcounties_by_year_grouped`)

**Location in code:** Lines 6555-6654

**What it generates:**
- Grouped boxplot: Years on X-axis, one box per county within each year
- Fixed north-to-south county order
- Shared color scheme (tab20 palette)
- Legend outside plot frame

**Key parameters:**
- `box_width=0.10`
- `county_gap=0.06` (space between counties)
- `year_spacing=1.30` (space between year groups)

**Logic:**
1. Loads all `beast_seasonal_cuts_WY*.csv` files for each county
2. Aggregates data by county and year
3. Positions boxes with calculated offsets
4. Colors boxes by county using consistent palette

**Assessment:** ✅ **APPROPRIATE**
- Grouped design allows year-to-year and county-to-county comparison
- Consistent colors aid interpretation
- Parameters optimized to prevent overcrowding

---

### 4.2 Boxplots Aggregated by County (`plot_allyears_by_county`)

**Location in code:** Lines 6658-6737

**What it generates:**
- Vertical boxplot with counties on X-axis (north to south)
- Each box pools ALL years for that county
- Same color scheme as grouped plot

**Logic:**
- Concatenates arrays across years for each county
- Creates vertical boxplot
- Shows county-level patterns independent of year

**Assessment:** ✅ **APPROPRIATE**
- Complements the by-year plot
- Good for understanding spatial patterns
- Vertical orientation works well with county labels

---

### 4.3 Individual County Boxplots (`plot_n_cuttings_boxplots`, `plot_n_cp_season_boxplots`)

**Location in code:** Lines 6320-6453

**What it generates:**
- Horizontal boxplots for a single county
- One plot for `n_cuttings`, one for `n_cp_season`
- Auto-detects available years on disk

**Logic:**
- Searches for `beast_seasonal_cuts_WY*.csv` files
- Extracts metric values per year
- Creates horizontal boxplot with year labels

**Assessment:** ✅ **APPROPRIATE**
- Quick diagnostic for single-county analysis
- Auto-detection makes it user-friendly

---

## 5. Cut Dates with ET Visualization

### 5.1 Cut Dates and ET Bars (`plot_cut_dates_and_et`)

**Location in code:** Lines 6961-7017 (original), Lines 7354-7515 (revised with ET local min logic)

**What it generates:**
- Timeline showing:
  - Cut dates marked with '×' symbols at top
  - ET between cuts shown as colored bars
  - Different colors per UniqueID
  - Stacked × marks if multiple cuts on same date

**Revised Logic (ET estimation):**
```python
def _find_pre_cut_local_min_start(et, cut_date, lower_bound, rise_days=5, rise_eps=0.02):
    """Scan backward from cut_date to find local ET minimum."""
    for d in reversed(w.index):
        v = float(w.loc[d])
        if v <= min_val + 1e-12:
            min_val = v
            min_date = d
            consecutive_rise = 0
        else:
            if v > min_val + rise_eps:
                consecutive_rise += 1
            else:
                consecutive_rise = 0
            if consecutive_rise >= rise_days:
                break
    return min_date
```

**ET Calculation:**
1. For each cut date, find the preceding local minimum in ET
2. Sum ET from that minimum date to the cut date
3. This represents ET for that cutting cycle
4. Bars are dodged horizontally if multiple UIDs have cuts on same date

**Data used:**
- BEAST seasonal cuts CSV
- OpenET daily ET data
- Optional: Filter by `n_cp_season` value

**Assessment:** ⚠️ **REQUIRES VALIDATION**

**Strengths:**
- ✅ Visual representation of cutting schedule vs water use
- ✅ Per-cycle ET allows productivity analysis
- ✅ Handles multiple parcels on same plot with color coding

**Concerns:**
1. **Local minimum detection:** The 5-day rise detection (`rise_days=5`, `rise_eps=0.02`) is arbitrary
   - Alfalfa regrowth after cutting may not follow this pattern
   - ET minimum may not align with actual cutting date if irrigation continues
   
2. **ET sum to cut date:** Assumes cut occurs at change point date
   - Actual cutting may occur days before/after EVI change point
   
3. **Tail segment handling:** Post-last-cut ET included but may be incomplete

**Recommendations:**
- Sensitivity analysis: Show ET totals vs different `rise_days` values
- Compare summed ET to independent yield data
- Validate that ET drops actually correspond to harvest events

---

## Summary Assessment Table

| Figure | Appropriateness | Key Strengths | Main Concerns |
|--------|----------------|---------------|---------------|
| EVI Time Series | ✅ High | Clear processing pipeline, shows all stages | None significant |
| BEAST Decomposition | ✅ High | Established method, good QC | Requires domain expertise to interpret |
| EVI + Change Points | ⚠️ Medium | Direct visualization | CP→cut assumption unvalidated |
| Off-Phase ET Correction | ⚠️ Medium | Uncertainty quantification | Method choice arbitrary, assumptions need validation |
| Multi-Year ET | ✅ High | Good for trends | Same concerns as single-year |
| County/Year Boxplots | ✅ High | Good comparisons | None significant |
| Cut Dates + ET Bars | ⚠️ Medium | Productivity visualization | Local min detection arbitrary, timing uncertainty |

---

## General Recommendations

### High Priority
1. **Validate change point → harvest relationship** with ground truth data
2. **Sensitivity analysis** for correction parameters (window sizes, quantiles)
3. **Cross-validation** of off-phase corrections against independent ET measurements

### Medium Priority
4. **Method selection guidance:** Document when to use Method A vs B
5. **Uncertainty propagation:** Show how input uncertainty affects final ET estimates
6. **False positive analysis:** Quantify impact of incorrect change point detection

### Low Priority  
7. **Interactive versions:** Bokeh/Plotly for exploring individual parcels
8. **Batch summary reports:** Automated statistical summaries per county/year
9. **Data quality flags:** Highlight parcels with suspicious patterns

---

*Document generated from analysis of alfalfa_evi_jovyan.py*
*Date: 2026-02-02*
