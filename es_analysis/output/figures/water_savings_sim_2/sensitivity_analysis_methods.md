# Water Savings Sensitivity Analysis

## 1. Overview

This analysis quantifies the potential water savings from restricting late-season
alfalfa cuttings across California's Central and Southern San Joaquin Valley.
The sensitivity heatmap explores how savings vary across three dimensions:

1. **Segment ET** — total evapotranspiration consumed during cutting cycles (mm)
2. **Number of cuttings** — annual cutting count per parcel (1–14)
3. **Water year type** — San Joaquin Valley Water Year Index classification
4. **Cutoff date** — the policy date after which cuttings are considered "late"

## 2. Data

| Item | Value |
|------|-------|
| Parcel-years | 10,707 |
| Unique parcels | 1,792 |
| Counties | 10 (San Joaquin → Imperial, N→S) |
| Water years | WY2019–WY2024 |
| ET source | OpenET daily ETa (ensemble) |
| Cutting detection | BEAST change-point analysis on HLS EVI |
| Cutoff date | July 1 (default) |

**Input CSVs:**

- `late_cut_base_parcel_year.csv` — one row per parcel-year with n_cuttings,
  n_late_cuts, late_et_mm, total_et_mm, late_cut_dates, late_cycle_et_mm_list
- `late_cut_savings_parcel_year_mm.csv` — per-parcel-year savings under each
  cap scenario (saved_mm_cap0, saved_mm_cap1, etc.)

Water year types are assigned from the SJV Water Year Index
(CA DWR WSIHIST report):

| WY | Index | Type |
|----|-------|------|
| 2019 | 4.94 | Wet |
| 2020 | 2.35 | Dry |
| 2021 | 1.32 | Critical |
| 2022 | 1.56 | Critical |
| 2023 | 6.40 | Wet |
| 2024 | 3.49 | Above Normal |

## 3. Assumptions

### A. Cutoff date defines "late" cuts

All cuttings occurring on or after July 1 are classified as "late-season."
This date approximates the transition between peak-production and
shoulder-season growth in the San Joaquin Valley. The cutoff-date sweep
(panels i–l) tests the sensitivity of this choice from July 1 to December 1.

### B. Per-late-cut ET is empirically stable

Per-late-cut ET is driven by summer growing conditions — daily ET of ~5 mm/day
over ~30–40 day cutting cycles — and is relatively stable regardless of a
parcel's total seasonal ET or total number of cuttings:

| n_cuttings group | Mean per-late-cut ET | Median | n |
|------------------|---------------------|--------|---|
| 2–4 cuts | 205 mm (0.67 ac-ft/acre) | 204 mm | 331 |
| 5–8 cuts | 172 mm (0.56 ac-ft/acre) | 169 mm | 8,357 |
| 9–14 cuts | 154 mm (0.51 ac-ft/acre) | 153 mm | 1,593 |
| **All** | **170 mm (0.56 ac-ft/acre)** | **166 mm** | **10,285** |

The modest decline from 205 → 154 mm with increasing cuttings reflects
shorter inter-cut intervals at higher cutting frequencies. The standard
deviation is ~44 mm (CV ≈ 26%), confirming moderate stability.

However, per-cut ET does correlate moderately with total ET: the ratio
between high-ET-quartile and low-ET-quartile per-cut ET is ~1.35 at the
same n_cuttings. The model uses sqrt(ET/reference_ET) scaling to capture
this relationship.

### C. Feasibility bounds — diagonal band

Not all (n_cuttings, segment ET) combinations are physically possible.
The valid parameter space forms a **diagonal band**:

**Upper bound (ET ceiling):** A 2-cutting parcel cannot produce 2,000 mm
of segment ET. The ceiling is the observed p99 + 20% buffer per n_cuttings.

**Lower bound (ET floor):** A 14-cutting parcel cannot have only 150 mm
total ET — each cutting cycle requires a minimum ~50 mm of ET (20+ days
of growth at ≥ 2.5 mm/day). The floor is n_cuttings × 50 mm.

| n_cuttings | ET floor (mm) | ET ceiling (mm) | Valid band |
|------------|--------------|----------------|-----------|
| 1 | 50 | ~450 | 50–450 |
| 2 | 100 | ~930 | 100–930 |
| 5 | 250 | ~1,440 | 250–1,440 |
| 7 | 350 | ~1,690 | 350–1,690 |
| 10 | 500 | ~1,760 | 500–1,760 |
| 14 | 700 | ~2,500+ | 700–2,500 |

Cells outside the band appear as gray (physically implausible).

### D. Savings cap

Model-predicted savings are capped at 55% of total segment ET. This
reflects the physical constraint that late-season ET cannot exceed the
total — the empirical p95 of late_et_frac is 0.63, and 55% provides a
conservative bound for model-filled cells. Empirical cells (from real
data) are not subject to this cap.

### E. Cap scenarios

- **Cap 0**: Remove all late-season cuttings. Savings = sum of all late cutting-cycle ET.
- **Cap 1**: Allow one late cutting, remove the rest. Savings = sum of late cutting-cycle ET after the first.

### F. Model fills sparse cells

Grid cells with fewer than 5 observed parcel-years are filled by the
calibrated model. Cells with ≥ 5 observations use empirical means.
Model-filled cells appear in *italic* text; empirical cells in roman.

## 4. Methods

### 4.1 Sensitivity grid (panels a–h)

For each combination of (cap_k, wy_type, n_cuttings, et_bin):

1. **Bin segment ET** into 10 bins: 0–150, 150–300, 300–500, 500–700,
   700–900, 900–1100, 1100–1400, 1400–1700, 1700–2100, 2100–2500 mm.

2. **Count** parcel-years in each cell.

3. **If count ≥ 5** (empirical): compute the mean of `saved_mm_cap{K}`.

4. **If count < 5** (model-filled):
   - Check feasibility: if ET bin center is outside the
     [n_cuttings × 50, ceiling] band, mask the cell (NaN).
   - Compute expected n_late = round(n_cuttings × late_count_frac),
     where late_count_frac is blended from WY-type and n_cuttings-specific
     calibrations.
   - If n_late = 0: savings = 0.
   - If n_late > 0: savings = n_remove × per_late_cut_et(ET), where
     n_remove = n_late (cap 0) or max(0, n_late − cap_k),
     and per_late_cut_et = base_per_cut × sqrt(ET / reference_ET).
     base_per_cut is calibrated from observed data by n_cuttings;
     reference_ET is the mean total ET for that n_cuttings.
   - Savings are capped at 55% of segment ET (physical maximum).

5. **Convert** mm → ac-ft/acre (÷ 304.8).

6. **Annotate** each cell with the savings value and (n_late) count.

### 4.2 Calibration parameters

Derived from the full dataset by grouping:

**Late count fraction** (fraction of total cuts that are late, by WY type):

| WY Type | late_count_frac | n |
|---------|----------------|---|
| Critical | 0.330 | 3,584 |
| Dry | 0.332 | 1,792 |
| Above Normal | 0.295 | 1,747 |
| Wet | 0.328 | 3,584 |

**Late count fraction by n_cuttings** (selected):

| n_cuttings | late_count_frac |
|------------|----------------|
| 1 | 0.15 |
| 4 | 0.27 |
| 7 | 0.34 |
| 10 | 0.30 |
| 14 | 0.25 |

The blended fraction = (WY-type frac + n_cuttings frac) / 2.

### 4.3 Cutoff-date sweep (panels i–l)

For each parcel-year, the actual late_cut_dates are re-evaluated against
11 cutoff dates from July 1 to December 1 at ~15-day intervals:

Jul 1, Jul 16, Aug 1, Aug 16, Sep 1, Sep 16, Oct 1, Oct 16, Nov 1, Nov 16, Dec 1

For each (cutoff_date, county, wy_type) cell:
1. Parse late_cut_dates for each parcel-year.
2. Count cuts on or after the new cutoff → n_late_at_cutoff.
3. Compute mean n_late_at_cutoff across all parcel-years in that cell.
4. Cells with < 5 observations are left blank.

Counties are ordered north → south: San Joaquin, Stanislaus, Merced, Madera,
Fresno, Tulare, Kings, Kern, Riverside, Imperial.

Results are adjusted by a literature-calibrated county ratio to account
for the known N→S gradient in annual cutting frequency that our BEAST-detected
data under-represents (observed mean ≈ 6.4–7.6 across counties vs literature
6–10). The adjustment ratio = literature_cuts / observed_cuts:

| County | Observed mean | Literature cuts | Adj ratio |
|--------|-------------|-----------------|-----------|
| San Joaquin | 6.4 | 6 | 0.94 |
| Stanislaus | 7.0 | 7 | 1.01 |
| Merced | 7.3 | 7 | 0.95 |
| Fresno | 7.1 | 8 | 1.13 |
| Kern | 7.2 | 9 | 1.24 |
| Riverside | 7.6 | 9 | 1.19 |
| Imperial | 7.6 | 10 | 1.32 |

Adjusted n_late = round(empirical_n_late × adj_ratio). Savings are scaled
proportionally using the empirical per-late-cut ET for each cell. This
preserves the WY-type and seasonal timing patterns from the observed data
while correcting the county-level cutting count to match published values
(Putnam et al. 2008; Orloff et al. 2015).

## 5. Figure Captions

### Figure: Water Savings Sensitivity Heatmap

**Full title:** Water Savings Sensitivity — Cutoff: July 1

**Panels (a)–(d): Cap 0 savings by water year type.**
Mean water savings (ac-ft/acre) from removing all late-season cuttings
(cap 0), shown as a function of segment ET (x-axis, 10 bins from 0–150
to 2100–2500 mm) and number of cuttings per water year (y-axis, 1–14).
Each panel corresponds to a San Joaquin Valley water year type:
(a) Critical, (b) Dry, (c) Above Normal, (d) Wet.
Cell values show savings in ac-ft/acre with the expected number of late
cuts in parentheses. Italic values are model-filled (< 5 observations);
roman values are empirical means (≥ 5 observations). Gray cells indicate
physically implausible ET × cuttings combinations. The colorbar scale
(0–2.8 ac-ft/acre) is shared across panels (a)–(d).

**Panels (e)–(h): Cap 1 savings by water year type.**
As panels (a)–(d), but under a policy of capping late cuttings at one
(cap 1). Savings represent only the ET from the second and subsequent
late cuttings. Cap 1 savings are zero when n_late ≤ 1. The colorbar
scale (0–2.2 ac-ft/acre) is shared across panels (e)–(h).

### Figure 2: Cutoff-Date Sensitivity

**Full title:** Cutoff-Date Sensitivity: Late Cuts & Water Savings
by County (N → S)

This is a separate figure (cutoff_sensitivity.png) with 12 panels
(3 rows × 4 columns).

**Panels (a)–(d): Mean late-cut count by county and cutoff date.**
Mean number of late cuttings (rounded to nearest integer) as a function
of cutoff date (x-axis, July 1 to December 1 at 15-day intervals) and
county ordered north to south (y-axis, San Joaquin → Imperial). Each
panel corresponds to a water year type: (a) Critical, (b) Dry,
(c) Above Normal, (d) Wet. Values are literature-adjusted to reflect
the known N→S gradient in annual cutting frequency (6 cuts in San Joaquin
to 10 in Imperial). Southern counties show more late cuts at July 1
(n_late ≈ 3) than northern counties (n_late ≈ 2).

**Panels (e)–(h): Cap 0 savings by county and cutoff date.**
Mean water savings (ac-ft/acre) from removing all late cuttings at each
cutoff date, by county (N→S) and water year type. Savings are derived
from the literature-adjusted late-cut counts × observed per-late-cut ET.
Imperial (south) shows ~1.7 ac-ft/acre savings at July 1 vs ~1.1 for
San Joaquin (north), reflecting both more late cuts and the longer
southern growing season.

**Panels (i)–(l): Cap 1 savings by county and cutoff date.**
As panels (e)–(h), but under cap 1 (keep one late cutting). Savings
are zero wherever n_late ≤ 1. The cap 1 scenario shows greatest
differentiation between northern and southern counties because it
amplifies the effect of having more total late cuts.

## 6. Results

### 6.1 Water year type effects

Critical and Dry years show the highest savings potential:

| WY Type | Cap 0 mean | Cap 1 mean | % with late cuts |
|---------|-----------|-----------|-----------------|
| Critical | 1.35 ac-ft/acre | 0.68 ac-ft/acre | 99.2% |
| Dry | 1.34 | 0.71 | 99.3% |
| Above Normal | 1.07 | 0.46 | 88.7% |
| Wet | 1.27 | 0.65 | 94.9% |

Above Normal years have 11% fewer parcels with late cuts and lower mean
savings, partly because the single Above Normal year (WY2024) had reduced
ET across all counties.

### 6.2 Number of cuttings and late-cut count

The July 1 cutoff classifies a remarkably consistent fraction of cuts
as late: median n_late = 2, IQR [2, 3], mean = 2.3. The relationship
between total cuttings and late cuttings:

- **1–2 cuttings**: n_late rounds to 0–1. For 1-cut parcels, the single
  cut is before July 1 in ~85% of cases → near-zero savings.
- **3–5 cuttings**: n_late ≈ 1. Savings ≈ 1 × 190–213 mm ≈ 0.62–0.70
  ac-ft/acre.
- **6–8 cuttings** (bulk of data): n_late ≈ 2. Savings ≈ 2 × 164–180 mm
  ≈ 1.08–1.18 ac-ft/acre.
- **9–14 cuttings**: n_late ≈ 3–5. Savings reach 1.4–2.8 ac-ft/acre,
  but these parcels are rare (< 15% of dataset).

### 6.3 Segment ET

Segment ET (total ET across all cutting cycles in the water year) is
strongly correlated with n_cuttings:

| n_cuttings | Median ET | p95 ET | Max ET |
|------------|----------|--------|--------|
| 2 | 247 mm | 531 mm | 774 mm |
| 5 | 716 mm | 998 mm | 1,303 mm |
| 7 | 948 mm | 1,234 mm | 1,512 mm |
| 10 | 1,059 mm | 1,367 mm | 1,562 mm |

Within the empirical cells, higher segment ET is associated with modestly
higher savings because parcels in warmer locations or with longer growing
seasons have both more ET and more late cuts. However, the per-late-cut ET
is driven by summer growing conditions and remains relatively stable
(170 ± 44 mm), so **n_late_cuts is the primary driver of savings, not
total segment ET**.

### 6.4 County gradient (north → south)

After literature adjustment for the known N→S cutting frequency gradient,
the cutoff-date sensitivity (Figure 2) shows the expected geographic
pattern at the July 1 cutoff:

| County | Adj ratio | n_late (Jul 1) | Cap 0 savings |
|--------|-----------|---------------|---------------|
| San Joaquin (N) | 0.94 | 2 | 1.11 ac-ft/acre |
| Fresno | 1.13 | 3 | 1.73 |
| Kern | 1.24 | 3 | 1.56 |
| Riverside | 1.19 | 3 | 1.73 |
| Imperial (S) | 1.32 | 3 | 1.71 |

Southern counties have ~50% more late cuts and ~55% higher savings than
northern counties at the July 1 cutoff. This reflects the longer growing
season (Feb–Nov in Imperial vs Mar–Sep in San Joaquin) that supports
9–10 annual cuttings in the south versus 6 in the north (Putnam et al.
2008; Orloff et al. 2015). The gradient narrows at later cutoffs as the
growing season ends for all counties by October.

### 6.5 Cutoff date sensitivity

Moving the cutoff date later reduces the number of cuts classified as
late, and consequently reduces savings:

| Cutoff date | Mean n_late (all counties) | Approx. savings reduction |
|-------------|--------------------------|--------------------------|
| July 1 | 2.3 | baseline |
| August 1 | 1.5 | −35% |
| September 1 | 0.6 | −74% |
| October 1 | ~0 | −100% |

By October 1, virtually no cuts remain "late" because the alfalfa growing
season ends in September for most of the study area. The sharp decline
between August 16 and September 16 marks the effective end of the
late-cutting window.

### 6.6 Per-late-cut ET stability

The per-late-cut ET of ~170 mm (0.56 ac-ft/acre) is the fundamental
"unit cost" of each late-season cutting in water terms. This value is
remarkably stable across:

- **Counties**: 159 mm (Kern) to 176 mm (Riverside), CV = 4%
- **Water year types**: 165 mm (Dry) to 177 mm (Above Normal), CV = 3%
- **n_cuttings groups**: 154 mm (9–14 cuts) to 205 mm (2–4 cuts), CV = 13%

The largest variation is with n_cuttings, where fewer-cut parcels have
slightly longer cutting cycles and thus higher per-cut ET. This modest
variation (±20%) is incorporated in the model through n_cuttings-specific
calibration of per_late_cut_et.

## 7. Output Files

All outputs are saved to `es_analysis/output/figures/water_savings_sim_2/`:

| File | Description |
|------|-------------|
| `png/sensitivity_heatmap.png` | Figure 1: ET sensitivity (300 DPI) |
| `pdf/sensitivity_heatmap.pdf` | Figure 1: vector for publication |
| `png/cutoff_sensitivity.png` | Figure 2: Cutoff sensitivity (300 DPI) |
| `pdf/cutoff_sensitivity.pdf` | Figure 2: vector for publication |
| `sensitivity_grid.csv` | ET grid (1,120 cells: 262 empirical, 320 model-filled, 538 masked) |
| `cutoff_county_grid.csv` | Cutoff grid (440 cells, literature-adjusted) |

### Grid CSV columns

- `cap_k` — cap scenario (0 or 1)
- `wy_type` — water year type
- `n_cuttings` — cutting count (1–14)
- `et_bin_idx`, `et_bin_label`, `et_bin_center` — ET bin
- `mean_saved_mm`, `mean_saved_acft` — mean savings
- `mean_n_late` — mean number of late cuts
- `count` — number of observations in cell
- `is_model` — True if model-filled, False if empirical

### Cutoff grid CSV columns

- `cutoff_idx`, `cutoff_label` — cutoff date index and label
- `county`, `county_idx` — county name and N→S index
- `wy_type` — water year type
- `mean_n_late` — literature-adjusted mean late-cut count (integer)
- `mean_savings_cap0_acft` — cap 0 savings (ac-ft/acre)
- `mean_savings_cap1_acft` — cap 1 savings (ac-ft/acre)
- `adj_ratio` — literature_cuts / observed_cuts for that county
- `is_sparse` — True if < 5 observations
