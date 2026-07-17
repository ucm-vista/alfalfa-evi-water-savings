# Revisions — Cuttings vs. Air-Temperature Correlations

Pearson correlations between **number of alfalfa cuttings** and **air temperature**,
at each aggregation level (positive = warmer → more cuttings). Computed from
`data/cuttings_temperature.parquet` (WY2019–2024, 10 counties).

| Level (what's correlated) | n | Mean air temp (T_mean) | Max air temp (T_max) |
|---|---|---|---|
| **Per parcel-year** (the scatter figure) | 10,699 | **r = +0.078** (R² = 0.006) | **r = +0.074** (R² = 0.005) |
| **County–water-year means** | 60 | r = +0.340 (R² = 0.115) | r = +0.391 (R² = 0.153) |
| **County means** (the quadrant figure) | 10 | **r = +0.810** (R² = 0.657) | **r = +0.824** (R² = 0.679) |

The two figures correspond to the first and last rows:

- **`cuttings_temperature_scatter.png`** → per-parcel: **r ≈ 0.08** (T_mean) / **0.07** (T_max)
  — very weak; temperature explains <1% of field-to-field variation.
- **`cuttings_temperature_quadrant.png`** → county means: **r ≈ 0.81** (T_mean) / **0.82** (T_max)
  — strong; temperature explains ~66–68% of the between-county variation.

The jump from 0.08 → 0.81 as you aggregate is the core result: the cuttings–temperature
link is a **regional/spatial signal** (counties along the thermal gradient), not a
field-level one — within any given thermal zone, parcel-to-parcel differences
(management, water, stand age) dominate.

Note the county-level r = 0.81/0.82 reflects only 10 points, so its confidence interval
is wide even though it is significant. Maximum temperature correlates marginally higher
than mean temperature at every level, but the difference is small.

## Definitions
- **T_mean** = mean air temperature = (T_min + T_max)/2; **T_max** = maximum air temperature.
  Both are daily Daymet temperatures averaged over each parcel's detected cut-cycle windows.
- **Parcel-year** = one alfalfa field in one water year (WY; 1 Oct – 30 Sep).
- **Cuttings** = alfalfa harvest events detected from Harmonized Landsat Sentinel-2 (HLS)
  Enhanced Vegetation Index (EVI) time series via the BEAST change-point pipeline.

---

# Revisions — July 1 Cutoff Sensitivity to Alternative Cutoff Dates

Source: `output/figures/water_savings_sim_2/` (`cutoff_county_grid.csv`, literature-adjusted;
acreage from `basin_impact_data.csv`, basin total ≈ 540,000 alfalfa acres). Figure:
`water_savings_sim_2/png/cutoff_sensitivity.png`.

## Background
July 1 is the baseline "late-season" cutoff (`config.late_cutoff_month=7, late_cutoff_day=1`):
cuts on/after this date are "late" and are the candidates for restriction. The sweep re-counts
each parcel-year's detected cut dates against alternative cutoffs from **Jul 1 → Oct 1 at ~15-day
steps**. Because per-late-cut ET is nearly constant (~170 mm ≈ 0.56 ac-ft/acre), savings scale
almost linearly with how many cuts fall after the cutoff. (Note: the sweep can only test cutoffs
**on/after Jul 1**; the saved late-cut dataset retains only cycles already late under the Jul 1
baseline. Earlier cutoffs need a rebuild via `build_late_cut_dataset(cutoff_month=6, ...)`. Also,
the methods doc text says Jul 1–Dec 1 / 11 steps, but the figure/CSV actually span Jul 1–Oct 1 / 7
steps — reconcile before publication.)

## Mean & total potential water savings by cutoff
Cap 0 = remove all late cuts; Cap 1 = keep one late cut. Total = per-acre x county acreage, summed
(thousand acre-feet per year, kAF/yr).

| Cutoff | Mean cap 0 (ac-ft/acre) | Total cap 0 (kAF/yr) | % of Jul 1 | Mean cap 1 (ac-ft/acre) | Total cap 1 (kAF/yr) |
|---|---|---|---|---|---|
| **Jul 1** | 1.16 | **645** | 100% | 0.47 | 266 |
| Jul 16 | 0.95 | 526 | 82% | 0.32 | 178 |
| Aug 1 | 0.72 | 398 | 62% | 0.16 | 86 |
| Aug 16 | 0.47 | 260 | 40% | 0.02 | 9 |
| Sep 1 | 0.22 | 125 | 19% | 0.00 | 0 |
| Sep 16 | 0.03 | 18 | 3% | 0.00 | 0 |
| Oct 1 | 0.00 | 0 | 0% | 0.00 | 0 |

July 1 yields ~645 kAF/yr basin-wide under cap 0 (~266 kAF under cap 1). A one-month delay to Aug 1
drops this ~38% (to ~398 kAF); by Sep 1 only ~19% (~125 kAF) remains. The steepest fall is between
mid-August and mid-September, marking the effective end of the late-cutting window. Cap-1 savings
decay fastest (≈0 by Aug 16) because later cutoffs push most parcels to ≤1 late cut.

## County ranking — stable through early August, then collapses (does not reorder)
Spearman rank correlation of per-acre cap-0 savings vs. the July 1 ordering:

| vs Jul 1 | Jul 16 | Aug 1 | Aug 16 | Sep 1 | Sep 16 |
|---|---|---|---|---|---|
| rho | +0.92 | +0.76 | +0.44 | +0.14 | -0.51 |

The ranking is robust through ~Aug 1 (rho 0.76-0.92). Riverside and Imperial (south) are the top two
at every cutoff through Sep 1; only the #3 slot shuffles (Kings -> Merced by Aug 1). The decline
after mid-August is not genuine reordering — absolute savings collapse toward zero everywhere, so
ranks become tie/noise-dominated (rho undefined at Oct 1, all zeros).

## Water-year-type ranking — essentially invariant
Mean per-acre cap-0 savings (ac-ft/acre) by WY type, same order at every cutoff Jul 1 -> Sep 16:

| WY type | Jul 1 | Aug 1 | Sep 1 |
|---|---|---|---|
| Critical | 1.33 | 0.84 | 0.26 |
| Dry | 1.30 | 0.80 | 0.23 |
| Wet | 1.15 | 0.73 | 0.23 |
| Above Normal | 0.84 | 0.50 | 0.14 |

Order is **Critical > Dry > Wet > Above Normal** at every cutoff (only the all-zero Oct 1 column
flips meaninglessly). Drought years (Critical/Dry) always offer the most savings; Above-Normal the
least.

## Bottom line
Moving the cutoff later scales the *magnitude* of savings down steeply but leaves the
*who-saves-most* story intact — south > north among counties and Critical/Dry > Wet > Above-Normal
among year types hold robustly until savings vanish near October.

---

# Revisions — Potential Water Savings: Uncorrected vs. Corrected OpenET ET

Late-cut potential water savings computed two ways — from **uncorrected** OpenET actual ET
(ETa) and from **pass-timing–corrected** ET — reported by water year and by county. Cap 0 =
remove all late cuts (cut date ≥ Jul 1); Cap 1 = keep one late cut. All ET correction data are
Method A, 500 bootstrap replicates, from `alfalfa_run_6/et_correction/` (the same run behind
`et_correction_summary.png`).

## Figures
- **Uncorrected** (OpenET ETa): `water_savings/png/wy_savings_combined_stacked_uncorrected.png`
  (by WY) and `county_savings_combined_stacked_uncorrected.png` (by county).
- **Corrected**: `water_savings/png/wy_savings_combined_stacked_corrected.png` and
  `county_savings_combined_stacked_corrected.png`.
- Augmented parcel-year data: `water_savings/late_cut_savings_parcel_year_mm_with_corrected.csv`.

Figure titles now state the ET mode explicitly ("Uncorrected ET (OpenET ETa)" vs "Corrected ET").
The earlier ambiguous no-suffix files (titled "Corrected ET" but holding uncorrected data) were
removed.

## Method — deriving corrected savings
Each parcel-year's uncorrected late savings are scaled by that parcel's late-season (Jul–Sep)
corrected/open ET ratio:

    saved_corrected_mm_cap{k} = saved_mm_cap{k} × R_late
    R_late = Σ ET_corr(Jul,Aug,Sep) / Σ ET_open(Jul,Aug,Sep)   (per parcel-year)

`R_late` comes from `offphase_monthly_long_methodA…boot500`. Because late cuts fall inside the
Jul–Sep window, this proportional scaling closely tracks a per-cut-cycle recompute. All 10,707
parcel-years matched; mean `R_late` = 0.983 (mean late-window reduction 1.7%).

## The correction is small and only mildly seasonal
Unlike a large seasonal redistribution, the run-6 correction is modest across the whole year and
only slightly elevated in summer (total-weighted, 10 counties, WY2019–2024):

| Window | ET reduction |
|---|---|
| Whole water year (`et_correction_summary.png`) | 1.4% |
| Jul–Sep late-cut window | 1.7% |
| Peak month (Jul) | 2.1% |

Because the late-cut window sees essentially the same small correction as the annual mean,
**corrected and uncorrected savings are nearly identical** (≈1.7% apart). See Figure 5.

## By water year (Cap 0 = remove all late cuts)
Totals pooled across all 10 counties (thousand acre-feet per year, kAF/yr).

| WY (type) | Uncorrected (ac-ft/acre) | Corrected (ac-ft/acre) | Δ | Uncorrected total (kAF/yr) | Corrected total (kAF/yr) |
|---|---|---|---|---|---|
| WY2019 (Wet)          | 1.37 | 1.36 | −0.9% | 94.2 | 93.5 |
| WY2020 (Dry)          | 1.34 | 1.33 | −0.9% | 92.2 | 91.4 |
| WY2021 (Critical)     | 1.36 | 1.30 | −4.3% | 92.2 | 88.8 |
| WY2022 (Critical)     | 1.33 | 1.31 | −1.7% | 92.0 | 90.5 |
| WY2023 (Wet)          | 1.16 | 1.15 | −1.2% | 77.9 | 76.8 |
| WY2024 (Above Normal) | 1.07 | 1.06 | −1.4% | 71.2 | 70.3 |

## By county (Cap 0), COUNTY_ORDER
Per-county totals are the annual mean over WY2019–2024 (kAF/yr).

| County | Uncorrected (ac-ft/acre) | Corrected (ac-ft/acre) | Δ | Uncorrected total (kAF/yr) | Corrected total (kAF/yr) |
|---|---|---|---|---|---|
| San Joaquin | 1.24 | 1.20 | −3.6% | 15.0 | 14.4 |
| Stanislaus  | 1.28 | 1.27 | −1.1% |  5.1 |  5.0 |
| Merced      | 1.29 | 1.29 | −0.2% |  7.3 |  7.3 |
| Madera      | 1.21 | 1.20 | −1.0% |  2.0 |  2.0 |
| Fresno      | 1.27 | 1.25 | −1.7% |  7.5 |  7.4 |
| Tulare      | 1.29 | 1.28 | −0.6% |  6.2 |  6.1 |
| Kings       | 1.28 | 1.27 | −1.0% |  3.4 |  3.3 |
| Kern        | 1.18 | 1.16 | −1.7% | 11.3 | 11.1 |
| Riverside   | 1.35 | 1.32 | −2.1% | 18.4 | 18.0 |
| Imperial    | 1.22 | 1.21 | −0.5% | 10.5 | 10.5 |

## Bottom line
Pass-timing correction lowers estimated late-cut water savings by only ≈1–2% (cap 0; largest in
San Joaquin −3.6% and WY2021 −4.3%, negligible in Merced/Imperial). It changes no ranking — the
who-saves-most story (south > north; Critical/Dry > Wet > Above-Normal) is identical between the
uncorrected and corrected series. This is a **robustness result**: the water-savings conclusions
do not depend on the ET correction. Either series can be reported; we suggest the uncorrected
OpenET-native savings as primary with a one-line note that correction changes them by <2%.

---

## Figure 5 (main manuscript) — `et_corrections/png/et_correction_combined.png`

**ET correction: annual magnitude and seasonal distribution (10 counties, WY2019–2024).**
(a) Mean annual ET per county, actual OpenET (orange) vs. pass-timing–corrected (green); red
labels give the percent reduction, error bars are 90% bootstrap CIs on the corrected mean. The
correction lowers annual ET by only 0.3–2.2% across counties. (b) Mean ET by water-year month
(Oct→Sep), same actual-vs-corrected comparison; the correction is small in every month and only
mildly elevated through the Jul–Sep late-cut window (shaded), peaking at 2.1% in July. Because the
late-cut window experiences nearly the same small correction as the annual mean (1.7% vs 1.4%),
correcting for satellite pass timing changes potential late-cut water savings by <2% and does not
alter any county or water-year-type ranking (Method A, 500 bootstrap replicates).

---

# Revisions — ET Correction (Method A): Equation, Recovery Shape, Window, Aggregation

Specifics for the correction used in run-6 (**Method A**), from
`es_analysis/data_providers/et_provider.py` (`compute_daily_and_monthly_for_uid`).

## Correction equation
Per cutting at harvest day *h* with next clear Landsat pass *p*; off-phase days `t ∈ [h, p)`:

    Δf   = max( median(ETof[h−5 : h−1]) − quantile_0.25(ETof[h : h+5]), 0 ) × (1 + cloud_gap/16)
    δ(t) = Δf · ETo(t)                                   for each day t in [h, p)
    ET_corr(t)     = max( ET_open(t) − δ(t), 0 )                       # per day
    ET_corr(month) = max( Σ_{t∈month} ET_open − Σ_{t∈month} δ, 0 )     # summed to months

`Δf` is the observed drop in ET fraction (ETof = ETa/ETo, a Kc-like fraction): pre-cut median
(5-day window `[h−5, h−1]`) minus post-cut 0.25-quantile (5-day window `[h, h+5]`), inflated by
`(1 + cloud_gap/16)` when clouds delay the clear pass. `ETo` = daily reference ET (mm/day), so
`Δf·ETo` is the daily mm of over-counted ET removed. Multiple cuttings' `δ` add. 90% bootstrap CIs
(`n_boot = 500`) come from resampling the pre-/post-cut `ETof` values.

## Recovery shape — **stepwise / rectangular**
`Δf` is held **constant** across every day of the gap; the daily correction varies only because
`ETo` (weather) varies. It is **not** linear, **not** exponential, and **not** based on observed
EVI recovery. The deficit *magnitude* is empirical (from observed `ETof`), but its time profile is
flat.

## Recovery window length — **variable, data-driven**
The deficit is applied over the full off-phase gap = **[harvest date, first *clear* Landsat
pass)**, set by satellite revisit and cloud cover (nominally ~8–16 days; longer when clouds delay
the next clear scene). Not a fixed window.

## Daily then aggregated monthly — **yes**
`δ` is accumulated on a **daily** index (per off-phase day, summed across all cuttings), then
aggregated to months by summation: `ET_corr(month) = Σ_days ET_open − Σ_days δ`, clipped at ≥ 0.
Never a monthly-level approximation.

---

# Revisions — PlanetScope Independent Validation of Cut Dates

Independent cross-sensor check of the BEAST-detected cutting dates using **PlanetScope**
(~3 m, near-daily) EVI — a sensor entirely separate from the HLS (Harmonized Landsat–Sentinel-2)
data the cut detection is built on. Code + outputs: `es_analysis/planet_validation/`.

## Design
- **Parcels:** 2 each in San Joaquin (north), Kern (central), Imperial (south), WY2022, selected
  from the cuttings dataset behind `cuttings_analysis/*/boxplot_year_colored.png`
  (`data/multicounty_matched.parquet`; `cut_match_ratio=1.0`, `n_cuttings` 4–8).
- **Cut dates = TROUGHS only.** We validate the physical cut signature — the bare-field EVI
  **minimum** (`matched_minima_iso`), not the pre-harvest peak change-points. 2 troughs/parcel → 12 windows.
- **PlanetScope EVI:** same EVI formula as HLS (`2.5·(NIR−Red)/(NIR+6·Red−7.5·Blue+1)`), median over
  each parcel's clear pixels, read cloud-native (windowed COG clips via GDAL `/vsicurl`; no full
  scenes downloaded; free metadata search). Cloud filter matches the HLS EVI pipeline (scene ≤50%)
  with a stricter per-parcel clear-pixel gate (≥85%, since PlanetScope's UDM2 mask is coarser).
- **Metric:** PlanetScope post-cut EVI trough date − BEAST trough date (0 = perfect agreement).

## Results — all 12 troughs confirmed within ±5 days
| Region | County | Parcel | Cut | BEAST trough | PlanetScope trough | Offset (d) | EVI drop |
|---|---|---|---|---|---|---|---|
| North | San Joaquin | 3901548 | 1 | 2022-05-04 | 2022-05-06 | +2 | 0.47 |
| North | San Joaquin | 3901548 | 2 | 2022-08-19 | 2022-08-23 | +4 | 0.41 |
| North | San Joaquin | 3902388 | 1 | 2022-04-14 | 2022-04-17 | +3 | 0.49 |
| North | San Joaquin | 3902388 | 2 | 2022-07-27 | 2022-07-25 | −2 | 0.45 |
| Central | Kern | 1501484 | 1 | 2022-05-03 | 2022-04-30 | −3 | 0.42 |
| Central | Kern | 1501484 | 2 | 2022-06-29 | 2022-06-26 | −3 | 0.36 |
| Central | Kern | 1507861 | 1 | 2022-04-13 | 2022-04-11 | −2 | 0.42 |
| Central | Kern | 1507861 | 2 | 2022-07-09 | 2022-07-10 | +1 | 0.53 |
| South | Imperial | 1300103 | 1 | 2022-05-02 | 2022-05-01 | −1 | 0.35 |
| South | Imperial | 1300103 | 2 | 2022-07-11 | 2022-07-12 | +1 | **0.12** (weak) |
| South | Imperial | 1300275 | 1 | 2022-04-24 | 2022-04-21 | −3 | 0.41 |
| South | Imperial | 1300275 | 2 | 2022-08-31 | 2022-08-27 | −4 | **0.05** (weak) |

**Summary (125 PlanetScope observations):** mean |offset| = **2.4 d**, median |offset| = **2 d**,
mean bias = **−0.6 d** (no systematic lag), **100% within ±5 d**.

**Figures** (`planet_validation/figures/png/`):
- **`cutdate_validation_overview.png`** — comprehensive summary figure: (a) dumbbell timeline of all 12
  cuts (this-study date `|` vs PlanetScope date `●`, grouped/coloured by county; the two weak Imperial
  cases shown open `○`); (b) 1:1 agreement scatter with the ±5 d band (mean |Δ| 2.4 d, median 2 d, bias
  −0.6 d, 100% within ±5 d, n=12).
- Best / worst exemplars: **`south_Imperial_1300103_cut1.png`** (−1 d) and
  **`north_San_Joaquin_3901548_cut2.png`** (+4 d).
- Per-cut overlays for all 12: `<region>_<county>_<parcel>_cut<n>.png` (HLS EVI curve, this-study cut
  date, PlanetScope EVI + trough).

Data: `planet_evi_points.csv` (per-observation), `validation_summary.csv` (per-cut + overall).

## Figure captions
**`cutdate_validation_overview.png` — Independent PlanetScope validation of BEAST-detected alfalfa
cutting dates (WY2022; 6 parcels across San Joaquin, Kern, and Imperial counties).** (a) For each of
the 12 cut cycles (rows, grouped and coloured by county), the this-study cut date (BEAST EVI trough;
black tick) and the independently observed PlanetScope EVI-trough date (filled circle) are shown on a
calendar axis; the connecting line is the day-offset between them. The two weak-signal Imperial
late-season cases (per-cut EVI drop < 0.13) are drawn as open circles. (b) Agreement between the
this-study and PlanetScope cut dates (day-of-year), with the 1:1 line and a ±5-day band; points are
coloured by county (open = weak). All 12 cuts agree within ±5 days (mean absolute offset 2.4 d,
median 2 d, bias −0.6 d; n = 12).

**`south_Imperial_1300103_cut1.png` — Best-case validation example (Imperial County parcel 1300103,
cut 1).** HLS EVI (smoothed line plus cloud-screened observations) defines the cut cycle; the
this-study cut date (BEAST EVI trough; black line) is marked, and independent PlanetScope median-EVI
observations (orange) trace the same post-cut trough. The PlanetScope trough (dashed) precedes the
this-study cut date by 1 day (PlanetScope date − this-study cut date = −1 d).

**`north_San_Joaquin_3901548_cut2.png` — Worst-case validation example (San Joaquin County parcel
3901548, cut 2).** Panels as above; here the independent PlanetScope EVI trough follows the this-study
cut date by 4 days (PlanetScope date − this-study cut date = +4 d) — the largest offset among the 12
validated cuts, still within the ±5-day agreement band.

## Caveats
- **Two Imperial late-season cases are weak-signal** (parcel 1300103 cut 2 and 1300275 cut 2, EVI
  drop <0.13): by mid/late summer these fields show almost no cut cycle (EVI ~0.14–0.21), so the
  "trough" is near-noise. Both still land within ±5 d, but they are not strong confirmations —
  consistent with Imperial County's known degraded signal (see CLAUDE.md).
- PlanetScope and HLS differ spectrally, so absolute EVI differs; the validation is of trough
  **timing**, not EVI magnitude. Across the other 10 cases the two sensors' EVI trajectories track
  closely (e.g. Kern 1507861 cut 2).

## Bottom line
An independent 3 m, near-daily sensor reproduces the BEAST-detected alfalfa cut dates to within a
few days (mean 2.4 d, no bias), across the north–south extent of the study area. The cut-date
detection underpinning the ET and water-savings analyses is well supported by external data.
