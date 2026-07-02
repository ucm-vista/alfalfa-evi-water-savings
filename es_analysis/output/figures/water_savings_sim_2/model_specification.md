# Water Savings Sensitivity Model — Full Mathematical Specification

## 1. Model Overview

The model estimates water savings (mm or ac-ft/acre) from restricting
late-season alfalfa cuttings. It operates on a three-dimensional
parameter space:

- **N** = total number of cuttings per water year (1-14)
- **ET** = total segment evapotranspiration (mm), i.e., the cumulative
  ET across all cutting cycles in the water year
- **w** = water year type (Critical, Dry, Above Normal, Wet)

And for the cutoff-date analysis, adds:

- **d** = cutoff date (July 1 through December 1)
- **c** = county (San Joaquin through Imperial, north to south)

The model produces savings under two policy scenarios:

- **Cap 0** (k=0): Remove all late-season cuttings
- **Cap 1** (k=1): Allow one late cutting, remove the rest

---

## 2. Core Savings Equation

For a given cell (N, ET, w, k):

```
S(N, ET, w, k) = min( n_remove × e_cut(N, ET),  0.55 × ET )
```

where:

| Symbol | Definition |
|--------|-----------|
| S | Water savings (mm) |
| n_remove | Number of late cuts removed under cap k |
| e_cut | Per-late-cut ET (mm), a function of N and ET |
| 0.55 × ET | Physical cap: savings cannot exceed 55% of total ET |

### 2.1 Number of late cuts removed

```
n_late(N, w) = round( N × λ(N, w) )

n_remove(N, w, k) = { n_late         if k = 0
                     { max(0, n_late - k)   if k > 0
```

If n_late = 0, then S = 0 (no late cuts to remove).

### 2.2 Late-count fraction λ

The fraction of total cuts that are "late" (after the cutoff date),
blended from two calibrations:

```
λ(N, w) = ( λ_w(w) + λ_N(N) ) / 2
```

| Symbol | Definition | Source |
|--------|-----------|--------|
| λ_w(w) | Late fraction by WY type | Calibrated from 10,707 parcel-years |
| λ_N(N) | Late fraction by n_cuttings | Calibrated from 10,707 parcel-years |

### 2.3 Per-late-cut ET with sqrt scaling

```
e_cut(N, ET) = e_base(N) × √( ET / ET_ref(N) )
```

Clamped to [0.4, 1.6] × e_base to prevent extreme extrapolation:

```
e_cut(N, ET) = e_base(N) × clamp( √(ET / ET_ref(N)),  0.4,  1.6 )
```

| Symbol | Definition |
|--------|-----------|
| e_base(N) | Observed mean per-late-cut ET at n_cuttings = N |
| ET_ref(N) | Observed mean total ET at n_cuttings = N |
| √(ET/ET_ref) | Sublinear scaling — empirical ratio ~1.35 between low/high ET quartiles matches √2 ≈ 1.41 |

### 2.4 Unit conversion

```
S_acft = S_mm / 304.8        (mm → ac-ft/acre, pure depth conversion)
```

---

## 3. Feasibility Constraints

Not all (N, ET) combinations are physically possible. The model masks
cells outside a diagonal feasibility band:

```
ET_floor(N) ≤ ET ≤ ET_ceil(N)
```

| Bound | Formula | Rationale |
|-------|---------|-----------|
| ET_floor | N × 50 mm | Each cutting cycle requires ≥ 50 mm ET (~20 days × 2.5 mm/day minimum growth) |
| ET_ceil | p99(ET | N) × 1.2 | Observed 99th percentile + 20% buffer; no parcel with N cuts can physically produce more ET |

Cells where ET_floor > ET_ceil (impossible) or outside the band are
set to NaN (gray in figure).

---

## 4. Calibration Parameters

All parameters are derived from 10,707 parcel-years across 10 counties,
WY2019-2024. The model is calibrated once; individual cell predictions
use these fixed parameters.

### 4.1 Late-count fraction by WY type: λ_w(w)

| w | λ_w | n |
|---|-----|---|
| Critical | 0.3304 | 3,584 |
| Dry | 0.3321 | 1,792 |
| Above Normal | 0.2955 | 1,747 |
| Wet | 0.3282 | 3,584 |

Computed as: λ_w = mean(n_late_cuts) / mean(n_cuttings) for all
parcel-years in WY type w.

### 4.2 Late-count fraction by n_cuttings: λ_N(N)

| N | λ_N | n |
|---|-----|---|
| 1 | 0.1481 | 27 |
| 2 | 0.2368 | 95 |
| 3 | 0.2545 | 148 |
| 4 | 0.2820 | 289 |
| 5 | 0.3356 | 779 |
| 6 | 0.3443 | 1,676 |
| 7 | 0.3399 | 3,050 |
| 8 | 0.3198 | 3,050 |
| 9 | 0.3048 | 1,284 |
| 10 | 0.3004 | 263 |
| 11 | 0.2791 | 43 |
| 12 | 0.2500 | 3 |
| 13 | 0.33 | 0 (fallback) |
| 14 | 0.33 | 0 (fallback) |

Computed as: λ_N = mean(n_late_cuts) / N for parcel-years with
n_cuttings = N. Pattern: rises from 0.15 (1 cut) to peak 0.34
(6-7 cuts) then declines to 0.25 (12 cuts).

### 4.3 Per-late-cut ET: e_base(N)

| N | e_base (mm) | e_base (ac-ft/acre) | n |
|---|-------------|--------------------|----|
| 1 | 210.3 | 0.690 | 4 |
| 2 | 232.2 | 0.762 | 40 |
| 3 | 213.4 | 0.700 | 86 |
| 4 | 195.3 | 0.641 | 205 |
| 5 | 189.9 | 0.623 | 680 |
| 6 | 180.4 | 0.592 | 1,604 |
| 7 | 171.4 | 0.562 | 3,029 |
| 8 | 164.2 | 0.539 | 3,044 |
| 9 | 157.0 | 0.515 | 1,284 |
| 10 | 144.4 | 0.474 | 263 |
| 11 | 130.0 | 0.427 | 43 |
| 12 | 116.5 | 0.382 | 3 |
| 13 | 170.0 | 0.558 | 0 (fallback) |
| 14 | 170.0 | 0.558 | 0 (fallback) |

Computed as: e_base(N) = mean( late_et_mm / n_late_cuts ) for
parcel-years with n_cuttings = N and n_late_cuts > 0.

Physical interpretation: each late cutting cycle consumes ~170 mm
(~0.56 ac-ft/acre) of ET, driven by summer conditions (~5 mm/day
× ~35 day regrowth cycle). The decline from 232 → 117 mm with
increasing N reflects shorter inter-cut intervals at higher
cutting frequencies.

### 4.4 Reference total ET: ET_ref(N)

| N | ET_ref (mm) | n |
|---|------------|---|
| 1 | 178 | 27 |
| 2 | 268 | 95 |
| 3 | 378 | 148 |
| 4 | 521 | 289 |
| 5 | 693 | 779 |
| 6 | 834 | 1,676 |
| 7 | 948 | 3,050 |
| 8 | 1,030 | 3,050 |
| 9 | 1,077 | 1,284 |
| 10 | 1,085 | 263 |
| 11 | 1,092 | 43 |
| 12 | 1,129 | 3 |
| 13 | 1,690 | 0 (fallback = N × 130) |
| 14 | 1,820 | 0 (fallback = N × 130) |

Computed as: ET_ref(N) = mean(total_et_mm) for parcel-years with
n_cuttings = N. Used as the denominator in the sqrt scaling ratio.

### 4.5 ET ceiling: ET_ceil(N)

| N | ET_ceil (mm) | Source |
|---|-------------|--------|
| 1 | 408 | p99 × 1.2 |
| 2 | 904 | p99 × 1.2 |
| 3 | 1,002 | p99 × 1.2 |
| 4 | 1,241 | p99 × 1.2 |
| 5 | 1,432 | p99 × 1.2 |
| 6 | 1,529 | p99 × 1.2 |
| 7 | 1,604 | p99 × 1.2 |
| 8 | 1,690 | p99 × 1.2 |
| 9 | 1,766 | p99 × 1.2 |
| 10 | 1,753 | p99 × 1.2 |
| 11 | 1,796 | p99 × 1.2 |
| 12 | 1,788 | p99 × 1.2 |
| 13 | 3,250 | N × 250 fallback |
| 14 | 3,500 | N × 250 fallback |

### 4.6 ET floor: ET_floor(N)

```
ET_floor(N) = N × 50 mm
```

| N | ET_floor (mm) |
|---|--------------|
| 1 | 50 |
| 5 | 250 |
| 7 | 350 |
| 10 | 500 |
| 14 | 700 |

---

## 5. Cutoff-Date Model (Figure 4)

The cutoff-date model extends the core savings equation to sweep the
cutoff date from July 1 to December 1, with county-specific parameters.

### 5.1 Modeled cut timing

Cuts are modeled as evenly spaced across the water year (Oct 1 → Sep 30):

```
t_i = Oct 1 + (i + 0.5) × (365 / N_c)     for i = 0, 1, ..., N_c - 1
```

where N_c = literature-informed annual cuts for county c.

### 5.2 Literature-informed annual cuts: N_c

| County | N_c | Source |
|--------|-----|--------|
| San Joaquin | 8 | Putnam et al. 2008 |
| Stanislaus | 8 | " |
| Merced | 8 | " |
| Madera | 8 | " |
| Fresno | 9 | Orloff et al. 2015 |
| Tulare | 9 | " |
| Kings | 9 | " |
| Kern | 10 | " |
| Riverside | 12 | UC ANR Imperial/desert reports |
| Imperial | 12 | " |

### 5.3 County per-late-cut ET: e_c

| County | e_c (mm) | e_c (ac-ft/acre) |
|--------|---------|------------------|
| San Joaquin | 174 | 0.571 |
| Stanislaus | 164 | 0.538 |
| Merced | 161 | 0.528 |
| Madera | 176 | 0.577 |
| Fresno | 173 | 0.567 |
| Tulare | 167 | 0.548 |
| Kings | 172 | 0.564 |
| Kern | 159 | 0.522 |
| Riverside | 176 | 0.577 |
| Imperial | 175 | 0.574 |

Computed from observed data: e_c = mean(late_et_mm / n_late_cuts)
for parcel-years in county c with n_late_cuts > 0.

### 5.4 Number of late cuts at cutoff date d

```
n_late(c, d) = | { i : t_i ≥ d } |
```

i.e., count of modeled cut dates on or after cutoff date d.

### 5.5 WY-type scaling

```
n_late(c, d, w) = round( n_late(c, d) × φ_w )
```

where φ_w is a WY-type scaling factor derived from calibration:

| w | φ_w |
|---|-----|
| Critical | 1.028 |
| Dry | 1.033 |
| Above Normal | 0.919 |
| Wet | 1.021 |

Computed as: φ_w = λ_w(w) / mean(λ_w) across all WY types.

### 5.6 Cutoff savings

```
S_cap0(c, d, w) = n_late(c, d, w) × e_c

S_cap1(c, d, w) = max(0, n_late(c, d, w) - 1) × e_c
```

No sqrt scaling or savings cap applied here because the cutoff model
does not vary ET — per-cut ET is fixed at the county-calibrated value.

---

## 6. Worked Examples

### Example 1: ET sensitivity cell

**Input**: N=7 cuttings, ET=900 mm, w=Critical, k=0 (cap 0)

Step 1 — Late-count fraction:
```
λ(7, Critical) = (λ_w(Critical) + λ_N(7)) / 2
               = (0.3304 + 0.3399) / 2
               = 0.3352
```

Step 2 — Number of late cuts:
```
n_late = round(7 × 0.3352) = round(2.346) = 2
```

Step 3 — Per-cut ET with sqrt scaling:
```
e_cut = 171.4 × √(900 / 947.7)
      = 171.4 × √(0.9497)
      = 171.4 × 0.9745
      = 167.0 mm
```

Step 4 — Raw savings:
```
S_raw = 2 × 167.0 = 334.1 mm
```

Step 5 — Savings cap check:
```
0.55 × 900 = 495 mm > 334.1 → not capped
```

Step 6 — Final:
```
S = 334.1 mm = 334.1 / 304.8 = 1.10 ac-ft/acre
```

### Example 2: Cutoff-date cell

**Input**: county=Imperial, d=Aug 1, w=Critical

Step 1 — Generate 12 evenly-spaced cuts (Oct 1 → Sep 30):
```
interval = 365 / 12 = 30.4 days
t_0 = Oct 1 + 15.2 days = Oct 16
t_1 = Nov 16, t_2 = Dec 16, t_3 = Jan 16, t_4 = Feb 15,
t_5 = Mar 17, t_6 = Apr 17, t_7 = May 17, t_8 = Jun 17,
t_9 = Jul 17, t_10 = Aug 17, t_11 = Sep 16
```

Step 2 — Count cuts ≥ Aug 1:
```
Aug 17, Sep 16 → n_late_base = 2
```

Step 3 — WY-type scaling:
```
n_late = round(2 × 1.028) = round(2.056) = 2
```

Step 4 — Savings:
```
S_cap0 = 2 × 175 mm = 350 mm = 1.15 ac-ft/acre
S_cap1 = 1 × 175 mm = 175 mm = 0.57 ac-ft/acre
```

### Example 3: Savings cap binding

**Input**: N=14, ET=1000 mm, w=Dry, k=0

Step 1:
```
λ(14, Dry) = (0.3321 + 0.33) / 2 = 0.3311
n_late = round(14 × 0.3311) = round(4.635) = 5
```

Step 2:
```
e_cut = 170.0 × √(1000 / 1820) = 170.0 × 0.741 = 126.0 mm
```

Step 3:
```
S_raw = 5 × 126.0 = 630.0 mm
0.55 × 1000 = 550 mm < 630 → CAPPED
S = 550 mm = 1.80 ac-ft/acre
```

---

## 7. Assumptions Summary

| # | Assumption | Justification |
|---|-----------|---------------|
| 1 | July 1 defines "late" cuts | Transition from peak to shoulder season in SJV |
| 2 | Per-cut ET is stable (~170 mm) | Observed CV = 26%; driven by summer daily ET × cycle length |
| 3 | Per-cut ET scales as √(ET/ref) | Empirical ratio ~1.35 between quartiles matches √2 |
| 4 | Late ET ≤ 55% of total ET | p95 of observed late_et_frac = 0.63; 55% is conservative |
| 5 | ET floor = N × 50 mm | Minimum ~50 mm ET per cutting cycle for biomass production |
| 6 | ET ceiling = p99 × 1.2 | Physical max ET for a given cutting frequency |
| 7 | Blended λ = (λ_w + λ_N) / 2 | Equal weight to WY-type and n_cuttings effects |
| 8 | Evenly-spaced cuts (cutoff model) | Simplification; actual timing varies by phenology |
| 9 | Southern counties: 12 cuts/yr | Literature: Putnam 2008, Orloff 2015, UC ANR |
| 10 | Northern counties: 8 cuts/yr | Same sources |

---

## 8. Model Limitations

1. **No spatial ET variation in ET sensitivity model** — per-cut ET
   is calibrated by n_cuttings pooled across counties, not county-specific.
   The cutoff model uses county-specific per-cut ET.

2. **Integer rounding of n_late** — creates step discontinuities.
   At boundary values of N × λ ≈ k + 0.5, small parameter changes
   cause n_late to jump by 1.

3. **Evenly-spaced cuts** — real cuts cluster in spring-summer with
   wider spacing in winter. The uniform model overestimates winter
   cuts and underestimates summer cut density.

4. **No deficit irrigation** — the model assumes full ET demand is met.
   Under deficit irrigation, per-cut ET would be lower.

5. **Static calibration** — parameters are fixed from 2019-2024 data.
   Climate trends or management changes would shift the calibration.
