# Multi-Strategy Water Savings Simulation Plan

## 1. Objective

Extend the base water savings sensitivity model to simulate and compare
all feasible water management strategies for late-season alfalfa across
the Central and Southern San Joaquin Valley. The analysis answers:

> **"Given a parcel's county, cutting frequency, water year type, water
> price, and water source — what is the optimal strategy for balancing
> water savings against hay revenue, and how much water can be saved
> at the basin scale?"**

Our existing model (Figures 1-4) serves as the **base case** (Strategy S2:
remove late cuts under full irrigation). This extension adds deficit
irrigation, economic crossover analysis, and basin-scale aggregation.

---

## 2. Strategy Definitions

Eight strategies spanning from no intervention to maximum water savings:

### S0: Base case — full production
- All cuts taken, full irrigation throughout
- Water saved: 0
- Yield loss: 0%
- Revenue loss: $0/acre
- **Benchmark against which all others are measured**

### S1: Cap 1 — keep one late cut
- Remove all late cuts except the first one after July 1
- Full irrigation maintained
- Water saved: (n_late − 1) × e_cut ac-ft/acre
- Yield loss: 10-15% (1-2 fewer harvests)
- Revenue loss: (n_late − 1) × $162/acre

### S2: Cap 0 — remove all late cuts (our base model)
- Remove all cuts after July 1, continue irrigating
- Water saved: n_late × e_cut ac-ft/acre
- Yield loss: 20-25% (2-3 fewer harvests)
- Revenue loss: n_late × $162/acre

### S3: Deficit irrigation only (Montazar)
- Keep all cuts, withhold irrigation for ~60 days (Jul-Sep)
- Daily ET drops from ~7 to ~2 mm/d (factor 0.29, Montazar et al. 2026)
- Growers typically skip 1-2 cuts due to no growth
- Water saved: deficit_duration × (ET_full − ET_deficit) mm
- Yield loss: 10-15% (reduced growth + skipped cuts)
- Revenue loss: ~$200-300/acre

### S4: Selective deficit — deficit on final 2 cycles only
- Full irrigation through most of season, deficit only during last
  2 cutting cycles before September
- Shortest intervention with measurable savings
- Water saved: 0.3-0.6 ac-ft/acre
- Yield loss: ~5-8% (last 2 cuts lower yield)
- Revenue loss: ~$80-130/acre

### S5: Hybrid — deficit + cap 1
- Deficit irrigate during late season AND keep only 1 late cut
- Water saved: S_deficit + 1 late cut × e_cut × deficit_factor
- Yield loss: 20-30%
- Revenue loss: $350-550/acre

### S6: Hybrid — deficit + cap 0
- Deficit irrigate AND remove all late cuts — maximum savings
- Water saved: S_deficit + n_late × e_cut × deficit_factor
- Yield loss: 30-40%
- Revenue loss: $500-700/acre

### S7: Early termination — end season August 1
- Stop all irrigation August 1, resume October for fall growth
- Effectively eliminates 2-3 months of production
- Water saved: 1.2-2.0 ac-ft/acre
- Yield loss: 25-35%
- Revenue loss: $400-600/acre

---

## 3. Parameter Framework

### 3.1 Water parameters by county

| County | Source | Normal $/ac-ft | Drought $/ac-ft | SGMA pressure | GW pump $/ac-ft |
|--------|--------|---------------|----------------|---------------|----------------|
| San Joaquin | CVP + GW | 175 | 600 | Moderate | 80 |
| Stanislaus | TID/MID + GW | 150 | 500 | Moderate | 90 |
| Merced | Merced ID + GW | 175 | 700 | High | 110 |
| Madera | Madera ID + GW | 200 | 750 | High | 120 |
| Fresno | CVP/Friant + GW | 200 | 800 | High | 130 |
| Tulare | CVP/Friant + GW | 250 | 1000 | Very High | 170 |
| Kings | Kings R + GW | 250 | 1000 | Very High | 160 |
| Kern | SWP + GW + transfers | 350 | 1200 | Extreme | 220 |
| Riverside | Colorado R + GW | 200 | 600 | Moderate | 100 |
| Imperial | Colorado R (IID) | 22 | 22 | None | 50 |

### 3.2 Hay market parameters

| Cut timing | Yield (tons/ac) | Quality | Price ($/ton) | Revenue ($/acre) |
|-----------|----------------|---------|-------------|-----------------|
| Spring (Mar-May) | 1.3-1.5 | Supreme/Premium | 260-350 | 340-525 |
| Early summer (Jun-Jul) | 1.0-1.2 | Premium/Good | 210-260 | 210-310 |
| Late summer (Jul-Sep) | 0.8-1.0 | Good/Fair | 170-210 | 135-210 |
| Late season (Aug-Oct) | 0.6-0.8 | Fair/Utility | 140-180 | 85-145 |
| Deficit period | 0.3-0.5 | Utility/Stub | 110-140 | 35-70 |

Key: late cuts are **lowest yield, lowest quality, lowest price** — the
weakest link in the production chain and the best candidate for removal.

### 3.3 Deficit irrigation parameters (from Montazar et al. 2026)

| Parameter | Value | Source |
|-----------|-------|--------|
| ET_full (Imperial, summer) | 6-9 mm/d (mean 7) | Montazar Fig. 6 |
| ET_deficit (after cutoff) | 1-3 mm/d (mean 2) | Montazar Fig. 6 |
| Deficit ET factor | 0.29 (2/7) | Derived |
| Kc_act under deficit | 0.2-0.4 | Montazar Fig. 7 |
| Kc_act under full | 0.8-1.1 | Montazar Fig. 7 |
| NDVI under deficit | 0.25-0.40 | Montazar Fig. 7 |
| Deficit duration (IID DIP) | 60-78 days | Montazar Table 1 |
| Deficit-period ET savings | 150-200 mm (43-50%) | Montazar Section 3.3 |
| Seasonal ET savings | 231 mm (15%) | Montazar Section 3.5 |
| Yield loss | ~10-15% of seasonal | Montazar Section 3.3 |
| Recovery | Rapid after re-irrigation | Montazar Section 3.4 |

**Scaling to other counties** — deficit parameters vary with climate:

| County | ET_full (mm/d) | Deficit factor | Deficit duration (days) | Deficit savings (mm) |
|--------|---------------|---------------|------------------------|---------------------|
| San Joaquin | 5.0 | 0.35 | 45 | 146 |
| Stanislaus | 5.0 | 0.35 | 45 | 146 |
| Merced | 5.2 | 0.34 | 48 | 160 |
| Madera | 5.3 | 0.33 | 48 | 168 |
| Fresno | 5.5 | 0.33 | 50 | 184 |
| Tulare | 5.5 | 0.33 | 50 | 184 |
| Kings | 5.5 | 0.32 | 52 | 190 |
| Kern | 6.0 | 0.31 | 55 | 228 |
| Riverside | 6.5 | 0.30 | 60 | 273 |
| Imperial | 7.0 | 0.29 | 70 | 348 |

Rationale: Hotter counties (south) have higher ET_full, more severe
deficit suppression, and longer viable deficit windows.

### 3.4 Per-cut ET by county (from our calibration)

| County | e_cut (mm) | e_cut (ac-ft/acre) |
|--------|-----------|-------------------|
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

### 3.5 Literature annual cuts by county

| County | N_cuts | n_late (at Jul 1) | Late cut revenue ($/acre/cut) |
|--------|--------|-------------------|------------------------------|
| San Joaquin | 8 | 2 | $162 |
| Stanislaus | 8 | 2 | $162 |
| Merced | 8 | 2 | $162 |
| Madera | 8 | 2 | $162 |
| Fresno | 9 | 3 | $155 |
| Tulare | 9 | 3 | $155 |
| Kings | 9 | 3 | $155 |
| Kern | 10 | 3 | $148 |
| Riverside | 12 | 3 | $140 |
| Imperial | 12 | 3 | $135 |

Late cut revenue decreases south because: more cuts → shorter cycles →
less yield per cut + more heat stress → lower quality.

---

## 4. Savings Equations by Strategy

### Common variables:

```
e_c    = county per-cut ET (ac-ft/acre)
n_late = expected late cuts (integer)
P_hay  = revenue per late cut ($/acre)
P_w    = water price ($/ac-ft)
D      = deficit duration (days)
ET_f   = full-irrigation daily ET (mm/d)
f_d    = deficit ET factor (0.29-0.35)
```

### Strategy equations:

```
S0: S_water = 0,           R_loss = 0
S1: S_water = (n_late-1) × e_c,  R_loss = (n_late-1) × P_hay
S2: S_water = n_late × e_c,      R_loss = n_late × P_hay
S3: S_water = D × ET_f × (1-f_d) / 304.8,  R_loss = 1.5 × P_hay  (skip ~1.5 cuts)
S4: S_water = 2 × e_c × (1-f_d),           R_loss = 2 × P_hay × 0.3  (reduced yield)
S5: S_water = S3 + (n_late-1) × e_c × f_d, R_loss = S3_loss + (n_late-2) × P_hay
S6: S_water = S3 + n_late × e_c × f_d,     R_loss = S3_loss + (n_late-1) × P_hay
S7: S_water = (end_of_season - Aug1) × ET_f / 304.8, R_loss = 2.5 × P_hay
```

### Net economic benefit:

```
NEB(strategy) = S_water × P_w − R_loss
```

When NEB > 0, saving water is worth more than the lost hay.

---

## 5. Figure Plan

### Figure 5: Economic crossover curves (3 panels)
**"At what water price does each strategy become profitable?"**

- X-axis: Water price ($0 — $1200/ac-ft)
- Y-axis: Net economic benefit ($/acre) = water_saved × water_price − revenue_lost
- Lines: one per strategy (S0-S7), color-coded
- Vertical markers: typical normal and drought water prices for that region
- Three panels:
  - (a) Northern SJV: San Joaquin, Stanislaus, Merced (8 cuts)
  - (b) Central SJV: Fresno, Tulare, Kings, Kern (9-10 cuts)
  - (c) Desert: Riverside, Imperial (12 cuts)
- Where lines cross $0: the break-even water price for that strategy
- Key result: Cap 0 (S2) crosses $0 at ~$290/ac-ft; crossed by all
  SJV counties during drought but never by Imperial

### Figure 6: Strategy comparison heatmap
**"Which strategy is optimal for each county under each condition?"**

- Layout: 2 rows × 1 column (pooled WY types)
  - Row 0: Normal water year pricing
  - Row 1: Drought water year pricing
- Each panel: county (y, N→S) × strategy (x, S0-S7)
- Color: Net economic benefit ($/acre)
- Green = profitable, Red = net loss, White = break-even
- Shows at a glance: Kern in drought → S6 is optimal;
  Imperial always → S0 (economics) or S3 (policy)

### Figure 7: Savings decomposition (stacked bars)
**"Where do the savings come from?"**

- X-axis: Counties (N→S)
- Y-axis: Water savings (ac-ft/acre)
- Stacked bars showing:
  - Bottom (blue): cutting management savings (our base model)
  - Middle (teal): deficit irrigation savings (Montazar-calibrated)
  - Top (green): combined interaction savings (additive minus double-counting)
- Grouped by WY type (4 panels) or pooled
- Shows that cutting management is dominant in N counties (shorter deficit
  window) while deficit irrigation dominates in S counties (longer window)

### Figure 8: Seasonal ET trajectory by strategy
**"How does each strategy modify the annual ET profile?"**

- One representative panel per region (N, Central, Desert)
- X-axis: Month (Oct → Sep, full WY)
- Y-axis: Daily ET (mm/d)
- Lines:
  - S0: full ET trajectory (solid black)
  - S2: ET with late cuts removed (dashed blue) — flat after July 1
  - S3: ET with deficit (dotted green) — drops to 2mm/d in summer
  - S6: Combined (dash-dot red) — maximum reduction
- Shaded areas between lines = water saved by each mechanism
- Annotations: total savings in ac-ft/acre for each strategy

### Figure 9: Basin-scale impact
**"How much water at the regional scale?"**

- X-axis: Strategy (S0-S7)
- Y-axis: Total water savings (thousand ac-ft)
- Based on alfalfa acreage by county:
  - San Joaquin: ~80,000 acres
  - Fresno: ~60,000 acres
  - Kern: ~55,000 acres
  - Tulare: ~45,000 acres
  - Imperial: ~120,000 acres
  - Others: ~140,000 acres
  - **Total: ~500,000 acres**
- S2 (our base cap 0): ~500k × 1.3 ac-ft = ~650,000 ac-ft
- S6 (hybrid max): ~500k × 1.9 ac-ft = ~950,000 ac-ft
- Context: Lake Mead loses ~1.2 MAF/year to evaporation

---

## 6. Implementation Architecture

### 6.1 New file: `es_analysis/charts/water_savings/strategy_simulation.py`

```
Constants:
  STRATEGY_PARAMS = {S0..S7 definitions}
  COUNTY_WATER_PARAMS = {prices, sources, SGMA}
  HAY_MARKET_PARAMS = {yields, quality, prices by season}
  DEFICIT_PARAMS = {from Montazar, scaled by county}
  ALFALFA_ACREAGE = {by county, from USDA NASS}

Functions:
  compute_strategy_savings(county, n_cuts, wy_type, strategy) → dict
  compute_net_benefit(county, strategy, water_price) → float
  build_crossover_data(counties, strategies, price_range) → DataFrame
  build_strategy_heatmap_data(counties, strategies, price_scenarios) → DataFrame
  build_decomposition_data(counties, wy_types) → DataFrame
  build_seasonal_profile(county, strategy) → DataFrame
  build_basin_impact(counties, strategies) → DataFrame

Figure functions:
  economic_crossover_figure(data, out_dir) → (fig, axes, summary)
  strategy_heatmap_figure(data, out_dir) → (fig, axes, summary)
  savings_decomposition_figure(data, out_dir) → (fig, axes, summary)
  seasonal_et_profile_figure(data, out_dir) → (fig, axes, summary)
  basin_impact_figure(data, out_dir) → (fig, axes, summary)
```

### 6.2 Data flow

```
Our calibration data (10,707 parcel-years)
        │
        ├── Per-cut ET by county ──────────────────┐
        ├── n_late by county × WY type ────────────┤
        ├── Late fraction by n_cuttings ───────────┤
        │                                          │
Montazar et al. 2026                               ▼
        │                               ┌──────────────────┐
        ├── Deficit ET factor (0.29) ──>│  Strategy Engine  │
        ├── Deficit duration by county ─>│  (all 8 strategies│
        ├── Recovery dynamics ──────────>│   × 10 counties   │
        │                               │   × 4 WY types)   │
Water price data (SGMA, CVP, IID)       └──────┬───────────┘
        │                                      │
        ├── Normal prices by county ───────────┤
        ├── Drought prices ────────────────────┤
        │                                      ▼
Hay market data                        ┌───────────────┐
        │                              │ 5 Figures     │
        ├── Price by quality/season ──>│ + CSVs        │
        ├── Yield by cut timing ──────>│ + .md update  │
        └── Acreage by county ────────>└───────────────┘
```

### 6.3 What stays unchanged

| Component | Status |
|-----------|--------|
| Figures 1-4 | **Unchanged** — base case reference |
| `sensitivity_heatmap.py` existing functions | **Unchanged** |
| `model_specification.md` | **Extend** with strategy section |
| `sensitivity_analysis_methods.md` | **Extend** with economic results |
| All calibration parameters | **Reused**, not recomputed |

---

## 7. Key Analysis Questions This Answers

### 7.1 Water availability
- Surface vs groundwater: SGMA-constrained counties (Tulare, Kings, Kern)
  face structural water scarcity regardless of drought → strategies S2+
  become rational even in normal years as curtailments phase in
- Imperial: water is always cheap ($22/ac-ft) but limited by Colorado
  River allocations → policy-driven, not price-driven

### 7.2 When is saving water worth more than hay?
- Break-even: ~$290/ac-ft for cutting management, ~$310/ac-ft for deficit
- In drought: all SJV counties exceed break-even → all strategies profitable
- In normal years: only Kern consistently exceeds break-even
- Imperial: NEVER on economics alone → purely policy-driven (IID DIP)

### 7.3 Geographic gradient
- **Northern SJV** (SJ, Stan, Merced): 8 cuts, 2 late, shorter deficit
  window → cutting management saves 1.1 ac-ft/acre, deficit saves 0.48
- **Central SJV** (Fresno-Kern): 9-10 cuts, 3 late, longer deficit
  window → cutting saves 1.5, deficit saves 0.75
- **Desert** (Riverside, Imperial): 12 cuts, 3 late, longest deficit
  window → cutting saves 1.7, deficit saves 1.1

### 7.4 Seasonal effects
- Late cuts (Jul-Sep) are lowest quality: fair/utility grade, $140-180/ton
- Spring cuts (Mar-May) are highest quality: supreme/premium, $260-350/ton
- Deficit period cuts are barely viable: stub quality, $110-140/ton
- **Removing late cuts has the smallest revenue impact per unit of water saved**

### 7.5 Yield-quality-price interaction
- Late cut yield: 0.8-1.0 tons/acre at $180/ton = $145-180/acre
- Water cost of late cut: 0.56 ac-ft × $290 break-even = $162/acre
- **The margins are thin** — even small water price increases tip the balance
- Export hay markets demand quality that late cuts often don't meet

### 7.6 Basin-scale potential
- Total alfalfa acreage: ~500,000 acres across 10 counties
- S2 (cap 0) basin-wide: ~650,000 ac-ft savings
- S6 (hybrid max) basin-wide: ~950,000 ac-ft savings
- Context: California's total agricultural water use is ~34 MAF/year
- Alfalfa late-cut savings alone = 2-3% of total ag water use

---

## 8. Implementation Order

| Phase | What | Priority | Depends on |
|-------|------|----------|------------|
| A | Parameter tables + strategy engine | High | Nothing |
| B | Figure 5: Economic crossover | High | Phase A |
| C | Figure 7: Savings decomposition | High | Phase A |
| D | Figure 6: Strategy heatmap | Medium | Phase A |
| E | Figure 8: Seasonal ET profiles | Medium | Phase A |
| F | Figure 9: Basin-scale impact | Medium | Phase A |
| G | Documentation update | High | All figures |

Estimated output: 5 new figures + 1 new Python module + updated .md files.

---

## 9. Limitations and Caveats

1. **Water prices are estimates** — actual prices vary by district, year,
   contract type, and whether surface or groundwater. We use representative
   ranges, not exact prices.

2. **Hay prices vary significantly** — export vs domestic, drought vs
   surplus, regional vs national market. We use California averages.

3. **Deficit parameters scaled from Imperial** — Montazar's study is
   Imperial-specific. Scaling to SJV counties assumes similar crop-soil-
   atmosphere response, which may not hold under different soil types
   or less extreme heat.

4. **No stand persistence modeling** — repeated deficit irrigation can
   thin alfalfa stands over time, reducing yield in subsequent years.
   We model single-year impacts only.

5. **No groundwater recharge accounting** — water "saved" from reduced
   ET may reduce return flows to aquifers, partially offsetting basin-scale
   savings (the Burt/Perry critique).

6. **SGMA timeline uncertainty** — the pace and severity of groundwater
   curtailments varies by basin and is politically negotiated. We use
   current trajectory estimates.
