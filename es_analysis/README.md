# ES Analysis

Modular EVI and ET analysis toolkit for alfalfa cutting detection.

## Directory Structure

```
es_analysis/
├── data_providers/          # Data loading and transformation
│   ├── evi_provider.py      # EVI data operations
│   ├── beast_provider.py    # BEAST cutting detection
│   ├── et_provider.py       # ET data and corrections
│   ├── landsat_provider.py  # Landsat metadata
│   ├── statistics_provider.py  # Aggregated statistics
│   └── config.py            # Configuration and parameters
│
├── charts/                  # Individual chart scripts
│   ├── evi/                 # EVI visualizations
│   ├── beast/               # BEAST visualizations
│   ├── et_corrections/      # ET correction visualizations
│   ├── statistics/          # Statistics visualizations
│   └── multi_panel/         # Multi-panel visualizations
│
├── runners/                 # Execute related chart groups
│   ├── run_evi_plots.py
│   ├── run_beast_plots.py
│   ├── run_et_plots.py
│   └── run_all_plots.py
│
├── utils/                   # Shared utilities
│   ├── helpers.py           # Common helpers
│   ├── gapfill.py           # Quartic gap-fill
│   ├── smoothing.py         # Savitzky-Golay smoothing
│   └── plotting.py          # Plotting utilities
│
└── output/                  # Generated outputs
    ├── figures/
    ├── data/
    └── logs/
```

## Usage

### Run individual chart
```bash
cd es_analysis
python charts/evi/evi_raw_plot.py
```

### Run related charts
```bash
python runners/run_evi_plots.py
```

### Run all charts
```bash
python runners/run_all_plots.py
```

## Configuration

Edit `data_providers/config.py` to modify:
- Input data paths
- Output directories
- Processing parameters
- County and year selections

## Refactoring Plan

See [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for complete details including:
- Complete chart inventory with source references
- Data provider specifications
- Migration phases
- Data flow diagrams
- Parameter documentation

## Data Flow

1. **EVI Processing**: Load raw EVI → gap-fill → smooth
2. **BEAST Analysis**: Run BEAST → extract change points → detect cuttings
3. **ET Corrections**: Load ET data → apply corrections based on harvest dates
4. **Statistics**: Aggregate data → compute summaries → generate visualizations

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
- Rbeast
- joblib

## Original Source

Refactored from `alfalfa_evi_jovyan.py` (21,184 lines)

---

**Version:** 1.0.0
**Date:** 2026-02-02