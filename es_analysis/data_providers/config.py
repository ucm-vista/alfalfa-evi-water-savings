import os
from pathlib import Path
from typing import List, Optional, Tuple


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a repo-root .env into os.environ (no override).

    Kept dependency-free; python-dotenv is used if available but not required.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(_REPO_ROOT / ".env")
        return
    except Exception:
        pass
    env_file = _REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Repo root = three levels up from this file (…/es_analysis/data_providers/config.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_load_dotenv()

# Raw external data roots share the ancestor /home/jovyan/work. Override with $WORK_ROOT.
WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/home/jovyan/work"))

# Intermediate/derived data (BEAST outputs, county-year exports, stat CSVs) lives at the
# repo root by default — where the Zenodo intermediate bundle unpacks. Override with $DATA_ROOT.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_REPO_ROOT)))


class Config:
    """Configuration module for EVI analysis pipeline."""

    # ================== DATA PATHS ==================
    csv_path: Path = Path(os.environ.get(
        "EVI_CSV",
        str(WORK_ROOT / "emery_method_2/earthdata_downloader/vi_results/combined_evi_timeseries.csv"),
    ))
    # DEPRECATED: local copy from old emery_method. Use csv_path (emery_method_2) for all new work.
    combined_evi_csv_path: Path = WORK_ROOT / "emery_method/reporting/evi_analysis/combined_evi_timeseries.csv"
    output_dir: Path = Path("visualizations")

    # County-year exports
    county_year_root: Path = DATA_ROOT / "county_year_exports"
    county_year_root_new: Path = DATA_ROOT / "county_year_exports_new"
    plots_root: Path = Path("county_year_plots")

    # BEAST outputs
    beast_out_root: Path = DATA_ROOT / "beast_outputs"
    beast_out_root_new: Path = DATA_ROOT / "beast_outputs_new"

    # OpenET and reference data
    openet_root: Path = WORK_ROOT / "emery_method/reporting/actual_ET_new"
    openet_download_root: Path = WORK_ROOT / "emery_method/reporting/actual_ET_api"
    openet_api_daily_root: Path = WORK_ROOT / "emery_method/reporting/actual_ET_api/raw/daily"
    use_api_et: bool = True  # True = load per-parcel JSONs from openet_api_daily_root
    # Secret: loaded from environment / .env (see .env.example). Never hard-code.
    openet_api_key: str = os.environ.get("OPENET_API_KEY_1", "")
    eto_etof_root: Path = WORK_ROOT / "emery_method/reporting/ETo_ETof_LandSat_passes/OpenET_ETo_ETof_overlap"
    landsat_meta_csv: Path = (
        WORK_ROOT / "emery_method/reporting/ETo_ETof_LandSat_passes/LandSat_Meta_data/"
        "landsat89_pass_path38_44_row33_37_2019_2024_0-80cc.csv"
    )
    export_root: Path = Path("county_year_exports/offphase_corrections_exports")

    # Pre-computed cutting weather stats (shipped in the intermediate bundle)
    cutting_weather_stats_root: Path = DATA_ROOT / "Cutting_weather_stats"

    # Statistics export
    statistics_export_dir: Path = DATA_ROOT / "statistics_exports"

    # Parcel and county boundary shapefiles
    parcel_shp: Path = WORK_ROOT / "emery_method/shapefile/alfalfa_parcels_2018_2023_all_years.shp"
    county_boundary_shp: Path = WORK_ROOT / "emery_method/shapefile/CA_SanJoaq_Imperial_Counties_proj.shp"

    # Daymet data
    daymet_root: Path = WORK_ROOT / "emery_method/reporting/daymet_alfalfa"

    # GDD parameters (alfalfa)
    gdd_base_c: float = 5.0
    gdd_tmin_floor_c: float = 5.0     # 50 F
    gdd_tmax_cap_c: float = 30.0      # 86 F

    # ================== COUNTY SELECTION ==================
    selected_counties_original: List[str] = [
        "Fresno", "San_Joaquin", "Stanislaus", "Madera", "Kings", "Tulare",
        "Kern", "Los_Angeles", "San_Bernardino", "Imperial", "Riverside", "Merced",
    ]

    selected_counties_expanded: List[str] = [
        "Fresno", "San_Joaquin", "Stanislaus", "Madera", "Kings", "Tulare",
        "Kern", "Imperial", "Riverside", "Merced",
    ]

    # ================== WATER YEARS ==================
    valid_years: set = set(range(2019, 2024))
    water_years: List[int] = [2019, 2020, 2021, 2022, 2023, 2024]

    # ================== PROCESSING PARAMETERS ==================
    interp_window_days: int = 30
    sg_window: int = 15
    sg_poly: int = 3

    # EVI gap-fill and smoothing
    peak_window_days: int = 60
    delta_min: float = 0.10
    min_spacing_days: int = 22
    min_evi_max: float = 0.35
    cp_tolerance_days: int = 10

    # Fallow-like filter
    fallow_max_evi_threshold: float = 0.20
    fallow_std_threshold: float = 0.03

    # ================== BEAST PARAMETERS ==================
    beast_call_mode: str = "subproc"
    beast_period: float = 365.0

    # Seasonal BEAST sweep
    season_sweep_minlengths: List[int] = [20, 27, 35, 45]
    season_sweep_leftmargins: List[int] = [30, 35, 40, 45]
    season_scp_minmax: Tuple[int, int] = (6, 12)

    # Trend BEAST sweep
    trend_sweep_minlengths: List[int] = [90, 120, 180]
    trend_sweep_leftmargins: List[int] = [90, 120, 180]
    trend_scp_minmax: Tuple[int, int] = (0, 0)

    # ================== INCLUSIVE MINIMA PARAMETERS ==================
    strict_tol_steps: List[int] = [7, 10, 14, 18, 21, 28, 32]
    cp_boost_window_days: int = 10
    amp_frac_min: float = 0.12
    peak_window_days_range: List[int] = [45, 60, 75, 90]
    delta_min_range: List[float] = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    min_spacing_days_range: List[int] = [22, 26, 30]
    min_evi_max_range: List[float] = [0.50, 0.40, 0.35, 0.25]
    minima_only_when_cp_count_at_most: int = 0

    # ================== EVI QUALITY FILTERS ==================
    evi_filter_qa: bool = True                # drop rows where qa_status != "ok"
    evi_min_valid_pixel_fraction: float = 0.50  # drop if valid_pixels/total_pixels < this
    evi_cloud_cover_max: float = 50.0         # drop if scene cloud_cover > this %

    # ================== ET CORRECTION PARAMETERS ==================
    cloud_cover_max: float = 20.0
    r_days: int = 8
    pre_window: int = 5
    post_window: int = 5
    low_quantile: float = 0.25
    chosen_method: str = "A"
    ci_alpha: float = 0.10
    n_boot: int = 4000
    inflate_by_cloud_gap: bool = True

    # ================== COUNTY EXPECTED CUTS (PRIORS) ==================
    county_expected_cuts: dict = {
        "San Joaquin": (6, 12),
        "Stanislaus": (6, 12),
        "Madera": (6, 12),
        "Merced": (6, 12),
        "Fresno": (6, 12),
        "Kings": (6, 12),
        "Tulare": (6, 12),
        "Kern": (6, 12),
        "Riverside": (6, 12),
        "Imperial": (6, 12),
    }

    # ================== EVI CUT-WINDOW PARAMETERS ==================
    evi_mode: str = "gapfilled"
    summer_lookback_range_days: Tuple[int, int] = (28, 32)
    winter_lookback_range_days: Tuple[int, int] = (90, 120)
    min_gap_before_cut_days: int = 3
    min_segment_days: int = 22

    # Post-harvest gap merge (Layer 3)
    max_segment_gap_extension_days: int = 75

    # Desert-specific EVI minima thresholds (Layer 2)
    desert_counties: List[str] = ["Imperial", "Riverside"]
    desert_delta_min: float = 0.05
    desert_min_evi_max: float = 0.50
    cv_delta_min: float = 0.08
    cv_min_evi_max: float = 0.40

    # Segment ET filtering
    segment_et_filter_mode: str = "none"    # "none" | "absolute" | "relative" | "both"
    segment_et_abs_min_mm: float = 2.0
    segment_et_rel_min_frac: float = 0.25

    # Legacy ET-rise fallback parameters
    rise_days: int = 5
    rise_eps: float = 0.02

    # ================== LATE-WATER SAVINGS PARAMETERS ==================
    late_cutoff_month: int = 7
    late_cutoff_day: int = 1
    cap_values: Tuple[int, ...] = (0, 1, 2, 3)
    water_saving_out_dir: Path = DATA_ROOT / "water_saving_scenarios_stats"

    # ================== BEAST IMPROVEMENT PARAMETERS ==================
    # Improvement A: Whittaker smoother (replaces gap-fill + SG two-step)
    evi_smoothing_method: str = "whittaker"     # "sg" (original) or "whittaker"
    whittaker_lambda: float = 1e2               # light smooth for live EVI rendering (charts)
    whittaker_order: int = 2                    # difference order (2 = second-order smoothness)

    # Improvement B: BEAST occupancy curve for timing uncertainty
    use_occupancy_curve: bool = True            # compute timing sigma from occupancy peaks

    # Improvement C: Ensemble consensus (replaces best-run selection)
    beast_ensemble_mode: bool = True            # pool CPs across all sweep runs
    min_consensus_freq: float = 0.25            # min fraction of runs detecting a CP to keep it

    # Improvement D: Adaptive priors (two-pass)
    beast_adaptive_priors: bool = True          # estimate county-year priors from data
    beast_prior_sample_frac: float = 0.10       # fraction of parcels for first-pass estimation
    beast_prior_wide_range: Tuple[int, int] = (3, 15)  # wide prior for first pass

    # Improvement E: Probability filtering
    min_cp_probability: float = 0.5             # min BEAST probability to keep a CP

    # Tiered fallback: consensus → best-run → season-filtered minima
    beast_fallback_to_best_run: bool = True      # tier 2: try best-run when consensus yields 0 CPs
    growing_season_start_month: int = 3          # March — tier 3 season filter start
    growing_season_end_month: int = 10           # October — tier 3 season filter end

    # Low-EVI validation on CP-centric boost
    max_boost_evi: float = 0.45                 # max EVI at boosted minimum (must be a trough)
    require_peak_before_boost: bool = True       # boosted min must have preceding peak with delta

    # Whittaker-based interval estimation (replaces fixed lookback windows)
    use_whittaker_interval: bool = True
    whittaker_interval_lambda: float = 1e2   # light smooth for derivative analysis (not 1e4)
    whittaker_summer_max_lookback_days: int = 60
    whittaker_winter_max_lookback_days: int = 150
    whittaker_regrowth_slope_threshold: float = 0.002  # EVI/day

    # ================== THERMAL-TIME SEGMENT ESTIMATION ==================
    use_thermal_time_interval: bool = True
    thermal_time_trusted_summer_range: Tuple[int, int] = (20, 45)
    thermal_time_trusted_winter_range: Tuple[int, int] = (50, 150)

    # ================== BULK PROCESSING SETTINGS ==================
    max_uids_per_county_wy: Optional[int] = None
    uid_sample_seed: int = 42
    n_boot_bulk: int = 500
    min_cuttings: int = 1  # 0 = include all, 1 = exclude zero-cut parcels

    @classmethod
    def get_county_expected_range(cls, county: str) -> Tuple[int, int]:
        """Get expected cutting range for a county."""
        from .evi_provider import normalize_county_name
        return cls.county_expected_cuts.get(normalize_county_name(county), (6, 12))


config = Config()

# Module-level constant aliases for backward-compatible imports
BEAST_OUT_ROOT = config.beast_out_root_new