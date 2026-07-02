"""
Data providers module
Centralized data loading and transformation for EVI analysis
"""

from .config import Config, config, BEAST_OUT_ROOT
from .evi_provider import (
    EviDataProvider,
    normalize_county_name,
    normalize_county_names,
    water_year_bounds,
    water_year_bounds_for_years,
    daily_timeseries_water_year,
    build_daily_table,
)
from .beast_provider import BEASTDataProvider
from .et_provider import (
    get_harvest_dates_for_uid,
    compute_daily_and_monthly_for_uid,
    list_uids_for_county_wy,
)
from .et_stats_provider import run_et_correction_stats
from .landsat_provider import LandsatDataProvider
from .statistics_provider import (
    COUNTY_ORDER,
    aggregate_to_county_year_points,
    aggregate_to_county_overall_points,
    aggregate_to_county_year_points_four,
)
from .evi_cut_window_provider import (
    find_pre_cut_min_start_date_from_evi,
    compute_cut_cycle_segments,
    filter_segments_by_et,
    load_evi_for_uid_wy,
    load_evi_for_wy,
)
from .spatial_provider import (
    load_parcels_for_county,
    load_parcels_area_acres,
    load_county_boundaries,
)
from .daymet_provider import (
    compute_gdd5_for_parcels_over_segments,
    compute_gdd5_for_parcels_cut_window,
    compute_daymet_mean_for_parcels,
)
from .parcel_summary_provider import (
    build_parcel_summary_wy,
    build_multicounty_df_parcel_year,
    debug_one_parcel,
)
from .cutting_stats_provider import (
    run_cutting_weather_stats,
    compute_cutting_statistics,
    generate_cutting_narrative,
    export_cutting_stats_csvs,
    load_cutting_data_from_beast,
)
from .late_water_provider import (
    run_late_water_saving_workflow,
    build_late_cut_dataset,
    compute_et_by_cut_cycles,
    compute_cap_savings_per_row,
    add_cap_savings_columns,
    add_normalized_saving_columns,
    make_savings_summary,
    save_workflow_outputs,
    summarize_late_et_by_late_cut_count,
)
from .wy_type_provider import (
    SJ_VALLEY_WY_INDEX,
    WY_TYPE_ORDER,
    WY_TYPE_COLORS,
    SJ_VALLEY_COUNTIES,
    COLORADO_RIVER_COUNTIES,
    get_wy_type,
    get_wy_index,
    get_wy_color,
    get_wy_region,
    get_usdm_peak,
    add_wy_type_columns,
    add_usdm_columns,
    get_wy_types_present,
)

# Constant aliases from config (used by EVI chart scripts)
SG_WINDOW = config.sg_window
SG_POLY = config.sg_poly
INTERP_WINDOW_DAYS = config.interp_window_days
COUNTY_YEAR_EXPORT_ROOT = config.county_year_root_new

# Class name alias (charts use BeastDataProvider, provider defines BEASTDataProvider)
BeastDataProvider = BEASTDataProvider

__all__ = [
    "Config",
    "config",
    "EviDataProvider",
    "normalize_county_name",
    "normalize_county_names",
    "water_year_bounds",
    "water_year_bounds_for_years",
    "daily_timeseries_water_year",
    "build_daily_table",
    "BEASTDataProvider",
    "BeastDataProvider",
    "get_harvest_dates_for_uid",
    "compute_daily_and_monthly_for_uid",
    "list_uids_for_county_wy",
    "run_et_correction_stats",
    "LandsatDataProvider",
    "COUNTY_ORDER",
    "aggregate_to_county_year_points",
    "aggregate_to_county_overall_points",
    "aggregate_to_county_year_points_four",
    # EVI cut-window provider
    "find_pre_cut_min_start_date_from_evi",
    "compute_cut_cycle_segments",
    "filter_segments_by_et",
    "load_evi_for_uid_wy",
    "load_evi_for_wy",
    # Spatial provider
    "load_parcels_for_county",
    "load_parcels_area_acres",
    "load_county_boundaries",
    # Daymet provider
    "compute_gdd5_for_parcels_over_segments",
    "compute_gdd5_for_parcels_cut_window",
    "compute_daymet_mean_for_parcels",
    # Parcel summary provider
    "build_parcel_summary_wy",
    "build_multicounty_df_parcel_year",
    "debug_one_parcel",
    # Cutting stats provider
    "run_cutting_weather_stats",
    "compute_cutting_statistics",
    "generate_cutting_narrative",
    "export_cutting_stats_csvs",
    "load_cutting_data_from_beast",
    # Late-water savings provider
    "run_late_water_saving_workflow",
    "build_late_cut_dataset",
    "compute_et_by_cut_cycles",
    "compute_cap_savings_per_row",
    "add_cap_savings_columns",
    "add_normalized_saving_columns",
    "make_savings_summary",
    "save_workflow_outputs",
    "summarize_late_et_by_late_cut_count",
    # WY type provider
    "SJ_VALLEY_WY_INDEX",
    "WY_TYPE_ORDER",
    "WY_TYPE_COLORS",
    "SJ_VALLEY_COUNTIES",
    "COLORADO_RIVER_COUNTIES",
    "get_wy_type",
    "get_wy_index",
    "get_wy_color",
    "get_wy_region",
    "get_usdm_peak",
    "add_wy_type_columns",
    "add_usdm_columns",
    "get_wy_types_present",
    # Constants
    "SG_WINDOW",
    "SG_POLY",
    "INTERP_WINDOW_DAYS",
    "COUNTY_YEAR_EXPORT_ROOT",
    "BEAST_OUT_ROOT",
]
