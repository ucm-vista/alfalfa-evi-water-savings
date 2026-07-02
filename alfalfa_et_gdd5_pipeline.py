#!/usr/bin/env python3
"""
Alfalfa ET/GDD5 Post-BEAST Pipeline
====================================
Unified interactive orchestrator for the post-BEAST analysis pipeline.
Calls into existing es_analysis providers and charts with user-specified parameters.

Usage:
    # Interactive mode (prompts for everything)
    python alfalfa_et_gdd5_pipeline.py

    # CLI mode
    python alfalfa_et_gdd5_pipeline.py --counties Fresno Kings --wy 2022 --action summary

    # Single-parcel debug
    python alfalfa_et_gdd5_pipeline.py --action debug --county Fresno --wy 2022 --parcel 1036125

    # Full pipeline
    python alfalfa_et_gdd5_pipeline.py --action full --counties Fresno --wy 2019-2024
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# sys.path setup — same pattern as existing chart scripts
# Place evi_analysis/ on sys.path so 'es_analysis' is importable as a package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from es_analysis.data_providers.config import config
from es_analysis.data_providers.evi_provider import normalize_county_name
from es_analysis.data_providers.spatial_provider import COUNTY_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_WYS = list(range(2019, 2025))  # WY2019 through WY2024
ACTIONS = ["summary", "plots", "et_correction_plots", "et_stats", "debug", "late_water", "water_savings_plots", "wy_type_analysis", "export", "full", "heatmap", "anomaly_evi", "save_data", "run_versioned"]


# ---------------------------------------------------------------------------
# PipelineConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Collects all tunable parameters with defaults pulled from config."""

    # Selection
    counties: List[str] = field(default_factory=lambda: list(COUNTY_ORDER))
    water_years: List[int] = field(default_factory=lambda: list(ALL_WYS))
    parcel_id: Optional[str] = None

    # ET Correction
    chosen_method: str = "A"
    n_boot: int = 500
    cloud_cover_max: float = 20.0
    r_days: int = 8
    pre_window: int = 5
    post_window: int = 5
    ci_alpha: float = 0.10

    # Segment Estimation
    evi_mode: str = "gapfilled"
    min_segment_days: int = 20
    summer_lookback: Tuple[int, int] = (28, 32)
    winter_lookback: Tuple[int, int] = (90, 120)
    use_thermal_time: bool = True
    thermal_summer_range: Tuple[int, int] = (20, 45)
    thermal_winter_range: Tuple[int, int] = (50, 150)
    use_whittaker: bool = True
    whittaker_interval_lambda: float = 1e2  # light smooth for segment estimation

    # Late-Water Savings
    late_cutoff_month: int = 7
    late_cutoff_day: int = 1
    cap_values: Tuple[int, ...] = (0, 1, 2, 3)

    # Processing
    workers: int = 4
    output_dir: Path = field(
        default_factory=lambda: _REPO_ROOT / "es_analysis" / "output"
    )

    # Filtering
    min_cuttings: int = 1  # 0 = include all, 1 = exclude zero-cut parcels

    # WY Type Analysis
    class_mode: str = "sji"  # "sji", "usdm", or "mixed"
    sj_valley_only: bool = False
    wy_charts: Optional[List[str]] = None  # None = all 7 charts

    # Run output
    run_name: str = "alfalfa_run_2"

    # Plotting
    min_et_filter_mm: float = 0.0
    daymet_var: str = "gdd5"
    et_mode: str = "actual"  # "actual", "corrected", or "both"
    n_random_parcels: int = 3  # for et_correction_plots random sampling
    et_corr_county: Optional[str] = None  # single county for ET correction plots (None = prompt)


# ---------------------------------------------------------------------------
# BEAST status checker
# ---------------------------------------------------------------------------
def check_beast_status() -> Dict[str, Dict[str, object]]:
    """Scan beast_outputs_new/ and report per-county completion status.

    Returns dict: county -> {"complete": [wys], "missing": [wys], "n_parcels": int or None}
    """
    beast_root = _REPO_ROOT / "beast_outputs_new"
    status = {}
    for county in COUNTY_ORDER:
        # Directory names may use spaces (e.g. "San Joaquin")
        county_dir = beast_root / county
        if not county_dir.is_dir():
            county_dir = beast_root / county.replace(" ", "_")
        complete, missing = [], []
        n_parcels = None
        for wy in ALL_WYS:
            csv_path = county_dir / f"beast_seasonal_cuts_WY{wy}.csv"
            if csv_path.is_file():
                complete.append(wy)
                if n_parcels is None:
                    try:
                        df = pd.read_csv(csv_path, nrows=5000)
                        id_col = "UniqueID" if "UniqueID" in df.columns else "parcel_id"
                        n_parcels = df[id_col].nunique()
                    except Exception:
                        pass
            else:
                missing.append(wy)
        status[county] = {
            "complete": complete,
            "missing": missing,
            "n_parcels": n_parcels,
        }
    return status


def _beast_marker(info: dict) -> str:
    n_complete = len(info["complete"])
    total = len(ALL_WYS)
    if n_complete == total:
        return "complete"
    elif n_complete > 0:
        return f"{n_complete}/{total}"
    return "missing"


# ---------------------------------------------------------------------------
# Config override mechanism
# ---------------------------------------------------------------------------
def apply_config_overrides(cfg: PipelineConfig) -> None:
    """Temporarily push PipelineConfig values into the global config singleton."""
    config.chosen_method = cfg.chosen_method
    config.n_boot = cfg.n_boot
    config.n_boot_bulk = cfg.n_boot
    config.cloud_cover_max = cfg.cloud_cover_max
    config.r_days = cfg.r_days
    config.pre_window = cfg.pre_window
    config.post_window = cfg.post_window
    config.ci_alpha = cfg.ci_alpha
    config.evi_mode = cfg.evi_mode
    config.min_segment_days = cfg.min_segment_days
    config.summer_lookback_range_days = cfg.summer_lookback
    config.winter_lookback_range_days = cfg.winter_lookback
    config.use_thermal_time_interval = cfg.use_thermal_time
    config.thermal_time_trusted_summer_range = cfg.thermal_summer_range
    config.thermal_time_trusted_winter_range = cfg.thermal_winter_range
    config.use_whittaker_interval = cfg.use_whittaker
    config.whittaker_interval_lambda = cfg.whittaker_interval_lambda
    config.late_cutoff_month = cfg.late_cutoff_month
    config.late_cutoff_day = cfg.late_cutoff_day
    config.cap_values = cfg.cap_values
    config.min_cuttings = cfg.min_cuttings


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------
def action_summary(cfg: PipelineConfig) -> Optional[pd.DataFrame]:
    """Build parcel-year summary and print ET/GDD5 statistics."""
    from es_analysis.data_providers.parcel_summary_provider import (
        build_multicounty_matched,
        build_parcel_summary_matched,
    )

    apply_config_overrides(cfg)

    print("\n--- Building parcel-year summary ---")
    if len(cfg.counties) == 1 and len(cfg.water_years) == 1:
        df = build_parcel_summary_matched(
            cfg.counties[0],
            cfg.water_years[0],
            daymet_var=cfg.daymet_var,
            evi_mode=cfg.evi_mode,
            et_mode=cfg.et_mode,
            method=cfg.chosen_method,
        )
    else:
        df = build_multicounty_matched(
            daymet_var=cfg.daymet_var,
            wy_start=min(cfg.water_years),
            wy_end=max(cfg.water_years),
            counties=cfg.counties,
            evi_mode=cfg.evi_mode,
            et_mode=cfg.et_mode,
            method=cfg.chosen_method,
        )

    if df is None or df.empty:
        print("  No data returned.")
        return None

    et_col = "et_cum_minET_to_last_cut_mm"
    gdd_col = f"{cfg.daymet_var}_mean"
    n_cut_col = "n_cuttings"

    print(f"\n  Total parcel-year rows: {len(df)}")

    # Per-county summary
    if "county" in df.columns:
        print("\n  Per-County Summary:")
        print(f"  {'County':<16} {'N':>6} {'Med ET mm':>10} {'Med GDD5':>10} {'Med Cuts':>9}")
        print(f"  {'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*9}")
        for county in COUNTY_ORDER:
            sub = df[df["county"] == county]
            if sub.empty:
                continue
            med_et = sub[et_col].median() if et_col in sub.columns else float("nan")
            med_gdd = sub[gdd_col].median() if gdd_col in sub.columns else float("nan")
            med_cuts = sub[n_cut_col].median() if n_cut_col in sub.columns else float("nan")
            print(f"  {county:<16} {len(sub):>6} {med_et:>10.1f} {med_gdd:>10.1f} {med_cuts:>9.1f}")

    # Per n_cuttings
    if n_cut_col in df.columns and et_col in df.columns:
        print(f"\n  Per-Cutting-Count Summary:")
        print(f"  {'Cuts':>5} {'N':>6} {'Med ET mm':>10} {'Ratio/155':>10}")
        print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*10}")
        for nc in sorted(df[n_cut_col].dropna().unique()):
            nc = int(nc)
            sub = df[df[n_cut_col] == nc]
            med = sub[et_col].median()
            print(f"  {nc:>5} {len(sub):>6} {med:>10.1f} {med / 155:>10.2f}")

    # Overall distribution
    if et_col in df.columns:
        vals = df[et_col].dropna()
        if len(vals) > 0:
            print(f"\n  ET Distribution (mm):")
            for label, v in [
                ("min", vals.min()), ("p5", vals.quantile(0.05)),
                ("p25", vals.quantile(0.25)), ("median", vals.median()),
                ("p75", vals.quantile(0.75)), ("p95", vals.quantile(0.95)),
                ("max", vals.max()),
            ]:
                print(f"    {label:>7}: {v:>10.1f}")
            n_low = (vals < 50).sum()
            print(f"\n    Below 50mm per-cut: {n_low} ({100 * n_low / len(vals):.1f}%)")

    return df


def action_plots(cfg: PipelineConfig) -> None:
    """Generate per-county matched cutting plots (scatter + boxplot)."""
    from es_analysis.charts.statistics.cuttings_matched_plots import (
        generate_county_plots, scatter_all_years, scatter_by_year, boxplot_year_colored,
    )

    apply_config_overrides(cfg)

    from es_analysis.utils.run_output import ensure_run_dirs
    run_root = ensure_run_dirs(cfg.run_name)
    out_base = run_root / "cuttings_analysis"
    out_base.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating cuttings analysis plots ---")
    print(f"  et_mode={cfg.et_mode}, method={cfg.chosen_method}")
    all_frames = []
    for county in cfg.counties:
        print(f"  {county} ...", end=" ", flush=True)
        try:
            df = generate_county_plots(
                county,
                base_out=out_base,
                wy_start=min(cfg.water_years),
                wy_end=max(cfg.water_years),
                min_et_mm=cfg.min_et_filter_mm,
                et_mode=cfg.et_mode,
                method=cfg.chosen_method,
            )
            if not df.empty:
                all_frames.append(df)
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc}")

    # Combined all-counties plots
    if len(all_frames) > 1:
        df_combined = pd.concat(all_frames, ignore_index=True)
        agg_dir = out_base / "all_counties"
        agg_dir.mkdir(parents=True, exist_ok=True)
        label = f"All {len(cfg.counties)} Counties"
        print(f"\n  Aggregate ({len(df_combined)} parcel-years) ...", end=" ", flush=True)
        scatter_all_years(df_combined, label, agg_dir, et_mode=cfg.et_mode)
        scatter_by_year(df_combined, label, agg_dir, et_mode=cfg.et_mode)
        boxplot_year_colored(df_combined, label, agg_dir, et_mode=cfg.et_mode)
        print("done")

        # Spatial ridgeline plots (map + density ridges)
        try:
            import matplotlib.pyplot as plt
            from es_analysis.charts.statistics.spatial_ridgeline_plots import (
                spatial_ridgeline_cuttings,
                spatial_ridgeline_seg_et,
            )
            print("\n  Spatial ridgeline: cuttings ...", end=" ", flush=True)
            fig, _ = spatial_ridgeline_cuttings(df_combined, out_dir=agg_dir)
            plt.close(fig)
            print("done")
            print("  Spatial ridgeline: segment ET ...", end=" ", flush=True)
            fig, _ = spatial_ridgeline_seg_et(df_combined, out_dir=agg_dir)
            plt.close(fig)
            print("done")
        except Exception as exc:
            print(f"  Spatial ridgeline FAILED: {exc}")

    print(f"\n  Plots written to: {out_base}")


def action_et_correction_plots(cfg: PipelineConfig) -> None:
    """Generate ET correction plots: per-parcel (max cuts) + aggregate summary."""
    from es_analysis.charts.et_corrections.et_separate_daily_monthly_plot import (
        et_separate_daily_monthly,
    )
    from es_analysis.data_providers.et_provider import _load_seasonal_csv, _norm_county_name

    apply_config_overrides(cfg)

    from es_analysis.utils.run_output import ensure_run_dirs
    run_root = ensure_run_dirs(cfg.run_name)
    out_dir = run_root / "et_corrections"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which county to plot (single county for per-parcel plots)
    if cfg.et_corr_county:
        plot_counties = [cfg.et_corr_county]
    elif len(cfg.counties) == 1:
        plot_counties = cfg.counties
    else:
        # Non-interactive: default to all counties
        if not sys.stdin.isatty():
            plot_counties = list(cfg.counties)
        else:
            # Interactive selection: pick one county
            print("\n  ET correction per-parcel plots work best for a single county.")
            print("  Available counties:")
            for i, c in enumerate(cfg.counties, 1):
                print(f"    [{i}] {c}")
            print(f"    [a] All ({len(cfg.counties)} counties)")
            sel = input("  Select county (number or 'a'): ").strip()
            if sel.lower() == "a":
                plot_counties = list(cfg.counties)
            else:
                try:
                    idx = int(sel) - 1
                    plot_counties = [cfg.counties[idx]]
                except (ValueError, IndexError):
                    print(f"  Invalid selection '{sel}', defaulting to Fresno")
                    plot_counties = ["Fresno"]

    print("\n--- Generating ET correction plots ---")
    print(f"  et_mode={cfg.et_mode}, method={cfg.chosen_method}")
    print(f"  County: {', '.join(plot_counties)}")

    # --- Per-parcel: parcel with max cuttings per county ---
    results = []
    for county in plot_counties:
        if cfg.parcel_id:
            # User-specified parcel: plot for all WYs
            for wy in cfg.water_years:
                print(f"    {county} WY{wy} UID={cfg.parcel_id}", end=" ... ", flush=True)
                try:
                    result = et_separate_daily_monthly(
                        county=county, water_year=wy, uid=cfg.parcel_id,
                        method=cfg.chosen_method, et_mode=cfg.et_mode,
                        ci_alpha=cfg.ci_alpha, n_boot=cfg.n_boot,
                        cloud_cover_max=cfg.cloud_cover_max, output_dir=out_dir,
                    )
                    results.append(result)
                    print("done")
                except Exception as exc:
                    print(f"FAILED: {exc}")
        else:
            # Find parcel with max cuttings across all WYs for this county
            cn = _norm_county_name(county)
            max_uid, max_cuts, max_wy = None, 0, None
            for wy in cfg.water_years:
                try:
                    df_s = _load_seasonal_csv(cn, wy)
                    cut_col = "n_cp_season" if "n_cp_season" in df_s.columns else "n_cuttings"
                    idx_max = df_s[cut_col].idxmax()
                    row = df_s.loc[idx_max]
                    if float(row[cut_col]) > max_cuts:
                        max_cuts = float(row[cut_col])
                        max_uid = str(row["UniqueID"])
                        max_wy = wy
                except (FileNotFoundError, ValueError):
                    continue

            if max_uid is None:
                print(f"  {county}: no BEAST data found, skipping")
                continue

            print(f"  {county}: max-cuts parcel UID={max_uid} ({int(max_cuts)} cuts in WY{max_wy})")
            print(f"    Plotting WY{max_wy}", end=" ... ", flush=True)
            try:
                result = et_separate_daily_monthly(
                    county=county, water_year=max_wy, uid=max_uid,
                    method=cfg.chosen_method, et_mode=cfg.et_mode,
                    ci_alpha=cfg.ci_alpha, n_boot=cfg.n_boot,
                    cloud_cover_max=cfg.cloud_cover_max, output_dir=out_dir,
                )
                results.append(result)
                print("done")
            except Exception as exc:
                print(f"FAILED: {exc}")

    print(f"\n  Generated {len(results)} per-parcel plot(s)")

    # --- Aggregate ET correction summary ---
    print("\n  [Aggregate ET correction summary]")
    try:
        from es_analysis.charts.et_corrections.et_correction_summary_plot import (
            et_correction_summary,
        )
        from es_analysis.utils.run_output import get_run_root
        run_et_dir = get_run_root(cfg.run_name) / "et_correction"
        et_dir = run_et_dir if run_et_dir.exists() else None
        et_correction_summary(
            counties=cfg.counties,
            wy_start=min(cfg.water_years),
            wy_end=max(cfg.water_years),
            et_corr_dir=et_dir,
            out_dir=out_dir,
        )
    except Exception as exc:
        print(f"  Aggregate summary FAILED: {exc}")

    print(f"\n  All plots in: {out_dir}")


def action_et_stats(cfg: PipelineConfig) -> Optional[Dict]:
    """Run off-phase ET correction pipeline."""
    from es_analysis.data_providers.et_stats_provider import run_et_correction_stats
    from es_analysis.utils.run_output import ensure_run_dirs

    apply_config_overrides(cfg)

    run_root = ensure_run_dirs(cfg.run_name)

    print("\n--- Running ET correction stats ---")
    print(f"  Counties: {', '.join(cfg.counties)}")
    print(f"  WY range: {min(cfg.water_years)}-{max(cfg.water_years)}")
    print(f"  Bootstrap: {cfg.n_boot}, Method: {cfg.chosen_method}")

    results = run_et_correction_stats(
        counties=cfg.counties,
        wy_start=min(cfg.water_years),
        wy_end=max(cfg.water_years),
        n_boot=cfg.n_boot,
        chosen_method=cfg.chosen_method,
        cloud_cover_max=cfg.cloud_cover_max,
        r_days=cfg.r_days,
        pre_window=cfg.pre_window,
        post_window=cfg.post_window,
        ci_alpha=cfg.ci_alpha,
        out_dir=run_root / "et_correction",
        export_csv=True,
        max_workers=cfg.workers,
    )

    if results and "narrative" in results:
        narr = results["narrative"]
        if isinstance(narr, tuple) and len(narr) >= 1:
            print(f"\n  {narr[0]}")

    if results and "paths" in results:
        print(f"\n  Exported to: {results['paths']}")

    return results


def action_debug(cfg: PipelineConfig) -> Optional[pd.DataFrame]:
    """Single-parcel diagnostic with per-segment table and EVI/ET plot."""
    from es_analysis.data_providers.parcel_summary_provider import debug_one_parcel

    apply_config_overrides(cfg)

    if not cfg.parcel_id:
        print("  ERROR: --parcel is required for debug action.")
        return None
    if len(cfg.counties) != 1:
        print("  ERROR: debug requires exactly one --county.")
        return None
    if len(cfg.water_years) != 1:
        print("  ERROR: debug requires exactly one --wy.")
        return None

    county = cfg.counties[0]
    wy = cfg.water_years[0]
    print(f"\n--- Debug: parcel {cfg.parcel_id}, {county} WY{wy} ---")

    df = debug_one_parcel(
        county,
        wy,
        cfg.parcel_id,
        evi_mode=cfg.evi_mode,
        summer_lookback_range_days=cfg.summer_lookback,
        winter_lookback_range_days=cfg.winter_lookback,
        min_segment_days=cfg.min_segment_days,
        do_plot=True,
    )

    if df is not None and not df.empty:
        print(f"\n  Segments ({len(df)} rows):")
        print(df.to_string(index=False))
    else:
        print("  No segments returned.")
    return df


def action_late_water(cfg: PipelineConfig) -> Optional[Dict]:
    """Run late-water savings workflow."""
    import shutil
    from es_analysis.data_providers.late_water_provider import run_late_water_saving_workflow
    from es_analysis.utils.run_output import ensure_run_dirs

    apply_config_overrides(cfg)

    print("\n--- Running late-water savings analysis ---")
    print(f"  Cutoff: {cfg.late_cutoff_month}/{cfg.late_cutoff_day}")
    print(f"  Cap values: {cfg.cap_values}")

    results = run_late_water_saving_workflow(
        wy_start=min(cfg.water_years),
        wy_end=max(cfg.water_years),
        counties=cfg.counties,
        cutoff_month=cfg.late_cutoff_month,
        cutoff_day=cfg.late_cutoff_day,
        cap_values=cfg.cap_values,
        compute_area_acft=True,
        evi_mode=cfg.evi_mode,
        min_segment_days=cfg.min_segment_days,
        summer_lookback_range_days=cfg.summer_lookback,
        winter_lookback_range_days=cfg.winter_lookback,
        export_csv=True,
        et_mode=cfg.et_mode,
        method=cfg.chosen_method,
    )

    if results and "df_savings_mm" in results:
        sav = results["df_savings_mm"]
        print(f"\n  Savings summary ({len(sav)} rows):")
        print(sav.to_string(index=False))
    if results and "paths" in results:
        print(f"\n  Exported to: {results['paths']}")

    # Copy key CSVs into run-specific water_savings directory
    run_root = ensure_run_dirs(cfg.run_name)
    ws_src = Path(config.water_saving_out_dir)
    ws_dst = run_root / "water_savings"
    ws_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in ws_src.glob("*.csv"):
        shutil.copy2(f, ws_dst / f.name)
        copied += 1
    if copied:
        print(f"\n  Copied {copied} CSV(s) to run dir: {ws_dst}")

    return results


def action_water_savings_plots(cfg: PipelineConfig) -> None:
    """Generate aggregate water savings charts."""
    from es_analysis.runners.run_water_savings_plots import run_aggregate_plots
    from es_analysis.utils.run_output import ensure_run_dirs

    apply_config_overrides(cfg)

    run_root = ensure_run_dirs(cfg.run_name)
    out_dir = run_root / "water_savings"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_root / "water_savings"

    print("\n--- Generating Water Savings Plots ---")
    print(f"  Cutoff: {cfg.late_cutoff_month}/{cfg.late_cutoff_day}")
    print(f"  Cap values: {cfg.cap_values}")
    print(f"  ET mode: {cfg.et_mode}")
    print(f"  Data dir: {data_dir}")

    run_aggregate_plots(out_dir, cap_values=cfg.cap_values, et_mode=cfg.et_mode,
                        data_dir=data_dir)

    print(f"\n  Plots written to: {out_dir}")


def action_wy_type_analysis(cfg: PipelineConfig) -> None:
    """Generate Water Year Type analysis charts and stats."""
    from es_analysis.runners.run_wy_type_analysis import run_wy_type_analysis
    from es_analysis.utils.run_output import ensure_run_dirs

    apply_config_overrides(cfg)

    run_root = ensure_run_dirs(cfg.run_name)
    out_dir = run_root / "WY_types"
    stats_dir = run_root / "WY_stats"
    data_dir = run_root / "water_savings"
    et_dir = run_root / "et_correction"

    print("\n--- Water Year Type Analysis ---")
    print(f"  Counties: {', '.join(cfg.counties)}")
    print(f"  WY range: {min(cfg.water_years)}-{max(cfg.water_years)}")
    print(f"  Class mode: {getattr(cfg, 'class_mode', 'sji')}")
    print(f"  Data dir: {data_dir}")
    print(f"  ET dir:   {et_dir}")

    run_wy_type_analysis(
        counties=cfg.counties,
        wy_start=min(cfg.water_years),
        wy_end=max(cfg.water_years),
        class_mode=cfg.class_mode,
        sj_valley_only=cfg.sj_valley_only,
        output_dir=out_dir,
        stats_dir=stats_dir,
        charts=cfg.wy_charts,
        data_dir=data_dir,
        et_dir=et_dir,
    )

    print(f"\n  Figures in: {out_dir}")
    print(f"  Stats in:   {stats_dir}")


def action_export(cfg: PipelineConfig) -> Optional[Path]:
    """Export parcel-year DataFrame to CSV."""
    from es_analysis.data_providers.parcel_summary_provider import build_multicounty_matched

    apply_config_overrides(cfg)

    print("\n--- Exporting parcel-year data ---")
    df = build_multicounty_matched(
        daymet_var=cfg.daymet_var,
        wy_start=min(cfg.water_years),
        wy_end=max(cfg.water_years),
        counties=cfg.counties,
        evi_mode=cfg.evi_mode,
        et_mode=cfg.et_mode,
        method=cfg.chosen_method,
    )

    if df is None or df.empty:
        print("  No data to export.")
        return None

    from es_analysis.utils.run_output import ensure_run_dirs
    run_root = ensure_run_dirs(cfg.run_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_root / f"parcel_year_summary_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"  Saved to: {out_path}")
    return out_path


def action_heatmap(cfg: PipelineConfig) -> None:
    """Generate dual heatmap with independent colorbars and % annotations."""
    from es_analysis.charts.statistics.county_wy_et_heatmap_plot import county_wy_et_dual_heatmap
    from es_analysis.utils.run_output import ensure_run_dirs, load_dataframe, save_dataframe

    apply_config_overrides(cfg)

    print("\n--- Generating Dual Heatmap ---")

    # Try parquet first
    df = load_dataframe("multicounty_matched", cfg.run_name)
    if df is None:
        from es_analysis.data_providers.parcel_summary_provider import build_multicounty_matched
        print("  Building multicounty_matched from providers...")
        df = build_multicounty_matched(et_mode=cfg.et_mode)

    root = ensure_run_dirs(cfg.run_name)
    out_dir = root / "statistics"
    fig, axes, summary = county_wy_et_dual_heatmap(df, out_dir=out_dir)

    # Save pivot tables
    if "pivot_annual" in summary:
        save_dataframe(summary["pivot_annual"].reset_index(), "heatmap_pivot_annual", cfg.run_name)
    if "pivot_segment" in summary:
        save_dataframe(summary["pivot_segment"].reset_index(), "heatmap_pivot_segment", cfg.run_name)

    print(f"  Mean annual ET: {summary['mean_annual_et_mm']:.1f} mm")
    print(f"  Mean segment ET: {summary['mean_segment_et_mm']:.1f} mm")
    print(f"  Output: {out_dir}")


def action_anomaly_evi(cfg: PipelineConfig) -> None:
    """Generate EVI plots for anomaly parcels."""
    from es_analysis.charts.evi.evi_anomaly_plot import generate_anomaly_evi_plots
    from es_analysis.utils.run_output import ensure_run_dirs, get_run_root

    print("\n--- Generating Anomaly EVI Plots ---")

    root = ensure_run_dirs(cfg.run_name)
    out_dir = root / "test"

    # Use the CSV source (generate_anomaly_evi_plots reads CSV)
    anomaly_csv = (
        _REPO_ROOT / "es_analysis" / "output" / "figures" / "test" / "anomaly_parcels.csv"
    )

    result = generate_anomaly_evi_plots(anomaly_csv=anomaly_csv, out_dir=out_dir)
    print(f"  Output: {out_dir}")


def action_save_data(cfg: PipelineConfig) -> None:
    """Save all pipeline DataFrames as parquet for plot reproduction."""
    from es_analysis.runners.run_alfalfa_run_2 import step_save_all_data
    from es_analysis.utils.run_output import ensure_run_dirs

    apply_config_overrides(cfg)

    print("\n--- Saving All Data as Parquet ---")
    ensure_run_dirs(cfg.run_name)
    n = step_save_all_data(cfg.run_name, et_mode=cfg.et_mode)
    print(f"\n  Saved {n} parquet files to {cfg.run_name}")


def action_run_versioned(cfg: PipelineConfig) -> None:
    """Create a fresh versioned run directory with ALL data + ALL plots."""
    from es_analysis.runners.run_alfalfa_run_2 import run_versioned

    apply_config_overrides(cfg)

    print(f"\n--- Versioned Run: {cfg.run_name} ---")
    print(f"  This will create {cfg.run_name}/ with all data and all plots from scratch.")
    run_versioned(
        run_name=cfg.run_name,
        plots_only=False,
        et_mode=cfg.et_mode,
        class_mode=cfg.class_mode,
    )


def action_full(cfg: PipelineConfig) -> None:
    """Run all actions in sequence."""
    divider = "=" * 60

    print(f"\n{divider}")
    print("  FULL PIPELINE RUN")
    print(f"{divider}")

    print(f"\n{divider}")
    print("  Step 1/11: Summary Statistics")
    print(divider)
    action_summary(cfg)

    print(f"\n{divider}")
    print("  Step 2/11: Export Data")
    print(divider)
    action_export(cfg)

    print(f"\n{divider}")
    print("  Step 3/11: Cuttings Analysis Plots (scatter + boxplot)")
    print(divider)
    action_plots(cfg)

    print(f"\n{divider}")
    print("  Step 4/11: ET Correction Plots (daily + monthly per parcel)")
    print(divider)
    action_et_correction_plots(cfg)

    print(f"\n{divider}")
    print("  Step 5/11: ET Correction Stats")
    print(divider)
    action_et_stats(cfg)

    print(f"\n{divider}")
    print("  Step 6/11: Late-Water Savings (data + CSV)")
    print(divider)
    action_late_water(cfg)

    print(f"\n{divider}")
    print("  Step 7/11: Water Savings Plots")
    print(divider)
    action_water_savings_plots(cfg)

    print(f"\n{divider}")
    print("  Step 8/11: Water Year Type Analysis")
    print(divider)
    action_wy_type_analysis(cfg)

    print(f"\n{divider}")
    print("  Step 9/11: Save All Data (parquet)")
    print(divider)
    action_save_data(cfg)

    print(f"\n{divider}")
    print("  Step 10/11: Dual Heatmap")
    print(divider)
    action_heatmap(cfg)

    print(f"\n{divider}")
    print("  Step 11/11: Anomaly EVI Plots")
    print(divider)
    action_anomaly_evi(cfg)

    print(f"\n{divider}")
    print("  FULL PIPELINE COMPLETE")
    print(divider)


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------
def _input_choice(prompt: str, valid: range) -> int:
    """Read an integer choice from stdin."""
    while True:
        try:
            val = int(input(prompt).strip())
            if val in valid:
                return val
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number in {valid.start}-{valid.stop - 1}")


def _print_beast_status():
    """Print BEAST processing status for all counties."""
    status = check_beast_status()
    print("\n  BEAST Processing Status:")
    for i, county in enumerate(COUNTY_ORDER, 1):
        info = status[county]
        marker = _beast_marker(info)
        parcels = f"({info['n_parcels']} parcels)" if info["n_parcels"] else ""
        sym = {"complete": "+", "missing": "o"}.get(marker, "~")
        print(f"    [{i:>2}] {sym} {county:<16} {marker:<12} {parcels}")
    return status


def _select_counties_interactive() -> List[str]:
    """Prompt user to select counties from COUNTY_ORDER."""
    status = _print_beast_status()
    print(f"    [ a] All completed counties")
    print(f"    [ *] All 10 counties")
    sel = input("\n  Select counties (comma-separated numbers, or a/*): ").strip()

    if sel == "*":
        return list(COUNTY_ORDER)
    if sel.lower() == "a":
        return [c for c in COUNTY_ORDER if _beast_marker(status[c]) == "complete"]

    try:
        indices = [int(x.strip()) for x in sel.split(",")]
        return [COUNTY_ORDER[i - 1] for i in indices if 1 <= i <= len(COUNTY_ORDER)]
    except (ValueError, IndexError):
        print("  Invalid selection, using all counties.")
        return list(COUNTY_ORDER)


def _select_water_years_interactive() -> List[int]:
    """Prompt user for water year(s)."""
    print(f"\n  Available water years: {ALL_WYS[0]}-{ALL_WYS[-1]}")
    sel = input("  Enter WY (single, range like 2019-2024, or 'all'): ").strip()

    if sel.lower() == "all":
        return list(ALL_WYS)
    if "-" in sel:
        parts = sel.split("-")
        try:
            return list(range(int(parts[0]), int(parts[1]) + 1))
        except (ValueError, IndexError):
            pass
    try:
        return [int(sel)]
    except ValueError:
        print("  Invalid input, using all water years.")
        return list(ALL_WYS)


def _print_params(cfg: PipelineConfig) -> None:
    """Display current parameter values."""
    print(f"\n  Current Parameters:")
    print(f"    Counties:           {', '.join(cfg.counties)} ({len(cfg.counties)} selected)")
    print(f"    Water Years:        {min(cfg.water_years)}-{max(cfg.water_years)}")
    print(f"    ET Method:          {cfg.chosen_method}")
    print(f"    Bootstrap:          {cfg.n_boot}")
    seg_mode = []
    if cfg.use_thermal_time:
        seg_mode.append("thermal-time")
    if cfg.use_whittaker:
        seg_mode.append("whittaker")
    if not seg_mode:
        seg_mode.append("fixed-lookback")
    print(f"    Segment Mode:       {' + '.join(seg_mode)}")
    print(f"    Min Segment Days:   {cfg.min_segment_days}")
    print(f"    EVI Mode:           {cfg.evi_mode}")
    print(f"    Whittaker Lambda:   {cfg.whittaker_interval_lambda}")
    print(f"    Late Cutoff:        {cfg.late_cutoff_month}/{cfg.late_cutoff_day}")
    print(f"    Cap Values:         {', '.join(str(v) for v in cfg.cap_values)}")
    print(f"    Workers:            {cfg.workers}")
    print(f"    Output Dir:         {cfg.output_dir}")
    print(f"    Daymet Variable:    {cfg.daymet_var}")
    print(f"    Min ET Filter:      {cfg.min_et_filter_mm} mm")
    print(f"    ET Mode:            {cfg.et_mode}")
    print(f"    Random Parcels:     {cfg.n_random_parcels} (for ET correction plots)")
    print(f"    Cloud Cover Max:    {cfg.cloud_cover_max}%")
    print(f"    Min Cuttings:       {cfg.min_cuttings}")
    print(f"    WY Class Mode:      {cfg.class_mode}")
    print(f"    SJ Valley Only:     {'Yes' if cfg.sj_valley_only else 'No'}")


def _configure_params_interactive(cfg: PipelineConfig) -> PipelineConfig:
    """Sub-menu for editing parameters."""
    while True:
        _print_params(cfg)
        print()
        print("    [c] Change counties       [w] Change water years")
        print("    [m] Change ET method       [b] Change bootstrap")
        print("    [s] Change segment params  [t] Toggle thermal-time")
        print("    [e] Change ET mode         [l] Change late-water")
        print("    [o] Change output dir      [k] Change workers")
        print("    [v] Change daymet var      [x] Cloud cover max")
        print("    [y] WY class mode          [z] Whittaker lambda")
        print("    [n] Min cuttings filter")
        print("    [d] Set defaults           [r] Return to main menu")

        choice = input("\n  Select: ").strip().lower()

        if choice == "r":
            break
        elif choice == "c":
            cfg.counties = _select_counties_interactive()
        elif choice == "w":
            cfg.water_years = _select_water_years_interactive()
        elif choice == "m":
            sel = input("  ET Method [A/B]: ").strip().upper()
            if sel in ("A", "B"):
                cfg.chosen_method = sel
        elif choice == "b":
            try:
                cfg.n_boot = int(input("  Bootstrap replicates: ").strip())
            except ValueError:
                print("  Invalid number.")
        elif choice == "s":
            try:
                cfg.min_segment_days = int(input("  Min segment days: ").strip())
            except ValueError:
                print("  Invalid number.")
            sel = input("  EVI mode [gapfilled/smoothed]: ").strip().lower()
            if sel in ("gapfilled", "smoothed"):
                cfg.evi_mode = sel
        elif choice == "t":
            cfg.use_thermal_time = not cfg.use_thermal_time
            print(f"  Thermal-time: {'ON' if cfg.use_thermal_time else 'OFF'}")
        elif choice == "l":
            try:
                md = input("  Late cutoff (month,day e.g. 7,1): ").strip().split(",")
                cfg.late_cutoff_month = int(md[0])
                cfg.late_cutoff_day = int(md[1])
            except (ValueError, IndexError):
                print("  Invalid input.")
            try:
                caps = input("  Cap values (comma-separated, e.g. 0,1,2,3): ").strip()
                cfg.cap_values = tuple(int(x) for x in caps.split(","))
            except ValueError:
                print("  Invalid input.")
        elif choice == "e":
            sel = input("  ET mode [actual/corrected/both]: ").strip().lower()
            if sel in ("actual", "corrected", "both"):
                cfg.et_mode = sel
            try:
                n = input(f"  Random parcels for ET plots [{cfg.n_random_parcels}]: ").strip()
                if n:
                    cfg.n_random_parcels = int(n)
            except ValueError:
                print("  Invalid number.")
        elif choice == "o":
            p = input("  Output directory: ").strip()
            if p:
                cfg.output_dir = Path(p)
        elif choice == "k":
            try:
                cfg.workers = int(input("  Parallel workers: ").strip())
            except ValueError:
                print("  Invalid number.")
        elif choice == "v":
            sel = input("  Daymet variable [gdd5/tmax]: ").strip().lower()
            if sel in ("gdd5", "tmax"):
                cfg.daymet_var = sel
        elif choice == "x":
            try:
                cfg.cloud_cover_max = float(input("  Cloud cover max (%): ").strip())
            except ValueError:
                print("  Invalid number.")
        elif choice == "y":
            print("    sji   = SJV Water Year Index for all counties")
            print("    usdm  = USDM peak drought for all counties")
            print("    mixed = SJV for 8 SJ Valley + USDM for Imperial/Riverside")
            sel = input("  WY class mode [sji/usdm/mixed]: ").strip().lower()
            if sel in ("sji", "usdm", "mixed"):
                cfg.class_mode = sel
        elif choice == "z":
            try:
                cfg.whittaker_interval_lambda = float(input("  Whittaker lambda: ").strip())
            except ValueError:
                print("  Invalid number.")
        elif choice == "n":
            _ed_min_cuttings(cfg)
        elif choice == "d":
            new = PipelineConfig()
            cfg.counties = new.counties
            cfg.water_years = new.water_years
            cfg.chosen_method = new.chosen_method
            cfg.n_boot = new.n_boot
            cfg.evi_mode = new.evi_mode
            cfg.min_segment_days = new.min_segment_days
            cfg.use_thermal_time = new.use_thermal_time
            cfg.use_whittaker = new.use_whittaker
            cfg.whittaker_interval_lambda = new.whittaker_interval_lambda
            cfg.late_cutoff_month = new.late_cutoff_month
            cfg.late_cutoff_day = new.late_cutoff_day
            cfg.cap_values = new.cap_values
            cfg.workers = new.workers
            cfg.output_dir = new.output_dir
            cfg.daymet_var = new.daymet_var
            cfg.cloud_cover_max = new.cloud_cover_max
            cfg.class_mode = new.class_mode
            cfg.sj_valley_only = new.sj_valley_only
            cfg.wy_charts = new.wy_charts
            cfg.et_mode = new.et_mode
            cfg.min_et_filter_mm = new.min_et_filter_mm
            cfg.n_random_parcels = new.n_random_parcels
            cfg.min_cuttings = new.min_cuttings
            print("  Parameters reset to defaults.")

    return cfg


# ---------------------------------------------------------------------------
# Per-action sub-menu infrastructure
# ---------------------------------------------------------------------------
# Editors — each modifies cfg in-place and returns it
def _ed_counties(cfg):
    cfg.counties = _select_counties_interactive()
    return cfg

def _ed_water_years(cfg):
    cfg.water_years = _select_water_years_interactive()
    return cfg

def _ed_et_corr_county(cfg):
    """Select single county for ET correction plots."""
    print("  Select county for ET correction plots:")
    for i, c in enumerate(COUNTY_ORDER, 1):
        print(f"    [{i}] {c}")
    print(f"    [0] (prompt at runtime)")
    sel = input("  Select: ").strip()
    if sel == "0":
        cfg.et_corr_county = None
    else:
        try:
            idx = int(sel) - 1
            cfg.et_corr_county = COUNTY_ORDER[idx]
        except (ValueError, IndexError):
            print("  Invalid, keeping current.")
    return cfg

def _ed_method(cfg):
    sel = input("  ET Method [A/B]: ").strip().upper()
    if sel in ("A", "B"):
        cfg.chosen_method = sel
    return cfg

def _ed_n_boot(cfg):
    try:
        cfg.n_boot = int(input("  Bootstrap replicates: ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_daymet_var(cfg):
    sel = input("  Daymet variable [gdd5/tmax]: ").strip().lower()
    if sel in ("gdd5", "tmax"):
        cfg.daymet_var = sel
    return cfg

def _ed_et_mode(cfg):
    sel = input("  ET mode [actual/corrected/both]: ").strip().lower()
    if sel in ("actual", "corrected", "both"):
        cfg.et_mode = sel
    return cfg

def _ed_evi_mode(cfg):
    sel = input("  EVI mode [gapfilled/smoothed]: ").strip().lower()
    if sel in ("gapfilled", "smoothed"):
        cfg.evi_mode = sel
    return cfg

def _ed_min_et_filter(cfg):
    try:
        cfg.min_et_filter_mm = float(input("  Min ET filter (mm): ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_cloud_cover(cfg):
    try:
        cfg.cloud_cover_max = float(input("  Cloud cover max (%): ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_parcel_id(cfg):
    sel = input("  Parcel UniqueID (blank = auto-select max-cuts): ").strip()
    cfg.parcel_id = sel if sel else None
    return cfg

def _ed_ci_alpha(cfg):
    try:
        cfg.ci_alpha = float(input("  CI alpha (e.g. 0.10): ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_workers(cfg):
    try:
        cfg.workers = int(input("  Parallel workers: ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_n_random(cfg):
    try:
        cfg.n_random_parcels = int(input("  Random parcels: ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_r_days(cfg):
    try:
        cfg.r_days = int(input("  Reference window days: ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_pre_post_window(cfg):
    try:
        cfg.pre_window = int(input("  Pre-window days: ").strip())
        cfg.post_window = int(input("  Post-window days: ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_late_cutoff(cfg):
    try:
        md = input("  Late cutoff (month,day e.g. 7,1): ").strip().split(",")
        cfg.late_cutoff_month = int(md[0])
        cfg.late_cutoff_day = int(md[1])
    except (ValueError, IndexError):
        print("  Invalid input.")
    return cfg

def _ed_cap_values(cfg):
    try:
        caps = input("  Cap values (comma-separated, e.g. 0,1,2,3): ").strip()
        cfg.cap_values = tuple(int(x) for x in caps.split(","))
    except ValueError:
        print("  Invalid input.")
    return cfg

def _ed_class_mode(cfg):
    print("    sji   = SJV Water Year Index for all counties")
    print("    usdm  = USDM peak drought for all counties")
    print("    mixed = SJV for 8 SJ Valley + USDM for Imperial/Riverside")
    sel = input("  Classification mode [sji/usdm/mixed]: ").strip().lower()
    if sel in ("sji", "usdm", "mixed"):
        cfg.class_mode = sel
    return cfg

def _ed_sj_valley_only(cfg):
    cfg.sj_valley_only = not cfg.sj_valley_only
    print(f"  SJ Valley Only: {'ON (exclude Imperial/Riverside)' if cfg.sj_valley_only else 'OFF (all 10 counties)'}")
    return cfg

def _ed_wy_charts(cfg):
    all_charts = ["cuttings", "annual_et", "heatmap", "correction", "savings", "late_cut", "multipanel"]
    print("  Available charts:")
    for i, c in enumerate(all_charts, 1):
        sel = "*" if cfg.wy_charts is None or c in (cfg.wy_charts or []) else " "
        print(f"    [{i}] [{sel}] {c}")
    print(f"    [a] All charts")
    sel = input("  Select charts (comma-separated numbers or 'a'): ").strip()
    if sel.lower() == "a":
        cfg.wy_charts = None  # None means all
    else:
        try:
            indices = [int(x.strip()) for x in sel.split(",")]
            cfg.wy_charts = [all_charts[i - 1] for i in indices if 1 <= i <= len(all_charts)]
        except (ValueError, IndexError):
            print("  Invalid selection.")
    return cfg

def _ed_min_segment(cfg):
    try:
        cfg.min_segment_days = int(input("  Min segment days: ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_whittaker_lambda(cfg):
    try:
        cfg.whittaker_interval_lambda = float(input("  Whittaker lambda: ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

def _ed_thermal_time(cfg):
    cfg.use_thermal_time = not cfg.use_thermal_time
    print(f"  Thermal-time: {'ON' if cfg.use_thermal_time else 'OFF'}")
    return cfg

def _ed_min_cuttings(cfg):
    try:
        cfg.min_cuttings = int(input("  Min cuttings (0=include all, 1=exclude zero-cut): ").strip())
    except ValueError:
        print("  Invalid number.")
    return cfg

# Value formatters
def _fmt_counties(cfg):
    if len(cfg.counties) == len(COUNTY_ORDER):
        return f"All ({len(cfg.counties)})"
    elif len(cfg.counties) <= 3:
        return ", ".join(cfg.counties)
    return f"{', '.join(cfg.counties[:3])}, ... ({len(cfg.counties)})"

def _fmt_wys(cfg):
    if len(cfg.water_years) == 1:
        return str(cfg.water_years[0])
    return f"{min(cfg.water_years)}-{max(cfg.water_years)}"

def _fmt_wy_charts(cfg):
    if cfg.wy_charts is None:
        return "All (7)"
    return ", ".join(cfg.wy_charts)


def _run_action_menu(cfg: PipelineConfig, title: str, params: list, action_fn) -> None:
    """Generic per-action sub-menu: show params, allow editing, then run.

    params: list of (label, value_fn, editor_fn) tuples
        - label:     display name (str)
        - value_fn:  callable(cfg) -> str for current value display
        - editor_fn: callable(cfg) -> cfg for editing
    """
    while True:
        print(f"\n  --- {title} ---")
        for i, (label, val_fn, _) in enumerate(params, 1):
            print(f"    [{i:>2}] {label:<26} {val_fn(cfg)}")
        print()
        print(f"    [ r] Run     [ q] Back")

        sel = input("\n  Edit param #, [r]un, or [q]uit: ").strip().lower()

        if sel == "q":
            return
        elif sel == "r":
            action_fn(cfg)
            return
        else:
            try:
                idx = int(sel) - 1
                if 0 <= idx < len(params):
                    _, _, editor = params[idx]
                    editor(cfg)
                else:
                    print("  Invalid selection.")
            except ValueError:
                print("  Invalid selection.")


# Per-action parameter specs: (label, value_fn, editor_fn)
_PARAMS_SUMMARY = [
    ("Counties",         _fmt_counties,                          _ed_counties),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("Daymet Variable",  lambda c: c.daymet_var,                 _ed_daymet_var),
    ("EVI Mode",         lambda c: c.evi_mode,                   _ed_evi_mode),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
    ("ET Method",        lambda c: c.chosen_method,              _ed_method),
    ("Min Cuttings",     lambda c: str(c.min_cuttings),          _ed_min_cuttings),
]

_PARAMS_PLOTS = [
    ("Counties",         _fmt_counties,                          _ed_counties),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("Min ET Filter",    lambda c: f"{c.min_et_filter_mm} mm",   _ed_min_et_filter),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
    ("ET Method",        lambda c: c.chosen_method,              _ed_method),
    ("Min Cuttings",     lambda c: str(c.min_cuttings),          _ed_min_cuttings),
]

_PARAMS_ET_CORRECTION_PLOTS = [
    ("Plot County",      lambda c: c.et_corr_county or "(prompt at runtime)", _ed_et_corr_county),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("Parcel ID",        lambda c: c.parcel_id or "(auto: max-cuts)", _ed_parcel_id),
    ("N Random Parcels", lambda c: str(c.n_random_parcels),      _ed_n_random),
    ("Cloud Cover Max",  lambda c: f"{c.cloud_cover_max}%",      _ed_cloud_cover),
    ("ET Method",        lambda c: c.chosen_method,              _ed_method),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
    ("CI Alpha",         lambda c: str(c.ci_alpha),              _ed_ci_alpha),
    ("N Bootstrap",      lambda c: str(c.n_boot),                _ed_n_boot),
]

_PARAMS_ET_STATS = [
    ("Counties",         _fmt_counties,                          _ed_counties),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("N Bootstrap",      lambda c: str(c.n_boot),                _ed_n_boot),
    ("ET Method",        lambda c: c.chosen_method,              _ed_method),
    ("Cloud Cover Max",  lambda c: f"{c.cloud_cover_max}%",      _ed_cloud_cover),
    ("R Days",           lambda c: str(c.r_days),                _ed_r_days),
    ("Pre/Post Window",  lambda c: f"{c.pre_window}/{c.post_window}", _ed_pre_post_window),
    ("CI Alpha",         lambda c: str(c.ci_alpha),              _ed_ci_alpha),
    ("Workers",          lambda c: str(c.workers),               _ed_workers),
    ("Min Cuttings",     lambda c: str(c.min_cuttings),          _ed_min_cuttings),
]

_PARAMS_LATE_WATER = [
    ("Counties",         _fmt_counties,                          _ed_counties),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("Late Cutoff",      lambda c: f"{c.late_cutoff_month}/{c.late_cutoff_day}", _ed_late_cutoff),
    ("Cap Values",       lambda c: ", ".join(str(v) for v in c.cap_values), _ed_cap_values),
    ("EVI Mode",         lambda c: c.evi_mode,                   _ed_evi_mode),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
    ("ET Method",        lambda c: c.chosen_method,              _ed_method),
    ("Min Segment Days", lambda c: str(c.min_segment_days),      _ed_min_segment),
    ("Min Cuttings",     lambda c: str(c.min_cuttings),          _ed_min_cuttings),
]

_PARAMS_WATER_SAVINGS_PLOTS = [
    ("Cap Values",       lambda c: ", ".join(str(v) for v in c.cap_values), _ed_cap_values),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
    ("Late Cutoff",      lambda c: f"{c.late_cutoff_month}/{c.late_cutoff_day}", _ed_late_cutoff),
]

_PARAMS_WY_TYPE = [
    ("Counties",         _fmt_counties,                          _ed_counties),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("Classification",   lambda c: c.class_mode,                 _ed_class_mode),
    ("SJ Valley Only",   lambda c: "Yes" if c.sj_valley_only else "No", _ed_sj_valley_only),
    ("Charts",           _fmt_wy_charts,                         _ed_wy_charts),
    ("Min Cuttings",     lambda c: str(c.min_cuttings),          _ed_min_cuttings),
]

_PARAMS_EXPORT = [
    ("Counties",         _fmt_counties,                          _ed_counties),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("Daymet Variable",  lambda c: c.daymet_var,                 _ed_daymet_var),
    ("EVI Mode",         lambda c: c.evi_mode,                   _ed_evi_mode),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
    ("ET Method",        lambda c: c.chosen_method,              _ed_method),
]


def _ed_run_name(cfg):
    sel = input(f"  Run name [{cfg.run_name}]: ").strip()
    if sel:
        cfg.run_name = sel
    return cfg

_PARAMS_HEATMAP = [
    ("Counties",         _fmt_counties,                          _ed_counties),
    ("Water Years",      _fmt_wys,                               _ed_water_years),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
]

_PARAMS_ANOMALY_EVI = [
    ("Run Name",         lambda c: c.run_name,                   _ed_run_name),
]

_PARAMS_SAVE_DATA = [
    ("Run Name",         lambda c: c.run_name,                   _ed_run_name),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
]

_PARAMS_RUN_VERSIONED = [
    ("Run Name",         lambda c: c.run_name,                   _ed_run_name),
    ("ET Mode",          lambda c: c.et_mode,                    _ed_et_mode),
    ("WY Class Mode",    lambda c: c.class_mode,                 _ed_class_mode),
]


def _run_debug_menu(cfg: PipelineConfig) -> None:
    """Special sub-menu for single-parcel debug (requires county/wy/parcel)."""
    # Ensure single county
    if len(cfg.counties) != 1:
        print("\n  Debug requires a single county.")
        cfg.counties = _select_counties_interactive()[:1]
    # Ensure single WY
    if len(cfg.water_years) != 1:
        print("  Debug requires a single water year.")
        cfg.water_years = _select_water_years_interactive()[:1]

    params = [
        ("County",      lambda c: c.counties[0] if c.counties else "?", _ed_counties),
        ("Water Year",  lambda c: str(c.water_years[0]) if c.water_years else "?", _ed_water_years),
        ("Parcel ID",   lambda c: c.parcel_id or "(not set)", _ed_parcel_id),
        ("EVI Mode",    lambda c: c.evi_mode, _ed_evi_mode),
        ("Min Seg Days", lambda c: str(c.min_segment_days), _ed_min_segment),
        ("Thermal Time", lambda c: "ON" if c.use_thermal_time else "OFF", _ed_thermal_time),
    ]

    while True:
        print(f"\n  --- Debug Single Parcel ---")
        for i, (label, val_fn, _) in enumerate(params, 1):
            print(f"    [{i:>2}] {label:<26} {val_fn(cfg)}")
        print()
        if not cfg.parcel_id:
            print("    ** Parcel ID is required — edit [3] first **")
        print(f"    [ r] Run     [ q] Back")

        sel = input("\n  Edit param #, [r]un, or [q]uit: ").strip().lower()

        if sel == "q":
            return
        elif sel == "r":
            if not cfg.parcel_id:
                print("  ERROR: Set a parcel ID first.")
                continue
            # Force single county/wy
            cfg.counties = cfg.counties[:1]
            cfg.water_years = cfg.water_years[:1]
            action_debug(cfg)
            cfg.parcel_id = None  # reset after debug
            return
        else:
            try:
                idx = int(sel) - 1
                if 0 <= idx < len(params):
                    _, _, editor = params[idx]
                    editor(cfg)
                    # Re-enforce single selections for county/wy
                    if idx == 0:
                        cfg.counties = cfg.counties[:1]
                    elif idx == 1:
                        cfg.water_years = cfg.water_years[:1]
                else:
                    print("  Invalid selection.")
            except ValueError:
                print("  Invalid selection.")


def run_interactive(cfg: PipelineConfig) -> None:
    """Main interactive menu loop."""
    while True:
        print("\n" + "=" * 60)
        print("  Alfalfa ET/GDD5 Post-BEAST Pipeline")
        print("=" * 60)
        print()
        print("  Available actions:")
        print("    1. Summary Statistics      - Parcel-year summary, ET/GDD5 stats")
        print("    2. Cuttings Analysis Plots - Scatter, boxplot (with et_mode)")
        print("    3. ET Correction Plots     - Per-parcel daily+monthly two-panel")
        print("    4. ET Correction Stats     - Off-phase ET correction pipeline")
        print("    5. Debug Single Parcel     - Per-segment diagnostic + plots")
        print("    6. Late-Water Savings      - Cap scenario water savings (data)")
        print("    7. Water Savings Plots     - Cap comparison, frequency, heatmap")
        print("    8. WY Type Analysis        - Cuttings/ET/savings by water year type")
        print("    9. Export Data             - Parcel-year DataFrame to CSV")
        print("   10. Full Pipeline           - Run all above in sequence")
        print("   11. Configure Parameters    - View/edit tunable parameters")
        print("   12. BEAST Status            - Check county processing status")
        print("   13. Dual Heatmap            - County x WY heatmap (independent colorbars)")
        print("   14. Anomaly EVI Plots       - EVI curves for anomaly parcels")
        print("   15. Save Data (parquet)     - Persist all data for plot reproduction")
        print("   16. Run All (versioned)     - Fresh run dir with ALL data + ALL plots")
        print("    0. Exit")
        print()

        choice = _input_choice("  Select action [0-16]: ", range(0, 17))

        if choice == 0:
            print("  Goodbye.")
            break
        elif choice == 1:
            _run_action_menu(cfg, "Summary Statistics", _PARAMS_SUMMARY, action_summary)
        elif choice == 2:
            _run_action_menu(cfg, "Cuttings Analysis Plots", _PARAMS_PLOTS, action_plots)
        elif choice == 3:
            _run_action_menu(cfg, "ET Correction Plots", _PARAMS_ET_CORRECTION_PLOTS, action_et_correction_plots)
        elif choice == 4:
            _run_action_menu(cfg, "ET Correction Stats", _PARAMS_ET_STATS, action_et_stats)
        elif choice == 5:
            _run_debug_menu(cfg)
        elif choice == 6:
            _run_action_menu(cfg, "Late-Water Savings", _PARAMS_LATE_WATER, action_late_water)
        elif choice == 7:
            _run_action_menu(cfg, "Water Savings Plots", _PARAMS_WATER_SAVINGS_PLOTS, action_water_savings_plots)
        elif choice == 8:
            _run_action_menu(cfg, "WY Type Analysis", _PARAMS_WY_TYPE, action_wy_type_analysis)
        elif choice == 9:
            _run_action_menu(cfg, "Export Data", _PARAMS_EXPORT, action_export)
        elif choice == 10:
            action_full(cfg)
        elif choice == 11:
            cfg = _configure_params_interactive(cfg)
        elif choice == 12:
            _print_beast_status()
        elif choice == 13:
            _run_action_menu(cfg, "Dual Heatmap", _PARAMS_HEATMAP, action_heatmap)
        elif choice == 14:
            _run_action_menu(cfg, "Anomaly EVI Plots", _PARAMS_ANOMALY_EVI, action_anomaly_evi)
        elif choice == 15:
            _run_action_menu(cfg, "Save Data (parquet)", _PARAMS_SAVE_DATA, action_save_data)
        elif choice == 16:
            _run_action_menu(cfg, "Run All (versioned)", _PARAMS_RUN_VERSIONED, action_run_versioned)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alfalfa ET/GDD5 Post-BEAST Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--action",
        choices=ACTIONS,
        help="Action to run (omit for interactive menu)",
    )
    p.add_argument(
        "--counties",
        nargs="+",
        help='County name(s), e.g. Fresno Kings "San Joaquin", or "all"',
    )
    p.add_argument(
        "--county",
        help="Single county (alias for --counties with one value)",
    )
    p.add_argument(
        "--wy",
        help="Water year(s): single (2022), range (2019-2024), or 'all'",
    )
    p.add_argument("--parcel", help="Single parcel UniqueID (for debug action)")
    p.add_argument("--method", choices=["A", "B"], help="ET correction method")
    p.add_argument("--n-boot", type=int, help="Bootstrap replicates")
    p.add_argument(
        "--evi-mode", choices=["gapfilled", "smoothed"], help="EVI mode"
    )
    p.add_argument("--min-segment", type=int, help="Minimum segment days")
    p.add_argument(
        "--use-thermal", action="store_true", default=None,
        help="Enable thermal-time estimation",
    )
    p.add_argument(
        "--no-thermal", action="store_true", help="Disable thermal-time estimation",
    )
    p.add_argument("--whittaker-lambda", type=float, help="Whittaker lambda")
    p.add_argument("--late-cutoff", help="Month,Day (e.g. 7,1)")
    p.add_argument("--cap-values", help="Cap values, comma-separated (e.g. 0,1,2,3)")
    p.add_argument("--workers", type=int, help="Parallel workers")
    p.add_argument("--output-dir", type=Path, help="Output directory")
    p.add_argument("--min-et-filter", type=float, help="Min ET filter for plots (mm)")
    p.add_argument(
        "--daymet-var", choices=["gdd5", "tmax"], help="Daymet variable"
    )
    p.add_argument(
        "--et-mode", choices=["actual", "corrected", "both"],
        help="ET mode for plots: actual, corrected, or both",
    )
    p.add_argument(
        "--n-random", type=int,
        help="Number of random parcels for ET correction plots (default: 3)",
    )
    p.add_argument(
        "--class-mode", choices=["sji", "usdm", "mixed"],
        help="WY type classification mode (for wy_type_analysis action)",
    )
    p.add_argument(
        "--sj-valley-only", action="store_true",
        help="Exclude Imperial and Riverside (WY type analysis)",
    )
    p.add_argument(
        "--wy-charts", type=str,
        help="WY type charts (comma-separated: cuttings,annual_et,heatmap,correction,savings,late_cut,multipanel)",
    )
    p.add_argument(
        "--min-cuttings", type=int,
        help="Min cuttings filter (0=include all, 1=exclude zero-cut parcels)",
    )
    p.add_argument(
        "--run-name",
        default="alfalfa_run_2",
        help="Run output directory name (default: alfalfa_run_2)",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even when args are provided",
    )
    return p


def parse_water_years(wy_str: str) -> List[int]:
    """Parse water year argument: single int, range 'YYYY-YYYY', or 'all'."""
    if wy_str.lower() == "all":
        return list(ALL_WYS)
    if "-" in wy_str:
        parts = wy_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(wy_str)]


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build PipelineConfig from parsed CLI arguments."""
    cfg = PipelineConfig()

    # Counties
    counties_raw = args.counties or ([args.county] if args.county else None)
    if counties_raw:
        if len(counties_raw) == 1 and counties_raw[0].lower() == "all":
            cfg.counties = list(COUNTY_ORDER)
        else:
            cfg.counties = [normalize_county_name(c) for c in counties_raw]

    # Water years
    if args.wy:
        cfg.water_years = parse_water_years(args.wy)

    # Parcel
    if args.parcel:
        cfg.parcel_id = args.parcel

    # ET Correction
    if args.method:
        cfg.chosen_method = args.method
    if args.n_boot is not None:
        cfg.n_boot = args.n_boot

    # Segment
    if args.evi_mode:
        cfg.evi_mode = args.evi_mode
    if args.min_segment is not None:
        cfg.min_segment_days = args.min_segment
    if args.no_thermal:
        cfg.use_thermal_time = False
    elif args.use_thermal:
        cfg.use_thermal_time = True
    if args.whittaker_lambda is not None:
        cfg.whittaker_interval_lambda = args.whittaker_lambda

    # Late-water
    if args.late_cutoff:
        parts = args.late_cutoff.split(",")
        cfg.late_cutoff_month = int(parts[0])
        cfg.late_cutoff_day = int(parts[1])
    if args.cap_values:
        cfg.cap_values = tuple(int(x) for x in args.cap_values.split(","))

    # Processing
    if args.workers is not None:
        cfg.workers = args.workers
    if args.output_dir:
        cfg.output_dir = args.output_dir

    # Plotting
    if args.min_et_filter is not None:
        cfg.min_et_filter_mm = args.min_et_filter
    if args.daymet_var:
        cfg.daymet_var = args.daymet_var
    if args.et_mode:
        cfg.et_mode = args.et_mode
    if args.n_random is not None:
        cfg.n_random_parcels = args.n_random

    # WY type
    if args.class_mode:
        cfg.class_mode = args.class_mode
    if args.sj_valley_only:
        cfg.sj_valley_only = True
    if args.wy_charts:
        cfg.wy_charts = [c.strip() for c in args.wy_charts.split(",")]

    # Filtering
    if args.min_cuttings is not None:
        cfg.min_cuttings = args.min_cuttings

    # Run output
    if args.run_name:
        cfg.run_name = args.run_name

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)

    # Interactive mode: no action specified, or --interactive flag
    if args.action is None or args.interactive:
        run_interactive(cfg)
        return

    # CLI dispatch
    dispatch = {
        "summary": action_summary,
        "plots": action_plots,
        "et_correction_plots": action_et_correction_plots,
        "et_stats": action_et_stats,
        "debug": action_debug,
        "late_water": action_late_water,
        "water_savings_plots": action_water_savings_plots,
        "wy_type_analysis": action_wy_type_analysis,
        "export": action_export,
        "full": action_full,
        "heatmap": action_heatmap,
        "anomaly_evi": action_anomaly_evi,
        "save_data": action_save_data,
        "run_versioned": action_run_versioned,
    }

    fn = dispatch[args.action]
    fn(cfg)


if __name__ == "__main__":
    main()
