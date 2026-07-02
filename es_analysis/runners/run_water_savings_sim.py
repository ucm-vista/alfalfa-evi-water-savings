#!/usr/bin/env python3
"""Theoretical alfalfa water-savings sensitivity simulation.

Builds a scenario cube from an existing versioned run, using observed
parcel-year structure from the saved run data but allowing theoretical
segment-ET targets beyond the observed ET envelope.

The simulation is empirical-theoretical:
- empirical in parcel/county/WY structure, WY-type labels, and observed
  late-season ET fractions
- theoretical in the target segment ET axis and the extension of the
  cuttings axis beyond the observed envelope

Important limitation:
This workflow can only vary cutoff dates on or after the baseline
July 1 cutoff, because the saved late-cut dataset stores only cycles
already classified as late under that baseline cutoff.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.late_water_provider import wy_date
from es_analysis.data_providers.wy_type_provider import WY_TYPE_ORDER, add_wy_type_columns
from es_analysis.utils.publication_style import apply_style, save_pub_figure
from es_analysis.utils.run_output import get_run_root
from es_analysis.utils.units import mm_to_acft_per_acre

DEFAULT_ET_MAX_MM = 2500
DEFAULT_ET_STEP_MM = 150
DEFAULT_CUTOFFS = ['07-01', '07-15', '08-01', '08-15']
DEFAULT_CAPS = (0, 1)
DEFAULT_MAX_CUTTINGS = 14
DEFAULT_OUT_DIR = Path(__file__).parent.parent / 'output' / 'figures' / 'water_savings_sim'
REGION_ORDER = ['All counties', 'North', 'San Joaquin', 'South', 'Imperial']
REGION_FIGURE_ORDER = ['North', 'San Joaquin', 'South', 'Imperial']
COUNTY_TO_REGION = {
    'San Joaquin': 'North',
    'Stanislaus': 'North',
    'Merced': 'North',
    'Madera': 'North',
    'Fresno': 'San Joaquin',
    'Kings': 'San Joaquin',
    'Tulare': 'San Joaquin',
    'Kern': 'South',
    'Riverside': 'South',
    'Imperial': 'Imperial',
}
BLUE_GREEN_CMAP = LinearSegmentedColormap.from_list(
    'blue_green_acft',
    ['#eff8ff', '#d2edf2', '#a6dcd8', '#70c3b0', '#349b7c', '#0d6950'],
)


def build_et_grid(max_mm: int = DEFAULT_ET_MAX_MM, step_mm: int = DEFAULT_ET_STEP_MM) -> List[float]:
    vals = list(range(0, max_mm + 1, step_mm))
    if not vals or vals[-1] != max_mm:
        vals.append(max_mm)
    return [float(v) for v in vals]


DEFAULT_ET_GRID_MM = build_et_grid()


def _parse_cutoff(spec: str) -> Tuple[int, int, str]:
    month_s, day_s = spec.split('-', 1)
    month = int(month_s)
    day = int(day_s)
    if (month, day) < (7, 1):
        raise ValueError(
            f'Cutoff {spec} is earlier than 07-01. '
            'This simulation only supports cutoffs on or after July 1 from the saved run data.'
        )
    label = pd.Timestamp(2000, month, day).strftime('%b %-d') if sys.platform != 'win32' else pd.Timestamp(2000, month, day).strftime('%b %#d')
    return month, day, label


def _ensure_list(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        try:
            parsed = ast.literal_eval(txt)
        except (SyntaxError, ValueError):
            return []
        return _ensure_list(parsed)
    return [value]


def _parse_date_list(value) -> List[pd.Timestamp]:
    if isinstance(value, str) and 'Timestamp(' in value:
        matches = re.findall(r"Timestamp\('([^']+)'\)", value)
        return [pd.Timestamp(m) for m in matches]

    out = []
    for item in _ensure_list(value):
        if item is None:
            continue
        ts = pd.Timestamp(item)
        if pd.notna(ts):
            out.append(ts)
    return out


def _parse_float_list(value) -> List[float]:
    out = []
    for item in _ensure_list(value):
        try:
            val = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out.append(val)
    return out


def _assign_region(county: str) -> str:
    return COUNTY_TO_REGION.get(str(county), 'Other')


def _subset_late_cycles_for_cutoff(row: pd.Series, month: int, day: int) -> List[float]:
    cutoff = wy_date(int(row['WY']), month, day)
    dates = _parse_date_list(row['late_cut_dates'])
    vals = _parse_float_list(row['late_cycle_et_mm_list'])
    pairs = [(d, v) for d, v in zip(dates, vals) if pd.notna(d) and np.isfinite(v)]
    pairs.sort(key=lambda x: x[0])
    return [v for d, v in pairs if d >= cutoff]


def _cap_savings(values: Sequence[float], cap_k: int) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return 0.0
    if cap_k <= 0:
        return float(np.sum(vals))
    if len(vals) <= cap_k:
        return 0.0
    return float(np.sum(vals[cap_k:]))


def _load_run_inputs(run_name: str) -> pd.DataFrame:
    run_root = get_run_root(run_name)
    data_dir = run_root / 'data'
    df_mc = pd.read_parquet(data_dir / 'multicounty_matched.parquet')
    df_sav = pd.read_parquet(data_dir / 'late_cut_savings.parquet')
    df_mc['UniqueID'] = df_mc['UniqueID'].astype(str)
    df_sav['UniqueID'] = df_sav['UniqueID'].astype(str)

    keep_cols = ['UniqueID', 'county', 'WY', 'et_cum_minET_to_last_cut_mm']
    df_et = df_mc[keep_cols].drop_duplicates()

    df = df_sav.merge(df_et, on=['UniqueID', 'county', 'WY'], how='left')
    df['segment_et_mm'] = pd.to_numeric(df['et_cum_minET_to_last_cut_mm'], errors='coerce')
    df['segment_et_mm'] = df['segment_et_mm'].fillna(pd.to_numeric(df['total_et_mm'], errors='coerce'))
    df['n_cuttings_int'] = pd.to_numeric(df['n_cuttings'], errors='coerce').round().astype('Int64')
    df['region_group'] = df['county'].map(_assign_region)
    add_wy_type_columns(df)
    return df


def _add_cutoff_savings_columns(df: pd.DataFrame, cutoff_specs, cap_values: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    for month, day, _label in cutoff_specs:
        key = f'{month:02d}{day:02d}'
        late_lists = out.apply(lambda row: _subset_late_cycles_for_cutoff(row, month, day), axis=1)
        out[f'late_cycle_et_mm_list_cutoff_{key}'] = late_lists
        out[f'n_late_cuts_cutoff_{key}'] = late_lists.apply(len)
        out[f'late_et_mm_cutoff_{key}'] = late_lists.apply(lambda vals: float(np.sum(vals)) if vals else 0.0)
        for cap_k in cap_values:
            out[f'sim_saved_mm_cap{cap_k}_cutoff_{key}'] = late_lists.apply(lambda vals, k=cap_k: _cap_savings(vals, k))
    return out


def _nearest_available_cutting(available: Sequence[int], target: int) -> Optional[int]:
    avail = [int(v) for v in available if pd.notna(v)]
    if not avail:
        return None
    return min(avail, key=lambda v: (abs(v - target), v))


def build_scenario_cube(
    df: pd.DataFrame,
    et_grid_mm: Sequence[float],
    cutoff_specs,
    cap_values: Iterable[int] = DEFAULT_CAPS,
    max_cuttings: int = DEFAULT_MAX_CUTTINGS,
) -> pd.DataFrame:
    records = []
    wy_types_present = [wt for wt in WY_TYPE_ORDER if wt in set(df['wy_type'].dropna())]
    wy_scenarios = [('Observed mix', None)] + [(wt, wt) for wt in wy_types_present]
    region_scenarios = [('All counties', None)] + [
        (region, region) for region in REGION_FIGURE_ORDER if region in set(df['region_group'].dropna())
    ]

    for month, day, cutoff_label in cutoff_specs:
        cutoff_key = f'{month:02d}{day:02d}'
        late_cuts_col = f'n_late_cuts_cutoff_{cutoff_key}'
        for cap_k in cap_values:
            saved_col = f'sim_saved_mm_cap{cap_k}_cutoff_{cutoff_key}'
            denom = pd.to_numeric(df['segment_et_mm'], errors='coerce')
            ratio = np.where(denom > 0, pd.to_numeric(df[saved_col], errors='coerce') / denom, np.nan)
            ratio = np.clip(ratio, 0.0, 1.0)
            work = df.copy()
            work['sim_ratio'] = ratio

            for region_label, region_filter in region_scenarios:
                region_sub = work.copy() if region_filter is None else work[work['region_group'] == region_filter].copy()

                for wy_label, wy_filter in wy_scenarios:
                    wy_sub = region_sub.copy() if wy_filter is None else region_sub[region_sub['wy_type'] == wy_filter].copy()
                    valid_pool = wy_sub[np.isfinite(wy_sub['sim_ratio'])].copy()
                    available_cuttings = sorted(valid_pool['n_cuttings_int'].dropna().astype(int).unique().tolist())

                    for n_cuttings in range(1, max_cuttings + 1):
                        source_cuttings = _nearest_available_cutting(available_cuttings, n_cuttings)
                        if source_cuttings is None:
                            valid = valid_pool.iloc[0:0].copy()
                        else:
                            valid = valid_pool[valid_pool['n_cuttings_int'] == source_cuttings].copy()

                        late_counts = pd.to_numeric(valid[late_cuts_col], errors='coerce') if not valid.empty else pd.Series(dtype=float)

                        for et_target_mm in et_grid_mm:
                            record = {
                                'cutoff_label': cutoff_label,
                                'cutoff_month': month,
                                'cutoff_day': day,
                                'wy_type_scenario': wy_label,
                                'region_scenario': region_label,
                                'cap_k': int(cap_k),
                                'n_cuttings': int(n_cuttings),
                                'source_n_cuttings': int(source_cuttings) if source_cuttings is not None else np.nan,
                                'cuttings_extrapolated': bool(source_cuttings is not None and source_cuttings != n_cuttings),
                                'et_target_mm': float(et_target_mm),
                                'n_samples': int(len(valid)),
                                'n_counties': int(valid['county'].nunique()) if not valid.empty else 0,
                                'n_wys': int(valid['WY'].nunique()) if not valid.empty else 0,
                                'n_unique_parcels': int(valid['UniqueID'].nunique()) if not valid.empty else 0,
                                'observed_segment_et_min_mm': float(valid['segment_et_mm'].min()) if not valid.empty else np.nan,
                                'observed_segment_et_max_mm': float(valid['segment_et_mm'].max()) if not valid.empty else np.nan,
                                'observed_saved_min_mm': float(valid[saved_col].min()) if not valid.empty else np.nan,
                                'observed_saved_max_mm': float(valid[saved_col].max()) if not valid.empty else np.nan,
                                'late_cuts_min': float(late_counts.min()) if not late_counts.empty else np.nan,
                                'late_cuts_mean': float(late_counts.mean()) if not late_counts.empty else np.nan,
                                'late_cuts_max': float(late_counts.max()) if not late_counts.empty else np.nan,
                            }
                            if valid.empty:
                                record.update({
                                    'savings_min_mm': np.nan,
                                    'savings_mean_mm': np.nan,
                                    'savings_median_mm': np.nan,
                                    'savings_max_mm': np.nan,
                                    'savings_range_mm': np.nan,
                                    'savings_min_acft_per_acre': np.nan,
                                    'savings_mean_acft_per_acre': np.nan,
                                    'savings_max_acft_per_acre': np.nan,
                                    'savings_range_acft_per_acre': np.nan,
                                })
                                records.append(record)
                                continue

                            theoretical = valid['sim_ratio'].to_numpy(dtype=float) * float(et_target_mm)
                            theoretical_acft = mm_to_acft_per_acre(theoretical)
                            record.update({
                                'savings_min_mm': float(np.min(theoretical)),
                                'savings_mean_mm': float(np.mean(theoretical)),
                                'savings_median_mm': float(np.median(theoretical)),
                                'savings_max_mm': float(np.max(theoretical)),
                                'savings_range_mm': float(np.max(theoretical) - np.min(theoretical)),
                                'savings_min_acft_per_acre': float(np.min(theoretical_acft)),
                                'savings_mean_acft_per_acre': float(np.mean(theoretical_acft)),
                                'savings_max_acft_per_acre': float(np.max(theoretical_acft)),
                                'savings_range_acft_per_acre': float(np.max(theoretical_acft) - np.min(theoretical_acft)),
                            })
                            records.append(record)

    return pd.DataFrame.from_records(records)


def _format_savings_annotation(min_v: float, max_v: float, late_min: float, late_max: float, include_late: bool) -> str:
    if not (np.isfinite(min_v) and np.isfinite(max_v)):
        return ''
    savings_txt = f'{min_v:.2f}|{max_v:.2f}'
    if not include_late or not (np.isfinite(late_min) and np.isfinite(late_max)):
        return savings_txt
    if int(round(late_min)) == int(round(late_max)):
        late_txt = f'L{int(round(late_min))}'
    else:
        late_txt = f'L{int(round(late_min))}-{int(round(late_max))}'
    return f'{late_txt}\n{savings_txt}'


def _format_late_cut_annotation(min_v: float, max_v: float) -> str:
    if not (np.isfinite(min_v) and np.isfinite(max_v)):
        return ''
    if int(round(min_v)) == int(round(max_v)):
        return f'{int(round(min_v))}'
    return f'{int(round(min_v))}|{int(round(max_v))}'


def _draw_heatmap(
    ax: plt.Axes,
    pivot_value: pd.DataFrame,
    pivot_ann: pd.DataFrame,
    title: str,
    show_y: bool,
    show_x: bool,
    cmap,
    vmin: float,
    vmax: float,
    xlabel: str,
    ylabel: str,
    ann_fontsize: float = 2.9,
) -> plt.Axes:
    data = pivot_value.values.astype(float)
    data_ma = np.ma.masked_invalid(data)
    im = ax.imshow(data_ma, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ann = pivot_ann.iloc[i, j]
            if isinstance(ann, str) and ann:
                val = data[i, j]
                text_color = 'white' if np.isfinite(val) and val > 0.62 * vmax else '#10231c'
                ax.text(j, i, ann, ha='center', va='center', fontsize=ann_fontsize, color=text_color, linespacing=0.88)
    ax.set_title(title, fontsize=9.5, pad=4)
    ax.set_xticks(np.arange(len(pivot_value.columns)))
    ax.set_yticks(np.arange(len(pivot_value.index)))
    if show_x:
        ax.set_xticklabels([str(int(v)) for v in pivot_value.columns], rotation=45, ha='right', fontsize=5.3)
        ax.set_xlabel(xlabel, fontsize=8.2)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticklabels([str(int(v)) for v in pivot_value.index], fontsize=6.3)
        ax.set_ylabel(ylabel, fontsize=8.2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color('black')
    return im


def _add_side_colorbar(fig: plt.Figure, axes: np.ndarray, im, label: str, tick_size: float = 7.0, label_size: float = 8.0):
    top = axes[0, -1].get_position().y1
    bottom = axes[-1, -1].get_position().y0
    right = axes[-1, -1].get_position().x1
    cax = fig.add_axes([right + 0.008, bottom, 0.014, top - bottom])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    return cbar


def plot_all_counties_wy_heatmaps(cube: pd.DataFrame, cap_k: int, out_dir: Path) -> None:
    apply_style()
    cutoff_order = list(dict.fromkeys(cube['cutoff_label'].tolist()))
    wy_order = ['Observed mix'] + [wt for wt in WY_TYPE_ORDER if wt in set(cube['wy_type_scenario'])]
    wy_order = [wt for wt in wy_order if wt in set(cube['wy_type_scenario'])]
    cuts = list(range(1, int(cube['n_cuttings'].max()) + 1))
    et_values = sorted(cube['et_target_mm'].unique().tolist())
    cap_df = cube[(cube['cap_k'] == cap_k) & (cube['region_scenario'] == 'All counties')].copy()

    n_rows = len(cutoff_order)
    n_cols = len(wy_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(18, 3.15 * n_cols), max(12, 2.7 * n_rows)), squeeze=False)

    vmax = float(np.nanmax(cap_df['savings_mean_acft_per_acre'])) if cap_df['savings_mean_acft_per_acre'].notna().any() else 1.0
    cmap = BLUE_GREEN_CMAP.with_extremes(bad='#f3f3f3')
    im = None
    for i, cutoff_label in enumerate(cutoff_order):
        for j, wy_label in enumerate(wy_order):
            ax = axes[i, j]
            sub = cap_df[(cap_df['cutoff_label'] == cutoff_label) & (cap_df['wy_type_scenario'] == wy_label)].copy()
            pivot_value = sub.pivot(index='n_cuttings', columns='et_target_mm', values='savings_mean_acft_per_acre').reindex(index=cuts, columns=et_values)
            pivot_min = sub.pivot(index='n_cuttings', columns='et_target_mm', values='savings_min_acft_per_acre').reindex(index=cuts, columns=et_values)
            pivot_max = sub.pivot(index='n_cuttings', columns='et_target_mm', values='savings_max_acft_per_acre').reindex(index=cuts, columns=et_values)
            pivot_late_min = sub.pivot(index='n_cuttings', columns='et_target_mm', values='late_cuts_min').reindex(index=cuts, columns=et_values)
            pivot_late_max = sub.pivot(index='n_cuttings', columns='et_target_mm', values='late_cuts_max').reindex(index=cuts, columns=et_values)
            ann = pd.DataFrame('', index=cuts, columns=et_values)
            for r in cuts:
                for c in et_values:
                    ann.loc[r, c] = _format_savings_annotation(
                        pivot_min.loc[r, c],
                        pivot_max.loc[r, c],
                        pivot_late_min.loc[r, c],
                        pivot_late_max.loc[r, c],
                        include_late=True,
                    )
            im = _draw_heatmap(
                ax,
                pivot_value,
                ann,
                title=f'{wy_label} | {cutoff_label}',
                show_y=(j == 0),
                show_x=(i == n_rows - 1),
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                xlabel='Target segment ET (mm, 150-mm bins)',
                ylabel='Cuttings',
                ann_fontsize=2.55,
            )

    fig.suptitle(
        f'Theoretical water-savings sensitivity across WY scenarios (cap {cap_k})\n'
        'Color = mean simulated savings (ac-ft/ac); cell text = late cuts and min|max savings',
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(left=0.055, right=0.905, bottom=0.085, top=0.90, wspace=0.11, hspace=0.18)
    _add_side_colorbar(fig, axes, im, 'Mean simulated savings (ac-ft/ac)')
    save_pub_figure(fig, f'water_savings_sim_cap{cap_k}_wy_heatmap_acft_per_acre', out_dir)


def plot_regional_savings_heatmaps(cube: pd.DataFrame, cap_k: int, out_dir: Path) -> None:
    apply_style()
    cutoff_order = list(dict.fromkeys(cube['cutoff_label'].tolist()))
    cuts = list(range(1, int(cube['n_cuttings'].max()) + 1))
    et_values = sorted(cube['et_target_mm'].unique().tolist())
    cap_df = cube[
        (cube['cap_k'] == cap_k)
        & (cube['wy_type_scenario'] == 'Observed mix')
        & (cube['region_scenario'].isin(REGION_FIGURE_ORDER))
    ].copy()

    n_rows = len(cutoff_order)
    n_cols = len(REGION_FIGURE_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17.0, 13.2), squeeze=False)

    vmax = float(np.nanmax(cap_df['savings_mean_acft_per_acre'])) if cap_df['savings_mean_acft_per_acre'].notna().any() else 1.0
    cmap = BLUE_GREEN_CMAP.with_extremes(bad='#f3f3f3')
    im = None
    for i, cutoff_label in enumerate(cutoff_order):
        for j, region_label in enumerate(REGION_FIGURE_ORDER):
            ax = axes[i, j]
            sub = cap_df[(cap_df['cutoff_label'] == cutoff_label) & (cap_df['region_scenario'] == region_label)].copy()
            pivot_value = sub.pivot(index='n_cuttings', columns='et_target_mm', values='savings_mean_acft_per_acre').reindex(index=cuts, columns=et_values)
            pivot_min = sub.pivot(index='n_cuttings', columns='et_target_mm', values='savings_min_acft_per_acre').reindex(index=cuts, columns=et_values)
            pivot_max = sub.pivot(index='n_cuttings', columns='et_target_mm', values='savings_max_acft_per_acre').reindex(index=cuts, columns=et_values)
            pivot_late_min = sub.pivot(index='n_cuttings', columns='et_target_mm', values='late_cuts_min').reindex(index=cuts, columns=et_values)
            pivot_late_max = sub.pivot(index='n_cuttings', columns='et_target_mm', values='late_cuts_max').reindex(index=cuts, columns=et_values)
            ann = pd.DataFrame('', index=cuts, columns=et_values)
            for r in cuts:
                for c in et_values:
                    ann.loc[r, c] = _format_savings_annotation(
                        pivot_min.loc[r, c],
                        pivot_max.loc[r, c],
                        pivot_late_min.loc[r, c],
                        pivot_late_max.loc[r, c],
                        include_late=True,
                    )
            im = _draw_heatmap(
                ax,
                pivot_value,
                ann,
                title=f'{region_label} | {cutoff_label}',
                show_y=(j == 0),
                show_x=(i == n_rows - 1),
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                xlabel='Target segment ET (mm, 150-mm bins)',
                ylabel='Cuttings',
                ann_fontsize=2.8,
            )

    fig.suptitle(
        f'Theoretical water-savings sensitivity by region (cap {cap_k})\n'
        'Observed WY mix; color = mean simulated savings (ac-ft/ac); cell text = late cuts and min|max savings',
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(left=0.06, right=0.905, bottom=0.085, top=0.905, wspace=0.12, hspace=0.18)
    _add_side_colorbar(fig, axes, im, 'Mean simulated savings (ac-ft/ac)')
    save_pub_figure(fig, f'water_savings_sim_cap{cap_k}_regional_heatmap_acft_per_acre', out_dir)


def plot_regional_late_cut_heatmaps(cube: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    cutoff_order = list(dict.fromkeys(cube['cutoff_label'].tolist()))
    cuts = list(range(1, int(cube['n_cuttings'].max()) + 1))
    et_values = sorted(cube['et_target_mm'].unique().tolist())
    df = cube[
        (cube['cap_k'] == min(DEFAULT_CAPS))
        & (cube['wy_type_scenario'] == 'Observed mix')
        & (cube['region_scenario'].isin(REGION_FIGURE_ORDER))
    ].copy()

    n_rows = len(cutoff_order)
    n_cols = len(REGION_FIGURE_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17.0, 13.2), squeeze=False)

    vmax = float(np.nanmax(df['late_cuts_mean'])) if df['late_cuts_mean'].notna().any() else 1.0
    cmap = BLUE_GREEN_CMAP.with_extremes(bad='#f3f3f3')
    im = None
    for i, cutoff_label in enumerate(cutoff_order):
        for j, region_label in enumerate(REGION_FIGURE_ORDER):
            ax = axes[i, j]
            sub = df[(df['cutoff_label'] == cutoff_label) & (df['region_scenario'] == region_label)].copy()
            pivot_value = sub.pivot(index='n_cuttings', columns='et_target_mm', values='late_cuts_mean').reindex(index=cuts, columns=et_values)
            pivot_min = sub.pivot(index='n_cuttings', columns='et_target_mm', values='late_cuts_min').reindex(index=cuts, columns=et_values)
            pivot_max = sub.pivot(index='n_cuttings', columns='et_target_mm', values='late_cuts_max').reindex(index=cuts, columns=et_values)
            ann = pd.DataFrame('', index=cuts, columns=et_values)
            for r in cuts:
                for c in et_values:
                    ann.loc[r, c] = _format_late_cut_annotation(pivot_min.loc[r, c], pivot_max.loc[r, c])
            im = _draw_heatmap(
                ax,
                pivot_value,
                ann,
                title=f'{region_label} | {cutoff_label}',
                show_y=(j == 0),
                show_x=(i == n_rows - 1),
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                xlabel='Target segment ET (mm, 150-mm bins)',
                ylabel='Cuttings',
                ann_fontsize=3.1,
            )

    fig.suptitle(
        'Late-cut response by region and cutoff\n'
        'Observed WY mix; color = mean contributing late cuts; cell text = min|max late cuts',
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(left=0.06, right=0.905, bottom=0.085, top=0.905, wspace=0.12, hspace=0.18)
    _add_side_colorbar(fig, axes, im, 'Mean contributing late cuts')
    save_pub_figure(fig, 'water_savings_sim_regional_late_cuts_heatmap', out_dir)


def write_assumptions(
    out_dir: Path,
    run_name: str,
    et_grid_mm: Sequence[float],
    cutoff_specs,
    caps: Sequence[int],
    max_cuttings: int,
) -> Path:
    notes = [
        'Water savings simulation assumptions',
        '==================================',
        f'Run source: {run_name}',
        '',
        'Inputs:',
        '- Saved parcel-year run data from alfalfa_run_6',
        '- Observed parcel/county/WY structure and WY types are preserved',
        '- Savings are simulated by scaling observed late-season savings fractions to theoretical target segment ET levels',
        '- Savings figures are presented in ac-ft/ac',
        '',
        'Important limitation:',
        '- Cutoff sensitivity is only supported for cutoff dates on or after Jul 1, because the saved late-cut dataset contains only cycles that were already late under the Jul 1 baseline cutoff.',
        '',
        'Regional grouping used for this simulation:',
        '- North: San Joaquin, Stanislaus, Merced, Madera',
        '- San Joaquin: Fresno, Kings, Tulare',
        '- South: Kern, Riverside',
        '- Imperial: Imperial',
        '',
        f'ET grid (mm): {list(et_grid_mm)}',
        f'Cutoff scenarios: {[label for _, _, label in cutoff_specs]}',
        f'Cap scenarios: {list(caps)}',
        f'Cuttings axis: 1 to {max_cuttings}',
        '',
        'Heatmap interpretation:',
        '- Savings heatmaps: color is mean simulated savings (ac-ft/ac)',
        '- Savings cell text: late cuts and min|max simulated savings (ac-ft/ac)',
        '- Late-cut heatmap: color is mean contributing late cuts',
        '- ET bins use 150-mm spacing from 0 to the requested maximum, with the exact maximum appended if needed',
        '- Cuttings above the observed range use the nearest observed cut-count class within each subset',
    ]
    path = out_dir / 'simulation_assumptions.txt'
    path.write_text('\n'.join(notes) + '\n')
    return path


def run_simulation(
    run_name: str = 'alfalfa_run_6',
    out_dir: Path = DEFAULT_OUT_DIR,
    et_grid_mm: Optional[Sequence[float]] = None,
    cutoff_specs_raw: Sequence[str] = DEFAULT_CUTOFFS,
    caps: Sequence[int] = DEFAULT_CAPS,
    max_cuttings: int = DEFAULT_MAX_CUTTINGS,
    et_max_mm: int = DEFAULT_ET_MAX_MM,
    et_step_mm: int = DEFAULT_ET_STEP_MM,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    if et_grid_mm is None:
        et_grid_mm = build_et_grid(max_mm=et_max_mm, step_mm=et_step_mm)
    else:
        et_grid_mm = [float(v) for v in et_grid_mm]

    cutoff_specs = [_parse_cutoff(spec) for spec in cutoff_specs_raw]

    df = _load_run_inputs(run_name)
    df = _add_cutoff_savings_columns(df, cutoff_specs, caps)

    cube = build_scenario_cube(
        df,
        et_grid_mm=et_grid_mm,
        cutoff_specs=cutoff_specs,
        cap_values=caps,
        max_cuttings=max_cuttings,
    )
    cube.to_csv(data_dir / 'water_savings_sim_cube.csv', index=False)
    cube.to_parquet(data_dir / 'water_savings_sim_cube.parquet', index=False)

    metadata = {
        'run_name': run_name,
        'et_grid_mm': list(et_grid_mm),
        'cutoffs': [spec for spec in cutoff_specs_raw],
        'caps': list(caps),
        'max_cuttings': int(max_cuttings),
        'et_max_mm': int(et_max_mm),
        'et_step_mm': int(et_step_mm),
        'n_rows_input': int(len(df)),
        'n_rows_cube': int(len(cube)),
    }
    (data_dir / 'water_savings_sim_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    write_assumptions(out_dir, run_name, et_grid_mm, cutoff_specs, caps, max_cuttings)

    for cap_k in caps:
        plot_all_counties_wy_heatmaps(cube, cap_k=cap_k, out_dir=out_dir)
        plot_regional_savings_heatmaps(cube, cap_k=cap_k, out_dir=out_dir)
    plot_regional_late_cut_heatmaps(cube, out_dir=out_dir)

    return cube


def main() -> int:
    parser = argparse.ArgumentParser(description='Run theoretical alfalfa water-savings sensitivity simulation.')
    parser.add_argument('--run-name', default='alfalfa_run_6', help='Versioned run name to load from output/figures/<run-name>/data')
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR), help='Output directory for simulation figures and data')
    parser.add_argument('--et-grid-mm', nargs='*', type=float, default=None, help='Explicit target segment-ET scenario grid in mm')
    parser.add_argument('--et-max-mm', type=int, default=DEFAULT_ET_MAX_MM, help='Maximum ET target when using generated ET bins')
    parser.add_argument('--et-step-mm', type=int, default=DEFAULT_ET_STEP_MM, help='ET bin width in mm when using generated ET bins')
    parser.add_argument('--cutoffs', nargs='*', default=list(DEFAULT_CUTOFFS), help='Cutoff scenarios as MM-DD, only >= 07-01 supported')
    parser.add_argument('--caps', nargs='*', type=int, default=list(DEFAULT_CAPS), help='Cap scenarios to simulate')
    parser.add_argument('--max-cuttings', type=int, default=DEFAULT_MAX_CUTTINGS, help='Upper bound of theoretical cuttings axis')
    args = parser.parse_args()

    cube = run_simulation(
        run_name=args.run_name,
        out_dir=Path(args.out_dir),
        et_grid_mm=args.et_grid_mm,
        cutoff_specs_raw=args.cutoffs,
        caps=args.caps,
        max_cuttings=args.max_cuttings,
        et_max_mm=args.et_max_mm,
        et_step_mm=args.et_step_mm,
    )
    print(f'Saved simulation cube with {len(cube)} rows to {Path(args.out_dir) / "data"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
