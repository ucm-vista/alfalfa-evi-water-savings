#!/usr/bin/env python3
"""Compact empirical water-savings model for alfalfa parcel-years.

This runner builds an empirical response surface from saved parcel-WY data,
using segment ET bins and the number of contributing late cuts after a cutoff.
It is intentionally more empirical and more compact than the earlier
cuttings-based theoretical simulation.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.late_water_provider import wy_date
from es_analysis.data_providers.wy_type_provider import add_wy_type_columns
from es_analysis.utils.publication_style import apply_style, save_pub_figure
from es_analysis.utils.run_output import get_run_root
from es_analysis.utils.units import mm_to_acft_per_acre

DEFAULT_OUT_DIR = Path(__file__).parent.parent / "output" / "figures" / "water_savings_sim"
DEFAULT_CUTOFFS = ["07-01", "07-15", "08-01", "08-15"]
DEFAULT_PRIMARY_CUTOFF = "07-01"
DEFAULT_CAP_K = 0
DEFAULT_ET_MAX_MM = 2500
DEFAULT_ET_STEP_MM = 150
DEFAULT_PRIOR_N = 6.0
DEFAULT_MIN_N = 3
REGION_ORDER = ["North", "San Joaquin", "South", "Imperial"]
WY_DISPLAY_ORDER = ["Wet", "Above Normal", "Dry", "Critical"]
COUNTY_TO_REGION = {
    "San Joaquin": "North",
    "Stanislaus": "North",
    "Merced": "North",
    "Madera": "North",
    "Fresno": "San Joaquin",
    "Kings": "San Joaquin",
    "Tulare": "San Joaquin",
    "Kern": "South",
    "Riverside": "South",
    "Imperial": "Imperial",
}
BLUE_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "blue_green_surface",
    ["#f7fcfd", "#d9f0f0", "#a6dcd8", "#69c2b0", "#2f9e7e", "#0b5d48"],
)


def _parse_cutoff(spec: str) -> Tuple[int, int, str]:
    month_s, day_s = spec.split("-", 1)
    month = int(month_s)
    day = int(day_s)
    if (month, day) < (7, 1):
        raise ValueError("Only cutoff dates on or after 07-01 are supported from the saved run data.")
    label = pd.Timestamp(2000, month, day).strftime("%b %-d") if sys.platform != "win32" else pd.Timestamp(2000, month, day).strftime("%b %#d")
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
    if isinstance(value, str) and "Timestamp(" in value:
        matches = re.findall(r"Timestamp\('([^']+)'\)", value)
        return [pd.Timestamp(m) for m in matches]
    out = []
    for item in _ensure_list(value):
        if item is None:
            continue
        try:
            ts = pd.Timestamp(item)
        except Exception:
            continue
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
    return COUNTY_TO_REGION.get(str(county), "Other")


def _subset_late_cycles_for_cutoff(row: pd.Series, month: int, day: int) -> List[float]:
    cutoff = wy_date(int(row["WY"]), month, day)
    dates = _parse_date_list(row["late_cut_dates"])
    vals = _parse_float_list(row["late_cycle_et_mm_list"])
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


def _build_et_bins(max_mm: int, step_mm: int) -> Tuple[np.ndarray, List[str], List[float]]:
    left_edges = list(range(0, max_mm, step_mm))
    right_edges = left_edges[1:] + [max_mm]
    edges = np.array(left_edges + [max_mm], dtype=float)
    labels = [f"{lo}-{hi}" for lo, hi in zip(left_edges, right_edges)]
    mids = [(lo + hi) / 2.0 for lo, hi in zip(left_edges, right_edges)]
    return edges, labels, mids


def _load_base_inputs(run_name: str) -> pd.DataFrame:
    run_root = get_run_root(run_name)
    data_dir = run_root / "data"
    df_mc = pd.read_parquet(data_dir / "multicounty_matched.parquet")
    df_sav = pd.read_parquet(data_dir / "late_cut_savings.parquet")
    df_mc["UniqueID"] = df_mc["UniqueID"].astype(str)
    df_sav["UniqueID"] = df_sav["UniqueID"].astype(str)
    keep_cols = ["UniqueID", "county", "WY", "et_cum_minET_to_last_cut_mm"]
    df_et = df_mc[keep_cols].drop_duplicates()
    df = df_sav.merge(df_et, on=["UniqueID", "county", "WY"], how="left")
    df["segment_et_mm"] = pd.to_numeric(df["et_cum_minET_to_last_cut_mm"], errors="coerce")
    df["segment_et_mm"] = df["segment_et_mm"].fillna(pd.to_numeric(df["total_et_mm"], errors="coerce"))
    df["n_cuttings_int"] = pd.to_numeric(df["n_cuttings"], errors="coerce").round().astype("Int64")
    df["region"] = df["county"].map(_assign_region)
    add_wy_type_columns(df)
    return df


def build_parcel_year_long(
    df: pd.DataFrame,
    cutoff_specs: Sequence[Tuple[int, int, str]],
    cap_values: Sequence[int],
    et_max_mm: int,
    et_step_mm: int,
) -> pd.DataFrame:
    edges, labels, mids = _build_et_bins(et_max_mm, et_step_mm)
    rows = []
    for _, row in df.iterrows():
        seg_et = row.get("segment_et_mm")
        if not np.isfinite(seg_et):
            continue
        seg_et_clipped = min(float(seg_et), float(et_max_mm) - 1e-9)
        et_bin = pd.cut(pd.Series([seg_et_clipped]), bins=edges, labels=labels, include_lowest=True, right=False)[0]
        if pd.isna(et_bin):
            continue
        et_bin = str(et_bin)
        et_bin_mid = mids[labels.index(et_bin)]
        for month, day, cutoff_label in cutoff_specs:
            late_vals = _subset_late_cycles_for_cutoff(row, month, day)
            late_count = int(len(late_vals))
            base = {
                "UniqueID": row["UniqueID"],
                "county": row["county"],
                "region": row["region"],
                "WY": int(row["WY"]),
                "wy_type": row["wy_type"],
                "cutoff_label": cutoff_label,
                "segment_et_mm": float(seg_et),
                "segment_et_bin": et_bin,
                "segment_et_bin_mid": float(et_bin_mid),
                "n_cuttings": int(row["n_cuttings_int"]) if pd.notna(row["n_cuttings_int"]) else np.nan,
                "late_cut_count": late_count,
            }
            for cap_k in cap_values:
                saved_mm = _cap_savings(late_vals, cap_k)
                rec = dict(base)
                rec["cap_k"] = int(cap_k)
                rec["saved_mm"] = float(saved_mm)
                rec["saved_acft_per_acre"] = float(mm_to_acft_per_acre(saved_mm))
                rows.append(rec)
    return pd.DataFrame.from_records(rows)


def _present_wy_types(df: pd.DataFrame) -> List[str]:
    present = set(df["wy_type"].dropna())
    return [wt for wt in WY_DISPLAY_ORDER if wt in present]


def build_surface_cube(parcel_long: pd.DataFrame, cap_k: int, prior_n: float) -> pd.DataFrame:
    df = parcel_long[parcel_long["cap_k"] == cap_k].copy()
    group_cols = ["cutoff_label", "region", "wy_type", "late_cut_count", "segment_et_bin", "segment_et_bin_mid"]
    cube = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_obs=("saved_acft_per_acre", "size"),
            n_wy=("WY", "nunique"),
            mean_savings_acft_per_acre=("saved_acft_per_acre", "mean"),
            mean_saved_mm=("saved_mm", "mean"),
            mean_segment_et_mm=("segment_et_mm", "mean"),
            mean_total_cuttings=("n_cuttings", "mean"),
        )
        .reset_index()
    )
    parent = (
        df.groupby(["cutoff_label", "region", "wy_type", "late_cut_count"], dropna=False)
        .agg(parent_mean_savings_acft_per_acre=("saved_acft_per_acre", "mean"))
        .reset_index()
    )
    cube = cube.merge(parent, on=["cutoff_label", "region", "wy_type", "late_cut_count"], how="left")
    cube["smoothed_savings_acft_per_acre"] = (
        cube["n_obs"] * cube["mean_savings_acft_per_acre"] + prior_n * cube["parent_mean_savings_acft_per_acre"]
    ) / (cube["n_obs"] + prior_n)
    return cube


def build_cutoff_summary(parcel_long: pd.DataFrame, cap_k: int) -> pd.DataFrame:
    df = parcel_long[parcel_long["cap_k"] == cap_k].copy()
    summary = (
        df.groupby(["cutoff_label", "region", "wy_type"], dropna=False)
        .agg(
            n_obs=("saved_acft_per_acre", "size"),
            mean_savings_acft_per_acre=("saved_acft_per_acre", "mean"),
            mean_saved_mm=("saved_mm", "mean"),
            mean_late_cuts=("late_cut_count", "mean"),
            mean_segment_et_mm=("segment_et_mm", "mean"),
        )
        .reset_index()
    )
    return summary


def compute_diagnostics(parcel_long: pd.DataFrame, cap_k: int) -> dict:
    df = parcel_long[parcel_long["cap_k"] == cap_k].copy()
    cor_cut = df[["n_cuttings", "saved_mm"]].corr().iloc[0, 1]
    cor_late = df[["late_cut_count", "saved_mm"]].corr().iloc[0, 1]
    cor_et = df[["segment_et_mm", "saved_mm"]].corr().iloc[0, 1]
    region_means = (
        df.groupby("region")[["segment_et_mm", "saved_mm", "late_cut_count", "n_cuttings"]]
        .mean()
        .round(3)
        .to_dict(orient="index")
    )
    return {
        "corr_total_cuttings_vs_saved_mm": None if pd.isna(cor_cut) else float(cor_cut),
        "corr_late_cuts_vs_saved_mm": None if pd.isna(cor_late) else float(cor_late),
        "corr_segment_et_vs_saved_mm": None if pd.isna(cor_et) else float(cor_et),
        "region_means": region_means,
    }


def _row_categories(cube: pd.DataFrame) -> List[Tuple[str, int]]:
    wy_types = _present_wy_types(cube)
    max_late = int(cube["late_cut_count"].max()) if len(cube) else 0
    rows = []
    for wy in wy_types:
        for late in range(max_late + 1):
            rows.append((wy, late))
    return rows


def _draw_region_surface(ax: plt.Axes, sub: pd.DataFrame, row_cats: List[Tuple[str, int]], et_bins: List[str], vmin: float, vmax: float, title: str, show_y: bool, show_x: bool, cmap) -> plt.Axes:
    pivot = pd.DataFrame(np.nan, index=pd.Index(row_cats, dtype=object), columns=et_bins)
    if not sub.empty:
        work = sub.copy()
        work["row_cat"] = list(zip(work["wy_type"], work["late_cut_count"]))
        table = work.pivot(index="row_cat", columns="segment_et_bin", values="smoothed_savings_acft_per_acre")
        table = table.reindex(index=row_cats, columns=et_bins)
        pivot.loc[:, :] = table.values

    data = np.ma.masked_invalid(pivot.values.astype(float))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xticks(np.arange(len(et_bins)))
    xticklabels = [b if i % 2 == 0 or i == len(et_bins) - 1 else "" for i, b in enumerate(et_bins)]
    if show_x:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
        ax.set_xlabel("Segment ET bin (mm)", fontsize=8.5)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticks(np.arange(len(row_cats)))
        ax.set_yticklabels([f"{wy} | {late}" for wy, late in row_cats], fontsize=6.4)
        ax.set_ylabel("WY type | late cuts after cutoff", fontsize=8.5)
    else:
        ax.set_yticks(np.arange(len(row_cats)))
        ax.set_yticklabels([])
    ax.set_xlabel("Segment ET bin (mm)", fontsize=8.5)
    ax.tick_params(length=0)
    for idx in range(1, len(_present_wy_types(sub))):
        ax.axhline(idx * (len(row_cats) / max(len(_present_wy_types(sub)), 1)) - 0.5, color="#6f8c86", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")
    return im


def plot_compact_surface(surface_cube: pd.DataFrame, cutoff_label: str, cap_k: int, out_dir: Path, min_n: int) -> None:
    apply_style()
    sub = surface_cube[(surface_cube["cutoff_label"] == cutoff_label) & (surface_cube["region"].isin(REGION_ORDER))].copy()
    sub.loc[sub["n_obs"] < min_n, "smoothed_savings_acft_per_acre"] = np.nan
    row_cats = _row_categories(sub)
    et_bins = sorted(sub["segment_et_bin"].unique().tolist(), key=lambda x: float(x.split("-")[0]))
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2), squeeze=False)
    vmax = float(np.nanmax(sub["smoothed_savings_acft_per_acre"])) if sub["smoothed_savings_acft_per_acre"].notna().any() else 1.0
    cmap = BLUE_GREEN_CMAP.with_extremes(bad="#f2f2f2")
    im = None
    for idx, region in enumerate(REGION_ORDER):
        i, j = divmod(idx, 2)
        ax = axes[i, j]
        reg = sub[sub["region"] == region].copy()
        if not reg.empty and reg["n_obs"].sum() > 0:
            mean_et = float(np.average(reg["mean_segment_et_mm"], weights=reg["n_obs"]))
            title = f"{region} | mean seg ET {mean_et:.0f} mm"
        else:
            title = region
        im = _draw_region_surface(ax, reg, row_cats, et_bins, 0.0, vmax, title, show_y=(j == 0), show_x=(i == 1), cmap=cmap)
    fig.suptitle(
        f"Empirical water-savings surface ({cutoff_label}, cap {cap_k})\n"
        "Color = smoothed mean savings (ac-ft/ac) per parcel-WY cell; blanks indicate too little data",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.17, right=0.91, bottom=0.12, top=0.90, wspace=0.12, hspace=0.16)
    top = axes[0, 1].get_position().y1
    bottom = axes[1, 1].get_position().y0
    right = axes[1, 1].get_position().x1
    cax = fig.add_axes([right + 0.01, bottom, 0.015, top - bottom])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Smoothed mean savings (ac-ft/ac)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save_pub_figure(fig, f"empirical_savings_surface_{cutoff_label.replace(' ', '').lower()}_cap{cap_k}_acft_per_acre", out_dir)


def _draw_summary_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str, cmap, vmin: float, vmax: float, fmt: str, ylabel: str, show_y: bool) -> plt.Axes:
    data = np.ma.masked_invalid(pivot.values.astype(float))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                txt = format(val, fmt)
                color = "white" if val > 0.62 * vmax else "#10231c"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), fontsize=8)
    if show_y:
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index), fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8.5)
    else:
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")
    return im


def plot_cutoff_summary(summary_df: pd.DataFrame, cap_k: int, out_dir: Path) -> None:
    apply_style()
    summary = summary_df[summary_df["region"].isin(REGION_ORDER)].copy()
    row_order = []
    for region in REGION_ORDER:
        for wy in _present_wy_types(summary):
            row_order.append(f"{region} | {wy}")
    summary["row_label"] = summary["region"] + " | " + summary["wy_type"]
    desired_cutoff_order = [_parse_cutoff(spec)[2] for spec in DEFAULT_CUTOFFS]
    cutoff_order = [c for c in desired_cutoff_order if c in set(summary["cutoff_label"])]
    piv_s = summary.pivot(index="row_label", columns="cutoff_label", values="mean_savings_acft_per_acre").reindex(index=row_order, columns=cutoff_order)
    piv_l = summary.pivot(index="row_label", columns="cutoff_label", values="mean_late_cuts").reindex(index=row_order, columns=cutoff_order)
    piv_l = piv_l.round(0)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 7.8), squeeze=False)
    axes = axes[0]
    cmap = BLUE_GREEN_CMAP.with_extremes(bad="#f2f2f2")
    im1 = _draw_summary_heatmap(axes[0], piv_s, f"Savings ({cap_k})", cmap, 0.0, float(np.nanmax(piv_s.values)) if np.isfinite(np.nanmax(piv_s.values)) else 1.0, ".2f", "Region | WY type", True)
    im2 = _draw_summary_heatmap(axes[1], piv_l, "Late cuts", cmap, 0.0, float(np.nanmax(piv_l.values)) if np.isfinite(np.nanmax(piv_l.values)) else 1.0, ".1f", "", False)

    fig.suptitle(
        f"Cutoff summary by region and WY type (cap {cap_k})\n"
        "Left = mean savings (ac-ft/ac); right = mean contributing late cuts",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.24, right=0.93, bottom=0.09, top=0.88, wspace=0.18)
    cax1 = fig.add_axes([axes[0].get_position().x1 + 0.01, axes[0].get_position().y0, 0.012, axes[0].get_position().height])
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Savings (ac-ft/ac)", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    cax2 = fig.add_axes([axes[1].get_position().x1 + 0.01, axes[1].get_position().y0, 0.012, axes[1].get_position().height])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("Late cuts", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    save_pub_figure(fig, f"empirical_cutoff_summary_cap{cap_k}", out_dir)


def write_notes(out_dir: Path, diagnostics: dict, primary_cutoff: str, et_max_mm: int, et_step_mm: int, prior_n: float, min_n: int, cap_k: int) -> None:
    lines = [
        "Compact empirical water-savings model",
        "===================================",
        f"Primary cutoff figure: {primary_cutoff}",
        f"Savings cap: {cap_k}",
        f"Segment ET bins: 0-{et_max_mm} mm in {et_step_mm}-mm steps",
        f"Shrinkage prior_n: {prior_n}",
        f"Minimum n for display: {min_n}",
        "",
        "Why this model changed:",
        f"- Corr(total cuttings, saved mm) = {diagnostics['corr_total_cuttings_vs_saved_mm']:.3f}" if diagnostics['corr_total_cuttings_vs_saved_mm'] is not None else "- Corr(total cuttings, saved mm) = NA",
        f"- Corr(late cuts after cutoff, saved mm) = {diagnostics['corr_late_cuts_vs_saved_mm']:.3f}" if diagnostics['corr_late_cuts_vs_saved_mm'] is not None else "- Corr(late cuts after cutoff, saved mm) = NA",
        f"- Corr(segment ET, saved mm) = {diagnostics['corr_segment_et_vs_saved_mm']:.3f}" if diagnostics['corr_segment_et_vs_saved_mm'] is not None else "- Corr(segment ET, saved mm) = NA",
        "- The surface therefore uses segment ET bin and late-cut count as the primary cell drivers, not total cuttings.",
        "",
        "Region means (from parcel-WY records at the selected cap):",
    ]
    for region in REGION_ORDER:
        stats = diagnostics["region_means"].get(region)
        if not stats:
            continue
        lines.append(
            f"- {region}: mean segment ET {stats['segment_et_mm']:.1f} mm, mean saved {stats['saved_mm']:.1f} mm, mean late cuts {stats['late_cut_count']:.2f}, mean total cuttings {stats['n_cuttings']:.2f}"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- The compact surface is per parcel-WY, grouped by WY type and late cuts after the cutoff.")
    lines.append("- Blank cells indicate insufficient support after smoothing and the minimum-n screen.")
    lines.append("- North/San Joaquin/South/Imperial panels retain the location-based ET gradient without exploding into many facet columns.")
    (out_dir / "empirical_model_notes.txt").write_text("\n".join(lines) + "\n")


def run_model(
    run_name: str,
    out_dir: Path,
    cutoff_specs_raw: Sequence[str],
    primary_cutoff_raw: str,
    cap_k: int,
    et_max_mm: int,
    et_step_mm: int,
    prior_n: float,
    min_n: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cutoff_specs = [_parse_cutoff(spec) for spec in cutoff_specs_raw]
    primary_cutoff = _parse_cutoff(primary_cutoff_raw)[2]
    df = _load_base_inputs(run_name)
    parcel_long = build_parcel_year_long(df, cutoff_specs=cutoff_specs, cap_values=[cap_k], et_max_mm=et_max_mm, et_step_mm=et_step_mm)
    surface_cube = build_surface_cube(parcel_long, cap_k=cap_k, prior_n=prior_n)
    cutoff_summary = build_cutoff_summary(parcel_long, cap_k=cap_k)
    diagnostics = compute_diagnostics(parcel_long, cap_k=cap_k)

    parcel_long.to_parquet(data_dir / "empirical_parcel_year_long.parquet", index=False)
    surface_cube.to_parquet(data_dir / "empirical_surface_cube.parquet", index=False)
    surface_cube.to_csv(data_dir / "empirical_surface_cube.csv", index=False)
    cutoff_summary.to_parquet(data_dir / "empirical_cutoff_summary.parquet", index=False)
    cutoff_summary.to_csv(data_dir / "empirical_cutoff_summary.csv", index=False)
    (data_dir / "empirical_model_config.json").write_text(json.dumps({
        "run_name": run_name,
        "cutoffs": list(cutoff_specs_raw),
        "primary_cutoff": primary_cutoff_raw,
        "cap_k": int(cap_k),
        "et_max_mm": int(et_max_mm),
        "et_step_mm": int(et_step_mm),
        "prior_n": float(prior_n),
        "min_n": int(min_n),
        "n_parcel_year_rows": int(len(parcel_long)),
        "n_surface_rows": int(len(surface_cube)),
    }, indent=2) + "\n")
    write_notes(out_dir, diagnostics, primary_cutoff, et_max_mm, et_step_mm, prior_n, min_n, cap_k)

    plot_compact_surface(surface_cube, cutoff_label=primary_cutoff, cap_k=cap_k, out_dir=out_dir, min_n=min_n)
    plot_cutoff_summary(cutoff_summary, cap_k=cap_k, out_dir=out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compact empirical water-savings model.")
    parser.add_argument("--run-name", default="alfalfa_run_6")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cutoffs", nargs="*", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--primary-cutoff", default=DEFAULT_PRIMARY_CUTOFF)
    parser.add_argument("--cap-k", type=int, default=DEFAULT_CAP_K)
    parser.add_argument("--et-max-mm", type=int, default=DEFAULT_ET_MAX_MM)
    parser.add_argument("--et-step-mm", type=int, default=DEFAULT_ET_STEP_MM)
    parser.add_argument("--prior-n", type=float, default=DEFAULT_PRIOR_N)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    args = parser.parse_args()
    run_model(
        run_name=args.run_name,
        out_dir=Path(args.out_dir),
        cutoff_specs_raw=args.cutoffs,
        primary_cutoff_raw=args.primary_cutoff,
        cap_k=args.cap_k,
        et_max_mm=args.et_max_mm,
        et_step_mm=args.et_step_mm,
        prior_n=args.prior_n,
        min_n=args.min_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
