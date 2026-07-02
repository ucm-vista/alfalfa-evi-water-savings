"""Multi-strategy water savings simulation with economic analysis.

Extends the base water savings model (Figures 1-4 in sensitivity_heatmap.py)
with 8 management strategies, economic crossover analysis, and basin-scale
aggregation.  Produces 5 figures:

Figure 5 -- Economic crossover curves (3 panels: North, Central, Desert)
Figure 6 -- Strategy comparison heatmap (2 panels: Normal, Drought pricing)
Figure 7 -- Water savings decomposition (stacked bars, 10 counties N->S)
Figure 8 -- Seasonal ET profile under alternative strategies (3 panels)
Figure 9 -- Basin-scale water savings potential (stacked bars by strategy)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.utils.publication_style import (
    apply_style,
    save_pub_figure,
    add_panel_label,
    DOUBLE_COL_WIDTH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COUNTIES_NS = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]

COUNTY_GROUPS = {
    "North SJV": ["San Joaquin", "Stanislaus", "Merced"],
    "Central SJV": ["Madera", "Fresno", "Tulare", "Kings", "Kern"],
    "Desert": ["Riverside", "Imperial"],
}

# Dominant conservation-payment program by region.
# Imperial: IID Deficit Irrigation Program / QSA fallowing — real, ~$600/ac.
# Riverside: PVID-MWD forbearance + Coachella SGMA (different mechanism).
# SJV (North + Central): SGMA-driven programs (DWR LandFlex, MLRP) — payments
# vary by basin; values used here are illustrative estimates, not from an
# authoritative per-county program schedule.
GROUP_PROGRAM_LABEL = {
    "North SJV": "SGMA incentive",
    "Central SJV": "SGMA incentive",
    "Desert": "DIP / QSA incentive",
}

# Featured county per region — drawn as dotted reference lines on the
# crossover panels so within-group price spread is visible alongside the
# group-average Normal/Drought benchmarks.
FEATURED_COUNTY = {
    "North SJV": "San Joaquin",
    "Central SJV": "Kern",
    "Desert": "Imperial",
}

STRATEGIES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]

STRATEGY_LABELS = {
    "S0": "S0: Base case",
    "S1": "S1: Cap 1",
    "S2": "S2: Cap 0",
    "S3": "S3: Deficit only",
    "S4": "S4: Selective deficit",
    "S5": "S5: Hybrid def+cap1",
    "S6": "S6: Hybrid def+cap0",
    "S7": "S7: Early termination",
}

STRATEGY_COLORS = {
    "S0": "#9E9E9E",
    "S1": "#1976D2",
    "S2": "#0D47A1",
    "S3": "#2E7D32",
    "S4": "#F9A825",
    "S5": "#E65100",
    "S6": "#B71C1C",
    "S7": "#4A148C",
}

STRATEGY_LINESTYLES = {
    "S0": ":",
    "S1": "--",
    "S2": "-",
    "S3": "-.",
    "S4": "--",
    "S5": "-.",
    "S6": "-",
    "S7": ":",
}

# ---------------------------------------------------------------------------
# County parameters
# ---------------------------------------------------------------------------
COUNTY_WATER_PARAMS = {
    "San Joaquin": {"price_normal": 175, "price_drought": 600, "sgma": "Moderate",
                    "incentive_per_acre": 100},
    "Stanislaus":  {"price_normal": 150, "price_drought": 500, "sgma": "Moderate",
                    "incentive_per_acre": 100},
    "Merced":      {"price_normal": 175, "price_drought": 700, "sgma": "High",
                    "incentive_per_acre": 150},
    "Madera":      {"price_normal": 200, "price_drought": 750, "sgma": "High",
                    "incentive_per_acre": 150},
    "Fresno":      {"price_normal": 200, "price_drought": 800, "sgma": "High",
                    "incentive_per_acre": 200},
    "Tulare":      {"price_normal": 250, "price_drought": 1000, "sgma": "Very High",
                    "incentive_per_acre": 250},
    "Kings":       {"price_normal": 250, "price_drought": 1000, "sgma": "Very High",
                    "incentive_per_acre": 250},
    "Kern":        {"price_normal": 350, "price_drought": 1200, "sgma": "Extreme",
                    "incentive_per_acre": 300},
    "Riverside":   {"price_normal": 200, "price_drought": 600, "sgma": "Moderate",
                    "incentive_per_acre": 200},
    "Imperial":    {"price_normal": 22, "price_drought": 22, "sgma": "None",
                    "dip_payment_per_acre": 600, "qsa_value_per_acft": 650,
                    "incentive_per_acre": 600},
}

COUNTY_DEFICIT_PARAMS = {
    "San Joaquin": {"et_full_mmd": 5.0, "deficit_factor": 0.35, "deficit_days": 45},
    "Stanislaus":  {"et_full_mmd": 5.0, "deficit_factor": 0.35, "deficit_days": 45},
    "Merced":      {"et_full_mmd": 5.2, "deficit_factor": 0.34, "deficit_days": 48},
    "Madera":      {"et_full_mmd": 5.3, "deficit_factor": 0.33, "deficit_days": 48},
    "Fresno":      {"et_full_mmd": 5.5, "deficit_factor": 0.33, "deficit_days": 50},
    "Tulare":      {"et_full_mmd": 5.5, "deficit_factor": 0.33, "deficit_days": 50},
    "Kings":       {"et_full_mmd": 5.5, "deficit_factor": 0.32, "deficit_days": 52},
    "Kern":        {"et_full_mmd": 6.0, "deficit_factor": 0.31, "deficit_days": 55},
    "Riverside":   {"et_full_mmd": 6.5, "deficit_factor": 0.30, "deficit_days": 60},
    "Imperial":    {"et_full_mmd": 7.0, "deficit_factor": 0.29, "deficit_days": 70},
}

COUNTY_PER_CUT_ET_MM = {
    "San Joaquin": 174, "Stanislaus": 164, "Merced": 161,
    "Madera": 176, "Fresno": 173, "Tulare": 167,
    "Kings": 172, "Kern": 159, "Riverside": 176, "Imperial": 175,
}

LITERATURE_ANNUAL_CUTS = {
    "San Joaquin": 8, "Stanislaus": 8, "Merced": 8, "Madera": 8,
    "Fresno": 9, "Tulare": 9, "Kings": 9, "Kern": 10,
    "Riverside": 12, "Imperial": 12,
}

COUNTY_ALFALFA_ACRES = {
    "San Joaquin": 80000, "Stanislaus": 45000, "Merced": 50000,
    "Madera": 20000, "Fresno": 60000, "Tulare": 45000,
    "Kings": 35000, "Kern": 55000, "Riverside": 30000, "Imperial": 120000,
}

# Hay market parameters
LATE_CUT_YIELD_TONS = 0.9       # tons/acre per late cut
LATE_CUT_HAY_PRICE = 180        # $/ton for late-season fair quality
DEFICIT_CUT_YIELD_TONS = 0.4    # tons/acre during deficit
DEFICIT_CUT_HAY_PRICE = 130     # $/ton for stub quality
MM_PER_FOOT = 304.8             # mm to ac-ft/acre conversion


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CMAP_BLUE_GREEN = LinearSegmentedColormap.from_list(
    "BlueGreen",
    ["#E0F7FA", "#80CBC4", "#388E3C", "#1B5E20"],
    N=256,
)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------
def compute_strategy(
    county: str,
    strategy: str,
    n_late: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute water_saved (ac-ft/acre) and revenue_lost ($/acre) for a
    given county and strategy.

    Parameters
    ----------
    county : str
        County name (must be in COUNTIES_NS).
    strategy : str
        Strategy code (S0 through S7).
    n_late : int, optional
        Number of late cuts.  If None, estimated as ~33% of annual cuts.

    Returns
    -------
    water_saved : float
        Water savings in ac-ft/acre.
    revenue_lost : float
        Revenue lost in $/acre.
    """
    e_c = COUNTY_PER_CUT_ET_MM[county] / MM_PER_FOOT  # ac-ft/acre per cut
    n_cuts = LITERATURE_ANNUAL_CUTS[county]
    if n_late is None:
        n_late = round(n_cuts * 0.33)
    dp = COUNTY_DEFICIT_PARAMS[county]
    deficit_savings_mm = dp["deficit_days"] * dp["et_full_mmd"] * (1 - dp["deficit_factor"])
    deficit_savings_acft = deficit_savings_mm / MM_PER_FOOT

    # Revenue per fair-quality late cut (full revenue when harvested as planned).
    late_rev = LATE_CUT_YIELD_TONS * LATE_CUT_HAY_PRICE          # 0.9 t × $180 = $162/cut
    # Revenue per deficit-affected ("stub") cut — reduced yield + lower price.
    deficit_cut_rev = DEFICIT_CUT_YIELD_TONS * DEFICIT_CUT_HAY_PRICE  # 0.4 t × $130 = $52/cut
    # Revenue gap when a fair cut is downgraded to stub quality (used for S3-S6).
    deficit_cut_rev_loss = late_rev - deficit_cut_rev             # $110/cut downgraded

    if strategy == "S0":
        return 0.0, 0.0
    elif strategy == "S1":
        ws = max(0, n_late - 1) * e_c
        rl = max(0, n_late - 1) * late_rev
        return ws, rl
    elif strategy == "S2":
        ws = n_late * e_c
        rl = n_late * late_rev
        return ws, rl
    elif strategy == "S3":
        # Deficit only: ~1.5 cuts downgraded fair → stub
        ws = deficit_savings_acft
        rl = 1.5 * deficit_cut_rev_loss
        return ws, rl
    elif strategy == "S4":
        # Selective deficit: 30% of the fair→stub gap on 2 cuts
        ws = 2 * e_c * (1 - dp["deficit_factor"])
        rl = 2 * deficit_cut_rev_loss * 0.3
        return ws, rl
    elif strategy == "S5":
        # Hybrid def+cap1: 1.5 stub-downgraded cuts + (n_late-2) fully lost
        ws = deficit_savings_acft + max(0, n_late - 1) * e_c * dp["deficit_factor"]
        rl = 1.5 * deficit_cut_rev_loss + max(0, n_late - 2) * late_rev
        return ws, rl
    elif strategy == "S6":
        # Hybrid def+cap0: 1.5 stub-downgraded cuts + (n_late-1) fully lost
        ws = deficit_savings_acft + n_late * e_c * dp["deficit_factor"]
        rl = 1.5 * deficit_cut_rev_loss + max(0, n_late - 1) * late_rev
        return ws, rl
    elif strategy == "S7":
        ws = 60 * dp["et_full_mmd"] / MM_PER_FOOT
        rl = 2.5 * late_rev
        return ws, rl
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _group_average_params(counties: List[str]) -> Dict:
    """Average county parameters for a group of counties."""
    n = len(counties)
    avg_per_cut = sum(COUNTY_PER_CUT_ET_MM[c] for c in counties) / n
    avg_n_cuts = sum(LITERATURE_ANNUAL_CUTS[c] for c in counties) / n
    avg_et_full = sum(COUNTY_DEFICIT_PARAMS[c]["et_full_mmd"] for c in counties) / n
    avg_deficit_factor = sum(COUNTY_DEFICIT_PARAMS[c]["deficit_factor"] for c in counties) / n
    avg_deficit_days = sum(COUNTY_DEFICIT_PARAMS[c]["deficit_days"] for c in counties) / n
    avg_price_normal = sum(COUNTY_WATER_PARAMS[c]["price_normal"] for c in counties) / n
    avg_price_drought = sum(COUNTY_WATER_PARAMS[c]["price_drought"] for c in counties) / n
    avg_incentive = sum(
        COUNTY_WATER_PARAMS[c].get("incentive_per_acre", 0) for c in counties
    ) / n
    return {
        "per_cut_mm": avg_per_cut,
        "n_cuts": avg_n_cuts,
        "et_full_mmd": avg_et_full,
        "deficit_factor": avg_deficit_factor,
        "deficit_days": avg_deficit_days,
        "price_normal": avg_price_normal,
        "price_drought": avg_price_drought,
        "incentive_per_acre": avg_incentive,
    }


def _compute_strategy_from_params(
    per_cut_mm: float,
    n_cuts: float,
    et_full_mmd: float,
    deficit_factor: float,
    deficit_days: float,
    strategy: str,
) -> Tuple[float, float]:
    """Compute strategy savings/loss from raw parameters (for group averages)."""
    e_c = per_cut_mm / MM_PER_FOOT
    n_late = round(n_cuts * 0.33)
    deficit_savings_mm = deficit_days * et_full_mmd * (1 - deficit_factor)
    deficit_savings_acft = deficit_savings_mm / MM_PER_FOOT

    late_rev = LATE_CUT_YIELD_TONS * LATE_CUT_HAY_PRICE          # $162/cut fair quality
    deficit_cut_rev = DEFICIT_CUT_YIELD_TONS * DEFICIT_CUT_HAY_PRICE  # $52/cut stub quality
    deficit_cut_rev_loss = late_rev - deficit_cut_rev             # $110/cut downgraded

    if strategy == "S0":
        return 0.0, 0.0
    elif strategy == "S1":
        ws = max(0, n_late - 1) * e_c
        rl = max(0, n_late - 1) * late_rev
        return ws, rl
    elif strategy == "S2":
        ws = n_late * e_c
        rl = n_late * late_rev
        return ws, rl
    elif strategy == "S3":
        ws = deficit_savings_acft
        rl = 1.5 * deficit_cut_rev_loss
        return ws, rl
    elif strategy == "S4":
        ws = 2 * e_c * (1 - deficit_factor)
        rl = 2 * deficit_cut_rev_loss * 0.3
        return ws, rl
    elif strategy == "S5":
        ws = deficit_savings_acft + max(0, n_late - 1) * e_c * deficit_factor
        rl = 1.5 * deficit_cut_rev_loss + max(0, n_late - 2) * late_rev
        return ws, rl
    elif strategy == "S6":
        ws = deficit_savings_acft + n_late * e_c * deficit_factor
        rl = 1.5 * deficit_cut_rev_loss + max(0, n_late - 1) * late_rev
        return ws, rl
    elif strategy == "S7":
        ws = 60 * et_full_mmd / MM_PER_FOOT
        rl = 2.5 * late_rev
        return ws, rl
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def daily_et_profile(et_full_mmd: float, day_of_wy: np.ndarray) -> np.ndarray:
    """Sinusoidal ET model: peaks in June-July.

    Parameters
    ----------
    et_full_mmd : float
        Peak daily ET (mm/d) for the county.
    day_of_wy : array-like
        Day of water year (0 = Oct 1, 365 = Sep 30).

    Returns
    -------
    et : ndarray
        Daily ET (mm/d).
    """
    day_of_wy = np.asarray(day_of_wy, dtype=float)
    et_min = et_full_mmd * 0.15
    et = et_min + (et_full_mmd - et_min) * np.maximum(
        0, np.sin(np.pi * (day_of_wy - 60) / 300)
    )
    return et


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------
def build_crossover_data(
    price_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build crossover data: NEB vs water price for each group/strategy."""
    if price_range is None:
        price_range = np.linspace(0, 1200, 241)

    rows = []
    for group_name, counties in COUNTY_GROUPS.items():
        params = _group_average_params(counties)
        incentive = params["incentive_per_acre"]
        for strat in STRATEGIES:
            ws, rl = _compute_strategy_from_params(
                params["per_cut_mm"], params["n_cuts"],
                params["et_full_mmd"], params["deficit_factor"],
                params["deficit_days"], strat,
            )
            inc_amt = incentive if strat != "S0" else 0.0
            for price in price_range:
                neb = ws * price - rl
                neb_inc = neb + inc_amt
                rows.append({
                    "group": group_name,
                    "strategy": strat,
                    "water_price": price,
                    "water_saved_acft": ws,
                    "revenue_lost": rl,
                    "neb": neb,
                    "neb_incentive": neb_inc,
                    "incentive_per_acre": inc_amt,
                })
    return pd.DataFrame(rows)


def build_strategy_heatmap_data() -> pd.DataFrame:
    """Build heatmap data: NEB for each county x strategy x price scenario."""
    rows = []
    for county in COUNTIES_NS:
        for strat in STRATEGIES:
            ws, rl = compute_strategy(county, strat)
            wp = COUNTY_WATER_PARAMS[county]
            p_norm = wp["price_normal"]
            p_drought = wp["price_drought"]
            incentive = wp.get("incentive_per_acre", 0)
            neb_normal = ws * p_norm - rl
            neb_drought = ws * p_drought - rl
            # With incentive: grower receives payment for participating
            # (DIP in Imperial, SGMA/conservation programs in SJV)
            # Incentive only applies to strategies that save water (not S0)
            neb_incentive = neb_normal + (incentive if strat != "S0" else 0)
            rows.append({
                "county": county,
                "strategy": strat,
                "water_saved_acft": ws,
                "revenue_lost": rl,
                "neb_normal": neb_normal,
                "neb_drought": neb_drought,
                "neb_incentive": neb_incentive,
            })
    return pd.DataFrame(rows)


def build_decomposition_data() -> pd.DataFrame:
    """Build savings decomposition: cutting, deficit, and interaction terms."""
    rows = []
    for county in COUNTIES_NS:
        ws_cutting, _ = compute_strategy(county, "S2")   # Cap 0 only
        ws_deficit, _ = compute_strategy(county, "S3")    # Deficit only
        ws_combined, _ = compute_strategy(county, "S6")   # Hybrid cap0 + deficit
        # Interaction: combined minus simple sum (negative because deficit
        # reduces per-cut ET, so there is some double-counting)
        ws_interaction = ws_combined - (ws_cutting + ws_deficit)
        # Clamp interaction to be non-negative for stacking
        ws_interaction = max(0.0, ws_interaction)
        # If combined < sum, reduce cutting and deficit proportionally
        if ws_combined < (ws_cutting + ws_deficit):
            scale = ws_combined / max(ws_cutting + ws_deficit, 1e-9)
            ws_cutting_adj = ws_cutting * scale
            ws_deficit_adj = ws_deficit * scale
            ws_interaction = 0.0
        else:
            ws_cutting_adj = ws_cutting
            ws_deficit_adj = ws_deficit

        rows.append({
            "county": county,
            "cutting_savings_acft": ws_cutting_adj,
            "deficit_savings_acft": ws_deficit_adj,
            "interaction_savings_acft": ws_interaction,
            "total_savings_acft": ws_combined,
        })
    return pd.DataFrame(rows)


def build_seasonal_profile_data() -> pd.DataFrame:
    """Build seasonal ET profiles for representative counties under 4 strategies."""
    days = np.arange(366)
    rows = []
    for group_name, counties in COUNTY_GROUPS.items():
        params = _group_average_params(counties)
        et_full = params["et_full_mmd"]
        deficit_factor = params["deficit_factor"]
        deficit_days = params["deficit_days"]

        # S0: full ET
        et_s0 = daily_et_profile(et_full, days)

        # S2: Cap 0 -- remove late-cut cycles after July 1
        # July 1 in WY = day 274 (Oct 1 = day 0, so Jan 1 = day 92,
        # Jul 1 = day 274)
        jul1_doy = 274
        # After Jul 1, ET drops because we remove cutting-cycle irrigation
        # Approximate: ET goes to maintenance level (~40% of full)
        et_s2 = et_s0.copy()
        et_s2[days >= jul1_doy] = et_s0[days >= jul1_doy] * 0.40

        # S3: Deficit only -- ET drops during deficit window
        # Deficit starts ~mid-July, lasts deficit_days
        deficit_start = jul1_doy + 14  # ~Jul 15 in WY
        deficit_end = deficit_start + int(deficit_days)
        et_s3 = et_s0.copy()
        mask_deficit = (days >= deficit_start) & (days < deficit_end)
        et_s3[mask_deficit] = et_s0[mask_deficit] * deficit_factor

        # S6: Hybrid deficit + cap 0 -- maximum reduction
        et_s6 = et_s0.copy()
        et_s6[days >= jul1_doy] = et_s0[days >= jul1_doy] * 0.40
        et_s6[mask_deficit] = et_s0[mask_deficit] * deficit_factor * 0.6

        for d in range(366):
            rows.append({
                "group": group_name,
                "day_of_wy": d,
                "et_s0": et_s0[d],
                "et_s2": et_s2[d],
                "et_s3": et_s3[d],
                "et_s6": et_s6[d],
            })
    return pd.DataFrame(rows)


def build_basin_impact_data() -> pd.DataFrame:
    """Build basin-scale impact: total water savings by strategy and county."""
    rows = []
    for county in COUNTIES_NS:
        acres = COUNTY_ALFALFA_ACRES[county]
        for strat in STRATEGIES:
            ws, _ = compute_strategy(county, strat)
            total_af = ws * acres
            # Determine group
            for grp_name, grp_counties in COUNTY_GROUPS.items():
                if county in grp_counties:
                    group = grp_name
                    break
            rows.append({
                "county": county,
                "group": group,
                "strategy": strat,
                "water_saved_per_acre_acft": ws,
                "acreage": acres,
                "total_savings_acft": total_af,
                "total_savings_kaf": total_af / 1000.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 5: Economic crossover curves
# ---------------------------------------------------------------------------
def economic_crossover_figure(
    out_dir: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Economic crossover: NEB vs water price for 3 regional groups.

    Two-row layout:
      Row 1 (a-c): market pricing only
      Row 2 (d-f): with DIP / SGMA conservation incentive payment
    """
    apply_style()
    data = build_crossover_data()

    group_names = list(COUNTY_GROUPS.keys())
    fig = plt.figure(figsize=(18, 12.6))
    gs = GridSpec(
        2, 3,
        wspace=0.25, hspace=0.38,
        left=0.06, right=0.98, top=0.91, bottom=0.11,
    )
    axes = np.empty((2, 3), dtype=object)

    row_specs = [
        ("neb", "Market pricing only", False),
        ("neb_incentive", None, True),  # row label is group-specific (DIP vs SGMA)
    ]

    # First pass: compute symmetric y-limits per group across both rows
    group_ylims: Dict[str, Tuple[float, float]] = {}
    for group_name in group_names:
        gdf = data[data["group"] == group_name]
        all_y = np.concatenate([gdf["neb"].values, gdf["neb_incentive"].values])
        ymin = float(np.nanmin(all_y))
        ymax = float(np.nanmax(all_y))
        pad = 0.08 * (ymax - ymin)
        group_ylims[group_name] = (ymin - pad, ymax + pad)

    panel_letter = 0
    for ri, (neb_col, row_label, with_incentive) in enumerate(row_specs):
        for gi, group_name in enumerate(group_names):
            ax = fig.add_subplot(gs[ri, gi])
            axes[ri, gi] = ax
            gdf = data[data["group"] == group_name]

            counties = COUNTY_GROUPS[group_name]
            params = _group_average_params(counties)

            for strat in STRATEGIES:
                sdf = gdf[gdf["strategy"] == strat]
                lw = 2.4 if strat in ("S2", "S6") else 1.5
                ax.plot(
                    sdf["water_price"], sdf[neb_col],
                    color=STRATEGY_COLORS[strat],
                    linestyle=STRATEGY_LINESTYLES[strat],
                    linewidth=lw,
                    label=STRATEGY_LABELS[strat],
                    zorder=3 if strat in ("S2", "S6") else 2,
                )

            ax.set_ylim(*group_ylims[group_name])

            ax.axhline(0, color="#444444", linewidth=0.9, linestyle="-", zorder=1)

            ymin_ax, ymax_ax = ax.get_ylim()
            ax.axhspan(0, ymax_ax, color="#FFF3E0", alpha=0.45, zorder=0)
            ax.axhspan(ymin_ax, 0, color="#ECEFF1", alpha=0.55, zorder=0)

            ax.axvline(
                params["price_normal"], color="#1565C0", linewidth=1.4,
                linestyle="--", alpha=0.85, zorder=1,
            )
            ax.axvline(
                params["price_drought"], color="#C62828", linewidth=1.4,
                linestyle="--", alpha=0.85, zorder=1,
            )

            top = ymax_ax - 0.06 * (ymax_ax - ymin_ax)
            ax.text(
                params["price_normal"] + 18, top, "Normal",
                fontsize=9, color="#1565C0", va="top", ha="left",
                fontweight="bold",
            )
            ax.text(
                params["price_drought"] + 18, top, "Drought",
                fontsize=9, color="#C62828", va="top", ha="left",
                fontweight="bold",
            )

            # Featured-county individual price markers (dotted, distinct
            # accent color) so within-group price spread is visible without
            # disturbing the existing group-average analysis.
            featured = FEATURED_COUNTY[group_name]
            fp = COUNTY_WATER_PARAMS[featured]
            fp_norm = fp["price_normal"]
            fp_drought = fp["price_drought"]
            featured_color = "#6A1B9A"  # purple, distinct from blue/red avg lines

            # Mid-y for vertical-rotated labels; small horizontal offset
            # places the text just to the left of the dotted line so the
            # line itself remains visible and labels read bottom-to-top.
            y_mid = (ymin_ax + ymax_ax) / 2.0
            x_offset = -16  # ~1.3% of the 0–1200 price axis, to the left
            label_bbox = dict(
                boxstyle="round,pad=0.18", facecolor="white",
                edgecolor=featured_color, linewidth=0.5, alpha=0.82,
            )

            def _draw_featured_line(x_at: float, label: str) -> None:
                ax.axvline(
                    x_at, color=featured_color, linewidth=1.5,
                    linestyle=":", alpha=0.95, zorder=2,
                )
                # If the line is too close to the left edge (e.g. Imperial
                # at $22), place the label to the right of the line instead.
                tx = x_at + (x_offset if x_at > 80 else -x_offset)
                ax.text(
                    tx, y_mid, label,
                    fontsize=8.5, color=featured_color,
                    ha="center", va="center",
                    rotation=90, rotation_mode="anchor",
                    fontweight="bold",
                    bbox=label_bbox, zorder=4,
                )

            if fp_drought == fp_norm:
                _draw_featured_line(
                    fp_norm, f"{featured}: ${fp_norm:.0f} (QSA-fixed)",
                )
            else:
                _draw_featured_line(
                    fp_norm, f"{featured} Normal: ${fp_norm:.0f}",
                )
                _draw_featured_line(
                    fp_drought, f"{featured} Drought: ${fp_drought:.0f}",
                )

            if with_incentive:
                inc = params["incentive_per_acre"]
                ax.text(
                    0.985, 0.05,
                    f"+ ${inc:.0f}/ac incentive",
                    transform=ax.transAxes,
                    fontsize=9.5, color="#BF360C", ha="right", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E1",
                              edgecolor="#FFB300", linewidth=0.8, alpha=0.95),
                )

            ax.set_xlabel("Water price ($/ac-ft)", fontsize=11)
            if gi == 0:
                ax.set_ylabel("Net economic benefit ($/acre)", fontsize=11)
            panel_tag = chr(ord('a') + panel_letter)
            panel_letter += 1
            n_counties_in_group = len(counties)
            this_row_label = (
                row_label
                if row_label is not None
                else f"+ {GROUP_PROGRAM_LABEL[group_name]}"
            )
            ax.set_title(
                f"({panel_tag}) {group_name} (n={n_counties_in_group}) — {this_row_label}",
                fontsize=11.5, fontweight="bold", pad=8,
            )
            ax.tick_params(axis="both", labelsize=9, length=3.5, width=0.7)
            _style_ax(ax)

            if ri == 0 and gi == 2:
                leg = ax.legend(
                    fontsize=8.5, loc="upper right",
                    frameon=True, framealpha=0.45,
                    edgecolor="#bbbbbb", fancybox=False,
                    handlelength=2.6, borderpad=0.5,
                    labelspacing=0.45,
                )
                leg.get_frame().set_facecolor("#FFFFFF")
                leg.get_frame().set_alpha(0.45)

    fig.suptitle(
        "Economic Crossover: When Does Saving Water Beat Hay Revenue?",
        fontsize=15, y=0.975, fontweight="bold",
    )

    # Region-composition footnote + program-source caveat
    region_lines = [
        f"{name} (n={len(cs)}): {', '.join(cs)}"
        for name, cs in COUNTY_GROUPS.items()
    ]
    footnote = (
        "Regions: " + "  |  ".join(region_lines) + "\n"
        "Incentive: Imperial = IID Deficit Irrigation Program / QSA "
        "(authoritative, ~$600/ac).  Riverside = PVID-MWD forbearance + "
        "Coachella SGMA (illustrative).  SJV counties = SGMA-driven programs "
        "(DWR LandFlex, MLRP) — illustrative estimates, not official rates."
    )
    fig.text(
        0.5, 0.012, footnote,
        ha="center", va="bottom", fontsize=8.5, color="#333333",
    )

    # Break-even prices summary (both market-only and incentive-shifted)
    summary = {
        "chart": "economic_crossover",
        "breakeven_prices": {},
        "breakeven_prices_with_incentive": {},
    }
    for group_name in group_names:
        gdf = data[data["group"] == group_name]
        be = {}
        be_inc = {}
        for strat in STRATEGIES:
            if strat == "S0":
                continue
            sdf = gdf[gdf["strategy"] == strat].sort_values("water_price")
            prices = sdf["water_price"].values
            for col, target in (("neb", be), ("neb_incentive", be_inc)):
                nebs = sdf[col].values
                cross_idx = np.where(np.diff(np.sign(nebs)))[0]
                if len(cross_idx) > 0:
                    i = cross_idx[0]
                    p0, p1 = prices[i], prices[i + 1]
                    n0, n1 = nebs[i], nebs[i + 1]
                    be_price = p0 - n0 * (p1 - p0) / (n1 - n0) if (n1 - n0) != 0 else p0
                    target[strat] = round(float(be_price), 1)
                else:
                    target[strat] = None
        summary["breakeven_prices"][group_name] = be
        summary["breakeven_prices_with_incentive"][group_name] = be_inc

    if out_dir is not None:
        save_pub_figure(fig, "economic_crossover", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# Figure 6: Strategy comparison heatmap
# ---------------------------------------------------------------------------
def strategy_heatmap_figure(
    out_dir: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Strategy comparison heatmap: county x strategy, normal vs drought."""
    apply_style()
    data = build_strategy_heatmap_data()

    fig = plt.figure(figsize=(16, 14.6))
    gs = GridSpec(
        3, 2,
        width_ratios=[1, 0.03],
        height_ratios=[1, 1, 1],
        wspace=0.05, hspace=0.35,
        left=0.10, right=0.92, top=0.93, bottom=0.10,
    )
    axes = np.empty(3, dtype=object)

    scenarios = [
        ("neb_normal", "Normal Water Year Pricing"),
        ("neb_drought", "Drought Water Year Pricing"),
        ("neb_incentive",
         "With Conservation Incentive  (Imperial: DIP / QSA  ·  SJV + Riverside: SGMA programs)"),
    ]
    county_labels = [c.replace("San Joaquin", "S. Joaquin") for c in COUNTIES_NS]
    strat_labels = [STRATEGY_LABELS[s] for s in STRATEGIES]

    # Compute global color range across all scenarios
    all_neb = np.concatenate([
        data["neb_normal"].values, data["neb_drought"].values,
        data["neb_incentive"].values,
    ])
    vabs = max(abs(np.nanmin(all_neb)), abs(np.nanmax(all_neb)))
    # Diverging colormap with reversed semantics:
    # negative (loss)    → deep red → orange → yellow → cream
    # zero               → cream / white
    # positive (profit)  → light blue → mid blue → dark navy
    cmap = LinearSegmentedColormap.from_list(
        "OrangeYellowCreamBlueNavy",
        ["#BF360C", "#FB8C00", "#FFC107", "#FFF176", "#FFFDE7",
         "#E3F2FD", "#64B5F6", "#1976D2", "#0D47A1"],
        N=256,
    )

    for ri, (neb_col, title_text) in enumerate(scenarios):
        ax = fig.add_subplot(gs[ri, 0])
        axes[ri] = ax

        # Build 2D array: counties (rows) x strategies (cols)
        n_counties = len(COUNTIES_NS)
        n_strats = len(STRATEGIES)
        grid = np.full((n_counties, n_strats), np.nan)
        for _, row in data.iterrows():
            ci = COUNTIES_NS.index(row["county"])
            si = STRATEGIES.index(row["strategy"])
            grid[ci, si] = row[neb_col]

        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(
            grid, cmap=cmap, aspect="auto", norm=norm,
            interpolation="nearest",
        )

        # Grid lines
        for x in range(n_strats + 1):
            ax.axvline(x - 0.5, color="white", linewidth=0.6)
        for y in range(n_counties + 1):
            ax.axhline(y - 0.5, color="white", linewidth=0.6)

        # Cell text
        for i in range(n_counties):
            for j in range(n_strats):
                v = grid[i, j]
                if not np.isfinite(v):
                    continue
                norm_v = norm(v)
                tc = "white" if (norm_v < 0.18 or norm_v > 0.86) else "#1A1A1A"
                ax.text(
                    j, i, f"${v:.0f}",
                    ha="center", va="center", fontsize=7.5, color=tc,
                    fontweight="bold" if abs(v) > 300 else "normal",
                )

        ax.set_yticks(range(n_counties))
        ax.set_yticklabels(county_labels, fontsize=9.5)
        ax.set_xticks(range(n_strats))
        ax.set_xticklabels(strat_labels, fontsize=8.5, rotation=35, ha="right")
        ax.set_ylabel("County (N to S)", fontsize=10.5, labelpad=5)
        ax.set_title(title_text, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(axis="both", length=3.5, width=0.7, pad=3)
        _style_ax(ax)
        add_panel_label(ax, chr(ord("a") + ri), x=-0.08, y=1.06)

        # Colorbar
        cbar_ax = fig.add_subplot(gs[ri, 1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Net economic benefit ($/acre)", fontsize=9.5)
        cbar.ax.tick_params(labelsize=8.5, length=2.5, width=0.6)
        cbar.outline.set_linewidth(0.6)

    fig.suptitle(
        "Strategy Comparison: Net Economic Benefit by County",
        fontsize=14.5, y=0.975, fontweight="bold",
    )

    # Region-composition footnote + program-source caveat
    region_lines = [
        f"{name} (n={len(cs)}): {', '.join(cs)}"
        for name, cs in COUNTY_GROUPS.items()
    ]
    footnote = (
        "Regions: " + "  |  ".join(region_lines) + "\n"
        "Incentive: Imperial = IID Deficit Irrigation Program / QSA "
        "(authoritative, ~$600/ac).  Riverside = PVID-MWD forbearance + "
        "Coachella SGMA (illustrative).  SJV counties = SGMA-driven programs "
        "(DWR LandFlex, MLRP) — illustrative estimates, not official rates."
    )
    fig.text(
        0.5, 0.012, footnote,
        ha="center", va="bottom", fontsize=8.5, color="#333333",
    )

    # Build summary with optimal strategies per county
    summary = {"chart": "strategy_heatmap", "optimal": {}}
    for county in COUNTIES_NS:
        cdata = data[data["county"] == county]
        best_norm = cdata.loc[cdata["neb_normal"].idxmax(), "strategy"]
        best_drought = cdata.loc[cdata["neb_drought"].idxmax(), "strategy"]
        summary["optimal"][county] = {
            "normal": best_norm, "drought": best_drought,
        }

    if out_dir is not None:
        save_pub_figure(fig, "strategy_heatmap", out_dir)
        csv_path = Path(out_dir) / "strategy_heatmap_data.csv"
        data.to_csv(csv_path, index=False)
        print(f"  Heatmap CSV: {csv_path}")
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# Figure 7: Savings decomposition
# ---------------------------------------------------------------------------
def savings_decomposition_figure(
    out_dir: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Stacked bar chart of water savings decomposition by county."""
    apply_style()
    data = build_decomposition_data()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(COUNTIES_NS))
    bar_width = 0.65

    color_cutting = "#0072B2"   # blue
    color_deficit = "#009E73"   # teal
    color_interaction = "#56B4E9"  # light blue / green

    cutting = data["cutting_savings_acft"].values
    deficit = data["deficit_savings_acft"].values
    interaction = data["interaction_savings_acft"].values

    bars_cutting = ax.bar(
        x, cutting, bar_width,
        label="Cutting management (Cap 0)", color=color_cutting, edgecolor="white",
        linewidth=0.5, zorder=3,
    )
    bars_deficit = ax.bar(
        x, deficit, bar_width, bottom=cutting,
        label="Deficit irrigation (Montazar)", color=color_deficit, edgecolor="white",
        linewidth=0.5, zorder=3,
    )
    bars_interaction = ax.bar(
        x, interaction, bar_width, bottom=cutting + deficit,
        label="Combined interaction", color=color_interaction, edgecolor="white",
        linewidth=0.5, zorder=3,
    )

    # Total labels on top
    totals = data["total_savings_acft"].values
    for i, total in enumerate(totals):
        ax.text(
            i, total + 0.02, f"{total:.2f}",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
        )

    county_labels = [c.replace("San Joaquin", "S. Joaquin") for c in COUNTIES_NS]
    ax.set_xticks(x)
    ax.set_xticklabels(county_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Water savings (ac-ft/acre)", fontsize=8)
    ax.set_xlabel("County (N to S)", fontsize=8)
    ax.set_title(
        "Water Savings Decomposition by County (N to S)",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.legend(
        fontsize=7, loc="upper left",
        frameon=True, framealpha=0.9,
        edgecolor="#cccccc", fancybox=False,
    )
    ax.set_ylim(0, max(totals) * 1.15)
    ax.tick_params(axis="both", labelsize=7, length=2, width=0.5)
    _style_ax(ax)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)

    summary = {
        "chart": "savings_decomposition",
        "counties": COUNTIES_NS,
        "totals_acft": {c: round(t, 3) for c, t in zip(COUNTIES_NS, totals)},
        "max_total_acft": round(float(max(totals)), 3),
    }

    if out_dir is not None:
        save_pub_figure(fig, "savings_decomposition", out_dir)
        csv_path = Path(out_dir) / "savings_decomposition_data.csv"
        data.to_csv(csv_path, index=False)
        print(f"  Decomposition CSV: {csv_path}")
        summary["out_dir"] = str(out_dir)

    return fig, ax, summary


# ---------------------------------------------------------------------------
# Figure 8: Seasonal ET profile
# ---------------------------------------------------------------------------
def _wy_day_to_month_ticks():
    """Return tick positions and labels for WY months (Oct through Sep)."""
    # Day 0 = Oct 1. Approximate month starts:
    # Oct=0, Nov=31, Dec=61, Jan=92, Feb=122, Mar=150, Apr=181, May=211,
    # Jun=242, Jul=272, Aug=303, Sep=334
    positions = [0, 31, 61, 92, 122, 150, 181, 211, 242, 272, 303, 334]
    labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
              "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    return positions, labels


def seasonal_et_profile_figure(
    out_dir: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray, Dict]:
    """Seasonal ET profiles under 4 strategies for 3 regional groups."""
    apply_style()
    data = build_seasonal_profile_data()

    group_names = list(COUNTY_GROUPS.keys())
    fig = plt.figure(figsize=(17, 5.5))
    gs = GridSpec(
        1, 3,
        wspace=0.22,
        left=0.06, right=0.98, top=0.86, bottom=0.14,
    )
    axes = np.empty(3, dtype=object)

    tick_pos, tick_labels = _wy_day_to_month_ticks()
    # July 1 in WY
    jul1_doy = 274

    strat_plot_cfg = [
        ("et_s0", "S0: Full ET", "#000000", "-", 1.2),
        ("et_s2", "S2: Cap 0", "#0072B2", "--", 1.0),
        ("et_s3", "S3: Deficit only", "#009E73", "-.", 1.0),
        ("et_s6", "S6: Hybrid max", "#CC79A7", "-", 1.0),
    ]

    for gi, group_name in enumerate(group_names):
        ax = fig.add_subplot(gs[0, gi])
        axes[gi] = ax
        gdf = data[data["group"] == group_name].sort_values("day_of_wy")
        days = gdf["day_of_wy"].values

        # Plot each strategy
        for col, label, color, ls, lw in strat_plot_cfg:
            vals = gdf[col].values
            ax.plot(days, vals, color=color, linestyle=ls, linewidth=lw,
                    label=label, zorder=3)

        # Shade savings areas
        et_s0 = gdf["et_s0"].values
        et_s2 = gdf["et_s2"].values
        et_s6 = gdf["et_s6"].values

        ax.fill_between(
            days, et_s2, et_s0,
            where=et_s0 > et_s2,
            color="#0072B2", alpha=0.12, zorder=1,
        )
        ax.fill_between(
            days, et_s6, et_s2,
            where=et_s2 > et_s6,
            color="#CC79A7", alpha=0.12, zorder=1,
        )

        # Key date markers
        ax.axvline(jul1_doy, color="#888888", linewidth=0.6, linestyle=":",
                    zorder=1)
        ax.text(jul1_doy + 2, ax.get_ylim()[1] * 0.95, "Jul 1",
                fontsize=5, color="#888888", va="top")

        # Compute savings annotations
        params = _group_average_params(COUNTY_GROUPS[group_name])
        savings_s2 = sum(et_s0 - et_s2) / MM_PER_FOOT
        savings_s6 = sum(et_s0 - et_s6) / MM_PER_FOOT
        ax.text(
            0.97, 0.97,
            f"S2 saves {savings_s2:.2f} ac-ft/ac\n"
            f"S6 saves {savings_s6:.2f} ac-ft/ac",
            transform=ax.transAxes,
            fontsize=5.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=45, ha="right")
        ax.set_xlabel("Month", fontsize=7.5)
        if gi == 0:
            ax.set_ylabel("Daily ET (mm/d)", fontsize=7.5)
        ax.set_title(f"({chr(ord('a') + gi)}) {group_name}", fontsize=9,
                      fontweight="bold", pad=6)
        ax.set_xlim(0, 365)
        ax.set_ylim(0, None)
        ax.tick_params(axis="both", labelsize=6.5, length=2, width=0.5)
        _style_ax(ax)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)

        if gi == 0:
            ax.legend(
                fontsize=5.5, loc="upper left",
                frameon=True, framealpha=0.9,
                edgecolor="#cccccc", fancybox=False,
            )

    fig.suptitle(
        "Seasonal ET Profile Under Alternative Strategies",
        fontsize=10, y=0.96,
    )

    summary = {
        "chart": "seasonal_et_profile",
        "groups": group_names,
    }

    if out_dir is not None:
        save_pub_figure(fig, "seasonal_et_profile", out_dir)
        summary["out_dir"] = str(out_dir)

    return fig, axes, summary


# ---------------------------------------------------------------------------
# Figure 9: Basin-scale impact
# ---------------------------------------------------------------------------
def basin_impact_figure(
    out_dir: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Basin-scale water savings: stacked bars by strategy, colored by county."""
    apply_style()
    data = build_basin_impact_data()

    fig, ax = plt.subplots(figsize=(14, 7.5))
    x = np.arange(len(STRATEGIES))
    bar_width = 0.7

    # Color palette for counties
    county_colors = {
        "San Joaquin": "#1f77b4",
        "Stanislaus":  "#aec7e8",
        "Merced":      "#2ca02c",
        "Madera":      "#98df8a",
        "Fresno":      "#ff7f0e",
        "Tulare":      "#ffbb78",
        "Kings":       "#d62728",
        "Kern":        "#ff9896",
        "Riverside":   "#9467bd",
        "Imperial":    "#c5b0d5",
    }

    # Build stacked bar data
    bottom = np.zeros(len(STRATEGIES))
    for county in COUNTIES_NS:
        cdata = data[data["county"] == county].set_index("strategy")
        vals = np.array([cdata.loc[s, "total_savings_kaf"] for s in STRATEGIES])
        ax.bar(
            x, vals, bar_width, bottom=bottom,
            label=county, color=county_colors[county],
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        bottom += vals

    # Reference lines
    # IID QSA transfer obligation
    ax.axhline(
        200, color="#D32F2F", linewidth=0.8, linestyle="--", zorder=2,
    )
    ax.text(
        len(STRATEGIES) - 0.5, 205, "IID QSA transfer (200 kaf)",
        fontsize=5.5, color="#D32F2F", ha="right", va="bottom",
    )
    # Lake Mead annual evaporation
    ax.axhline(
        800, color="#1565C0", linewidth=0.8, linestyle="--", zorder=2,
    )
    ax.text(
        len(STRATEGIES) - 0.5, 810, "Lake Mead evaporation (800 kaf)",
        fontsize=5.5, color="#1565C0", ha="right", va="bottom",
    )

    # Total labels on top of each bar
    for si, strat in enumerate(STRATEGIES):
        total = bottom[si]
        if total > 0:
            ax.text(
                si, total + 8, f"{total:.0f}",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            )

    strat_xlabels = [STRATEGY_LABELS[s] for s in STRATEGIES]
    ax.set_xticks(x)
    ax.set_xticklabels(strat_xlabels, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("Total water savings (thousand ac-ft)", fontsize=8)
    ax.set_xlabel("Strategy", fontsize=8)
    ax.set_title(
        "Basin-Scale Water Savings Potential by Strategy",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.legend(
        fontsize=6, loc="upper left", ncol=2,
        frameon=True, framealpha=0.9,
        edgecolor="#cccccc", fancybox=False,
        title="County", title_fontsize=7,
    )
    ax.set_ylim(0, max(bottom) * 1.12)
    ax.tick_params(axis="both", labelsize=7, length=2, width=0.5)
    _style_ax(ax)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)

    # Summary: total by strategy
    strategy_totals = {}
    for strat in STRATEGIES:
        sdf = data[data["strategy"] == strat]
        strategy_totals[strat] = round(float(sdf["total_savings_kaf"].sum()), 1)

    summary = {
        "chart": "basin_impact",
        "strategy_totals_kaf": strategy_totals,
        "total_acreage": sum(COUNTY_ALFALFA_ACRES.values()),
    }

    if out_dir is not None:
        save_pub_figure(fig, "basin_scale_impact", out_dir)
        csv_path = Path(out_dir) / "basin_impact_data.csv"
        data.to_csv(csv_path, index=False)
        print(f"  Basin impact CSV: {csv_path}")
        summary["out_dir"] = str(out_dir)

    return fig, ax, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Multi-strategy water savings simulation (Figures 5-9)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: water_savings_sim_2)",
    )
    args = parser.parse_args()

    out = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).parent.parent.parent
        / "output" / "figures" / "water_savings_sim_2"
    )
    out.mkdir(parents=True, exist_ok=True)

    # Figure 5: Economic crossover
    print("=== Figure 5: Economic Crossover ===")
    _, _, s5 = economic_crossover_figure(out_dir=out)
    for grp, be in s5["breakeven_prices"].items():
        print(f"  {grp}: {be}")

    # Figure 6: Strategy heatmap
    print("\n=== Figure 6: Strategy Heatmap ===")
    _, _, s6 = strategy_heatmap_figure(out_dir=out)
    for county, opt in s6["optimal"].items():
        print(f"  {county}: normal={opt['normal']}, drought={opt['drought']}")

    # Figure 7: Savings decomposition
    print("\n=== Figure 7: Savings Decomposition ===")
    _, _, s7 = savings_decomposition_figure(out_dir=out)
    for county, total in s7["totals_acft"].items():
        print(f"  {county}: {total:.3f} ac-ft/acre")

    # Figure 8: Seasonal ET profile
    print("\n=== Figure 8: Seasonal ET Profile ===")
    _, _, s8 = seasonal_et_profile_figure(out_dir=out)
    print(f"  Groups: {s8['groups']}")

    # Figure 9: Basin-scale impact
    print("\n=== Figure 9: Basin-Scale Impact ===")
    _, _, s9 = basin_impact_figure(out_dir=out)
    for strat, total in s9["strategy_totals_kaf"].items():
        print(f"  {strat}: {total:.0f} thousand ac-ft")
    print(f"  Total acreage: {s9['total_acreage']:,} acres")

    print("\nAll 5 figures generated successfully.")
