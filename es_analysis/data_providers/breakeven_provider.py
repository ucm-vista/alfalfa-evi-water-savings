"""Breakeven economics provider for last-cutting water analysis.

Computes the breakeven water price at which skipping the last alfalfa
cutting becomes economically rational, given per-cutting ET, forage
price, harvest cost, and yield assumptions. Produces a breakeven table
and a contour decision-surface plot.

Exports:
    run_breakeven_analysis  -- main entry point
    breakeven_water_price   -- standalone breakeven calculator
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import config
from ..utils.units import mm_to_acft_per_acre


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Economic parameter grids
WATER_PRICES = np.linspace(200, 600, 100)       # $/ac-ft
FORAGE_PRICES = np.array([150, 200, 250, 300])  # $/ton
HARVEST_COSTS = np.array([50, 65, 80])           # $/ton
LATE_YIELD_TON_PER_ACRE = 0.75                   # central estimate
LATE_YIELD_SENSITIVITY = [0.5, 0.75, 1.0]        # ton/acre


# ---------------------------------------------------------------------------
# Breakeven function
# ---------------------------------------------------------------------------

def breakeven_water_price(
    et_acft_per_acre: float,
    yield_ton_per_acre: float,
    forage_price_per_ton: float,
    harvest_cost_per_ton: float,
) -> float:
    """Water price ($/ac-ft) above which skipping a cutting is rational.

    The breakeven price equates water cost with net revenue from the
    cutting. Above this price, the water saved by skipping exceeds
    the foregone profit.

    breakeven = net_revenue / et_acft_per_acre
    where net_revenue = yield * (forage_price - harvest_cost)

    Args:
        et_acft_per_acre: ET consumed by the last cutting (ac-ft/acre).
        yield_ton_per_acre: Expected yield of the last cutting (ton/acre).
        forage_price_per_ton: Market price of forage ($/ton).
        harvest_cost_per_ton: Cost of harvesting ($/ton).

    Returns:
        Breakeven water price in $/ac-ft. Returns np.inf if
        et_acft_per_acre <= 0.
    """
    net_revenue = yield_ton_per_acre * (forage_price_per_ton - harvest_cost_per_ton)
    if et_acft_per_acre <= 0:
        return np.inf
    return net_revenue / et_acft_per_acre


# ---------------------------------------------------------------------------
# Kern County ET loading
# ---------------------------------------------------------------------------

def _load_kern_et(late_water_csv: Path) -> dict:
    """Load Kern County per-cutting ET data and compute summary stats.

    Args:
        late_water_csv: Path to late_cut_savings_parcel_year_mm.csv.

    Returns:
        Dict with keys: median, mean, p25, p75, n_records, n_parcels,
        all in ac-ft/acre.

    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If no Kern County data found.
    """
    late_water_csv = Path(late_water_csv)
    if not late_water_csv.exists():
        raise FileNotFoundError(
            f"Late-water CSV not found: {late_water_csv}"
        )

    df = pd.read_csv(late_water_csv)
    print(f"  Loaded {len(df):,} rows from {late_water_csv.name}")

    # Filter to Kern County
    df_kern = df[df["county"] == "Kern"].copy()
    if df_kern.empty:
        raise ValueError("No Kern County data found in late-water CSV")

    print(f"  Kern County: {len(df_kern):,} rows, "
          f"{df_kern['UniqueID'].nunique()} unique parcels")

    # Extract late ET in mm and convert to ac-ft/acre
    late_et_mm = df_kern["late_et_mm"].dropna()
    # Filter to rows with positive ET (skip zero-late-cut rows)
    late_et_mm = late_et_mm[late_et_mm > 0]
    late_et_acft = mm_to_acft_per_acre(late_et_mm)

    if late_et_acft.empty:
        raise ValueError("No positive late_et_mm values for Kern County")

    stats = {
        "median": float(late_et_acft.median()),
        "mean": float(late_et_acft.mean()),
        "p25": float(late_et_acft.quantile(0.25)),
        "p75": float(late_et_acft.quantile(0.75)),
        "n_records": int(len(late_et_acft)),
        "n_parcels": int(df_kern["UniqueID"].nunique()),
    }

    print(f"\n  Kern County late-cutting ET (ac-ft/acre):")
    print(f"    Median: {stats['median']:.4f}")
    print(f"    Mean:   {stats['mean']:.4f}")
    print(f"    P25:    {stats['p25']:.4f}")
    print(f"    P75:    {stats['p75']:.4f}")
    print(f"    N:      {stats['n_records']}")

    return stats


# ---------------------------------------------------------------------------
# Breakeven table
# ---------------------------------------------------------------------------

def _build_breakeven_table(kern_et_stats: dict) -> pd.DataFrame:
    """Build breakeven water price table for all parameter combinations.

    Args:
        kern_et_stats: Dict with median, p25, p75 ET in ac-ft/acre.

    Returns:
        DataFrame with columns: forage_price, harvest_cost,
        et_acft_median, breakeven_water_price_median,
        breakeven_at_p25_et, breakeven_at_p75_et.
    """
    rows = []
    for fp in FORAGE_PRICES:
        for hc in HARVEST_COSTS:
            bp_med = breakeven_water_price(
                kern_et_stats["median"],
                LATE_YIELD_TON_PER_ACRE,
                fp, hc,
            )
            bp_p25 = breakeven_water_price(
                kern_et_stats["p25"],
                LATE_YIELD_TON_PER_ACRE,
                fp, hc,
            )
            bp_p75 = breakeven_water_price(
                kern_et_stats["p75"],
                LATE_YIELD_TON_PER_ACRE,
                fp, hc,
            )
            rows.append({
                "forage_price": float(fp),
                "harvest_cost": float(hc),
                "yield_ton_per_acre": LATE_YIELD_TON_PER_ACRE,
                "et_acft_median": kern_et_stats["median"],
                "breakeven_water_price_median": bp_med,
                "breakeven_at_p25_et": bp_p25,
                "breakeven_at_p75_et": bp_p75,
            })

    table = pd.DataFrame(rows)

    print(f"\n  Breakeven table ({len(table)} scenarios):")
    for _, row in table.iterrows():
        print(f"    Forage ${row['forage_price']:.0f}/ton, "
              f"Harvest ${row['harvest_cost']:.0f}/ton -> "
              f"Breakeven ${row['breakeven_water_price_median']:.0f}/ac-ft "
              f"[${row['breakeven_at_p75_et']:.0f}-"
              f"${row['breakeven_at_p25_et']:.0f} at P25-P75 ET]")

    return table


# ---------------------------------------------------------------------------
# Contour / decision surface plot
# ---------------------------------------------------------------------------

def plot_breakeven_surface(
    kern_et_stats: dict,
    out_dir: Path,
    dpi: int = 300,
) -> Path:
    """Create 2D contour decision surface for keep vs skip cutting.

    X-axis: water price ($/ac-ft), Y-axis: forage price ($/ton).
    Color: net value of keeping the cutting (green = keep, red = skip).
    Black contour line at net_value = 0 (breakeven).
    Dashed lines show sensitivity to harvest cost.

    Args:
        kern_et_stats: Dict with median ET in ac-ft/acre.
        out_dir: Output directory.
        dpi: Plot resolution.

    Returns:
        Path to saved contour plot PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid
    water_prices = np.linspace(200, 600, 200)
    forage_prices = np.linspace(150, 300, 200)
    WP, FP = np.meshgrid(water_prices, forage_prices)

    et_acft = kern_et_stats["median"]
    yield_t = LATE_YIELD_TON_PER_ACRE

    # Central harvest cost ($65/ton)
    harvest_cost_central = 65.0
    net_value = yield_t * (FP - harvest_cost_central) - et_acft * WP

    fig, ax = plt.subplots(figsize=(10, 7))

    # Filled contour
    levels = np.linspace(-150, 150, 31)
    cf = ax.contourf(
        WP, FP, net_value,
        levels=levels,
        cmap="RdYlGn",
        extend="both",
    )

    # Breakeven line (central)
    cs_central = ax.contour(
        WP, FP, net_value,
        levels=[0],
        colors="black",
        linewidths=2.5,
    )
    ax.clabel(cs_central, fmt="Breakeven", fontsize=9, inline=True)

    # Sensitivity: harvest cost = $50/ton
    net_value_low_hc = yield_t * (FP - 50.0) - et_acft * WP
    cs_low = ax.contour(
        WP, FP, net_value_low_hc,
        levels=[0],
        colors="black",
        linewidths=1.5,
        linestyles="dashed",
    )
    ax.clabel(cs_low, fmt="HC=$50", fontsize=8, inline=True)

    # Sensitivity: harvest cost = $80/ton
    net_value_high_hc = yield_t * (FP - 80.0) - et_acft * WP
    cs_high = ax.contour(
        WP, FP, net_value_high_hc,
        levels=[0],
        colors="black",
        linewidths=1.5,
        linestyles="dotted",
    )
    ax.clabel(cs_high, fmt="HC=$80", fontsize=8, inline=True)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Net value of cutting ($/acre)", fontsize=11)

    # Labels
    ax.set_xlabel("Water price ($/ac-ft)", fontsize=12)
    ax.set_ylabel("Forage price ($/ton)", fontsize=12)
    ax.set_title(
        "Breakeven: Keep vs Skip Last Cutting (Kern County)",
        fontsize=13,
        fontweight="bold",
    )

    # Annotation with assumptions
    assumptions = (
        f"ET = {et_acft:.3f} ac-ft/acre (Kern median)\n"
        f"Yield = {yield_t} ton/acre\n"
        f"Solid: HC=${harvest_cost_central:.0f}/ton\n"
        f"Dashed: HC=$50, Dotted: HC=$80"
    )
    ax.text(
        0.02, 0.02, assumptions,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    plot_path = out_dir / "breakeven_surface.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  [saved] Breakeven surface: {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_breakeven_analysis(
    late_water_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    dpi: int = 300,
) -> dict:
    """Run breakeven economics analysis for the last alfalfa cutting.

    Loads Kern County per-cutting ET data, computes breakeven water
    prices across forage price and harvest cost scenarios, and
    generates a contour decision-surface plot.

    Args:
        late_water_csv: Path to late_cut_savings_parcel_year_mm.csv.
            Defaults to config.water_saving_out_dir / filename.
        out_dir: Output directory for plots and CSVs. Defaults to
            config.statistics_export_dir / "breakeven".
        dpi: Plot resolution.

    Returns:
        Dict with keys:
            kern_et_stats: Dict with median, mean, p25, p75 (ac-ft/acre).
            breakeven_table: DataFrame of breakeven scenarios.
            plot_path: Path to contour plot PNG.
            n_kern_parcels: Number of unique Kern parcels used.
    """
    if late_water_csv is None:
        late_water_csv = (
            config.water_saving_out_dir / "late_cut_savings_parcel_year_mm.csv"
        )
    if out_dir is None:
        out_dir = config.statistics_export_dir / "breakeven"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("BREAKEVEN ECONOMICS ANALYSIS")
    print(f"{'='*60}")

    # --- Load Kern County ET ---
    print(f"\n{'~'*40}")
    print("Loading Kern County per-cutting ET")
    print(f"{'~'*40}")
    kern_et_stats = _load_kern_et(late_water_csv)

    # --- Breakeven table ---
    print(f"\n{'~'*40}")
    print("Computing breakeven water prices")
    print(f"{'~'*40}")
    table = _build_breakeven_table(kern_et_stats)

    # Export table
    table_csv = out_dir / "breakeven_table.csv"
    table.to_csv(table_csv, index=False)
    print(f"  [saved] Breakeven table: {table_csv}")

    # --- Contour plot ---
    print(f"\n{'~'*40}")
    print("Generating decision surface plot")
    print(f"{'~'*40}")
    plot_path = plot_breakeven_surface(kern_et_stats, out_dir, dpi=dpi)

    print(f"\n{'='*60}")
    print("BREAKEVEN ANALYSIS COMPLETE")
    print(f"{'='*60}")

    return {
        "kern_et_stats": kern_et_stats,
        "breakeven_table": table,
        "plot_path": plot_path,
        "n_kern_parcels": kern_et_stats["n_parcels"],
    }
