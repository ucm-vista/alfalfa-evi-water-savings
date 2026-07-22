"""Runner: Publication-quality figures (Phase 5).

Usage:
    python -m es_analysis.runners.run_publication_figures [options]

Generates all Phase 5 publication figures: polished existing plots,
conceptual diagram, timing gap analysis, and water savings choropleth.
"""
import argparse
import calendar
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from ..utils.publication_style import (
    apply_style,
    save_pub_figure,
    add_panel_label,
    WONG_PALETTE,
    DOUBLE_COL_WIDTH,
)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_OUT_DIR = "es_analysis/output/publication_figures"
_DEFAULT_SAVINGS_CSV = (
    "water_saving_scenarios_stats/water_saving_summary_county_by_cap.csv"
)


# ---------------------------------------------------------------------------
# FIG-02: Conceptual Diagram
# ---------------------------------------------------------------------------

def _synthetic_evi_curve(n_days=365):
    """Build a synthetic EVI time series with 3 cutting cycles.

    Returns (days, evi) arrays where each cycle has sigmoid recovery
    from ~0.15 to ~0.55 followed by an abrupt drop at cutting.
    """
    # Define cutting days (day-of-year)
    cut_days = [100, 165, 235]
    evi = np.full(n_days, 0.15)

    # Build EVI curve: sigmoid recovery after each cut/start
    starts = [30] + [c + 2 for c in cut_days]  # recovery start points
    ends = cut_days + [300]

    for s, e in zip(starts, ends):
        if s >= n_days or e > n_days:
            break
        length = e - s
        t = np.linspace(-6, 6, length)
        sigmoid = 0.15 + 0.40 / (1 + np.exp(-t))
        evi[s:e] = sigmoid

    # Dormant winter period
    evi[:30] = 0.12 + 0.01 * np.random.default_rng(42).standard_normal(30)
    evi[300:] = 0.12 + 0.01 * np.random.default_rng(43).standard_normal(n_days - 300)

    # Clip to reasonable range
    evi = np.clip(evi, 0.08, 0.60)

    return np.arange(n_days), evi, cut_days


def _synthetic_et_curves(n_days=365, cut_days=None):
    """Build synthetic 'Actual ET' and 'OpenET estimate' curves.

    Actual ET drops sharply at cutting and recovers gradually.
    OpenET misses the post-cutting dip because the satellite captured
    EVI after partial recovery.
    """
    if cut_days is None:
        cut_days = [100, 165, 235]

    days = np.arange(n_days)

    # Base ET seasonal envelope (bell-shaped)
    envelope = 3.0 + 4.0 * np.exp(-0.5 * ((days - 180) / 70) ** 2)

    # Actual ET: drops at cutting, recovers over ~25 days
    actual = envelope.copy()
    for cd in cut_days:
        for d in range(cd, min(cd + 30, n_days)):
            recovery = (d - cd) / 30.0
            actual[d] *= 0.3 + 0.7 * recovery

    # OpenET estimate: misses the dip (sees satellite image 5-8 days later)
    openet = envelope.copy()
    for cd in cut_days:
        delay = 6  # satellite sees partial recovery
        for d in range(cd + delay, min(cd + 30, n_days)):
            recovery = (d - cd) / 30.0
            openet[d] *= 0.55 + 0.45 * recovery
        # OpenET does NOT see the first few days of drop
        # (it interpolates through the gap)

    return days, actual, openet


def plot_conceptual_diagram(out_dir, dpi=300):
    """FIG-02: 2-panel conceptual diagram of ET correction method.

    Panel (a): EVI time series with cutting events and satellite passes.
    Panel (b): Actual ET vs OpenET estimate with correction delta shading.
    """
    apply_style()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(DOUBLE_COL_WIDTH, 5.0),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35},
    )

    # --- Panel (a): EVI + satellite timing ---
    ax_evi = axes[0]
    days, evi, cut_days = _synthetic_evi_curve()

    # EVI curve
    ax_evi.plot(
        days, evi,
        color=WONG_PALETTE[3],  # bluish green
        linewidth=1.2,
        label="EVI",
    )

    # Cutting events (vertical dashed lines)
    for i, cd in enumerate(cut_days):
        ax_evi.axvline(
            cd,
            color=WONG_PALETTE[6],  # vermillion
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
        )
        ax_evi.text(
            cd, 0.58, "Cut",
            color=WONG_PALETTE[6],
            fontsize=6,
            ha="center",
            va="bottom",
        )

    # Satellite overpasses (inverted triangles at top)
    rng = np.random.default_rng(7)
    base_passes = np.arange(10, 360, 16)
    # Add some jitter and remove a few to show gaps
    passes = base_passes + rng.integers(-2, 3, size=len(base_passes))
    # Remove passes near some cuts to create timing gap
    mask = np.ones(len(passes), dtype=bool)
    mask[5] = False   # remove one near first cut
    mask[10] = False  # remove one near second cut
    passes = passes[mask]
    passes = passes[(passes > 0) & (passes < 365)]

    ax_evi.scatter(
        passes,
        np.full_like(passes, 0.61, dtype=float),
        marker="v",
        s=18,
        color=WONG_PALETTE[5],  # blue
        zorder=5,
        label="Satellite overpass",
    )

    # Annotate one timing gap
    gap_cut = cut_days[1]  # second cutting
    # Find closest pass AFTER this cut
    after_passes = passes[passes > gap_cut]
    if len(after_passes) > 0:
        next_pass = after_passes[0]
        gap_days_val = next_pass - gap_cut
        mid = (gap_cut + next_pass) / 2
        ax_evi.annotate(
            "",
            xy=(next_pass, 0.50),
            xytext=(gap_cut, 0.50),
            arrowprops=dict(
                arrowstyle="<->",
                color=WONG_PALETTE[0],
                linewidth=0.8,
            ),
        )
        ax_evi.text(
            mid, 0.52,
            f"Timing gap:\n{gap_days_val} days",
            fontsize=6,
            ha="center",
            va="bottom",
            color=WONG_PALETTE[0],
        )

    ax_evi.set_xlabel("Day of year")
    ax_evi.set_ylabel("EVI")
    ax_evi.set_xlim(0, 365)
    ax_evi.set_ylim(0.05, 0.65)
    ax_evi.legend(fontsize=6, loc="lower right", framealpha=0.9)
    add_panel_label(ax_evi, "a")

    # --- Panel (b): ET correction ---
    ax_et = axes[1]
    days_et, actual_et, openet_et = _synthetic_et_curves(cut_days=cut_days)

    ax_et.plot(
        days_et, actual_et,
        color=WONG_PALETTE[3],  # bluish green
        linewidth=1.2,
        label="Actual ET",
    )
    ax_et.plot(
        days_et, openet_et,
        color=WONG_PALETTE[1],  # orange
        linewidth=1.2,
        linestyle="--",
        label="OpenET estimate",
    )

    # Shade overestimation regions (where OpenET > Actual)
    overest = openet_et > actual_et
    ax_et.fill_between(
        days_et,
        actual_et,
        openet_et,
        where=overest,
        alpha=0.3,
        color=WONG_PALETTE[1],  # light orange
        label="Overestimation",
    )

    # Annotate one correction delta region
    anno_cut = cut_days[0]
    anno_x = anno_cut + 10
    if anno_x < len(actual_et):
        y_actual = actual_et[anno_x]
        y_openet = openet_et[anno_x]
        mid_y = (y_actual + y_openet) / 2
        ax_et.annotate(
            "Correction\ndelta",
            xy=(anno_x, mid_y),
            xytext=(anno_x + 40, mid_y + 1.5),
            fontsize=6,
            ha="center",
            arrowprops=dict(
                arrowstyle="->",
                color=WONG_PALETTE[0],
                linewidth=0.8,
            ),
        )

    # Cutting event markers
    for cd in cut_days:
        ax_et.axvline(
            cd,
            color=WONG_PALETTE[6],  # vermillion
            linestyle="--",
            linewidth=0.7,
            alpha=0.6,
        )

    ax_et.set_xlabel("Day of year")
    ax_et.set_ylabel("ET (mm/day)")
    ax_et.set_xlim(0, 365)
    ax_et.legend(fontsize=6, loc="upper right", framealpha=0.9)
    add_panel_label(ax_et, "b")

    save_pub_figure(fig, "fig02_conceptual_diagram", out_dir, dpi)
    print("  FIG-02 conceptual diagram complete.")


# ---------------------------------------------------------------------------
# FIG-03: Timing Gap
# ---------------------------------------------------------------------------

def plot_timing_gap(gap_df, out_dir, dpi=300):
    """FIG-03: 2-panel timing gap figure.

    Panel (a): Overall histogram of gap_days (0-16 day bins).
    Panel (b): Box/violin plot of gap_days by month (Mar-Oct).

    Args:
        gap_df: DataFrame from compute_timing_gaps() with columns
            gap_days, month.
        out_dir: Output directory.
        dpi: Figure DPI.
    """
    apply_style()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL_WIDTH, 3.5),
        gridspec_kw={"wspace": 0.35},
    )

    # Clean data
    gd = gap_df["gap_days"].dropna().values

    # --- Panel (a): Overall histogram ---
    ax_hist = axes[0]
    bins = np.arange(0, 18, 1)  # 0 to 17 (Landsat revisit ~16 days)
    ax_hist.hist(
        gd,
        bins=bins,
        color=WONG_PALETTE[5],  # blue
        edgecolor="white",
        linewidth=0.4,
    )

    median_gap = float(np.median(gd))
    ax_hist.axvline(
        median_gap,
        color=WONG_PALETTE[6],  # vermillion
        linestyle="--",
        linewidth=1.0,
    )
    ax_hist.text(
        median_gap + 0.3,
        ax_hist.get_ylim()[1] * 0.9,
        f"Median: {median_gap:.1f} d",
        fontsize=6,
        color=WONG_PALETTE[6],
        va="top",
    )

    ax_hist.set_xlabel("Days between cutting and nearest overpass")
    ax_hist.set_ylabel("Number of cutting events")
    ax_hist.set_title("Overall Distribution", fontsize=8)
    add_panel_label(ax_hist, "a")

    # --- Panel (b): By month (boxplot, growing season Mar-Oct) ---
    ax_month = axes[1]

    growing_months = list(range(3, 11))  # Mar=3 through Oct=10
    month_data = []
    month_labels = []
    for m in growing_months:
        vals = gap_df.loc[gap_df["month"] == m, "gap_days"].dropna().values
        if len(vals) > 0:
            month_data.append(vals)
            month_labels.append(calendar.month_abbr[m])
        else:
            month_data.append(np.array([]))
            month_labels.append(calendar.month_abbr[m])

    # Filter out empty months for boxplot
    non_empty = [(d, l) for d, l in zip(month_data, month_labels) if len(d) > 0]
    if non_empty:
        data_arrays, labels = zip(*non_empty)
        bp = ax_month.boxplot(
            data_arrays,
            patch_artist=True,
            widths=0.6,
        )
        for box in bp["boxes"]:
            box.set_facecolor(WONG_PALETTE[5])
            box.set_alpha(0.7)
            box.set_edgecolor(WONG_PALETTE[0])
        for median_line in bp["medians"]:
            median_line.set_color(WONG_PALETTE[6])
            median_line.set_linewidth(1.2)
        for whisker in bp["whiskers"]:
            whisker.set_color(WONG_PALETTE[0])
        for cap in bp["caps"]:
            cap.set_color(WONG_PALETTE[0])
        for flier in bp["fliers"]:
            flier.set_markerfacecolor(WONG_PALETTE[5])
            flier.set_markeredgecolor(WONG_PALETTE[5])
            flier.set_markersize(2)
            flier.set_alpha(0.4)

        ax_month.set_xticklabels(labels, fontsize=6)

    ax_month.set_xlabel("Month")
    ax_month.set_ylabel("Gap (days)")
    ax_month.set_title("By Month", fontsize=8)
    add_panel_label(ax_month, "b")

    save_pub_figure(fig, "fig03_timing_gap", out_dir, dpi)
    print("  FIG-03 timing gap complete.")


# ---------------------------------------------------------------------------
# FIG-04: Water Savings Choropleth
# ---------------------------------------------------------------------------

def plot_water_savings_choropleth(out_dir, dpi=300, cap=0, savings_csv=None):
    """FIG-04: County choropleth of water savings intensity.

    Colors 10 study counties by ac-ft/acre water saved under the given
    cap scenario.  Non-study counties (Los Angeles, San Bernardino) show
    as grey "No data".

    Args:
        out_dir: Output directory.
        dpi: Figure DPI.
        cap: Cap scenario (0 = all late cuttings removed).
        savings_csv: Override path to county savings CSV.
    """
    # Lazy imports -- geopandas is slow to load
    import geopandas as gpd  # noqa: F811
    from ..data_providers.spatial_provider import load_county_boundaries

    apply_style()

    # Load savings data
    csv_path = Path(savings_csv) if savings_csv else Path(_DEFAULT_SAVINGS_CSV)
    if not csv_path.exists():
        print(f"  WARNING: Savings CSV not found: {csv_path}")
        return
    savings_df = pd.read_csv(csv_path)

    # Filter to requested cap scenario
    if "cut_cap_k" in savings_df.columns:
        savings_df = savings_df[savings_df["cut_cap_k"] == cap].copy()

    if savings_df.empty:
        print(f"  WARNING: No data for cap={cap} in {csv_path}")
        return

    # Normalize county names in savings
    savings_df["county"] = (
        savings_df["county"]
        .astype(str)
        .str.replace("_", " ")
        .str.strip()
        .str.title()
    )

    # Load ALL county boundaries (include Los Angeles + San Bernardino)
    all_counties = [
        "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
        "Tulare", "Kings", "Kern", "Riverside", "Imperial",
        "Los Angeles", "San Bernardino",
    ]
    gdf = load_county_boundaries(counties=all_counties)
    if gdf is None:
        print("  WARNING: County boundaries not found, skipping choropleth")
        return

    # Merge savings onto geometry
    merged = gdf.merge(
        savings_df[["county", "water_saved_acft_per_acre"]],
        left_on="COUNTY_norm",
        right_on="county",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 6))

    merged.plot(
        column="water_saved_acft_per_acre",
        ax=ax,
        legend=True,
        legend_kwds={"label": "Water saved (ac-ft/acre)", "shrink": 0.7},
        cmap="YlOrRd",
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    # County name labels at representative point
    for _, row in merged.iterrows():
        if row.geometry and not row.geometry.is_empty:
            rp = row.geometry.representative_point()
            ax.text(
                rp.x, rp.y,
                row["COUNTY_norm"],
                fontsize=6,
                ha="center",
                va="center",
            )

    ax.set_axis_off()
    cap_desc = "All Late Cuttings Removed" if cap == 0 else f"Cap {cap}"
    ax.set_title(
        f"Water Savings Intensity by County ({cap_desc})",
        fontsize=9,
    )

    save_pub_figure(fig, "fig04_water_savings_choropleth", out_dir, dpi)
    print("  FIG-04 choropleth complete.")


# ---------------------------------------------------------------------------
# FIG-01: Polish existing Phase 3/4 figures
# ---------------------------------------------------------------------------

def polish_existing_figures(out_dir, dpi=300):
    """Re-render Phase 3 and Phase 4 plots at publication quality.

    Calls apply_style() to set Wong palette and journal rcParams globally,
    then calls the existing plot functions which inherit those rcParams.
    Each section is independently guarded so missing input files produce
    warnings instead of crashes.

    Figures are saved to polished/ AND to the standard png/ + pdf/
    subdirectories (via save_pub_figure) so that all publication outputs
    consistently appear in both formats.

    Args:
        out_dir: Root output directory.
        dpi: Figure DPI.

    Returns:
        List of output file paths (for counting in summary).
    """
    # All imports are lazy to avoid slow loading when polish is skipped
    from ..utils.publication_style import apply_style as _apply_style
    from ..runners.run_statistical_tests import (
        plot_correction_distribution,
        plot_correction_violin_by_county,
    )
    from ..data_providers.statistical_tests_provider import (
        load_parcel_year_data,
        run_all_statistical_tests,
    )

    _apply_style()
    polished_dir = Path(out_dir) / "polished"
    polished_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    # Helper: intercept plt.close so we can additionally save via
    # save_pub_figure (PNG in png/ + PDF in pdf/) before the figure is
    # released.  The original plot functions already save a PNG to the
    # out_dir they receive (polished/), so the interceptor adds the
    # dual-format copies.
    _captured_figs = []
    _real_close = plt.close

    def _capture_close(fig=None):
        """Store figure reference instead of closing immediately."""
        if fig is None:
            fig = plt.gcf()
        _captured_figs.append(fig)

    def _run_with_capture(plot_fn, pub_name, *args, **kwargs):
        """Call a plot function, capture its figure, dual-save, then close."""
        _captured_figs.clear()
        plt.close = _capture_close
        try:
            result = plot_fn(*args, **kwargs)
        finally:
            plt.close = _real_close

        for fig in _captured_figs:
            save_pub_figure(fig, pub_name, out_dir, dpi)
        _captured_figs.clear()
        return result

    # --- Phase 3: correction distribution + violin ---
    print("  [polish] Phase 3 plots...")
    parcel_csv = Path(
        "statistics_exports/"
        "offphase_parcel_year_methodA_cc20_ci90_boot500_WY2019-2024.csv"
    )
    if not parcel_csv.exists():
        print(f"  WARNING: {parcel_csv} not found, skipping Phase 3 polish")
    else:
        try:
            df = load_parcel_year_data(parcel_csv)
            failures_csv = Path(
                "statistics_exports/"
                "offphase_failures_methodA_cc20_ci90_boot500_WY2019-2024.csv"
            )
            results = run_all_statistical_tests(
                parcel_year_csv=parcel_csv,
                failures_csv=failures_csv,
                skip_hl=True,  # fast: skip Hodges-Lehmann (~30s)
            )
            cld_dict = results["posthoc"]["cld"]
            df = results["df"]

            p1 = _run_with_capture(
                plot_correction_distribution,
                "correction_distribution",
                df, cld_dict, polished_dir, dpi=dpi,
            )
            outputs.append(p1)
            p2 = _run_with_capture(
                plot_correction_violin_by_county,
                "correction_violin_by_county",
                df, cld_dict, polished_dir, dpi=dpi,
            )
            outputs.append(p2)
            print(f"  [polish] Phase 3: 2 plots saved to {polished_dir}")
        except Exception as e:
            print(f"  WARNING: Phase 3 polish failed ({e}), skipping")

    # (Phase 4 breakeven-surface figure removed with the simulation layer.)

    print(f"  [polish] Total polished: {len(outputs)} figures")
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the publication figures runner.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Generate Phase 5 publication-quality figures.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for figure exports (default: 300).",
    )
    parser.add_argument(
        "--skip-conceptual", action="store_true",
        help="Skip FIG-02 conceptual diagram.",
    )
    parser.add_argument(
        "--skip-timing", action="store_true",
        help="Skip FIG-03 timing gap figure.",
    )
    parser.add_argument(
        "--skip-choropleth", action="store_true",
        help="Skip FIG-04 choropleth map.",
    )
    parser.add_argument(
        "--skip-polish", action="store_true",
        help="Skip FIG-01 polish of existing Phase 3/4 plots.",
    )
    parser.add_argument(
        "--timing-counties", nargs="+", default=None,
        help="Counties for timing gap (default: all 10).",
    )
    parser.add_argument(
        "--timing-years", nargs="+", type=int, default=None,
        help="Water years for timing gap (default: all).",
    )
    parser.add_argument(
        "--cap-scenario", type=int, default=0,
        help="Cap level for choropleth (default: 0 = max savings).",
    )
    parser.add_argument(
        "--savings-csv", type=str, default=None,
        help=f"Path to savings CSV for choropleth (default: {_DEFAULT_SAVINGS_CSV}).",
    )

    args = parser.parse_args()

    t0 = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PUBLICATION FIGURES RUNNER (Phase 5)")
    print(f"{'='*60}")
    print(f"Output:           {out_dir}")
    print(f"DPI:              {args.dpi}")
    print(f"Skip conceptual:  {args.skip_conceptual}")
    print(f"Skip timing:      {args.skip_timing}")
    print(f"Skip choropleth:  {args.skip_choropleth}")
    print(f"Skip polish:      {args.skip_polish}")

    figure_count = 0

    try:
        # ---------------------------------------------------------------
        # FIG-02: Conceptual Diagram
        # ---------------------------------------------------------------
        if not args.skip_conceptual:
            print(f"\n{'='*60}")
            print("FIG-02: CONCEPTUAL DIAGRAM")
            print(f"{'='*60}")

            plot_conceptual_diagram(out_dir, dpi=args.dpi)
            figure_count += 1
        else:
            print("\n  [skipped] FIG-02 conceptual diagram")

        # ---------------------------------------------------------------
        # FIG-03: Timing Gap
        # ---------------------------------------------------------------
        if not args.skip_timing:
            print(f"\n{'='*60}")
            print("FIG-03: TIMING GAP")
            print(f"{'='*60}")

            from ..data_providers.publication_figures_provider import (
                compute_timing_gaps,
            )

            gap_df = compute_timing_gaps(
                counties=args.timing_counties,
                water_years=args.timing_years,
            )

            if not gap_df.empty:
                plot_timing_gap(gap_df, out_dir, dpi=args.dpi)
                figure_count += 1

                # Export gap data as CSV
                csv_path = out_dir / "timing_gap_data.csv"
                gap_df.to_csv(csv_path, index=False)
                print(f"  Exported: {csv_path} ({csv_path.stat().st_size / 1024:.0f} KB)")
            else:
                print("  WARNING: No timing gap data produced, skipping FIG-03 plot")
        else:
            print("\n  [skipped] FIG-03 timing gap")

        # ---------------------------------------------------------------
        # FIG-04: Choropleth
        # ---------------------------------------------------------------
        if not args.skip_choropleth:
            print(f"\n{'='*60}")
            print("FIG-04: WATER SAVINGS CHOROPLETH")
            print(f"{'='*60}")

            plot_water_savings_choropleth(
                out_dir,
                dpi=args.dpi,
                cap=args.cap_scenario,
                savings_csv=args.savings_csv,
            )
            figure_count += 1
        else:
            print("\n  [skipped] FIG-04 choropleth")

        # ---------------------------------------------------------------
        # FIG-01: Polish existing Phase 3/4 plots
        # ---------------------------------------------------------------
        if not args.skip_polish:
            print(f"\n{'='*60}")
            print("FIG-01: POLISH EXISTING FIGURES")
            print(f"{'='*60}")

            polished = polish_existing_figures(out_dir, dpi=args.dpi)
            figure_count += len(polished)
        else:
            print("\n  [skipped] FIG-01 polish")

        # ---------------------------------------------------------------
        # Final summary
        # ---------------------------------------------------------------
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Elapsed:  {elapsed:.1f}s")
        print(f"Output:   {out_dir}")
        print(f"Figures:  {figure_count}")

        # Count output files
        for subdir_name in ["png", "pdf", "polished"]:
            subdir = out_dir / subdir_name
            if subdir.exists():
                n_files = len(list(subdir.iterdir()))
                print(f"  {subdir_name:10s}: {n_files} files")

        return 0

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
