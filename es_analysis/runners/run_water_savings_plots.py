#!/usr/bin/env python3
"""CLI runner for water savings charts.

Generates aggregate water savings plots from pre-computed late-water
savings CSVs. All plots are county- and/or WY-level aggregates.

Examples:
    # All aggregate charts with actual ET
    python -m es_analysis.runners.run_water_savings_plots --aggregate

    # Corrected ET mode
    python -m es_analysis.runners.run_water_savings_plots --aggregate --et-mode corrected

    # Both (actual + corrected overlaid)
    python -m es_analysis.runners.run_water_savings_plots --aggregate --et-mode both

    # Specific chart types
    python -m es_analysis.runners.run_water_savings_plots --cap-comparison --heatmap

    # Everything
    python -m es_analysis.runners.run_water_savings_plots --all --et-mode both
"""

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _safe_parse_list(x):
    """Parse stringified list, handling nan values."""
    if not isinstance(x, str):
        return []
    cleaned = re.sub(r'\bnan\b', 'float("nan")', x)
    cleaned = cleaned.replace("\n", ",")
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return []


def _load_savings_csv(data_dir=None):
    """Load the savings parcel-year CSV.

    Args:
        data_dir: Optional override directory. Falls back to
            config.water_saving_out_dir when None.
    """
    import pandas as pd
    from es_analysis.data_providers.config import config

    base = Path(data_dir) if data_dir is not None else config.water_saving_out_dir
    csv_path = base / "late_cut_savings_parcel_year_mm.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Savings CSV not found: {csv_path}\n"
            "Run the late-water savings workflow first:\n"
            "  python -m es_analysis.runners.run_late_water_workflow"
        )
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if "cycle_et" in col and "list" in col:
            df[col] = df[col].apply(_safe_parse_list)
    return df


def _load_base_csv(data_dir=None):
    """Load the base parcel-year CSV.

    Args:
        data_dir: Optional override directory. Falls back to
            config.water_saving_out_dir when None.
    """
    import pandas as pd
    from es_analysis.data_providers.config import config

    base = Path(data_dir) if data_dir is not None else config.water_saving_out_dir
    csv_path = base / "late_cut_base_parcel_year.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Base CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def _load_county_wy_summary(data_dir=None):
    """Load the county x WY summary CSV.

    Args:
        data_dir: Optional override directory. Falls back to
            config.water_saving_out_dir when None.
    """
    import pandas as pd
    from es_analysis.data_providers.config import config

    base = Path(data_dir) if data_dir is not None else config.water_saving_out_dir
    csv_path = base / "water_saving_summary_county_WY_by_cap.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"County-WY summary CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def run_aggregate_plots(
    out_dir: Path,
    cap_values=(0, 1, 2, 3),
    et_mode: str = "actual",
    data_dir=None,
):
    """Generate all aggregate water savings charts.

    Args:
        out_dir: Directory for output figures.
        cap_values: Cap values to plot.
        et_mode: "actual", "corrected", or "both".
        data_dir: Optional directory to read input CSVs from.
            Falls back to config.water_saving_out_dir when None.
    """
    import matplotlib
    matplotlib.use("Agg")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. Cap comparison bar
    print("\n--- Cap Comparison Bar Chart ---")
    try:
        from es_analysis.charts.water_savings.savings_cap_comparison_bar import (
            savings_cap_comparison_bar,
        )
        df_sav = _load_savings_csv(data_dir=data_dir)
        _, _, summary = savings_cap_comparison_bar(
            df_sav, cap_values=cap_values, et_mode=et_mode, out_dir=out_dir,
        )
        results.append(("cap_comparison", summary))
        print(f"  Overall means: {summary['overall_means']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. Late-cut frequency
    print("\n--- Late-Cut Frequency Bar ---")
    try:
        from es_analysis.charts.water_savings.late_cut_frequency_bar import (
            late_cut_frequency_bar,
        )
        df_base = _load_base_csv(data_dir=data_dir)
        _, _, summary = late_cut_frequency_bar(
            df_base, et_mode=et_mode, out_dir=out_dir,
        )
        results.append(("frequency", summary))
        print(f"  Distribution: {summary['distribution']}")
        print(f"  Pct with late cuts: {summary['pct_with_late_cuts']:.1f}%")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 3. Heatmap (one per cap value 0 and 1)
    print("\n--- Savings Heatmap ---")
    try:
        from es_analysis.charts.water_savings.savings_heatmap import savings_heatmap
        df_cy = _load_county_wy_summary(data_dir=data_dir)
        for cap_k in [0, 1]:
            _, _, summary = savings_heatmap(
                df_cy, cap_k=cap_k, et_mode=et_mode, out_dir=out_dir,
            )
            results.append((f"heatmap_cap{cap_k}", summary))
            print(f"  Cap={cap_k}: mean={summary['overall_mean']:.4f}, max={summary['max_county_wy']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 4. Savings by water year (combined cap0 + cap1)
    print("\n--- Savings by Water Year (combined) ---")
    try:
        from es_analysis.charts.water_savings.savings_by_wy_bar import (
            savings_by_wy_bar_combined,
        )
        df_sav = _load_savings_csv(data_dir=data_dir)
        _, _, summary = savings_by_wy_bar_combined(
            df_sav, cap_values=(0, 1), et_mode=et_mode, out_dir=out_dir,
        )
        results.append(("by_wy_combined", summary))
        print(f"  Combined: caps={summary['cap_values']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 5. Late ET share
    print("\n--- Late ET Share ---")
    try:
        from es_analysis.charts.water_savings.late_et_share_bar import late_et_share_bar
        df_base = _load_base_csv(data_dir=data_dir)
        if "total_et_mm" in df_base.columns:
            for group in ["county", "WY"]:
                _, _, summary = late_et_share_bar(
                    df_base, group_by=group, et_mode=et_mode, out_dir=out_dir,
                )
                results.append((f"et_share_{group}", summary))
                print(f"  By {group}: mean={summary['overall_mean_pct']:.1f}%")
        else:
            print("  SKIPPED: total_et_mm column not in base CSV. Re-run workflow.")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 6. All-counties savings bar (cap=0 and cap=1)
    print("\n--- All-Counties Savings Bar ---")
    try:
        from es_analysis.charts.water_savings.savings_by_all_counties_bar import (
            savings_by_all_counties_bar,
        )
        df_sav = _load_savings_csv(data_dir=data_dir)
        _, _, summary = savings_by_all_counties_bar(
            df_sav, cap_values=(0, 1), et_mode=et_mode, out_dir=out_dir,
        )
        results.append(("all_counties_bar", summary))
        for k in [0, 1]:
            key = f"cap{k}_by_county"
            if key in summary:
                print(f"  Cap={k}: {summary[key]}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 7. Cutoff-date sensitivity (county x cutoff x WY type)
    print("\n--- Cutoff Sensitivity ---")
    try:
        from es_analysis.charts.water_savings.sensitivity_heatmap import (
            cutoff_sensitivity_figure,
        )
        df_base = _load_base_csv(data_dir=data_dir)
        df_sav = _load_savings_csv(data_dir=data_dir)
        _, _, summary = cutoff_sensitivity_figure(
            df_base, df_savings=df_sav, out_dir=out_dir,
        )
        results.append(("cutoff_sensitivity", summary))
        print(f"  Cells: {summary['n_cells']} "
              f"(non-sparse: {summary['n_non_sparse']})")
    except Exception as e:
        print(f"  FAILED: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate water savings charts (aggregate only).",
    )
    parser.add_argument("--aggregate", action="store_true",
                        help="Generate all aggregate charts")
    parser.add_argument("--all", action="store_true",
                        help="Generate all charts (same as --aggregate)")
    parser.add_argument("--cap-comparison", action="store_true",
                        help="Cap comparison bar chart only")
    parser.add_argument("--frequency", action="store_true",
                        help="Late-cut frequency chart only")
    parser.add_argument("--heatmap", action="store_true",
                        help="Savings heatmap only")
    parser.add_argument("--by-wy", action="store_true",
                        help="Savings by water year only")
    parser.add_argument("--et-share", action="store_true",
                        help="Late ET share chart only")
    parser.add_argument("--all-counties", action="store_true",
                        help="All-counties savings bar (cap0/cap1)")
    parser.add_argument("--cutoff", action="store_true",
                        help="Cutoff-date sensitivity figure (county x cutoff x WY type)")
    parser.add_argument("--et-mode", type=str, default="actual",
                        choices=["actual", "corrected", "both"],
                        help="ET mode (default: actual)")
    parser.add_argument("--method", type=str, default="A",
                        choices=["A", "B"],
                        help="ET correction method (default: A)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory to read input CSVs from (overrides config default)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Named run (e.g. alfalfa_run_6). Auto-resolves --data-dir and --output-dir from saved run data.")

    args = parser.parse_args()

    # --run-name sets defaults for --data-dir and --output-dir
    if args.run_name:
        from es_analysis.utils.run_output import get_run_root
        run_root = get_run_root(args.run_name)
        if not args.data_dir:
            args.data_dir = str(run_root / "water_savings")
        if not args.output_dir:
            args.output_dir = str(run_root / "water_savings")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent / "output" / "figures" / "water_savings"
    )
    data_dir = Path(args.data_dir) if args.data_dir else None

    do_all = args.aggregate or args.all
    any_specific = args.cap_comparison or args.frequency or args.heatmap or args.by_wy or args.et_share or args.all_counties or args.cutoff

    if not do_all and not any_specific:
        parser.error("Specify --aggregate, --all, or specific chart flags")

    if do_all:
        print("=" * 60)
        print("  AGGREGATE WATER SAVINGS CHARTS")
        print(f"  ET mode: {args.et_mode}")
        if data_dir:
            print(f"  Data dir: {data_dir}")
        print("=" * 60)
        run_aggregate_plots(out_dir, et_mode=args.et_mode, data_dir=data_dir)
    else:
        import matplotlib
        matplotlib.use("Agg")
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.cap_comparison:
            from es_analysis.charts.water_savings.savings_cap_comparison_bar import savings_cap_comparison_bar
            df = _load_savings_csv(data_dir=data_dir)
            _, _, s = savings_cap_comparison_bar(df, et_mode=args.et_mode, out_dir=out_dir)
            print(f"Cap comparison: {s['overall_means']}")

        if args.frequency:
            from es_analysis.charts.water_savings.late_cut_frequency_bar import late_cut_frequency_bar
            df = _load_base_csv(data_dir=data_dir)
            _, _, s = late_cut_frequency_bar(df, et_mode=args.et_mode, out_dir=out_dir)
            print(f"Frequency: {s['distribution']}")

        if args.heatmap:
            from es_analysis.charts.water_savings.savings_heatmap import savings_heatmap
            df = _load_county_wy_summary(data_dir=data_dir)
            for cap in [0, 1]:
                _, _, s = savings_heatmap(df, cap_k=cap, et_mode=args.et_mode, out_dir=out_dir)
                print(f"Heatmap cap={cap}: mean={s['overall_mean']:.4f}")

        if args.by_wy:
            from es_analysis.charts.water_savings.savings_by_wy_bar import savings_by_wy_bar_combined
            df = _load_savings_csv(data_dir=data_dir)
            _, _, s = savings_by_wy_bar_combined(df, cap_values=(0, 1), et_mode=args.et_mode, out_dir=out_dir)
            print(f"By WY combined: caps={s['cap_values']}")

        if args.et_share:
            from es_analysis.charts.water_savings.late_et_share_bar import late_et_share_bar
            df = _load_base_csv(data_dir=data_dir)
            for g in ["county", "WY"]:
                _, _, s = late_et_share_bar(df, group_by=g, et_mode=args.et_mode, out_dir=out_dir)
                print(f"ET share by {g}: mean={s['overall_mean_pct']:.1f}%")

        if args.all_counties:
            from es_analysis.charts.water_savings.savings_by_all_counties_bar import savings_by_all_counties_bar
            df = _load_savings_csv(data_dir=data_dir)
            _, _, s = savings_by_all_counties_bar(df, cap_values=(0, 1), et_mode=args.et_mode, out_dir=out_dir)
            print(f"All-counties: {s.get('cap0_by_county', {})}")

        if args.cutoff:
            from es_analysis.charts.water_savings.sensitivity_heatmap import (
                cutoff_sensitivity_figure,
            )
            df_sav = _load_savings_csv(data_dir=data_dir)
            df_base = _load_base_csv(data_dir=data_dir)
            _, _, s = cutoff_sensitivity_figure(
                df_base, df_savings=df_sav, out_dir=out_dir,
            )
            print(f"Cutoff sensitivity: {s['n_cells']} cells "
                  f"(non-sparse: {s['n_non_sparse']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
