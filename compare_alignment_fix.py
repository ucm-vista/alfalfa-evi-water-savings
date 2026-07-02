#!/usr/bin/env python3
"""Generate before/after comparison boxplots for alignment fix.

Produces side-by-side boxplots using pre_realign (before) and
current (after) BEAST CSVs for a given county.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from es_analysis.data_providers.config import config
from es_analysis.data_providers.evi_provider import normalize_county_name
from es_analysis.data_providers.parcel_summary_provider import build_multicounty_matched
from es_analysis.data_providers.beast_provider import BEASTDataProvider


def load_beast_cuttings(county: str, wy: int, suffix: str = "") -> pd.DataFrame:
    """Load BEAST CSV and return parcel_id + n_cuttings."""
    county_norm = normalize_county_name(county)
    name = f"beast_seasonal_cuts_WY{wy}{suffix}.csv"
    csv_path = config.beast_out_root_new / county_norm.replace("_", " ") / name
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df[["parcel_id", "n_cuttings"]].copy()


def build_summary(county: str, wys: list, suffix: str = "") -> pd.DataFrame:
    """Build parcel-year summary using BEAST CSVs with given suffix."""
    frames = []
    for wy in wys:
        bc = load_beast_cuttings(county, wy, suffix)
        if bc.empty:
            continue
        bc["WY"] = wy
        bc["county"] = county
        frames.append(bc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    county = "San Joaquin"
    wys = list(range(2019, 2025))
    out_dir = Path("es_analysis/output/figures/alfalfa_run_3/test_et_gdd")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BEFORE (pre_realign) data...")
    before = build_summary(county, wys, suffix="_pre_realign")
    print(f"  {len(before)} rows")

    print("Loading AFTER (fixed) data...")
    after = build_summary(county, wys, suffix="")
    print(f"  {len(after)} rows")

    if before.empty or after.empty:
        print("ERROR: missing data")
        return

    # Now build full pipeline summaries for before/after
    # We need ET and GDD5 from the multicounty_matched builder
    # For the AFTER case, it'll use the current (repaired) CSVs
    print("\nBuilding AFTER multicounty summary...")
    df_after = build_multicounty_matched(
        wy_start=2019, wy_end=2024,
        counties=[county], et_mode="actual",
    )
    print(f"  {len(df_after)} rows, median ET={df_after['et_cum_minET_to_last_cut_mm'].median():.1f}")

    # For BEFORE: swap CSVs back temporarily
    county_norm = normalize_county_name(county)
    beast_dir = config.beast_out_root_new / county_norm.replace("_", " ")
    print("\nSwapping to pre_realign CSVs for BEFORE summary...")
    for wy in wys:
        cur = beast_dir / f"beast_seasonal_cuts_WY{wy}.csv"
        bak = beast_dir / f"beast_seasonal_cuts_WY{wy}_pre_realign.csv"
        tmp = beast_dir / f"beast_seasonal_cuts_WY{wy}_fixed.csv"
        if cur.exists() and bak.exists():
            cur.rename(tmp)
            bak.rename(cur)

    df_before = build_multicounty_matched(
        wy_start=2019, wy_end=2024,
        counties=[county], et_mode="actual",
    )
    print(f"  {len(df_before)} rows, median ET={df_before['et_cum_minET_to_last_cut_mm'].median():.1f}")

    # Swap back
    print("Restoring fixed CSVs...")
    for wy in wys:
        cur = beast_dir / f"beast_seasonal_cuts_WY{wy}.csv"
        tmp = beast_dir / f"beast_seasonal_cuts_WY{wy}_fixed.csv"
        bak = beast_dir / f"beast_seasonal_cuts_WY{wy}_pre_realign.csv"
        if tmp.exists():
            cur.rename(bak)
            tmp.rename(cur)

    # --- Generate comparison plots ---
    et_col = "et_cum_minET_to_last_cut_mm"
    gdd_col = "gdd5_mean"
    cut_col = "n_cuttings"

    YEAR_COLORS = {
        2019: "#E69F00", 2020: "#56B4E9", 2021: "#009E73",
        2022: "#F0E442", 2023: "#0072B2", 2024: "#D55E00",
    }

    for val_col, xlabel, fname in [
        (et_col, "Cumulative segment ET (actual, mm)", "boxplot_et"),
        (gdd_col, "Cumulative GDD5 (\u00b0C\u00b7day)", "boxplot_gdd5"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Unified cut values across both panels
        all_cut_vals = sorted(set(
            df_before[cut_col].dropna().unique().astype(int).tolist()
            + df_after[cut_col].dropna().unique().astype(int).tolist()
        ))

        for ax_idx, (df, label) in enumerate([
            (df_before, "BEFORE (early-exit tolerance)"),
            (df_after, "AFTER (max-matches tolerance)"),
        ]):
            ax = axes[ax_idx]
            cut_vals = sorted(df[cut_col].dropna().unique().astype(int))

            # Data points behind
            for wy, color in YEAR_COLORS.items():
                sub = df[df["WY"] == wy]
                if sub.empty:
                    continue
                jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(sub))
                ax.scatter(sub[val_col], sub[cut_col].values + jitter,
                           s=6, alpha=0.35, color=color, edgecolor="none",
                           label=str(wy), zorder=2)

            # Box-and-whiskers on top
            box_data = [df[df[cut_col] == c][val_col].dropna().values for c in cut_vals]
            ax.boxplot(box_data, positions=cut_vals, widths=0.6, vert=False,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="white", edgecolor="black",
                                     linewidth=0.8, alpha=0.85),
                       medianprops=dict(color="black", linewidth=1.2),
                       whiskerprops=dict(linewidth=0.8),
                       capprops=dict(linewidth=0.8),
                       zorder=5)

            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Number of cuttings", fontsize=9)
            ax.set_yticks(all_cut_vals)
            ax.set_yticklabels([str(c) for c in all_cut_vals])
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.7)

            # Annotate counts
            for c in cut_vals:
                n = (df[cut_col] == c).sum()
                ax.text(ax.get_xlim()[0] + 5, c + 0.35, f"n={n}",
                        fontsize=6, va="bottom", color="grey")

        n_before = len(df_before)
        n_after = len(df_after)
        n1_before = (df_before[cut_col] == 1).sum()
        n1_after = (df_after[cut_col] == 1).sum()
        fig.suptitle(
            f"{county} \u2014 Alignment Fix Comparison\n"
            f"Before: {n_before} rows, 1-cut={n1_before} ({100*n1_before/n_before:.1f}%) | "
            f"After: {n_after} rows, 1-cut={n1_after} ({100*n1_after/n_after:.1f}%)",
            fontsize=10, y=1.04,
        )
        fig.tight_layout()

        for ext in ("png", "pdf"):
            path = out_dir / f"{fname}_comparison.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir / fname}_comparison.png")

    # Summary stats
    print(f"\n=== Summary ===")
    print(f"Before: median ET={df_before[et_col].median():.1f}mm, "
          f"median GDD5={df_before[gdd_col].median():.1f}, "
          f"1-cut={n1_before} ({100*n1_before/n_before:.1f}%)")
    print(f"After:  median ET={df_after[et_col].median():.1f}mm, "
          f"median GDD5={df_after[gdd_col].median():.1f}, "
          f"1-cut={n1_after} ({100*n1_after/n_after:.1f}%)")


if __name__ == "__main__":
    main()
