"""EVI plots for anomaly parcels: raw observations, gap-filled, smoothed, with cutting dates.

For each anomaly parcel, produces a single-panel figure showing:
  - Scatter: raw EVI observations
  - Line: quartic gap-filled EVI
  - Line: Whittaker-smoothed EVI
  - Vertical dashed lines at BEAST-detected cutting dates
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from es_analysis.data_providers.beast_provider import BEASTDataProvider
from es_analysis.data_providers.config import config
from es_analysis.data_providers.evi_provider import EviDataProvider, normalize_county_name
from es_analysis.utils.publication_style import (
    SINGLE_COL_WIDTH,
    WONG_PALETTE,
    apply_style,
    save_pub_figure,
)


def plot_anomaly_evi(
    uid: str,
    county: str,
    wy: int,
    n_cuttings: int,
    anomaly_type: str,
    output_dir: Path,
    evi_provider: Optional[EviDataProvider] = None,
    beast_provider: Optional[BEASTDataProvider] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """Plot EVI time series for a single anomaly parcel.

    Args:
        uid: Parcel UniqueID.
        county: County name.
        wy: Water year.
        n_cuttings: Number of detected cuttings.
        anomaly_type: Anomaly classification string.
        output_dir: Base output directory (test/{County}/WY{wy}/).
        evi_provider: Reusable EviDataProvider (created if None).
        beast_provider: Reusable BEASTDataProvider (created if None).

    Returns:
        (fig, ax, info_dict)
    """
    apply_style()

    county_norm = normalize_county_name(county)

    # --- Load EVI data ---
    if evi_provider is None:
        evi_provider = EviDataProvider()
        evi_provider.load_data()

    daily_df = evi_provider.create_daily_timeseries(
        evi_provider.data, county_norm, uid, [wy]
    )

    # Quartic gap-fill (separate call to get distinct trace)
    gapfilled = EviDataProvider.quartic_gapfill(daily_df)

    # Whittaker smooth (separate call for distinct trace) — use lower lambda
    # to preserve cutting-cycle oscillations (1e2 instead of default 1e4)
    whittaker = EviDataProvider.smooth_whittaker(daily_df, lmbda=1e2)

    # --- Load BEAST cut dates ---
    if beast_provider is None:
        beast_provider = BEASTDataProvider()

    cut_dates = []
    beast_df = beast_provider.load_seasonal_cuts_csv(county_norm, wy)
    if beast_df is not None:
        uid_col = "UniqueID" if "UniqueID" in beast_df.columns else "parcel_id"
        parcel_rows = beast_df[beast_df[uid_col].astype(str) == str(uid)]
        if not parcel_rows.empty and "matched_minima_iso" in parcel_rows.columns:
            raw = parcel_rows.iloc[0]["matched_minima_iso"]
            if pd.notna(raw) and str(raw).strip():
                for ds in str(raw).split(";"):
                    ds = ds.strip()
                    if ds:
                        try:
                            cut_dates.append(pd.Timestamp(ds))
                        except ValueError:
                            pass

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 1.8, 3.0))

    dates = daily_df["date"]
    raw_evi = daily_df["mean_evi"]

    # Raw observations (scatter)
    obs_mask = raw_evi.notna()
    ax.scatter(
        dates[obs_mask], raw_evi[obs_mask],
        s=15, alpha=0.7, color=WONG_PALETTE[5],
        label="Raw observations", zorder=3,
    )

    # Gap-filled line
    ax.plot(
        dates, gapfilled,
        color=WONG_PALETTE[1], lw=1.2, alpha=0.8,
        label="Quartic gap-fill",
    )

    # Whittaker smooth line
    ax.plot(
        dates, whittaker,
        color=WONG_PALETTE[3], lw=1.8,
        label="Whittaker smooth",
    )

    # Cutting date vertical lines
    for cd in cut_dates:
        ax.axvline(cd, color=WONG_PALETTE[6], ls="--", alpha=0.7, lw=0.8)
    # Single legend entry for cut dates
    if cut_dates:
        ax.axvline(
            cut_dates[0], color=WONG_PALETTE[6], ls="--", alpha=0.0,
            label=f"Cut dates ({len(cut_dates)})",
        )

    # Text box with metadata
    info_text = f"n_cuts={n_cuttings}  type={anomaly_type}"
    ax.text(
        0.02, 0.95, info_text,
        transform=ax.transAxes, fontsize=6,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="grey"),
    )

    ax.set_title(f"{county_norm} | UID {uid} | WY{wy}", fontsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    fig.tight_layout()

    # Save
    county_dir = output_dir / county_norm / f"WY{wy}"
    county_dir.mkdir(parents=True, exist_ok=True)
    save_pub_figure(fig, f"UID{uid}_evi_combined", county_dir)

    info = {
        "uid": uid,
        "county": county_norm,
        "wy": wy,
        "n_observations": int(obs_mask.sum()),
        "n_cut_dates": len(cut_dates),
        "anomaly_type": anomaly_type,
    }

    return fig, ax, info


def generate_anomaly_evi_plots(
    anomaly_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Dict:
    """Generate EVI plots for all anomaly parcels.

    Args:
        anomaly_csv: Path to anomaly_parcels.csv. If None, uses default location.
        out_dir: Output directory for test/ plots. If None, uses default.

    Returns:
        Summary dict with per-parcel info.
    """
    if anomaly_csv is None:
        anomaly_csv = (
            Path(__file__).parent.parent.parent
            / "output" / "figures" / "test" / "anomaly_parcels.csv"
        )

    if out_dir is None:
        out_dir = Path(__file__).parent.parent.parent / "output" / "figures" / "test"

    df = pd.read_csv(anomaly_csv)
    print(f"\n--- Generating anomaly EVI plots for {len(df)} parcels ---")

    # Shared providers for efficiency
    evi_provider = EviDataProvider()
    evi_provider.load_data()
    beast_provider = BEASTDataProvider()

    results = []
    for _, row in df.iterrows():
        uid = str(row["UniqueID"])
        county = str(row["county"])
        wy = int(row["WY"])
        n_cuts = int(row["n_cuttings"])
        atype = str(row["anomaly_type"])

        print(f"  {county} WY{wy} UID={uid} ({atype})", end=" ... ", flush=True)
        try:
            _, _, info = plot_anomaly_evi(
                uid=uid,
                county=county,
                wy=wy,
                n_cuttings=n_cuts,
                anomaly_type=atype,
                output_dir=out_dir,
                evi_provider=evi_provider,
                beast_provider=beast_provider,
            )
            results.append(info)
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({"uid": uid, "county": county, "wy": wy, "error": str(exc)})

    summary = {
        "n_total": len(df),
        "n_success": sum(1 for r in results if "error" not in r),
        "n_failed": sum(1 for r in results if "error" in r),
        "results": results,
    }

    print(f"\n  Done: {summary['n_success']} succeeded, {summary['n_failed']} failed")
    return summary


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    generate_anomaly_evi_plots()
