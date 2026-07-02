"""Plotting utilities for EVI visualization."""

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Matplotlib configuration
mpl.rcParams.update({
    "figure.figsize": (12, 10),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_series(daily_df: pd.DataFrame, filled: pd.Series, smoothed: pd.Series,
                county: str, parcel: str, years: List[int], win_days: int,
                sg_w: int, sg_p: int, output_dir: Path) -> Path:
    """Plot EVI time series with original, gap-filled, and smoothed values.

    Args:
        daily_df: DataFrame with 'date' and 'mean_evi' columns containing original daily data.
        filled: Series with gap-filled EVI values.
        smoothed: Series with smoothed EVI values.
        county: County name for the plot title.
        parcel: Parcel ID for the plot title.
        years: List of water years for the plot title.
        win_days: Gap-fill window size for the plot title.
        sg_w: Savitzky-Golay window size for the plot title.
        sg_p: Savitzky-Golay polynomial order for the plot title.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot file.

    Raises:
        ValueError: If daily_df is empty.
    """
    from .helper import water_year_bounds_multi

    if daily_df.empty:
        raise ValueError("No data available for the selected filters.")

    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()

    ax.plot(daily_df["date"], daily_df["mean_evi"], label="Original (raw)", linewidth=1.2)
    raw_mask = ~daily_df["mean_evi"].isna()
    ax.scatter(daily_df.loc[raw_mask, "date"], daily_df.loc[raw_mask, "mean_evi"],
               s=12, alpha=0.9, label="Raw obs (points)")

    ax.plot(daily_df["date"], filled, label="Gap-filled (quartic)", linewidth=1.4, alpha=0.9)
    ax.plot(daily_df["date"], smoothed, label="Smoothed (SG)", linewidth=1.6, alpha=0.95)

    start, end = water_year_bounds_multi(years)
    ax.set_xlim(start, end)

    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    title = f"{county} | parcel {parcel} | WY {min(years)}–{max(years)} | win={win_days}d, SG={sg_w}/{sg_p}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / (title.replace(" ", "_").replace("/", "_").replace("|", "_") + ".png")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.show()
    plt.close(fig)
    return outfile


def plot_evi_with_cps(df_parcel: pd.DataFrame, cp_dates: pd.DatetimeIndex,
                      county: str, wy: int, parcel: str,
                      output_dir: Path, interp_window: int = 30,
                      sg_window: int = 15, sg_poly: int = 3,
                      label_suffix: str = "") -> Path:
    """Plot EVI time series with vertical lines at change point dates.

    Args:
        df_parcel: DataFrame with EVI data including original_mean_evi, gapfilled_mean_evi,
                   and smoothed_mean_evi columns.
        cp_dates: DatetimeIndex of change point dates to mark.
        county: County name for the plot title.
        wy: Water year for the plot title.
        parcel: Parcel ID for the plot title.
        output_dir: Directory to save the plot.
        interp_window: Gap-fill window size for the plot title.
        sg_window: Savitzky-Golay window size for the plot title.
        sg_poly: Savitzky-Golay polynomial order for the plot title.
        label_suffix: Additional label text for the plot title.

    Returns:
        Path to the saved plot file.
    """
    from .helper import water_year_bounds

    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()

    ax.plot(df_parcel["date"], df_parcel["original_mean_evi"], label="Original (raw)", linewidth=1.0)
    m = ~df_parcel["original_mean_evi"].isna()
    ax.scatter(df_parcel.loc[m, "date"], df_parcel.loc[m, "original_mean_evi"],
               s=10, alpha=0.9, label="Raw obs")
    ax.plot(df_parcel["date"], df_parcel["gapfilled_mean_evi"],
            label="Gap-filled (quartic)", linewidth=1.2, alpha=0.9)
    ax.plot(df_parcel["date"], df_parcel["smoothed_mean_evi"],
            label="Smoothed (SG)", linewidth=1.5, alpha=0.95)

    for d in cp_dates:
        ax.axvline(d, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8)

    start, end = water_year_bounds(wy)
    ax.set_xlim(start, end)
    ax.set_xlabel("Date")
    ax.set_ylabel("EVI")
    ax.set_title(f"{county} | Parcel {parcel} | WY {wy} (EVI + CPs{label_suffix})")
    ax.legend()
    ax.grid(True, alpha=0.2)

    outfile = output_dir / f"WY{wy}_parcel_{parcel}_EVI_CP.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.show()
    plt.close(fig)
    return outfile


def plot_decomposition(result, county: str, wy: int, parcel: str, output_dir: Optional[Path] = None):
    """Plot BEAST decomposition display.

    Args:
        result: BEAST result object from Rbeast.beast().
        county: County name for the plot title.
        wy: Water year for the plot title.
        parcel: Parcel ID for the plot title.
        output_dir: Optional directory to save the plot. If None, plot is not saved.
    """
    try:
        import Rbeast as rb
    except ImportError:
        raise ImportError("Rbeast package is required for decomposition plots.")

    plt.figure(figsize=(12, 10))
    rb.plot(result, ncpStat='median')
    plt.suptitle(f"BEAST EVI Decomposition\nCounty={county}, WY={wy}, Parcel={parcel}",
                 y=0.98, fontsize=14)
    plt.subplots_adjust(top=0.90, hspace=0.0, wspace=0.0)
    plt.show()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        outfile = output_dir / f"WY{wy}_parcel_{parcel}_BEAST_decomp.png"
        plt.savefig(outfile, dpi=150)
        plt.close()
    else:
        plt.show()
        from IPython.display import display
        display(result)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to the specified path.

    Args:
        fig: Matplotlib figure to save.
        path: Path where the figure should be saved.
        dpi: DPI for the saved image.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")


def plot_county_year_boxplots(county: str, years: List[int], output_dir: Path) -> Optional[Path]:
    """Generate boxplots of n_cuttings distribution by water year for a county.

    Args:
        county: County name to plot.
        years: List of water years to include.
        output_dir: Directory containing seasonal CSVs and to save the plot.

    Returns:
        Path to the saved plot file, or None if no data found.
    """
    from .helper import norm_county_name

    county_norm = norm_county_name(county)
    cdir = output_dir / county_norm
    files = [cdir / f"beast_seasonal_cuts_WY{y}.csv" for y in years]
    data, ys = [], []
    for y, f in zip(years, files):
        if f.exists():
            d = pd.read_csv(f)
            vals = d["n_cuttings"].dropna().astype(int).values
            if vals.size:
                data.append(vals)
                ys.append(y)

    if not data:
        print(f"No seasonal CSVs for {county_norm} across {list(years)}.")
        return None

    import numpy as np
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(ys) + 2)))
    bp = ax.boxplot(data, vert=False, tick_labels=[str(y) for y in ys], patch_artist=True,
                    showfliers=False, whis=(0, 100))
    for patch in bp['boxes']:
        patch.set(facecolor="#dddddd", alpha=0.85, edgecolor="#aaaaaa")
    for whisk in bp['whiskers']:
        whisk.set(color="#999999")
    for cap in bp['caps']:
        cap.set(color="#999999")
    for i, vals in enumerate(data, start=1):
        mu = float(np.mean(vals))
        ax.scatter([mu], [i], s=28, c="black", zorder=5)

    ax.set_title(f"{county_norm}: Yearly Cutting Distribution (grey=min..max; •=mean)")
    ax.set_xlabel("Cuttings")
    ax.set_ylabel("Water Year")
    fig.tight_layout()
    out = cdir / "seasonal_cuts_boxplot.png"
    fig.savefig(out, dpi=150)
    plt.show()
    plt.close(fig)
    print("Saved boxplot:", out)
    return out