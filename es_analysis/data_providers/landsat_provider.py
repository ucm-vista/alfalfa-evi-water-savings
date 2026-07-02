"""Landsat provider module for satellite pass data.
Source lines: 50-54, 720, 2232
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from .config import config
from .evi_provider import normalize_county_name, water_year_bounds


class LandsatDataProvider:
    """Provider for Landsat satellite pass data."""

    def __init__(
        self,
        landsat_meta_csv: Optional[Path] = None,
    ):
        """Initialize Landsat data provider.

        Args:
            landsat_meta_csv: Path to Landsat metadata CSV
        """
        self.landsat_meta_csv = landsat_meta_csv or config.landsat_meta_csv

        if not self.landsat_meta_csv.exists():
            raise FileNotFoundError(f"Landsat metadata CSV not found: {self.landsat_meta_csv}")

    def load_landsat_passes(
        self,
        county: str,
        wy: int,
        cloud_cover_max: Optional[float] = None,
        min_spacing_days: int = 15,
        thin_track: bool = True,
    ) -> pd.DataFrame:
        """Load Landsat passes for a county and water year.

        Args:
            county: County name
            wy: Water year
            cloud_cover_max: Maximum cloud cover percentage (None = no filter)
            min_spacing_days: Minimum spacing between passes for thinning
            thin_track: Whether to thin to dominant WRS path/row track

        Returns:
            DataFrame with Landsat pass information
        """
        county_norm = normalize_county_name(county)

        df = pd.read_csv(self.landsat_meta_csv, sep=None, engine="python")

        low = {c.lower(): c for c in df.columns}
        needed = {"date_only", "cloud_cover", "county"}

        if not needed.issubset(set(low)):
            raise ValueError(f"Landsat CSV missing: {needed}")

        df = df.rename(
            columns={
                low["date_only"]: "date_only",
                low["cloud_cover"]: "cloud_cover",
                low["county"]: "county",
            }
        )

        df["county"] = df["county"].astype(str).str.strip().str.title()
        df["date_only"] = pd.to_datetime(df["date_only"], errors="coerce").dt.normalize()
        df["cloud_cover"] = pd.to_numeric(df["cloud_cover"], errors="coerce")

        start, end = water_year_bounds(wy)
        df = df.loc[
            df["county"].eq(county_norm) & df["date_only"].between(start, end)
        ].copy()

        if df.empty:
            return df

        has_wrs = {"wrs_path", "wrs_row"}.issubset(df.columns)

        if has_wrs:
            df["wrs_path"] = pd.to_numeric(df["wrs_path"], errors="coerce")
            df["wrs_row"] = pd.to_numeric(df["wrs_row"], errors="coerce")
            df = df.dropna(subset=["wrs_path", "wrs_row"])

            if not df.empty:
                grp = (
                    df.groupby(["wrs_path", "wrs_row"], as_index=False)
                    .agg(n=("date_only", "nunique"), med_cc=("cloud_cover", "median"))
                    .sort_values(["n", "med_cc"], ascending=[False, True])
                )
                best = grp.iloc[0]
                df = df[
                    (df["wrs_path"] == best["wrs_path"]) &
                    (df["wrs_row"] == best["wrs_row"])
                ].copy()

        if cloud_cover_max is not None:
            df = df.loc[df["cloud_cover"] <= float(cloud_cover_max)].copy()
            if df.empty:
                return df

        if has_wrs and not df.empty:
            df = df.drop_duplicates(subset=["date_only", "wrs_path", "wrs_row"])

        df = df.sort_values(["date_only", "cloud_cover"]).drop_duplicates(
            subset=["date_only"], keep="first"
        )

        if thin_track:
            selected = []
            last_date = None

            for _, row in df.iterrows():
                d = row["date_only"]
                if last_date is None or (d - last_date).days >= min_spacing_days:
                    selected.append(row)
                    last_date = d

            df = (
                pd.DataFrame(selected).reset_index(drop=True)
                if selected
                else df.iloc[0:0].copy()
            )

        keep_cols = ["date_only", "cloud_cover"]
        if has_wrs:
            keep_cols += ["wrs_path", "wrs_row"]

        return df[keep_cols].copy()

    @staticmethod
    def get_pass_dates(df: pd.DataFrame) -> np.ndarray:
        """Extract pass dates from Landsat DataFrame.

        Args:
            df: DataFrame from load_landsat_passes

        Returns:
            Array of Timestamp dates
        """
        if df.empty:
            return np.array([], dtype="datetime64[ns]")

        return pd.to_datetime(df["date_only"].unique())

    def get_first_clear_pass_after(
        self,
        county: str,
        wy: int,
        h: pd.Timestamp,
        cloud_cover_max: float,
    ) -> tuple[Optional[pd.Timestamp], Optional[float]]:
        """Get first clear Landsat pass after a date.

        Args:
            county: County name
            wy: Water year
            h: Reference date
            cloud_cover_max: Maximum cloud cover

        Returns:
            Tuple of (pass_date, cloud_cover)
        """
        passes_df = self.load_landsat_passes(
            county, wy, cloud_cover_max=cloud_cover_max, thin_track=True
        )
        pass_dates = self.get_pass_dates(passes_df)

        if pass_dates.size == 0:
            return (None, None)

        after = pass_dates[pass_dates > h]
        if after.size == 0:
            return (None, None)

        p = pd.to_datetime(after.min()).normalize()
        cc = passes_df.loc[
            pd.to_datetime(passes_df["date_only"]).dt.normalize().eq(p), "cloud_cover"
        ]

        return (p, float(cc.iloc[0]) if len(cc) else None)

    def get_first_any_pass_after(
        self,
        county: str,
        wy: int,
        h: pd.Timestamp,
    ) -> Optional[pd.Timestamp]:
        """Get first Landsat pass (any cloud cover) after a date.

        Args:
            county: County name
            wy: Water year
            h: Reference date

        Returns:
            Pass date or None
        """
        passes_df = self.load_landsat_passes(
            county, wy, cloud_cover_max=None, thin_track=False
        )
        pass_dates = self.get_pass_dates(passes_df)

        if pass_dates.size == 0:
            return None

        after = pass_dates[pass_dates > h]
        if after.size == 0:
            return None

        return pd.to_datetime(after.min()).normalize()