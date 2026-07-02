"""Spatial data provider for parcel shapefiles and county boundaries.

Handles loading and filtering of parcel geometries and county boundary
polygons, with area computation in acres via CA Albers (EPSG:3310).

Source: alfalfa_evi_jovyan.py lines 9250-9276, 16944-16987,
        county_map_and_ridgelines_plot.py lines 462-509
"""

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import geopandas as gpd

from .config import config
from .evi_provider import normalize_county_name
from ..utils.units import M2_PER_ACRE


# Canonical N-to-S county display order
COUNTY_ORDER = [
    "San Joaquin", "Stanislaus", "Merced", "Madera", "Fresno",
    "Tulare", "Kings", "Kern", "Riverside", "Imperial",
]


def load_parcels_for_county(
    county: str,
    parcel_shp: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Load parcel geometries for a single county.

    Reads the parcel shapefile, filters to the requested county,
    normalizes the UniqueID column, and reprojects to EPSG:4326.

    Args:
        county: County name (will be normalized).
        parcel_shp: Path to parcel shapefile. Defaults to config.parcel_shp.

    Returns:
        GeoDataFrame with columns [UniqueID, COUNTY_norm, geometry].
    """
    if parcel_shp is None:
        parcel_shp = config.parcel_shp

    county_norm = normalize_county_name(county)
    gdf = gpd.read_file(parcel_shp)

    if "COUNTY" not in gdf.columns:
        raise ValueError("Parcel shapefile must have a 'COUNTY' column.")

    if gdf.crs is None:
        print("[warn] Parcel shapefile has no CRS; assuming EPSG:4269 (NAD83).")
        gdf.set_crs(epsg=4269, inplace=True)
    gdf = gdf.to_crs(epsg=4326)

    gdf["COUNTY_norm"] = (
        gdf["COUNTY"]
        .astype(str)
        .str.replace("_", " ")
        .str.strip()
        .str.title()
    )
    gdf = gdf[gdf["COUNTY_norm"] == county_norm].copy()

    if "UniqueID" in gdf.columns:
        gdf["UniqueID"] = gdf["UniqueID"].astype(str)
    elif "parcel_id" in gdf.columns:
        gdf["UniqueID"] = gdf["parcel_id"].astype(str)
    else:
        raise ValueError("Parcel shapefile must have 'UniqueID' or 'parcel_id'.")

    return gdf[["UniqueID", "COUNTY_norm", "geometry"]].copy()


def load_parcels_area_acres(
    counties: Iterable[str],
    parcel_shp: Optional[Path] = None,
) -> pd.DataFrame:
    """Load parcel geometries and compute area in acres.

    Uses CA Albers (EPSG:3310) for accurate area calculation in m^2,
    then converts to acres.

    Args:
        counties: Iterable of county names to include.
        parcel_shp: Path to parcel shapefile. Defaults to config.parcel_shp.

    Returns:
        DataFrame with columns [UniqueID, county, area_acres],
        deduplicated by (UniqueID, county).
    """
    if parcel_shp is None:
        parcel_shp = config.parcel_shp

    gdf = gpd.read_file(parcel_shp)

    if "COUNTY" not in gdf.columns:
        raise ValueError("Parcel shapefile must have a 'COUNTY' column.")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4269)
    gdf = gdf.to_crs(epsg=4326)

    if "UniqueID" in gdf.columns:
        gdf["UniqueID"] = gdf["UniqueID"].astype(str)
    elif "parcel_id" in gdf.columns:
        gdf["UniqueID"] = gdf["parcel_id"].astype(str)
    else:
        raise ValueError("Parcel shapefile must have 'UniqueID' or 'parcel_id'.")

    gdf["county"] = (
        gdf["COUNTY"]
        .astype(str)
        .str.replace("_", " ")
        .str.strip()
        .str.title()
    )

    keep = {normalize_county_name(c) for c in counties}
    gdf = gdf[gdf["county"].isin(keep)].copy()

    # area: use CA Albers (EPSG:3310) for m^2 then convert to acres
    g_area = gdf.to_crs(epsg=3310)
    area_m2 = g_area.geometry.area.astype(float)
    area_acres = area_m2 / M2_PER_ACRE

    out = pd.DataFrame({
        "UniqueID": gdf["UniqueID"].astype(str).values,
        "county": gdf["county"].astype(str).values,
        "area_acres": area_acres.values,
    }).drop_duplicates(subset=["UniqueID", "county"])

    return out


def county_land_area_sqmi(
    county_boundary_shp: Optional[Path] = None,
    counties: Optional[List[str]] = None,
) -> dict:
    """Return {county: land_area_sq_mi} from the county boundary shapefile.

    Uses the census ALAND attribute (land area in m^2) when present;
    falls back to geometry area in EPSG:3310.
    """
    if county_boundary_shp is None:
        county_boundary_shp = config.county_boundary_shp
    if counties is None:
        counties = COUNTY_ORDER

    p = Path(county_boundary_shp)
    if not p.exists():
        return {}

    gdf = gpd.read_file(p)
    name_col = next(
        (c for c in ["NAME", "COUNTY", "County", "Name", "COUNTY_NAME"] if c in gdf.columns),
        None,
    )
    if name_col is None:
        return {}

    gdf["_norm"] = gdf[name_col].astype(str).str.replace("_", " ").str.strip().str.title()
    keep = {normalize_county_name(c) for c in counties}
    gdf = gdf[gdf["_norm"].isin(keep)].copy()

    if "ALAND" in gdf.columns:
        area_m2 = gdf["ALAND"].astype(float)
    else:
        g = gdf.to_crs(epsg=3310) if gdf.crs is not None else gdf
        area_m2 = g.geometry.area.astype(float)

    SQM_PER_SQMI = 2_589_988.110336
    sqmi = (area_m2 / SQM_PER_SQMI).values
    return {n: float(a) for n, a in zip(gdf["_norm"].values, sqmi)}


def load_county_boundaries(
    county_boundary_shp: Optional[Path] = None,
    counties: Optional[List[str]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Load county boundary polygons from shapefile.

    Reprojects to EPSG:4326 and filters to the requested counties
    (defaults to COUNTY_ORDER).

    Args:
        county_boundary_shp: Path to county boundary shapefile.
            Defaults to config.county_boundary_shp.
        counties: List of county names to include.
            Defaults to COUNTY_ORDER.

    Returns:
        GeoDataFrame with columns [COUNTY_norm, geometry],
        or None if the file is not found or cannot be loaded.
    """
    if county_boundary_shp is None:
        county_boundary_shp = config.county_boundary_shp
    if counties is None:
        counties = COUNTY_ORDER

    p = Path(county_boundary_shp)
    if not p.exists():
        print(f"[warn] County boundary shapefile not found: {p}")
        return None

    try:
        gdf = gpd.read_file(p)

        # find the county name column
        possible_cols = [
            "COUNTY", "County", "NAME", "Name",
            "COUNTY_NAM", "COUNTY_NAME",
        ]
        name_col = None
        for c in possible_cols:
            if c in gdf.columns:
                name_col = c
                break
        if name_col is None:
            print(
                f"[warn] County boundary shapefile has no recognizable "
                f"name column; columns: {list(gdf.columns)}"
            )
            return None

        gdf["COUNTY_norm"] = (
            gdf[name_col]
            .astype(str)
            .str.replace("_", " ")
            .str.strip()
            .str.title()
        )
        counties_norm = [normalize_county_name(c) for c in counties]
        gdf = gdf[gdf["COUNTY_norm"].isin(counties_norm)].copy()

        if gdf.crs is None:
            print("[warn] County boundary shapefile has no CRS; using as-is.")
        else:
            gdf = gdf.to_crs(epsg=4326)

        return gdf[["COUNTY_norm", "geometry"]].copy()
    except Exception as e:
        print(f"[warn] Error loading county boundaries: {e}")
        return None
