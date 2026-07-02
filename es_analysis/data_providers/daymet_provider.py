"""Daymet netCDF data provider for climate variables and GDD computation.

Handles loading Daymet v4 daily netCDF files, mapping parcels to the
Daymet grid, and computing spatially-aggregated climate metrics over
parcel-specific cut-cycle segments.

Source: alfalfa_evi_jovyan.py lines 9228-9727, 11600-11992
Uses the latest/most refined version of each function.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

from .config import config
from .evi_provider import water_year_bounds
from .evi_cut_window_provider import compute_daymet_window_dates


# ---------------------------------------------------------------------------
# NetCDF file loading
# ---------------------------------------------------------------------------

def open_local_nc(path: Path) -> xr.Dataset:
    """Open a local netCDF4 file with decode_times fallback.

    Args:
        path: Path to the netCDF file.

    Returns:
        xarray Dataset.
    """
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_times=True)
    except Exception:
        return xr.open_dataset(path, engine="netcdf4", decode_times=False)


def open_daymet_file(
    var: str,
    year: int,
    root: Optional[Path] = None,
) -> xr.Dataset:
    """Find and open a Daymet v4 daily netCDF file for a given variable and year.

    Searches for files matching ``daymet_v4_daily_na_{var}_{year}_*.nc``.

    Args:
        var: Daymet variable name (e.g. "tmax", "tmin", "prcp").
        year: Calendar year.
        root: Directory to search. Defaults to config.daymet_root.

    Returns:
        xarray Dataset.
    """
    if root is None:
        root = Path(config.daymet_root)
    matches = sorted(root.glob(f"daymet_v4_daily_na_{var}_{year}_*.nc"))
    if not matches:
        raise FileNotFoundError(
            f"No Daymet file found for {var} {year} under {root}"
        )
    return open_local_nc(matches[0])


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def extract_lat_lon_2d(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 2D lat/lon reference arrays from a Daymet dataset.

    Handles 2D and 3D (time, y, x) coordinate arrays.

    Args:
        ds: xarray Dataset with 'lat' and 'lon' variables.

    Returns:
        Tuple of (lat_ref, lon_ref) as 2D numpy arrays.
    """
    if "lat" not in ds or "lon" not in ds:
        raise ValueError("Daymet dataset missing 'lat'/'lon' coordinates.")
    lat_ref = ds["lat"].values
    lon_ref = ds["lon"].values
    if lat_ref.ndim == 3:
        lat_ref = lat_ref[0]
    if lon_ref.ndim == 3:
        lon_ref = lon_ref[0]
    return lat_ref, lon_ref


def approx_buffer_deg(
    lat_ref: np.ndarray,
    lon_ref: np.ndarray,
) -> float:
    """Estimate Daymet grid cell size in degrees for spatial buffering.

    Args:
        lat_ref: 2D latitude array (ny, nx).
        lon_ref: 2D longitude array (ny, nx).

    Returns:
        Approximate cell size in degrees.
    """
    ny, nx = lat_ref.shape
    dlat_candidates, dlon_candidates = [], []
    if ny > 1:
        dlat_candidates.append(
            np.nanmedian(np.abs(lat_ref[1:, :] - lat_ref[:-1, :]))
        )
        dlon_candidates.append(
            np.nanmedian(np.abs(lon_ref[1:, :] - lon_ref[:-1, :]))
        )
    if nx > 1:
        dlat_candidates.append(
            np.nanmedian(np.abs(lat_ref[:, 1:] - lat_ref[:, :-1]))
        )
        dlon_candidates.append(
            np.nanmedian(np.abs(lon_ref[:, 1:] - lon_ref[:, :-1]))
        )
    cell_deg = float(
        np.nanmax([np.nanmax(dlat_candidates), np.nanmax(dlon_candidates)])
    )
    return cell_deg


def build_parcel_to_indices(
    parcels: gpd.GeoDataFrame,
    lat_ref: np.ndarray,
    lon_ref: np.ndarray,
    buffer_deg: float,
) -> Dict[str, np.ndarray]:
    """Map parcel geometries to Daymet grid flat indices.

    For each parcel, buffers the geometry by buffer_deg, finds candidate
    grid cells within the bounding box, and does a point-in-polygon test.

    Args:
        parcels: GeoDataFrame with 'UniqueID' and 'geometry' columns.
        lat_ref: 2D latitude array (ny, nx).
        lon_ref: 2D longitude array (ny, nx).
        buffer_deg: Buffer distance in degrees.

    Returns:
        Dict mapping UniqueID -> array of flat grid indices.
    """
    flat_lat = lat_ref.ravel()
    flat_lon = lon_ref.ravel()

    uid_to_geom = (
        parcels.drop_duplicates(subset="UniqueID")
        .set_index("UniqueID")["geometry"]
    )
    parcel_to_idx: Dict[str, np.ndarray] = {}

    for uid, geom in uid_to_geom.items():
        if geom is None or geom.is_empty:
            continue

        poly = geom.buffer(buffer_deg)
        minx, miny, maxx, maxy = poly.bounds

        bbox_mask = (
            (flat_lon >= minx) & (flat_lon <= maxx)
            & (flat_lat >= miny) & (flat_lat <= maxy)
        )
        idx_candidates = np.where(bbox_mask)[0]
        if idx_candidates.size == 0:
            continue

        inside_idx: List[int] = []
        for i in idx_candidates:
            if poly.contains(Point(float(flat_lon[i]), float(flat_lat[i]))):
                inside_idx.append(int(i))

        if inside_idx:
            parcel_to_idx[str(uid)] = np.array(inside_idx, dtype=int)

    return parcel_to_idx


# ---------------------------------------------------------------------------
# Daily Daymet loading for a water year
# ---------------------------------------------------------------------------

def load_daymet_daily_flat_for_wy(
    var: str,
    wy: int,
    root: Optional[Path] = None,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, List, object]:
    """Load daily Daymet variable for a water year, flattened spatially.

    Args:
        var: Daymet variable name.
        wy: Water year.
        root: Daymet data root directory.

    Returns:
        Tuple of (time_wy, vals_flat, lat_ref, lon_ref, dsets, ds_all).
          - time_wy: DatetimeIndex (nt,)
          - vals_flat: ndarray (nt, ny*nx)
          - lat_ref, lon_ref: 2D arrays
          - dsets: list of opened datasets (for cleanup)
          - ds_all: concatenated dataset (for cleanup)
    """
    if root is None:
        root = Path(config.daymet_root)

    wy_start, wy_end = water_year_bounds(wy)
    years = sorted({wy_start.year, wy_end.year})

    dsets = [open_daymet_file(var, y, root=root) for y in years]
    ref = dsets[0]
    if var not in ref.variables:
        for ds in dsets:
            try:
                ds.close()
            except Exception:
                pass
        raise KeyError(
            f"Variable '{var}' not found in Daymet file for year {years[0]}"
        )

    lat_ref, lon_ref = extract_lat_lon_2d(ref)

    if len(dsets) == 1:
        ds_all = ref
    else:
        ds_all = xr.concat(
            dsets, dim="time",
            data_vars="minimal", coords="minimal", compat="override",
        )

    time_vals = ds_all["time"].values
    if not np.issubdtype(time_vals.dtype, np.datetime64):
        raise ValueError(
            "Daymet time axis is not datetime64. Ensure decode_times works."
        )

    time_all = pd.to_datetime(time_vals).normalize()
    mask_wy = (time_all >= wy_start) & (time_all <= wy_end)
    if not mask_wy.any():
        raise ValueError(f"No Daymet timestamps in WY{wy} for {var}.")

    idx = np.where(mask_wy)[0]
    time_wy = pd.DatetimeIndex(time_all[mask_wy])

    da = ds_all[var].isel(time=idx)
    vals = da.values
    if vals.ndim != 3:
        raise ValueError(f"Unexpected Daymet dims for {var}: {vals.shape}")

    nt, ny, nx = vals.shape
    if lat_ref.shape != (ny, nx) or lon_ref.shape != (ny, nx):
        raise ValueError(f"Lat/lon shape mismatch for {var} in WY{wy}.")

    vals_flat = vals.reshape(nt, ny * nx)
    return time_wy, vals_flat, lat_ref, lon_ref, dsets, ds_all


def _close_datasets(dsets: List, ds_all=None):
    """Helper to close xarray datasets safely."""
    if ds_all is not None:
        try:
            ds_all.close()
        except Exception:
            pass
    for ds in dsets:
        try:
            ds.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Daymet aggregation: mean over segments
# ---------------------------------------------------------------------------

def compute_daymet_mean_for_parcels_over_segments(
    var: str,
    wy: int,
    parcels: gpd.GeoDataFrame,
    parcel_segments: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    root: Optional[Path] = None,
) -> Dict[str, float]:
    """Compute parcel-mean Daymet variable averaged over union of segment dates.

    Args:
        var: Daymet variable name.
        wy: Water year.
        parcels: GeoDataFrame with parcel geometries.
        parcel_segments: Dict mapping UniqueID -> list of (start, end) segments.
        root: Daymet data root directory.

    Returns:
        Dict mapping UniqueID -> mean value.
    """
    time_wy, vals_flat, lat_ref, lon_ref, dsets, ds_all = (
        load_daymet_daily_flat_for_wy(var, wy, root=root)
    )
    buffer_deg = approx_buffer_deg(lat_ref, lon_ref)
    p2i = build_parcel_to_indices(parcels, lat_ref, lon_ref, buffer_deg)

    results: Dict[str, float] = {}

    for uid, segs in parcel_segments.items():
        uid = str(uid)
        if uid not in p2i or not segs:
            results[uid] = np.nan
            continue

        mask = np.zeros(len(time_wy), dtype=bool)
        for s, e in segs:
            if pd.isna(s) or pd.isna(e) or s > e:
                continue
            mask |= (
                (time_wy >= pd.to_datetime(s))
                & (time_wy <= pd.to_datetime(e))
            )

        if not mask.any():
            results[uid] = np.nan
            continue

        idx_space = p2i[uid]
        daily_parcel = vals_flat[:, idx_space].mean(axis=1)
        results[uid] = float(np.nanmean(daily_parcel[mask]))

    _close_datasets(dsets, ds_all)
    return results


def compute_daymet_mean_for_parcels(
    var: str,
    wy: int,
    parcels: gpd.GeoDataFrame,
    month_start: int,
    month_end: int,
    parcel_windows: Optional[
        Dict[str, Union[
            Tuple[pd.Timestamp, pd.Timestamp],
            List[Tuple[pd.Timestamp, pd.Timestamp]],
        ]]
    ] = None,
    root: Optional[Path] = None,
) -> Dict[str, float]:
    """Compute parcel-mean Daymet variable, flexible windowing.

    If parcel_windows is None, uses the fixed month window
    (month_start..month_end) and computes temporal mean then spatial mean.

    If parcel_windows is provided (uid -> segments or single window),
    uses segment-aligned daily aggregation.

    Args:
        var: Daymet variable name.
        wy: Water year.
        parcels: GeoDataFrame with parcel geometries.
        month_start: Start month for fixed windowing (1-12).
        month_end: End month for fixed windowing (1-12).
        parcel_windows: Optional per-parcel segment windows.
        root: Daymet data root directory.

    Returns:
        Dict mapping UniqueID -> mean value.
    """
    if root is None:
        root = Path(config.daymet_root)

    if parcel_windows is None:
        # Fixed month-window: temporal mean then spatial mean
        start_date, end_date = compute_daymet_window_dates(
            wy, month_start, month_end
        )
        years = sorted({start_date.year, end_date.year})

        dsets = [open_daymet_file(var, y, root=root) for y in years]
        ref = dsets[0]

        if var not in ref.variables:
            _close_datasets(dsets)
            raise KeyError(
                f"Variable '{var}' not found in Daymet file for year {years[0]}"
            )

        lat_ref, lon_ref = extract_lat_lon_2d(ref)

        if len(dsets) == 1:
            ds_all = ref
        else:
            ds_all = xr.concat(
                dsets, dim="time",
                data_vars="minimal", coords="minimal", compat="override",
            )

        time_vals = ds_all["time"].values
        if not np.issubdtype(time_vals.dtype, np.datetime64):
            _close_datasets(dsets, ds_all)
            raise ValueError(
                "Daymet time axis is not datetime64. Ensure decode_times works."
            )

        time = pd.to_datetime(time_vals).normalize()
        mask_t = (time >= start_date) & (time <= end_date)
        if not mask_t.any():
            _close_datasets(dsets, ds_all)
            raise ValueError(
                f"No Daymet timestamps in range "
                f"{start_date.date()}-{end_date.date()} for {var}."
            )

        da = ds_all[var].sel(time=mask_t).mean(dim="time")  # 2D
        vals = da.values
        if vals.ndim != 2:
            _close_datasets(dsets, ds_all)
            raise ValueError(
                f"Unexpected Daymet dims for {var}: {vals.shape}"
            )

        ny, nx = vals.shape
        if lat_ref.shape != (ny, nx) or lon_ref.shape != (ny, nx):
            _close_datasets(dsets, ds_all)
            raise ValueError(
                f"Lat/lon shape {lat_ref.shape} does not match "
                f"data shape {(ny, nx)} for {var}."
            )

        flat_vals = vals.ravel()
        buffer_deg = approx_buffer_deg(lat_ref, lon_ref)
        p2i = build_parcel_to_indices(parcels, lat_ref, lon_ref, buffer_deg)

        results: Dict[str, float] = {}
        for uid, idx_space in p2i.items():
            arr = flat_vals[idx_space]
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            results[str(uid)] = float(arr.mean())

        _close_datasets(dsets, ds_all)
        return results

    # Segment-aligned: daily aggregation per parcel
    time_wy, vals_flat, lat_ref, lon_ref, dsets, ds_all = (
        load_daymet_daily_flat_for_wy(var, wy, root=root)
    )
    buffer_deg = approx_buffer_deg(lat_ref, lon_ref)
    p2i = build_parcel_to_indices(parcels, lat_ref, lon_ref, buffer_deg)

    results = {}

    for uid, w in parcel_windows.items():
        uid = str(uid)
        if uid not in p2i:
            results[uid] = np.nan
            continue

        if isinstance(w, tuple) and len(w) == 2:
            segs = [w]
        else:
            segs = list(w)

        if not segs:
            results[uid] = np.nan
            continue

        mask = np.zeros(len(time_wy), dtype=bool)
        for s, e in segs:
            if pd.isna(s) or pd.isna(e) or s > e:
                continue
            s = pd.to_datetime(s).normalize()
            e = pd.to_datetime(e).normalize()
            mask |= (time_wy >= s) & (time_wy <= e)

        if not mask.any():
            results[uid] = np.nan
            continue

        idx_space = p2i[uid]
        daily_parcel = vals_flat[:, idx_space].mean(axis=1)
        results[uid] = float(np.nanmean(daily_parcel[mask]))

    _close_datasets(dsets, ds_all)
    return results


# ---------------------------------------------------------------------------
# GDD5 computation over segments
# ---------------------------------------------------------------------------

def compute_gdd5_for_parcels_over_segments(
    wy: int,
    parcels: gpd.GeoDataFrame,
    parcel_segments: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    tbase: float = None,
    tmin_floor: float = None,
    tmax_cap: float = None,
    root: Optional[Path] = None,
) -> Dict[str, float]:
    """Compute GDD5 per parcel as mean across cut-cycle segment sums.

    For each parcel:
      - For each segment: sum daily parcel-mean GDD in that segment
      - Parcel value = mean of per-segment sums

    Args:
        wy: Water year.
        parcels: GeoDataFrame with parcel geometries.
        parcel_segments: Dict mapping UniqueID -> list of (start, end) segments.
        tbase: GDD base temperature (C). Defaults to config.gdd_base_c.
        tmin_floor: Minimum temperature floor (C). Defaults to config.gdd_tmin_floor_c.
        tmax_cap: Maximum temperature cap (C). Defaults to config.gdd_tmax_cap_c.
        root: Daymet data root directory.

    Returns:
        Dict mapping UniqueID -> mean GDD5 across segments.
    """
    if tbase is None:
        tbase = config.gdd_base_c
    if tmin_floor is None:
        tmin_floor = config.gdd_tmin_floor_c
    if tmax_cap is None:
        tmax_cap = config.gdd_tmax_cap_c
    if root is None:
        root = Path(config.daymet_root)

    wy_start, wy_end = water_year_bounds(wy)
    years = sorted({wy_start.year, wy_end.year})

    dsets_tmax = [open_daymet_file("tmax", y, root=root) for y in years]
    dsets_tmin = [open_daymet_file("tmin", y, root=root) for y in years]

    ref = dsets_tmax[0]
    lat_ref, lon_ref = extract_lat_lon_2d(ref)

    ds_tmax_all = xr.concat(
        dsets_tmax, dim="time",
        data_vars="minimal", coords="minimal", compat="override",
    )
    ds_tmin_all = xr.concat(
        dsets_tmin, dim="time",
        data_vars="minimal", coords="minimal", compat="override",
    )

    time_all = pd.to_datetime(ds_tmax_all["time"].values).normalize()
    mask_wy = (time_all >= wy_start) & (time_all <= wy_end)
    if not mask_wy.any():
        _close_datasets(dsets_tmax + dsets_tmin, ds_tmax_all)
        try:
            ds_tmin_all.close()
        except Exception:
            pass
        raise ValueError(f"No Daymet timestamps in WY{wy} for tmin/tmax.")

    idx = np.where(mask_wy)[0]
    time_wy = pd.DatetimeIndex(time_all[mask_wy])

    da_tmax = ds_tmax_all["tmax"].isel(time=idx)
    da_tmin = ds_tmin_all["tmin"].isel(time=idx)

    # Alfalfa GDD clamps
    tmax_c = da_tmax.clip(min=tmin_floor, max=tmax_cap)
    tmin_c = da_tmin.clip(min=tmin_floor, max=tmax_cap)
    tavg = (tmax_c + tmin_c) / 2.0
    daily_gdd = (tavg - tbase).clip(min=0.0)

    gdd_vals = daily_gdd.values
    if gdd_vals.ndim != 3:
        raise ValueError(f"Unexpected GDD5 dims: {gdd_vals.shape}")

    nt, ny, nx = gdd_vals.shape
    if lat_ref.shape != (ny, nx) or lon_ref.shape != (ny, nx):
        raise ValueError(f"Lat/lon shape mismatch for GDD5 WY{wy}.")

    gdd_flat = gdd_vals.reshape(nt, ny * nx)

    buffer_deg = approx_buffer_deg(lat_ref, lon_ref)
    p2i = build_parcel_to_indices(parcels, lat_ref, lon_ref, buffer_deg)

    results: Dict[str, float] = {}

    for uid, segs in parcel_segments.items():
        uid = str(uid)
        if uid not in p2i or not segs:
            results[uid] = np.nan
            continue

        idx_space = p2i[uid]
        daily_parcel_gdd = gdd_flat[:, idx_space].mean(axis=1)

        seg_sums: List[float] = []
        for s, e in segs:
            if pd.isna(s) or pd.isna(e) or s > e:
                continue
            m = (
                (time_wy >= pd.to_datetime(s))
                & (time_wy <= pd.to_datetime(e))
            )
            if not m.any():
                continue
            seg_sums.append(float(np.nansum(daily_parcel_gdd[m])))

        results[uid] = float(np.nanmean(seg_sums)) if seg_sums else np.nan

    _close_datasets(dsets_tmax + dsets_tmin, ds_tmax_all)
    try:
        ds_tmin_all.close()
    except Exception:
        pass

    return results


def compute_daily_gdd5_for_parcels(
    wy: int,
    parcels: gpd.GeoDataFrame,
    tbase: float = None,
    tmin_floor: float = None,
    tmax_cap: float = None,
    root: Optional[Path] = None,
) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series]]:
    """Compute daily GDD5 time series per parcel for a water year.

    Returns daily GDD5 as a pd.Series per parcel, suitable for backward
    accumulation in thermal-time segment estimation.

    Args:
        wy: Water year.
        parcels: GeoDataFrame with parcel geometries.
        tbase: GDD base temperature (C). Defaults to config.gdd_base_c.
        tmin_floor: Minimum temperature floor (C).
        tmax_cap: Maximum temperature cap (C).
        root: Daymet data root directory.

    Returns:
        Tuple of (time_index, gdd_dict) where gdd_dict maps
        UniqueID -> pd.Series of daily GDD5 indexed by date.
    """
    if tbase is None:
        tbase = config.gdd_base_c
    if tmin_floor is None:
        tmin_floor = config.gdd_tmin_floor_c
    if tmax_cap is None:
        tmax_cap = config.gdd_tmax_cap_c
    if root is None:
        root = Path(config.daymet_root)

    wy_start, wy_end = water_year_bounds(wy)
    years = sorted({wy_start.year, wy_end.year})

    dsets_tmax = [open_daymet_file("tmax", y, root=root) for y in years]
    dsets_tmin = [open_daymet_file("tmin", y, root=root) for y in years]

    ref = dsets_tmax[0]
    lat_ref, lon_ref = extract_lat_lon_2d(ref)

    ds_tmax_all = xr.concat(
        dsets_tmax, dim="time",
        data_vars="minimal", coords="minimal", compat="override",
    )
    ds_tmin_all = xr.concat(
        dsets_tmin, dim="time",
        data_vars="minimal", coords="minimal", compat="override",
    )

    time_all = pd.to_datetime(ds_tmax_all["time"].values).normalize()
    mask_wy = (time_all >= wy_start) & (time_all <= wy_end)
    idx_wy = np.where(mask_wy)[0]
    time_wy = pd.DatetimeIndex(time_all[mask_wy])

    da_tmax = ds_tmax_all["tmax"].isel(time=idx_wy)
    da_tmin = ds_tmin_all["tmin"].isel(time=idx_wy)

    tmax_c = da_tmax.clip(min=tmin_floor, max=tmax_cap)
    tmin_c = da_tmin.clip(min=tmin_floor, max=tmax_cap)
    tavg = (tmax_c + tmin_c) / 2.0
    daily_gdd = (tavg - tbase).clip(min=0.0)

    gdd_vals = daily_gdd.values
    nt, ny, nx = gdd_vals.shape
    gdd_flat = gdd_vals.reshape(nt, ny * nx)

    buffer_deg = approx_buffer_deg(lat_ref, lon_ref)
    p2i = build_parcel_to_indices(parcels, lat_ref, lon_ref, buffer_deg)

    results: Dict[str, pd.Series] = {}
    for uid, idx_space in p2i.items():
        daily_parcel_gdd = gdd_flat[:, idx_space].mean(axis=1)
        results[str(uid)] = pd.Series(daily_parcel_gdd, index=time_wy, dtype=float)

    _close_datasets(dsets_tmax + dsets_tmin, ds_tmax_all)
    try:
        ds_tmin_all.close()
    except Exception:
        pass

    return time_wy, results


def compute_gdd5_for_parcels_cut_window(
    wy: int,
    parcels: gpd.GeoDataFrame,
    parcel_windows: Dict[str, Union[
        Tuple[pd.Timestamp, pd.Timestamp],
        List[Tuple[pd.Timestamp, pd.Timestamp]],
    ]],
    tbase: float = None,
    tmin_floor: float = None,
    tmax_cap: float = None,
    root: Optional[Path] = None,
) -> Dict[str, float]:
    """Compute GDD5 per parcel summed over union of segment dates.

    Unlike compute_gdd5_for_parcels_over_segments (which averages across
    segment sums), this function sums GDD over the union of all segment
    dates per parcel.

    Args:
        wy: Water year.
        parcels: GeoDataFrame with parcel geometries.
        parcel_windows: Dict mapping uid -> segments or single (start, end).
        tbase: GDD base temperature (C).
        tmin_floor: Minimum temperature floor (C).
        tmax_cap: Maximum temperature cap (C).
        root: Daymet data root directory.

    Returns:
        Dict mapping UniqueID -> total GDD5 over segments.
    """
    if tbase is None:
        tbase = config.gdd_base_c
    if tmin_floor is None:
        tmin_floor = config.gdd_tmin_floor_c
    if tmax_cap is None:
        tmax_cap = config.gdd_tmax_cap_c
    if root is None:
        root = Path(config.daymet_root)

    wy_start, wy_end = water_year_bounds(wy)
    years = sorted({wy_start.year, wy_end.year})

    dsets_tmax = [open_daymet_file("tmax", y, root=root) for y in years]
    dsets_tmin = [open_daymet_file("tmin", y, root=root) for y in years]

    ref = dsets_tmax[0]
    lat_ref, lon_ref = extract_lat_lon_2d(ref)

    ds_tmax_all = xr.concat(
        dsets_tmax, dim="time",
        data_vars="minimal", coords="minimal", compat="override",
    )
    ds_tmin_all = xr.concat(
        dsets_tmin, dim="time",
        data_vars="minimal", coords="minimal", compat="override",
    )

    time_vals = ds_tmax_all["time"].values
    if not np.issubdtype(time_vals.dtype, np.datetime64):
        _close_datasets(dsets_tmax + dsets_tmin, ds_tmax_all)
        try:
            ds_tmin_all.close()
        except Exception:
            pass
        raise ValueError(
            "Daymet time axis is not datetime64 for tmax/tmin."
        )

    time_all = pd.to_datetime(time_vals).normalize()
    mask_wy = (time_all >= wy_start) & (time_all <= wy_end)
    if not mask_wy.any():
        _close_datasets(dsets_tmax + dsets_tmin, ds_tmax_all)
        try:
            ds_tmin_all.close()
        except Exception:
            pass
        raise ValueError(f"No Daymet timestamps in WY{wy} for tmin/tmax.")

    idx_wy = np.where(mask_wy)[0]
    time_wy = pd.DatetimeIndex(time_all[mask_wy])

    da_tmax = ds_tmax_all["tmax"].isel(time=idx_wy)
    da_tmin = ds_tmin_all["tmin"].isel(time=idx_wy)

    tmax_c = da_tmax.clip(min=tmin_floor, max=tmax_cap)
    tmin_c = da_tmin.clip(min=tmin_floor, max=tmax_cap)
    tavg = (tmax_c + tmin_c) / 2.0
    daily_gdd = (tavg - tbase).clip(min=0.0)

    gdd_vals = daily_gdd.values
    if gdd_vals.ndim != 3:
        raise ValueError(f"Unexpected GDD array dims: {gdd_vals.shape}")

    nt, ny, nx = gdd_vals.shape
    if lat_ref.shape != (ny, nx) or lon_ref.shape != (ny, nx):
        raise ValueError(
            f"Lat/lon shape {lat_ref.shape} does not match GDD data {(ny, nx)}."
        )

    gdd_flat = gdd_vals.reshape(nt, ny * nx)

    buffer_deg = approx_buffer_deg(lat_ref, lon_ref)
    p2i = build_parcel_to_indices(parcels, lat_ref, lon_ref, buffer_deg)

    results: Dict[str, float] = {}

    for uid, w in parcel_windows.items():
        uid = str(uid)
        if uid not in p2i:
            results[uid] = np.nan
            continue

        if isinstance(w, tuple) and len(w) == 2:
            segs = [w]
        else:
            segs = list(w)

        if not segs:
            results[uid] = np.nan
            continue

        mask = np.zeros(len(time_wy), dtype=bool)
        for s, e in segs:
            if pd.isna(s) or pd.isna(e) or s > e:
                continue
            s = max(pd.to_datetime(s).normalize(), wy_start)
            e = min(pd.to_datetime(e).normalize(), wy_end)
            if s > e:
                continue
            mask |= (time_wy >= s) & (time_wy <= e)

        if not mask.any():
            results[uid] = np.nan
            continue

        idx_space = p2i[uid]
        daily_parcel_gdd = gdd_flat[:, idx_space].mean(axis=1)
        results[uid] = float(np.nansum(daily_parcel_gdd[mask]))

    _close_datasets(dsets_tmax + dsets_tmin, ds_tmax_all)
    try:
        ds_tmin_all.close()
    except Exception:
        pass

    return results
