"""BEAST provider module for cutting detection.
Source lines: 811-1178, 1180-1880
"""

import json
import math
import os
import gc
import sys
import shutil
import tempfile
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Optional
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import config
from .evi_provider import normalize_county_name, water_year_bounds


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Module-level BEAST helpers (subprocess child script, numpy loader)
# ---------------------------------------------------------------------------

_CHILD_SCRIPT_PATH: Optional[str] = None
_BATCH_CHILD_SCRIPT_PATH: Optional[str] = None


def _ensure_beast_child_script() -> str:
    """Create (or reuse) a temporary Python script for subprocess BEAST."""
    global _CHILD_SCRIPT_PATH
    if _CHILD_SCRIPT_PATH and os.path.exists(_CHILD_SCRIPT_PATH):
        return _CHILD_SCRIPT_PATH
    code = r"""
import sys, os, json, numpy as np
import Rbeast as rb

def main():
    inp = sys.argv[1]; outdir = sys.argv[2]; comp = sys.argv[3]
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    os.makedirs(outdir, exist_ok=True)
    try:
        x = np.load(inp).astype(float)
        x = np.clip(x, -0.05, 1.05)
        if not np.isfinite(x).any():
            for nm in ("cp","cpPr","occ"): np.save(os.path.join(outdir, f"{comp}_{nm}.npy"), np.array([], float))
            open(os.path.join(outdir, "status.txt"), "w").write("empty"); return
        if not np.isfinite(x).all():
            idx = np.arange(x.size); good = np.isfinite(x)
            x = np.interp(idx, idx[good], x[good])
        if x.size < 60 or float(x.std()) < 1e-8:
            for nm in ("cp","cpPr","occ"): np.save(os.path.join(outdir, f"{comp}_{nm}.npy"), np.array([], float))
            open(os.path.join(outdir, "status.txt"), "w").write("unsafe"); return
        x = x + 1e-6*np.random.default_rng(0).normal(size=x.size)
        kwargs = dict(period=365.0, quiet=1); kwargs.update(params or {})
        res = rb.beast(x, **kwargs)
        blk = getattr(res, comp, None) if hasattr(res, comp) else res.get(comp, None) if isinstance(res, dict) else None
        def _get(name):
            if blk is None: return np.array([], float)
            try: v = getattr(blk, name, None)
            except Exception: v = None
            if v is None and isinstance(blk, dict): v = blk.get(name, [])
            return np.asarray(v if v is not None else [], float).ravel()
        np.save(os.path.join(outdir, f"{comp}_cp.npy"),   _get("cp"))
        np.save(os.path.join(outdir, f"{comp}_cpPr.npy"), _get("cpPr"))
        np.save(os.path.join(outdir, f"{comp}_occ.npy"),  _get("cpOccPr"))
        open(os.path.join(outdir, "status.txt"), "w").write("ok")
    except Exception as e:
        try: open(os.path.join(outdir, "status.txt"), "w").write(f"exception:{type(e).__name__}:{e}")
        except Exception: pass
        raise
if __name__ == "__main__":
    main()
"""
    p = Path(tempfile.gettempdir()) / "beast_child_runner.py"
    p.write_text(code, encoding="utf-8")
    _CHILD_SCRIPT_PATH = str(p)
    return _CHILD_SCRIPT_PATH


def _ensure_batch_child_script() -> str:
    """Create (or reuse) a batch BEAST subprocess script.

    Runs all sweep configs with a single Python startup + Rbeast import.
    Each config runs in a forked child process for SIGFPE isolation —
    if Rbeast crashes on one config, the others still complete.
    """
    global _BATCH_CHILD_SCRIPT_PATH
    if _BATCH_CHILD_SCRIPT_PATH and os.path.exists(_BATCH_CHILD_SCRIPT_PATH):
        return _BATCH_CHILD_SCRIPT_PATH
    code = r"""
import sys, os, json, numpy as np, multiprocessing as mp

def run_one(x, comp, params, outdir):
    import Rbeast as rb
    os.makedirs(outdir, exist_ok=True)
    try:
        kwargs = dict(period=365.0, quiet=1)
        kwargs.update(params or {})
        res = rb.beast(x, **kwargs)
        blk = getattr(res, comp, None) if hasattr(res, comp) else res.get(comp, None) if isinstance(res, dict) else None
        def _get(name):
            if blk is None: return np.array([], float)
            try: v = getattr(blk, name, None)
            except Exception: v = None
            if v is None and isinstance(blk, dict): v = blk.get(name, [])
            return np.asarray(v if v is not None else [], float).ravel()
        np.save(os.path.join(outdir, f"{comp}_cp.npy"),   _get("cp"))
        np.save(os.path.join(outdir, f"{comp}_cpPr.npy"), _get("cpPr"))
        np.save(os.path.join(outdir, f"{comp}_occ.npy"),  _get("cpOccPr"))
        open(os.path.join(outdir, "status.txt"), "w").write("ok")
    except Exception as e:
        for nm in ("cp","cpPr","occ"):
            np.save(os.path.join(outdir, f"{comp}_{nm}.npy"), np.array([], float))
        try: open(os.path.join(outdir, "status.txt"), "w").write(f"exception:{type(e).__name__}:{e}")
        except Exception: pass

def main():
    mp.set_start_method("fork", force=True)
    inp = sys.argv[1]; basedir = sys.argv[2]; comp = sys.argv[3]
    configs_file = sys.argv[4]
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    configs = json.loads(open(configs_file).read())
    x = np.load(inp).astype(float)
    x = np.clip(x, -0.05, 1.05)
    if not np.isfinite(x).any():
        for i in range(len(configs)):
            outdir = os.path.join(basedir, str(offset + i), comp)
            os.makedirs(outdir, exist_ok=True)
            for nm in ("cp","cpPr","occ"):
                np.save(os.path.join(outdir, f"{comp}_{nm}.npy"), np.array([], float))
            open(os.path.join(outdir, "status.txt"), "w").write("empty")
        return
    if not np.isfinite(x).all():
        idx = np.arange(x.size); good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good])
    if x.size < 60 or float(x.std()) < 1e-8:
        for i in range(len(configs)):
            outdir = os.path.join(basedir, str(offset + i), comp)
            os.makedirs(outdir, exist_ok=True)
            for nm in ("cp","cpPr","occ"):
                np.save(os.path.join(outdir, f"{comp}_{nm}.npy"), np.array([], float))
            open(os.path.join(outdir, "status.txt"), "w").write("unsafe")
        return
    x = x + 1e-6*np.random.default_rng(0).normal(size=x.size)
    # Fork a child for each config — isolates SIGFPE crashes
    for i, params in enumerate(configs):
        outdir = os.path.join(basedir, str(offset + i), comp)
        p = mp.Process(target=run_one, args=(x, comp, params, outdir))
        p.start()
        p.join(timeout=600)
        if p.is_alive():
            p.kill(); p.join()

if __name__ == "__main__":
    main()
"""
    p = Path(tempfile.gettempdir()) / "beast_batch_runner.py"
    p.write_text(code, encoding="utf-8")
    _BATCH_CHILD_SCRIPT_PATH = str(p)
    return _BATCH_CHILD_SCRIPT_PATH


def _np_load_or_empty(p: Path) -> np.ndarray:
    """Load a .npy file, returning an empty float array on failure."""
    try:
        return np.load(p, allow_pickle=False)
    except Exception:
        return np.array([], dtype=float)


# ---------------------------------------------------------------------------
# BEAST execution backends
# ---------------------------------------------------------------------------

def _beast_run_subproc(
    series: pd.Series, component: str, params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run BEAST in a subprocess for stability (avoids Rbeast segfaults)."""
    comp = component if component in {"season", "trend"} else "season"
    idx = series.index
    x = series.to_numpy(dtype=float)

    if (np.isfinite(x).sum() < 60) or (np.nanstd(x) < 1e-8):
        return dict(
            status="unsafe", reason="too_short_or_flat",
            cp_idx=np.array([], int),
            cp_dates=pd.DatetimeIndex([]),
            cp_pr=np.array([], float),
            occ=pd.Series([], dtype=float, index=idx),
        )

    wrk = Path(tempfile.mkdtemp(prefix="rb_child_"))
    inp = wrk / "in.npy"
    outdir = wrk / "out" / comp
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(inp, np.asarray(x, float))
    child = _ensure_beast_child_script()

    try:
        subprocess.run(
            [sys.executable, child, str(inp), str(outdir), comp,
             json.dumps(params or {})],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=600, check=False,
        )
        status_txt = (
            (outdir / "status.txt").read_text().strip()
            if (outdir / "status.txt").exists() else "unknown"
        )
        cp = _np_load_or_empty(outdir / f"{comp}_cp.npy")
        cpPr = _np_load_or_empty(outdir / f"{comp}_cpPr.npy")
        occ = _np_load_or_empty(outdir / f"{comp}_occ.npy")

        if cp.size:
            finite = np.isfinite(cp)
            cp = cp[finite]
            if cpPr.size == finite.size:
                cpPr = cpPr[finite]

        if cp.size:
            n = len(idx)
            cp_idx = np.rint(cp).astype(int) - 1
            keep = (cp_idx >= 0) & (cp_idx < n)
            cp_idx = cp_idx[keep]
            cp_dates = idx[cp_idx]
            cp_pr = (
                cpPr[keep] if cpPr.size == cp.size
                else np.full(cp_idx.shape, np.nan)
            )
        else:
            cp_idx = np.array([], int)
            cp_dates = pd.DatetimeIndex([])
            cp_pr = np.array([], float)

        if status_txt.startswith("ok"):
            status = "ok"
        elif status_txt.startswith("unsafe"):
            status = "unsafe"
        elif status_txt.startswith("exception"):
            status = "exception"
        else:
            status = "unknown"

        return dict(
            status=status, reason=status_txt,
            cp_idx=cp_idx, cp_dates=cp_dates, cp_pr=cp_pr,
            occ=pd.Series(
                occ[:len(idx)] if occ.size else np.zeros(len(idx), float),
                index=idx, dtype=float,
            ),
        )
    except subprocess.TimeoutExpired:
        return dict(
            status="exception", reason="timeout",
            cp_idx=np.array([], int),
            cp_dates=pd.DatetimeIndex([]),
            cp_pr=np.array([], float),
            occ=pd.Series(np.zeros(len(idx)), index=idx),
        )
    except Exception as e:
        return dict(
            status="exception", reason=f"{type(e).__name__}:{e}",
            cp_idx=np.array([], int),
            cp_dates=pd.DatetimeIndex([]),
            cp_pr=np.array([], float),
            occ=pd.Series(np.zeros(len(idx)), index=idx),
        )
    finally:
        shutil.rmtree(wrk, ignore_errors=True)


def _beast_run_direct(
    series: pd.Series, component: str, params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run BEAST directly via Rbeast import (faster but less stable)."""
    try:
        import Rbeast as rb
    except Exception as e:
        return dict(
            status="exception", reason=f"Rbeast_import:{e}",
            cp_idx=np.array([], int),
            cp_dates=pd.DatetimeIndex([]),
            cp_pr=np.array([], float),
            occ=pd.Series(np.zeros(len(series)), index=series.index),
        )
    comp = component if component in {"season", "trend"} else "season"
    x = series.to_numpy(dtype=float)
    idx = series.index

    if (np.isfinite(x).sum() < 60) or (np.nanstd(x) < 1e-8):
        return dict(
            status="unsafe", reason="too_short_or_flat",
            cp_idx=np.array([], int),
            cp_dates=pd.DatetimeIndex([]),
            cp_pr=np.array([], float),
            occ=pd.Series([], dtype=float, index=idx),
        )

    try:
        res = rb.beast(x, period=365.0, quiet=1, **(params or {}))
        blk = (
            getattr(res, comp, None) if hasattr(res, comp)
            else res.get(comp, None) if isinstance(res, dict)
            else None
        )

        def _get(name):
            if blk is None:
                return np.array([], float)
            try:
                v = getattr(blk, name, None)
            except Exception:
                v = None
            if v is None and isinstance(blk, dict):
                v = blk.get(name, [])
            return np.asarray(v if v is not None else [], float).ravel()

        cp, cpPr, occ = _get("cp"), _get("cpPr"), _get("cpOccPr")

        if cp.size:
            finite = np.isfinite(cp)
            cp = cp[finite]
            if cpPr.size == finite.size:
                cpPr = cpPr[finite]

        if cp.size:
            n = len(idx)
            cp_idx = np.rint(cp).astype(int) - 1
            keep = (cp_idx >= 0) & (cp_idx < n)
            cp_idx = cp_idx[keep]
            cp_dates = idx[cp_idx]
            cp_pr = (
                cpPr[keep] if cpPr.size == cp.size
                else np.full(cp_idx.shape, np.nan)
            )
        else:
            cp_idx = np.array([], int)
            cp_dates = pd.DatetimeIndex([])
            cp_pr = np.array([], float)

        return dict(
            status="ok", reason="ok",
            cp_idx=cp_idx, cp_dates=cp_dates, cp_pr=cp_pr,
            occ=pd.Series(
                occ[:len(idx)] if occ.size else np.zeros(len(idx), float),
                index=idx, dtype=float,
            ),
            result=res,
        )
    except Exception as e:
        return dict(
            status="exception", reason=f"{type(e).__name__}:{e}",
            cp_idx=np.array([], int),
            cp_dates=pd.DatetimeIndex([]),
            cp_pr=np.array([], float),
            occ=pd.Series(np.zeros(len(idx)), index=idx),
        )


def _beast_run(
    series: pd.Series, component: str, params: Dict[str, Any],
    mode: str = "subproc",
) -> Dict[str, Any]:
    """Dispatch to subprocess or direct BEAST backend."""
    if mode == "subproc":
        return _beast_run_subproc(series, component, params)
    return _beast_run_direct(series, component, params)


def _read_batch_results(
    wrk: Path, comp: str, idx: pd.DatetimeIndex,
    start: int, count: int,
) -> List[Dict[str, Any]]:
    """Read BEAST batch results from numbered subdirectories."""
    results = []
    for i in range(start, start + count):
        outdir = wrk / str(i) / comp
        status_file = outdir / "status.txt"

        if not status_file.exists():
            results.append(dict(
                status="exception", reason="batch_incomplete",
                cp_idx=np.array([], int),
                cp_dates=pd.DatetimeIndex([]),
                cp_pr=np.array([], float),
                occ=pd.Series(np.zeros(len(idx)), index=idx),
            ))
            continue

        status_txt = status_file.read_text().strip()
        cp = _np_load_or_empty(outdir / f"{comp}_cp.npy")
        cpPr = _np_load_or_empty(outdir / f"{comp}_cpPr.npy")
        occ = _np_load_or_empty(outdir / f"{comp}_occ.npy")

        if cp.size:
            finite = np.isfinite(cp)
            cp = cp[finite]
            if cpPr.size == finite.size:
                cpPr = cpPr[finite]

        if cp.size:
            n = len(idx)
            cp_idx = np.rint(cp).astype(int) - 1
            keep = (cp_idx >= 0) & (cp_idx < n)
            cp_idx = cp_idx[keep]
            cp_dates = idx[cp_idx]
            cp_pr = (
                cpPr[keep] if cpPr.size == cp.size
                else np.full(cp_idx.shape, np.nan)
            )
        else:
            cp_idx = np.array([], int)
            cp_dates = pd.DatetimeIndex([])
            cp_pr = np.array([], float)

        if status_txt.startswith("ok"):
            status = "ok"
        elif status_txt.startswith("unsafe"):
            status = "unsafe"
        elif status_txt.startswith("exception"):
            status = "exception"
        else:
            status = "unknown"

        results.append(dict(
            status=status, reason=status_txt,
            cp_idx=cp_idx, cp_dates=cp_dates, cp_pr=cp_pr,
            occ=pd.Series(
                occ[:len(idx)] if occ.size else np.zeros(len(idx), float),
                index=idx, dtype=float,
            ),
        ))
    return results


def _beast_run_batch_subproc(
    series: pd.Series,
    component: str,
    param_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run all BEAST sweep configs in a single subprocess with fork isolation.

    One Python startup + Rbeast import, then each config runs in a forked
    child process.  SIGFPE in one config doesn't affect others.
    16x less startup overhead than individual subprocesses.
    """
    comp = component if component in {"season", "trend"} else "season"
    idx = series.index
    x = series.to_numpy(dtype=float)
    n_configs = len(param_list)

    if (np.isfinite(x).sum() < 60) or (np.nanstd(x) < 1e-8):
        empty = dict(
            status="unsafe", reason="too_short_or_flat",
            cp_idx=np.array([], int),
            cp_dates=pd.DatetimeIndex([]),
            cp_pr=np.array([], float),
            occ=pd.Series([], dtype=float, index=idx),
        )
        return [empty] * n_configs

    wrk = Path(tempfile.mkdtemp(prefix="rb_batch_"))
    inp = wrk / "in.npy"
    configs_file = wrk / "configs.json"
    np.save(inp, np.asarray(x, float))
    configs_file.write_text(json.dumps(param_list))
    child = _ensure_batch_child_script()

    # Timeout: generous to handle CPU contention under parallel load.
    # Each config can take 10-120s under load; 16 configs run sequentially.
    timeout = max(600, 120 * n_configs)
    try:
        subprocess.run(
            [sys.executable, child, str(inp), str(wrk), comp,
             str(configs_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    results = _read_batch_results(wrk, comp, idx, 0, n_configs)
    shutil.rmtree(wrk, ignore_errors=True)
    return results


# ---------------------------------------------------------------------------
# Sweep parameter builders (from config)
# ---------------------------------------------------------------------------

def _build_season_sweep() -> List[Dict[str, Any]]:
    """Build SEASON_SWEEP list from config parameters.

    Uses the full cross-product of minlengths × margins (16 runs)
    instead of zip (4 runs) to give ensemble consensus enough
    statistical power to be robust against BEAST MCMC variability.
    """
    from itertools import product as _product
    sweep = []
    minlens = config.season_sweep_minlengths
    margins = config.season_sweep_leftmargins
    scp_mm = config.season_scp_minmax
    for ml, lm in _product(minlens, margins):
        sweep.append(dict(
            season="harmonic",
            sseg_minlength=ml,
            sseg_leftmargin=lm,
            scp_minmax=scp_mm,
        ))
    return sweep


def _build_trend_sweep() -> List[Dict[str, Any]]:
    """Build TREND_SWEEP list from config parameters."""
    sweep = []
    minlens = config.trend_sweep_minlengths
    margins = config.trend_sweep_leftmargins
    tcp_mm = config.trend_scp_minmax
    for ml, lm in zip(minlens, margins):
        sweep.append(dict(
            season="harmonic",
            tseg_minlength=ml,
            tseg_leftmargin=lm,
            scp_minmax=tcp_mm,
        ))
    return sweep


class BEASTDataProvider:
    """Provider for BEAST-based cutting detection."""

    def __init__(self, county_year_root: Optional[Path] = None, beast_out_root: Optional[Path] = None):
        """Initialize BEAST provider.

        Args:
            county_year_root: Root directory for county-year CSV files
            beast_out_root: Root directory for BEAST outputs
        """
        self.county_year_root = county_year_root or config.county_year_root_new
        self.beast_out_root = beast_out_root or config.beast_out_root_new
        self.beast_call_mode: str = config.beast_call_mode

    @property
    def output_dir(self) -> Path:
        """Base output directory for BEAST outputs."""
        return self.beast_out_root

    @staticmethod
    def normalize_county_name(county: str) -> str:
        """Normalize county name (delegate to evi_provider)."""
        return normalize_county_name(county)

    @staticmethod
    def water_year_bounds(wy: int):
        """Get water year start/end dates."""
        return water_year_bounds(wy)

    @staticmethod
    def get_selected_counties() -> List[str]:
        """Return list of selected counties from config."""
        return list(config.selected_counties_expanded)

    def detect_years_on_disk(self, county: str) -> List[int]:
        """Discover available water years from BEAST output CSVs on disk."""
        county_norm = normalize_county_name(county)
        county_dir = self.beast_out_root / county_norm
        if not county_dir.exists():
            return []
        years = []
        for f in county_dir.glob("beast_seasonal_cuts_WY*.csv"):
            try:
                y = int(f.stem.split("WY")[-1])
                years.append(y)
            except ValueError:
                pass
        return sorted(years)

    def load_seasonal_cuts_csv(self, county: str, wy: int) -> Optional[pd.DataFrame]:
        """Load BEAST seasonal cuts CSV for a county/year."""
        county_norm = normalize_county_name(county)
        csv_path = self.beast_out_root / county_norm / f"beast_seasonal_cuts_WY{wy}.csv"
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path)

    def gather_year_values(
        self, county: str, years: List[int], column: str = "n_cuttings"
    ) -> tuple:
        """Load a column's values from seasonal CSVs for each year.

        Returns:
            Tuple of (data_lists, valid_years) where data_lists[i] is the
            list of values for valid_years[i].
        """
        data_lists = []
        valid_years = []
        for y in years:
            df = self.load_seasonal_cuts_csv(county, y)
            if df is None or column not in df.columns:
                continue
            vals = df[column].dropna().tolist()
            if vals:
                data_lists.append(vals)
                valid_years.append(y)
        return data_lists, valid_years

    def find_trend_csv(self, county: str, span: Optional[str] = None) -> Optional[Path]:
        """Find trend CPs CSV for a county."""
        county_norm = normalize_county_name(county)
        county_dir = self.beast_out_root / county_norm
        if not county_dir.exists():
            return None
        if span:
            p = county_dir / f"beast_trend_cps_{span}.csv"
            if p.exists():
                return p
        # Try to find any trend CSV
        candidates = sorted(county_dir.glob("beast_trend_cps_*.csv"))
        return candidates[0] if candidates else None

    @staticmethod
    def explode_trend_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Convert wide-format trend CP dates to long format.

        Returns DataFrame with columns: parcel_id, cp_order, cp_date, year, doy
        """
        cp_cols = [c for c in df.columns if c.startswith("trend_cp_date_")]
        rows = []
        for _, row in df.iterrows():
            pid = row.get("parcel_id", "")
            for col in cp_cols:
                val = row[col]
                if pd.isna(val) or str(val).strip() == "":
                    continue
                order = int(col.replace("trend_cp_date_", ""))
                try:
                    dt = pd.Timestamp(val)
                    rows.append({
                        "parcel_id": pid,
                        "cp_order": order,
                        "cp_date": dt,
                        "year": dt.year,
                        "doy": dt.timetuple().tm_yday,
                    })
                except Exception:
                    pass
        return pd.DataFrame(rows)

    @staticmethod
    def tab_yearly_medians(long: pd.DataFrame) -> pd.DataFrame:
        """Compute yearly median DOY and IQR for trend CPs."""
        if long.empty or "doy" not in long.columns:
            return pd.DataFrame(columns=["year", "median_doy", "q25", "q75"])
        grp = long.groupby("year")["doy"]
        tab = pd.DataFrame({
            "year": grp.median().index,
            "median_doy": grp.median().values,
            "q25": grp.quantile(0.25).values,
            "q75": grp.quantile(0.75).values,
        })
        return tab.sort_values("year").reset_index(drop=True)

    def explode_trend_cps(
        self, csv_path: Path, year_min: int = 2019, year_max: int = 2023
    ) -> pd.DataFrame:
        """Expand trend CPs from CSV and filter by year range.

        Returns DataFrame with columns: year, doy
        """
        df = pd.read_csv(csv_path)
        long = self.explode_trend_dates(df)
        if long.empty:
            return pd.DataFrame(columns=["year", "doy"])
        mask = (long["year"] >= year_min) & (long["year"] <= year_max)
        return long.loc[mask, ["year", "doy"]].reset_index(drop=True)

    def run_beast(
        self,
        series: pd.Series,
        cp_component: str = "season",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run BEAST algorithm on an EVI series.

        Dispatches to subprocess or direct mode based on
        ``self.beast_call_mode``.

        Args:
            series: EVI time series with DatetimeIndex.
            cp_component: "season" or "trend".
            params: Optional BEAST keyword overrides.

        Returns:
            Dict with keys: status, reason, cp_idx, cp_dates, cp_pr, occ.
        """
        return _beast_run(
            series, cp_component, params or {},
            mode=self.beast_call_mode,
        )

    def load_county_year_csv(self, county: str, wy: int) -> pd.DataFrame:
        """Load county-year EVI CSV.

        Args:
            county: County name
            wy: Water year

        Returns:
            DataFrame with EVI data
        """
        county_norm = normalize_county_name(county)
        csv_path = self.county_year_root / county_norm / f"WY{wy}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing county-year file: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["date"])
        required_cols = {"date", "parcel_id", "original_mean_evi", "gapfilled_mean_evi", "smoothed_mean_evi"}
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        return df

    def series_for_beast(self, df_parcel: pd.DataFrame, col: str = "smoothed_mean_evi") -> pd.Series:
        """Prepare EVI series for BEAST analysis.

        Args:
            df_parcel: DataFrame with parcel data
            col: Column name to use (default: smoothed_mean_evi)

        Returns:
            Series with full daily index
        """
        s = df_parcel.set_index("date").sort_index()[col].astype(float)
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        return s.reindex(full_idx)

    @staticmethod
    def prep_series_for_beast(series: pd.Series) -> Tuple[Optional[pd.Series], Optional[str]]:
        """Prepare series for BEAST with validation.

        Args:
            series: EVI series

        Returns:
            Tuple of (prepared_series, error_message)
        """
        s = series.copy()
        try:
            s = s.interpolate(method="time", limit=90, limit_direction="both")
        except Exception:
            s = s.interpolate(limit=90, limit_direction="both")

        s = s.clip(-0.05, 1.05)

        if s.isna().any():
            s = s.ffill().bfill()

        vals = s.to_numpy()
        n_valid = int(np.isfinite(vals).sum())
        std = float(np.nanstd(vals))
        frac_nan_orig = float(series.isna().mean())

        if (n_valid < 60) or (std < 1e-8) or (frac_nan_orig > 0.70):
            return None, f"unsafe_for_beast(n_valid={n_valid}, std={std:.2e}, frac_nan={frac_nan_orig:.2f})"

        return s, None

    @staticmethod
    def is_fallow(series: pd.Series) -> bool:
        """Check if series represents fallow field.

        Args:
            series: EVI series

        Returns:
            True if field appears fallow
        """
        s = series.dropna()
        if s.empty:
            return True
        return (s.max() < config.fallow_max_evi_threshold) or (s.std() < config.fallow_std_threshold)

    @staticmethod
    def qualified_minima(
        dates: pd.DatetimeIndex,
        evi: np.ndarray,
        peak_window_days: int,
        delta_min: float,
        min_spacing_days: int,
        min_evi_max: float,
    ) -> List[pd.Timestamp]:
        """Find qualified local minima in EVI series.

        Args:
            dates: DatetimeIndex
            evi: EVI values array
            peak_window_days: Window size to find peaks
            delta_min: Minimum delta from peak
            min_spacing_days: Minimum spacing between minima
            min_evi_max: Maximum EVI value at minima

        Returns:
            List of minima dates
        """
        if len(dates) == 0:
            return []

        min_idx, _ = find_peaks(-evi)
        max_idx, _ = find_peaks(evi)

        cuts, last_kept = [], None

        for i in min_idx:
            d_min = dates[i]
            y_min = evi[i]

            if not np.isfinite(y_min) or y_min > min_evi_max:
                continue

            window_start = d_min - pd.Timedelta(days=peak_window_days)
            prev_peaks = [j for j in max_idx if (dates[j] < d_min) and (dates[j] >= window_start)]

            if not prev_peaks:
                continue

            j = max(prev_peaks)
            y_peak = evi[j]

            if not np.isfinite(y_peak) or (y_peak - y_min) < delta_min:
                continue

            if last_kept is not None and (d_min - last_kept).days < min_spacing_days:
                continue

            cuts.append(d_min)
            last_kept = d_min

        return cuts

    @staticmethod
    def dedupe_by_nearness(
        cands: List[pd.Timestamp],
        evi_series: pd.Series,
        min_gap_days: int = 7,
    ) -> List[pd.Timestamp]:
        """Remove near-duplicate minima by EVI value.

        Args:
            cands: Candidate minima dates
            evi_series: EVI values
            min_gap_days: Minimum gap between minima

        Returns:
            Deduplicated list of minima
        """
        if not cands:
            return []

        s = evi_series.reindex(sorted(evi_series.index))
        cands = sorted(set(pd.to_datetime(cands)))
        kept = []

        for t in cands:
            if not kept:
                kept.append(t)
                continue

            if (t - kept[-1]).days < min_gap_days:
                t_old = kept[-1]
                v_old = float(s.get(t_old, np.inf))
                v_new = float(s.get(t, np.inf))
                if v_new < v_old:
                    kept[-1] = t
            else:
                kept.append(t)

        return kept

    @staticmethod
    def argmin_in_window(
        evi_series: pd.Series,
        center: pd.Timestamp,
        half_window_days: int,
    ) -> Optional[pd.Timestamp]:
        """Find EVI minimum within window around center.

        Args:
            evi_series: EVI values
            center: Center date
            half_window_days: Half window size

        Returns:
            Date of minimum value
        """
        lo = center - pd.Timedelta(days=half_window_days)
        hi = center + pd.Timedelta(days=half_window_days)
        w = evi_series.loc[(evi_series.index >= lo) & (evi_series.index <= hi)]

        if w.empty:
            return None

        i = int(np.nanargmin(w.to_numpy()))
        return pd.to_datetime(w.index[i])

    def build_inclusive_minima(self, series: pd.Series) -> Dict[str, List[pd.Timestamp]]:
        """Build base and inclusive minima sets.

        Args:
            series: EVI series

        Returns:
            Dict with 'base' and 'union' minima lists
        """
        idx = series.index
        y = series.to_numpy()
        amp = float(np.nanpercentile(y, 90) - np.nanpercentile(y, 10))

        base = self.qualified_minima(
            idx, y,
            peak_window_days=config.peak_window_days,
            delta_min=max(config.delta_min, config.amp_frac_min * amp),
            min_spacing_days=config.min_spacing_days,
            min_evi_max=config.min_evi_max,
        )

        union = set(base)

        for pw in config.peak_window_days_range:
            for dm in config.delta_min_range:
                d_dyn = max(dm, config.amp_frac_min * amp)
                for sp in config.min_spacing_days_range:
                    for evmax in config.min_evi_max_range:
                        c = self.qualified_minima(
                            idx, y,
                            peak_window_days=pw,
                            delta_min=d_dyn,
                            min_spacing_days=sp,
                            min_evi_max=evmax,
                        )
                        union.update(c)

        union = self.dedupe_by_nearness(
            sorted(union),
            series,
            min_gap_days=min(config.min_spacing_days_range),
        )

        return {"base": base, "union": union}

    def _estimate_adaptive_priors(
        self,
        groups: Dict[str, pd.DataFrame],
        parcels: List[str],
        county_norm: str,
        evi_col: str,
        n_jobs: int,
    ) -> Tuple[int, int]:
        """First-pass: estimate county-year cutting priors from a sample.

        Runs BEAST with wide priors on a random sample of parcels, then
        uses the resulting cut-count distribution to set tighter priors.

        Args:
            groups: Dict of parcel_id -> DataFrame.
            parcels: Full parcel list.
            county_norm: Normalized county name.
            evi_col: EVI column to use.
            n_jobs: Parallel workers.

        Returns:
            Tuple (prior_lo, prior_hi) for the second pass.
        """
        rng = np.random.default_rng(config.uid_sample_seed)
        n_sample = max(5, int(len(parcels) * config.beast_prior_sample_frac))
        sample_pids = list(rng.choice(parcels, size=min(n_sample, len(parcels)), replace=False))
        wide_lo, wide_hi = config.beast_prior_wide_range

        def _sample_process(pid: str) -> Optional[int]:
            sub = groups[pid].copy()
            series = self.series_for_beast(sub, col=evi_col)
            s_ok, _ = self.prep_series_for_beast(series)
            if s_ok is None:
                return None
            # Run with wide priors, ensemble disabled for speed
            sweep = _build_season_sweep()
            sweep = [{**p, "scp_minmax": (wide_lo, wide_hi)} for p in sweep]
            # Quick single-run (first param set) to get approximate cut count
            run = _beast_run(s_ok, "season", sweep[0], mode=self.beast_call_mode)
            mins_pack = self.build_inclusive_minima(s_ok)
            d, _ = self._align_minima_with_beast(
                mins_pack["union"], run["cp_dates"], run["cp_pr"],
                tol_days=config.strict_tol_steps[-1],
            )
            return len(d) if d else 0

        counts = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_sample_process)(pid) for pid in sample_pids
        )
        valid_counts = [c for c in counts if c is not None and c > 0]

        if len(valid_counts) < 3:
            return config.get_county_expected_range(county_norm)

        arr = np.array(valid_counts, dtype=float)
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q75 - q25
        prior_lo = max(wide_lo, int(np.floor(q25 - 0.5 * iqr)))
        prior_hi = min(wide_hi, int(np.ceil(q75 + 0.5 * iqr)))
        if prior_lo >= prior_hi:
            prior_hi = prior_lo + 2

        return (prior_lo, prior_hi)

    def run_seasonal_for_year(
        self,
        county: str,
        year: int,
        evi_col: str = "smoothed_mean_evi",
        n_jobs: int = 4,
        debug: bool = False,
    ) -> Path:
        """Run BEAST cutting detection for all parcels in a water year.

        Supports two-pass adaptive priors (Improvement D): first pass
        estimates county-year priors from a sample, second pass runs
        all parcels with data-driven priors.

        Args:
            county: County name
            year: Water year
            evi_col: EVI column to use
            n_jobs: Parallel jobs
            debug: Save debug outputs

        Returns:
            Path to output CSV
        """
        county_norm = normalize_county_name(county)
        wy = int(year)
        df = self.load_county_year_csv(county_norm, wy)

        parcels = sorted(df["parcel_id"].astype(str).unique().tolist())
        groups = {pid: g for pid, g in df.groupby(df["parcel_id"].astype(str))}

        # Adaptive priors: two-pass estimation (Improvement D)
        scp_override = None
        if config.beast_adaptive_priors and len(parcels) >= 10:
            scp_override = self._estimate_adaptive_priors(
                groups, parcels, county_norm, evi_col, n_jobs,
            )

        def _process(pid: str) -> Dict[str, Any]:
            sub = groups[pid].copy()
            series = self.series_for_beast(sub, col=evi_col)

            # Fallow filter: skip BEAST entirely for bare soil / fallow parcels
            if self.is_fallow(series):
                res = {
                    "season_params": {}, "season_status": "fallow",
                    "season_reason": "fallow",
                    "season_cp_dates": pd.DatetimeIndex([]),
                    "season_cp_pr": np.array([], float),
                    "minima_base": [], "minima_union": [],
                    "cuts": [], "cut_prs": [],
                    "fallback_used": 0,
                    "cut_timing_sigmas": [], "cut_consensus_freqs": [],
                }
                row = self._row_from_result(pid, wy, county_norm, evi_col, res)
                return row

            s_ok, s_reason = self.prep_series_for_beast(series)

            if s_ok is None:
                res = {
                    "season_params": {}, "season_status": "unsafe", "season_reason": s_reason,
                    "season_cp_dates": pd.DatetimeIndex([]), "season_cp_pr": np.array([], float),
                    "minima_base": [], "minima_union": [], "cuts": [], "cut_prs": [],
                    "fallback_used": 0,
                    "cut_timing_sigmas": [], "cut_consensus_freqs": [],
                }
            else:
                res = self._seasonal_cuts(
                    s_ok, county_norm, scp_minmax_override=scp_override,
                )

            row = self._row_from_result(pid, wy, county_norm, evi_col, res)
            return row

        rows = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_process)(pid) for pid in parcels
        )

        out_dir = self.beast_out_root / county_norm
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"beast_seasonal_cuts_WY{wy}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        return out_csv

    @staticmethod
    def _align_minima_with_beast(
        minima: List[pd.Timestamp],
        cp_dates: pd.DatetimeIndex,
        cp_pr: np.ndarray,
        tol_days: int,
    ) -> Tuple[List[pd.Timestamp], List[float]]:
        """Align inclusive minima with BEAST change-points within tolerance.

        Each minima is matched to the closest unmatched CP within
        ``tol_days``. Returns the matched minima dates and their
        corresponding BEAST probabilities.

        Args:
            minima: Candidate minima dates (inclusive union).
            cp_dates: BEAST change-point dates.
            cp_pr: BEAST change-point probabilities.
            tol_days: Maximum days between minima and CP to match.

        Returns:
            Tuple of (matched_dates, matched_probs).
        """
        if not minima or not len(cp_dates):
            return [], []
        kept_d, kept_p, used = [], [], set()
        for m in sorted(minima):
            best_k, best_dt = None, None
            for k, c in enumerate(cp_dates):
                if k in used:
                    continue
                dt = abs((m - c).days)
                if dt <= tol_days and (best_dt is None or dt < best_dt):
                    best_k, best_dt = k, dt
            if best_k is not None:
                kept_d.append(m)
                kept_p.append(
                    float(cp_pr[best_k]) if best_k < len(cp_pr)
                    else float("nan")
                )
                used.add(best_k)
        return kept_d, kept_p

    def _seasonal_cuts(
        self, series: pd.Series, county: str,
        scp_minmax_override: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Compute seasonal cuts for a single parcel.

        Runs the BEAST sweep (ensemble or best-run), builds inclusive
        minima, boosts near CPs with EVI validation, then gates with
        widening tolerance.

        Args:
            series: EVI series.
            county: County name.
            scp_minmax_override: Optional prior override for adaptive priors.

        Returns:
            Dict with cutting results.
        """
        # Run all sweeps once — shared by consensus and best-run tiers
        precomputed = self._run_all_sweeps(
            series, scp_minmax_override=scp_minmax_override,
        )

        # Tier 1: Ensemble consensus (default when beast_ensemble_mode=True)
        if config.beast_ensemble_mode:
            sel = self._seasonal_consensus(
                series, county,
                scp_minmax_override=scp_minmax_override,
                precomputed=precomputed,
            )
            consensus_freq = sel.get("consensus_freq", np.array([], float))
            occ_mean = sel.get("occ_mean", pd.Series([], dtype=float))

            # Tier 2: Fall back to best-run if consensus yielded 0 CPs
            if len(sel["cp_dates"]) == 0 and config.beast_fallback_to_best_run:
                sel_br = self._seasonal_best_run(
                    series, county,
                    scp_minmax_override=scp_minmax_override,
                    precomputed=precomputed,
                )
                if len(sel_br["cp_dates"]) > 0:
                    # Keep occ_mean from consensus (averaged across runs)
                    sel = sel_br
                    consensus_freq = np.array([], float)
        else:
            sel = self._seasonal_best_run(
                series, county,
                scp_minmax_override=scp_minmax_override,
                precomputed=precomputed,
            )
            consensus_freq = np.array([], float)
            occ_mean = pd.Series(np.zeros(len(series)), index=series.index)

        cp_dates, cp_pr = sel["cp_dates"], sel["cp_pr"]
        minima_union = list(sel["minima_union"])

        # CP-centric boost: ensure a minima candidate near each CP
        # Validate that boosted minima are genuine harvest troughs
        boost_win = int(config.cp_boost_window_days)
        evi_series = series.copy()
        evi_arr = evi_series.to_numpy()
        evi_idx = evi_series.index
        for cpd in cp_dates:
            if any(abs((m - cpd).days) <= boost_win for m in minima_union):
                continue
            mstar = self.argmin_in_window(evi_series, cpd, boost_win)
            if mstar is None:
                continue
            # Validate: EVI at boost candidate must be below threshold
            evi_at_min = float(evi_series.get(mstar, np.nan))
            if not np.isfinite(evi_at_min) or evi_at_min > config.max_boost_evi:
                continue
            # Validate: must have a preceding peak with sufficient delta
            if config.require_peak_before_boost:
                peak_win_start = mstar - pd.Timedelta(days=config.peak_window_days)
                peak_mask = (evi_idx >= peak_win_start) & (evi_idx < mstar)
                if peak_mask.any():
                    peak_val = float(np.nanmax(evi_arr[peak_mask]))
                    amp = float(np.nanpercentile(evi_arr[np.isfinite(evi_arr)], 90)
                               - np.nanpercentile(evi_arr[np.isfinite(evi_arr)], 10))
                    delta_thresh = max(config.delta_min, config.amp_frac_min * amp)
                    if (peak_val - evi_at_min) < delta_thresh:
                        continue
                else:
                    continue  # no preceding data to validate
            minima_union.append(mstar)

        # dedupe after boost
        minima_union = self.dedupe_by_nearness(
            sorted(minima_union), evi_series,
            min_gap_days=min(config.min_spacing_days_range),
        )

        # STRICT CP gating with widening tolerances — pick tol with most matches
        cuts, probs = [], []
        for tol in config.strict_tol_steps:
            d, p = self._align_minima_with_beast(
                minima_union, cp_dates, cp_pr, tol_days=tol,
            )
            if len(d) > len(cuts):
                cuts, probs = d, p

        # Track which tier produced the final result
        # 0 = tier 1 (consensus), 1 = tier 2 (best-run), 2 = tier 3 (minima-only)
        fallback_used = 0
        if not cuts and len(sel["cp_dates"]) > 0:
            # Alignment failed but CPs exist — still tier 1 or 2
            pass
        elif cuts:
            # Check if we fell through to best-run (tier 2)
            if config.beast_ensemble_mode and config.beast_fallback_to_best_run:
                if "score" in sel:  # best-run result has 'score' key
                    fallback_used = 1

        # Tier 3: Season-filtered minima fallback — if both consensus and
        # best-run yielded 0 matched cuts, use relaxed minima_union filtered
        # to growing season. minima_union includes sweep-expanded candidates
        # (min_evi_max up to 0.50) and CP-boosted minima, capturing far more
        # genuine troughs than the strict minima_base (min_evi_max=0.35).
        if not cuts and minima_union:
            exp_lo, exp_hi = config.get_county_expected_range(county)
            n_union = len(minima_union)
            if config.minima_only_when_cp_count_at_most >= 0 or n_union >= exp_lo:
                # Apply growing-season filter to remove winter dormancy
                season_filtered = self.season_filter_minima(
                    minima_union,
                    start_month=config.growing_season_start_month,
                    end_month=config.growing_season_end_month,
                )
                if season_filtered:
                    cuts = season_filtered
                else:
                    # All minima are winter — use unfiltered as last resort
                    cuts = list(minima_union)
                probs = [float("nan")] * len(cuts)
                fallback_used = 2

        # Compute timing uncertainty from occupancy curve (Improvement B)
        cut_timing_sigmas = []
        if config.use_occupancy_curve and len(occ_mean) > 0:
            for cut_date in cuts:
                sigma = self._compute_occ_timing_sigma(occ_mean, cut_date)
                cut_timing_sigmas.append(sigma)
        else:
            cut_timing_sigmas = [float("nan")] * len(cuts)

        # Consensus frequencies for the matched cuts (Improvement C)
        # Map each matched cut back to its closest consensus CP frequency
        cut_consensus_freqs = []
        if len(consensus_freq) > 0 and len(cuts) > 0:
            for cut_date in cuts:
                best_freq = 0.0
                for k, cpd in enumerate(cp_dates):
                    if abs((cut_date - cpd).days) <= config.strict_tol_steps[-1]:
                        if k < len(consensus_freq):
                            best_freq = max(best_freq, float(consensus_freq[k]))
                cut_consensus_freqs.append(best_freq)
        else:
            cut_consensus_freqs = [float("nan")] * len(cuts)

        return {
            "season_params": sel["params"],
            "season_status": sel["status"],
            "season_reason": sel["reason"],
            "season_cp_dates": cp_dates,
            "season_cp_pr": cp_pr,
            "minima_base": sel["minima_base"],
            "minima_union": minima_union,
            "cuts": cuts,
            "cut_prs": probs,
            "fallback_used": int(fallback_used),
            "cut_timing_sigmas": cut_timing_sigmas,
            "cut_consensus_freqs": cut_consensus_freqs,
        }

    @staticmethod
    def _compute_occ_timing_sigma(
        occ: pd.Series, cp_date: pd.Timestamp, half_search_days: int = 30,
    ) -> float:
        """Compute timing uncertainty (sigma in days) from occupancy curve.

        Measures the full width at half maximum (FWHM) of the occupancy
        peak around a changepoint, then converts to sigma assuming
        Gaussian: sigma ≈ FWHM / 2.355.

        Args:
            occ: Occupancy probability series (DatetimeIndex, values 0-1).
            cp_date: The changepoint date to measure around.
            half_search_days: How far to look from the CP.

        Returns:
            Timing uncertainty in days. NaN if occupancy is flat/missing.
        """
        lo = cp_date - pd.Timedelta(days=half_search_days)
        hi = cp_date + pd.Timedelta(days=half_search_days)
        window = occ.loc[(occ.index >= lo) & (occ.index <= hi)]
        if window.empty:
            return float("nan")
        peak_val = float(window.max())
        if peak_val < 0.05:
            return float("nan")
        half_max = peak_val / 2.0
        above = window >= half_max
        if not above.any():
            return float("nan")
        first_above = above.idxmax()
        last_above = above[::-1].idxmax()
        fwhm_days = max(1.0, (last_above - first_above).days)
        return fwhm_days / 2.355

    def _run_all_sweeps(
        self,
        series: pd.Series,
        scp_minmax_override: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run all 16 BEAST sweeps once and return results for reuse.

        Both ``_seasonal_consensus`` and ``_seasonal_best_run`` can
        accept the same precomputed results, avoiding duplicate BEAST
        calls when tiered fallback is triggered.

        MCMC seeds are NOT fixed — BEAST explores freely with its own
        internal RNG for better coverage of the posterior landscape.

        Args:
            series: Prepared EVI series.
            scp_minmax_override: Optional prior override for adaptive priors.

        Returns:
            Tuple of (sweep_params, run_results) — parallel lists.
        """
        sweep = _build_season_sweep()
        if scp_minmax_override is not None:
            sweep = [{**p, "scp_minmax": scp_minmax_override} for p in sweep]

        if self.beast_call_mode == "subproc":
            # Batch mode: single subprocess startup for all 16 configs
            run_results = _beast_run_batch_subproc(
                series, "season", sweep,
            )
        else:
            run_results = []
            for i, params in enumerate(sweep):
                run = _beast_run(
                    series, "season", params, mode=self.beast_call_mode,
                )
                run_results.append(run)

        return sweep, run_results

    @staticmethod
    def season_filter_minima(
        minima: List[pd.Timestamp],
        start_month: int = 3,
        end_month: int = 10,
    ) -> List[pd.Timestamp]:
        """Filter minima to growing season (default March-October).

        Removes winter dormancy fluctuations that are not real cuttings.

        Args:
            minima: Candidate minima dates.
            start_month: First month of growing season (inclusive).
            end_month: Last month of growing season (inclusive).

        Returns:
            Filtered list of minima within growing season.
        """
        return [m for m in minima if start_month <= m.month <= end_month]

    def _seasonal_consensus(
        self, series: pd.Series, county: str,
        scp_minmax_override: Optional[Tuple[int, int]] = None,
        precomputed: Optional[Tuple[List[Dict], List[Dict]]] = None,
    ) -> Dict[str, Any]:
        """Ensemble consensus across all BEAST parameter sweeps.

        Instead of picking a single 'best' run, pools CPs from all runs,
        clusters by proximity, and returns consensus detections scored by
        detection frequency and mean probability.

        Args:
            series: Prepared EVI series.
            county: County name (for expected-cut priors).
            scp_minmax_override: Optional prior override for adaptive priors.
            precomputed: Optional (sweep_params, run_results) from
                ``_run_all_sweeps`` to avoid re-running BEAST.

        Returns:
            Dict with keys: cp_dates, cp_pr, consensus_freq, occ_mean,
            minima_base, minima_union, params (dict of 'ensemble'),
            status, reason.
        """
        if precomputed is not None:
            sweep, run_results = precomputed
        else:
            sweep, run_results = self._run_all_sweeps(
                series, scp_minmax_override=scp_minmax_override,
            )

        n_runs = len(sweep)
        all_cps = []  # (date, probability, run_index)
        occ_curves = []

        for i, run in enumerate(run_results):
            if run["status"] == "exception":
                continue
            for d, p in zip(run["cp_dates"], run["cp_pr"]):
                all_cps.append((pd.Timestamp(d), float(p), i))
            if run["occ"] is not None and len(run["occ"]) > 0:
                occ_curves.append(run["occ"])

        # Average occupancy curve across all runs
        if occ_curves:
            occ_mean = pd.concat(occ_curves, axis=1).mean(axis=1)
        else:
            occ_mean = pd.Series(
                np.zeros(len(series)), index=series.index, dtype=float,
            )

        if not all_cps:
            mins_pack = self.build_inclusive_minima(series)
            return dict(
                cp_dates=pd.DatetimeIndex([]),
                cp_pr=np.array([], float),
                consensus_freq=np.array([], float),
                occ_mean=occ_mean,
                minima_base=mins_pack["base"],
                minima_union=mins_pack["union"],
                params={"mode": "ensemble", "n_runs": n_runs},
                status="ok", reason="no_cps_detected",
            )

        # Cluster CPs by proximity
        sorted_cps = sorted(all_cps, key=lambda x: x[0])
        tol = config.cp_tolerance_days
        clusters = []
        current = [sorted_cps[0]]
        for cp in sorted_cps[1:]:
            if abs((cp[0] - current[-1][0]).days) <= tol:
                current.append(cp)
            else:
                clusters.append(current)
                current = [cp]
        clusters.append(current)

        # Score each cluster
        n_successful = max(1, n_runs - sum(
            1 for i, p in enumerate(sweep)
            if not any(c[2] == i for c in all_cps)
        ))
        consensus_cps = []
        for cluster in clusters:
            dates = sorted(c[0] for c in cluster)
            probs = [c[1] for c in cluster]
            run_ids = set(c[2] for c in cluster)
            # Consensus date = median of cluster dates
            median_date = dates[len(dates) // 2]
            mean_prob = float(np.mean(probs))
            freq = len(run_ids) / n_successful
            consensus_cps.append({
                "date": median_date,
                "prob": mean_prob,
                "freq": freq,
                "n_detections": len(run_ids),
            })

        # Filter by minimum consensus frequency and probability
        min_freq = config.min_consensus_freq
        min_prob = config.min_cp_probability
        filtered = [
            c for c in consensus_cps
            if c["freq"] >= min_freq and c["prob"] >= min_prob
        ]
        filtered.sort(key=lambda c: c["date"])

        cp_dates = pd.DatetimeIndex([c["date"] for c in filtered])
        cp_pr = np.array([c["prob"] for c in filtered])
        cp_freq = np.array([c["freq"] for c in filtered])

        mins_pack = self.build_inclusive_minima(series)

        return dict(
            cp_dates=cp_dates,
            cp_pr=cp_pr,
            consensus_freq=cp_freq,
            occ_mean=occ_mean,
            minima_base=mins_pack["base"],
            minima_union=mins_pack["union"],
            params={"mode": "ensemble", "n_runs": n_runs},
            status="ok", reason="ok",
        )

    def _seasonal_best_run(
        self, series: pd.Series, county: str,
        scp_minmax_override: Optional[Tuple[int, int]] = None,
        precomputed: Optional[Tuple[List[Dict], List[Dict]]] = None,
    ) -> Dict[str, Any]:
        """Sweep BEAST seasonal parameters and select the best run.

        Scores each run by counting base-minima that match a CP within
        tolerance, penalised by deviation from county expected range.

        Args:
            series: EVI series.
            county: County name.
            scp_minmax_override: Optional prior override for adaptive priors.
            precomputed: Optional (sweep_params, run_results) from
                ``_run_all_sweeps`` to avoid re-running BEAST.

        Returns:
            Best run dict with keys: score, params, status, reason,
            cp_dates, cp_pr, minima_base, minima_union.
        """
        exp_lo, exp_hi = config.get_county_expected_range(county)
        cp_tol = config.cp_tolerance_days

        # build minima sets once
        mins_pack = self.build_inclusive_minima(series)
        minima_base = mins_pack["base"]
        minima_union = mins_pack["union"]

        if precomputed is not None:
            sweep, run_results = precomputed
        else:
            sweep, run_results = self._run_all_sweeps(
                series, scp_minmax_override=scp_minmax_override,
            )
        best_pack = None

        for params, run in zip(sweep, run_results):
            cp_dates, cp_pr = run["cp_dates"], run["cp_pr"]

            # score using BASE minima (keeps sweep behavior stable)
            used = set()
            matches = 0
            for m in minima_base:
                for k, c in enumerate(cp_dates):
                    if k in used:
                        continue
                    if abs((m - c).days) <= cp_tol:
                        matches += 1
                        used.add(k)
                        break
            n = len(minima_base)
            penalty = (max(0, exp_lo - n) + max(0, n - exp_hi)) * 0.5
            score = (
                matches - penalty
                - 0.01 * abs(params.get("sseg_minlength", 30) - 30)
            )

            pack = dict(
                score=score, params=params,
                status=run["status"], reason=run["reason"],
                cp_dates=cp_dates, cp_pr=cp_pr,
                minima_base=minima_base, minima_union=minima_union,
            )
            if best_pack is None or score > best_pack["score"]:
                best_pack = pack

        return best_pack

    def trend_cps(self, series: pd.Series) -> Dict[str, Any]:
        """Sweep BEAST trend parameters and select the best run.

        Args:
            series: EVI series spanning multiple water years.

        Returns:
            Best run dict with keys: params, cp_dates, cp_pr, score,
            status, reason.
        """
        sweep = _build_trend_sweep()
        best = None
        for params in sweep:
            run = _beast_run(
                series, "trend", params, mode=self.beast_call_mode,
            )
            cp_dates, cp_pr = run["cp_dates"], run["cp_pr"]
            score = len(cp_dates) + (
                0.1 * float(np.nanmean(cp_pr)) if len(cp_pr) else 0.0
            )
            pack = dict(
                params=params, cp_dates=cp_dates, cp_pr=cp_pr,
                score=score, status=run["status"], reason=run["reason"],
            )
            if best is None or pack["score"] > best["score"]:
                best = pack
        return best

    @staticmethod
    def _row_from_result(
        pid: str,
        wy: int,
        county_norm: str,
        evi_col: str,
        res: dict,
    ) -> dict:
        """Create result row from BEAST output.

        Args:
            pid: Parcel ID
            wy: Water year
            county_norm: Normalized county name
            evi_col: EVI column used
            res: BEAST result dict

        Returns:
            Result row dict
        """
        cp_dates = res["season_cp_dates"]
        cp_pr = res["season_cp_pr"]
        cuts = res["cuts"]
        cut_prs = res["cut_prs"]
        timing_sigmas = res.get("cut_timing_sigmas", [])
        consensus_freqs = res.get("cut_consensus_freqs", [])

        return {
            "ok": res["season_status"] in {"ok", "unsafe", "fallow"},
            "error_msg": "" if res["season_status"] in {"ok", "unsafe", "fallow"} else res.get("season_reason", ""),
            "beast_status": res["season_status"],
            "season_status": res["season_status"],
            "parcel_id": pid,
            "water_year": wy,
            "county": county_norm,
            "evi_series": evi_col,
            "n_cp_season": int(len(cp_dates)),
            "n_cuttings": int(len(cuts)),
            "season_cp_dates_iso": ";".join(pd.to_datetime(cp_dates).strftime("%Y-%m-%d").to_list()),
            "season_cp_probs": ";".join([f"{p:.3f}" for p in cp_pr]),
            "matched_minima_iso": ";".join(pd.to_datetime(cuts).strftime("%Y-%m-%d").to_list()),
            "matched_minima_probs": ";".join([f"{p:.3f}" for p in cut_prs]),
            "matched_timing_sigma_days": ";".join([f"{s:.1f}" if np.isfinite(s) else "nan" for s in timing_sigmas]),
            "matched_consensus_freq": ";".join([f"{f:.3f}" if np.isfinite(f) else "nan" for f in consensus_freqs]),
            "fallback_used": int(res["fallback_used"]),
            "season_params_used": json.dumps(res["season_params"]),
        }