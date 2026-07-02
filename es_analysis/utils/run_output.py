"""Versioned run output directory management and parquet persistence.

Creates a structured output directory for a named run, saves/loads
DataFrames as .parquet files for fast plot reproduction without
re-running the full pipeline.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

# Root for all figure output
_OUTPUT_ROOT = Path(__file__).parent.parent / "output" / "figures"

# Subdirectories created inside each run
_SUBDIRS = [
    "data",
    "WY_stats",
    "WY_types",
    "cuttings_analysis/all_counties",
    "et_corrections",
    "statistics",
    "water_savings",
    "test",
]


def get_run_root(run_name: str = "alfalfa_run_2") -> Path:
    """Return the root path for a named run (does not create it)."""
    return _OUTPUT_ROOT / run_name


def ensure_run_dirs(run_name: str = "alfalfa_run_2") -> Path:
    """Create the full directory structure for a named run.

    Returns the run root path.
    """
    root = get_run_root(run_name)
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def save_dataframe(
    df: pd.DataFrame,
    name: str,
    run_name: str = "alfalfa_run_2",
) -> Path:
    """Save a DataFrame as parquet in the run's data/ directory.

    Args:
        df: DataFrame to save.
        name: Logical name (without .parquet extension).
        run_name: Run directory name.

    Returns:
        Path to the saved parquet file.
    """
    data_dir = get_run_root(run_name) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_dataframe(
    name: str,
    run_name: str = "alfalfa_run_2",
) -> Optional[pd.DataFrame]:
    """Load a parquet file from the run's data/ directory.

    Returns None if the file does not exist.
    """
    path = get_run_root(run_name) / "data" / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def list_saved_data(run_name: str = "alfalfa_run_2") -> List[str]:
    """List available parquet file names (without extension)."""
    data_dir = get_run_root(run_name) / "data"
    if not data_dir.exists():
        return []
    return sorted(p.stem for p in data_dir.glob("*.parquet"))
