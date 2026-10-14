"""
src/data/loader.py

Data loading and timestamp parsing utilities for NASA IMS Bearing datasets.

Functions:
    - parse_timestamp: Parses NASA IMS filename into datetime.
    - list_ims_files: Finds and sorts all IMS snapshot data files.
    - detect_gaps: Detects time intervals longer than expected between consecutive files.
    - load_snapshot: Reads a single ASCII snapshot into a NumPy array of shape (N, n_channels).
    - load_timeline_metadata: Builds a metadata DataFrame with file indices, timestamps, and intervals.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Regex matching IMS snapshot files: YYYY.MM.DD.HH.MM.SS (no extension)
IMS_FILENAME_PATTERN = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"


def parse_timestamp(filename_or_path: str | Path) -> datetime:
    """
    Parse IMS snapshot timestamp from filename (e.g. '2004.02.12.10.32.39').

    Parameters
    ----------
    filename_or_path : str | Path
        Filename or path containing YYYY.MM.DD.HH.MM.SS.

    Returns
    -------
    datetime
        Parsed datetime object.
    """
    name = Path(filename_or_path).name
    return datetime.strptime(name, TIMESTAMP_FORMAT)


def list_ims_files(interim_dir: str | Path) -> list[Path]:
    """
    Scan interim directory recursively for valid IMS data files and sort them chronologically.

    Parameters
    ----------
    interim_dir : str | Path
        Path to interim dataset directory (e.g. data/interim/2nd_test).

    Returns
    -------
    list[Path]
        Sorted list of Path objects for all valid IMS data files.
    """
    path = Path(interim_dir)
    if not path.exists():
        raise FileNotFoundError(f"Interim directory not found: {path}")

    files = [
        p for p in path.rglob("*")
        if p.is_file() and IMS_FILENAME_PATTERN.match(p.name)
    ]
    # Sort strictly by timestamp parsed from name
    return sorted(files, key=lambda p: parse_timestamp(p.name))


def detect_gaps(
    files: Sequence[Path],
    nominal_interval_minutes: float = 10.0,
    threshold_ratio: float = 1.5,
) -> pd.DataFrame:
    """
    Detect time gaps between consecutive files where elapsed time exceeds nominal interval.

    Parameters
    ----------
    files : Sequence[Path]
        Chronologically sorted list of IMS files.
    nominal_interval_minutes : float
        Nominal collection interval in minutes (default 10 min).
    threshold_ratio : float
        Multiplier above which an interval is flagged as a gap (default 1.5x -> 15 min).

    Returns
    -------
    pd.DataFrame
        DataFrame listing detected gaps with start_idx, end_idx, timestamps, and gap_minutes.
    """
    if len(files) < 2:
        return pd.DataFrame(columns=["from_idx", "to_idx", "from_file", "to_file", "gap_minutes"])

    records = []
    threshold_sec = nominal_interval_minutes * 60.0 * threshold_ratio

    for i in range(len(files) - 1):
        t1 = parse_timestamp(files[i].name)
        t2 = parse_timestamp(files[i + 1].name)
        delta_sec = (t2 - t1).total_seconds()

        if delta_sec > threshold_sec:
            records.append({
                "from_idx": i,
                "to_idx": i + 1,
                "from_file": files[i].name,
                "to_file": files[i + 1].name,
                "from_time": t1,
                "to_time": t2,
                "gap_minutes": delta_sec / 60.0,
            })

    return pd.DataFrame(records)


def load_snapshot(
    file_path: str | Path,
    expected_points: int = 20480,
    n_channels: int | None = None,
) -> np.ndarray:
    """
    Load a single IMS ASCII snapshot file.

    Parameters
    ----------
    file_path : str | Path
        Path to the snapshot file.
    expected_points : int
        Expected number of rows (default 20,480).
    n_channels : int | None
        Expected number of columns / channels. If None, inferred automatically.

    Returns
    -------
    np.ndarray
        Array of shape (expected_points, n_channels) with float values.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot file not found: {path}")

    # IMS ASCII files are tab- or whitespace-separated
    arr = pd.read_csv(path, sep=r"\s+", header=None, engine="c").to_numpy(dtype=np.float64)

    if expected_points is not None and len(arr) != expected_points:
        raise ValueError(
            f"File {path.name} has {len(arr)} points, expected {expected_points}."
        )

    if n_channels is not None and arr.shape[1] != n_channels:
        raise ValueError(
            f"File {path.name} has {arr.shape[1]} channels, expected {n_channels}."
        )

    return arr


def load_timeline_metadata(interim_dir: str | Path) -> pd.DataFrame:
    """
    Construct a complete timeline metadata DataFrame for all files in an interim directory.

    Columns:
        - idx: 0-based file index
        - filename: snapshot filename
        - path: Path object
        - timestamp: datetime
        - elapsed_hours: hours from first snapshot
        - delta_minutes: interval to next snapshot
        - is_gap: boolean flag if interval to next exceeds 15 min
    """
    files = list_ims_files(interim_dir)
    if not files:
        return pd.DataFrame()

    timestamps = [parse_timestamp(f.name) for f in files]
    t0 = timestamps[0]

    elapsed_hours = [(t - t0).total_seconds() / 3600.0 for t in timestamps]

    delta_minutes: list[float | None] = []
    is_gap: list[bool] = []
    for i in range(len(timestamps) - 1):
        d_min = (timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0
        delta_minutes.append(d_min)
        is_gap.append(d_min > 15.0)

    # Last element has no next delta
    delta_minutes.append(0.0)
    is_gap.append(False)

    return pd.DataFrame({
        "idx": range(len(files)),
        "filename": [f.name for f in files],
        "path": files,
        "timestamp": timestamps,
        "elapsed_hours": elapsed_hours,
        "delta_minutes": delta_minutes,
        "is_gap": is_gap,
    })

