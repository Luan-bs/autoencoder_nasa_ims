"""
src/data/features.py

Feature engineering for NASA IMS Bearing vibration signals.

Computes time-domain and frequency-domain (FFT) features for each snapshot file
and each channel, adhering to the project specification in configs/data.yaml.

Mathematical Formulas:
----------------------
1. Time Domain:
   - RMS (Root Mean Square):
       RMS = sqrt( (1 / N) * sum(x_i^2) )
   - Peak (Maximum Absolute Amplitude):
       Peak = max( |x_i| )
   - Peak-to-Peak:
       P2P = max(x_i) - min(x_i)
   - Crest Factor:
       CF = Peak / RMS
   - Kurtosis (Pearson, normal distribution = 3.0):
       Kurtosis = ( (1 / N) * sum( (x_i - mean)^4 ) ) / ( (1 / N) * sum( (x_i - mean)^2 ) )^2

2. Frequency Domain (FFT Band Energies):
   - Sampling rate: fs = 20,000 Hz, points per file: N = 20,480
   - Frequency resolution: df = fs / N = 20000 / 20480 ~ 0.97656 Hz
   - Single-sided amplitude spectrum: |X(f)| = |rfft(x)|
   - Band Energy in [f_low, f_high]:
       E_band = sum_{f in [f_low, f_high]} ( |X(f)|^2 )
   - Configured Bands:
       - low   : [0, 500] Hz
       - mid   : [500, 2000] Hz
       - high  : [2000, 5000] Hz
       - bpfo  : [216, 256] Hz   (outer race defect ~236.4 Hz +- 8.5%)
       - bpfi  : [272, 322] Hz   (inner race defect ~296.9 Hz +- 8.5%)
       - bsf2x : [255, 302] Hz   (roller element defect ~278.4 Hz +- 8.5%)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

# Default FFT bands according to configs/data.yaml
DEFAULT_FFT_BANDS: dict[str, tuple[float, float]] = {
    "low": (0.0, 500.0),
    "mid": (500.0, 2000.0),
    "high": (2000.0, 5000.0),
    "bpfo": (216.0, 256.0),
    "bpfi": (272.0, 322.0),
    "bsf2x": (255.0, 302.0),
}


def compute_time_features(
    signal: np.ndarray,
    fisher_kurtosis: bool = False,
) -> dict[str, float]:
    """
    Compute time-domain statistical features for a 1D vibration signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array of vibration amplitudes.
    fisher_kurtosis : bool
        If True, Fisher's excess kurtosis (normal = 0.0).
        If False, Pearson kurtosis (normal = 3.0, standard in PHM literature).

    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'rms', 'peak', 'p2p', 'crest_factor', 'kurtosis'.
    """
    x = np.asarray(signal, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    p2p = float(np.ptp(x))
    crest_factor = float(peak / (rms + 1e-12))
    kurt = float(scipy_kurtosis(x, fisher=fisher_kurtosis))

    return {
        "rms": rms,
        "peak": peak,
        "p2p": p2p,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
    }


def compute_fft_features(
    signal: np.ndarray,
    fs: float = 20000.0,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """
    Compute frequency-domain FFT band energy features for a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D vibration signal.
    fs : float
        Sampling rate in Hz (default 20,000 Hz).
    bands : dict[str, tuple[float, float]] | None
        Dictionary mapping band names to (f_min, f_max) tuples.

    Returns
    -------
    dict[str, float]
        Dictionary mapping f'energy_{band_name}' to band energy sum.
    """
    x = np.asarray(signal, dtype=np.float64)
    bands = bands or DEFAULT_FFT_BANDS

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Amplitude spectrum |X(f)|
    fft_mag = np.abs(np.fft.rfft(x))
    fft_sq = fft_mag**2

    features = {}
    for name, (f_min, f_max) in bands.items():
        mask = (freqs >= f_min) & (freqs <= f_max)
        features[f"energy_{name}"] = float(np.sum(fft_sq[mask]))

    return features


def extract_snapshot_features(
    snapshot: np.ndarray,
    fs: float = 20000.0,
    channel_names: Sequence[str] | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    fisher_kurtosis: bool = False,
) -> dict[str, float]:
    """
    Extract all time and frequency features for a single multi-channel snapshot.

    Parameters
    ----------
    snapshot : np.ndarray
        2D array of shape (N, n_channels).
    fs : float
        Sampling rate in Hz.
    channel_names : Sequence[str] | None
        Names for each channel column (e.g. ['b1', 'b2', 'b3', 'b4']).
    bands : dict | None
        FFT bands dictionary.
    fisher_kurtosis : bool
        Kurtosis definition flag.

    Returns
    -------
    dict[str, float]
        Flat dictionary of features named '{channel}_{feature_name}'.
    """
    snapshot = np.asarray(snapshot)
    n_channels = snapshot.shape[1]
    if channel_names is None:
        channel_names = [f"b{i+1}" for i in range(n_channels)]

    out: dict[str, float] = {}
    for ch_idx, ch_name in enumerate(channel_names):
        sig = snapshot[:, ch_idx]
        t_feats = compute_time_features(sig, fisher_kurtosis=fisher_kurtosis)
        f_feats = compute_fft_features(sig, fs=fs, bands=bands)

        for k, v in t_feats.items():
            out[f"{ch_name}_{k}"] = v
        for k, v in f_feats.items():
            out[f"{ch_name}_{k}"] = v

    return out


def extract_features_df(
    files: Sequence[Path],
    fs: float = 20000.0,
    channel_names: Sequence[str] | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    cache_path: str | Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Extract features for a list of snapshot files, returning a combined DataFrame.
    Supports caching to CSV or Parquet to avoid recomputing.

    Parameters
    ----------
    files : Sequence[Path]
        Sorted list of snapshot files.
    fs : float
        Sampling rate in Hz.
    channel_names : Sequence[str] | None
        Channel labels.
    bands : dict | None
        FFT bands dictionary.
    cache_path : str | Path | None
        If provided and file exists, loads cached features. If provided and file does not exist,
        computes and saves to cache_path.
    show_progress : bool
        Whether to display a progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['idx', 'filename', 'timestamp', ...] + all channel features.
    """
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists():
            if cp.suffix == ".parquet":
                return pd.read_parquet(cp)
            return pd.read_csv(cp)

    from src.data.loader import load_snapshot, parse_timestamp

    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(enumerate(files), total=len(files), desc="Extracting features")
    else:
        iterator = enumerate(files)

    records = []
    for idx, f in iterator:
        ts = parse_timestamp(f.name)
        data = load_snapshot(f)
        feats = extract_snapshot_features(
            data,
            fs=fs,
            channel_names=channel_names,
            bands=bands,
        )
        row = {
            "idx": idx,
            "filename": f.name,
            "timestamp": ts,
            **feats,
        }
        records.append(row)

    df = pd.DataFrame(records)

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.suffix == ".parquet":
            df.to_parquet(cp, index=False)
        else:
            df.to_csv(cp, index=False)

    return df

