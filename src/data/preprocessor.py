"""
src/data/preprocessor.py

Temporal splitting, feature selection, StandardScaler normalization (fitted on train only),
and windowing for NASA IMS Bearing datasets.

Rules enforced:
  1. Temporal split strictly follows configs/labeling.yaml.
  2. Grey zone and post-failure motor shutdown files are excluded from train/val/test.
  3. Materializes CSV splits into data/processed/{train,val,test}/.
  4. StandardScaler is fitted ONLY on the train split.
  5. Sliding windows of length L never cross timestamp gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELING_CONFIG_PATH = PROJECT_ROOT / "configs" / "labeling.yaml"

META_COLUMNS = ["idx", "filename", "timestamp"]


@dataclass
class PreparedSplits:
    """Container for normalized, windowed train/val/test splits and full timeline."""

    set_name: str
    channel: str
    feature_names: list[str]
    window_size: int
    scaler: StandardScaler

    # Raw split DataFrames (unscaled)
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    df_timeline: pd.DataFrame  # All active files (0..test_end_idx) for timeline plotting

    # Scaled & windowed arrays
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_timeline: np.ndarray

    # Corresponding file indices (last file index of each window)
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    idx_timeline: np.ndarray


def load_labeling_config(set_name: str = "2nd_test") -> dict:
    """Load split boundaries for *set_name* from configs/labeling.yaml."""
    if not LABELING_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Labeling config not found: {LABELING_CONFIG_PATH}")
    with LABELING_CONFIG_PATH.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    sets = cfg.get("sets", {})
    if set_name not in sets:
        raise ValueError(f"Unknown set '{set_name}' in labeling.yaml.")
    split_cfg = sets[set_name]
    if split_cfg.get("train_end_idx") is None:
        raise ValueError(f"Split indices for '{set_name}' are not filled in labeling.yaml.")
    return split_cfg


def select_feature_columns(df: pd.DataFrame, channel: str = "b1") -> list[str]:
    """
    Return feature column names for a specific bearing channel ('b1', 'b2', 'b3', 'b4')
    or 'all' for all channels.
    """
    all_feat_cols = [c for c in df.columns if c not in META_COLUMNS]
    if channel == "all":
        return all_feat_cols
    prefix = f"{channel.lower()}_"
    cols = [c for c in all_feat_cols if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No feature columns found with prefix '{prefix}'. Available: {all_feat_cols}")
    return cols


def materialize_splits(
    set_name: str = "2nd_test",
    features_csv: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the full features CSV according to configs/labeling.yaml and save
    train, val, and test CSVs into:
      - data/processed/train/<set_name>_train.csv
      - data/processed/val/<set_name>_val.csv
      - data/processed/test/<set_name>_test.csv

    Returns (df_train, df_val, df_test).
    """
    proc_root = Path(processed_dir) if processed_dir else (PROJECT_ROOT / "data" / "processed")
    src_csv = Path(features_csv) if features_csv else (proc_root / f"{set_name}_features.csv")

    if not src_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {src_csv}")

    df = pd.read_csv(src_csv)
    df = df.sort_values("idx").reset_index(drop=True)

    split_cfg = load_labeling_config(set_name)
    train_end = int(split_cfg["train_end_idx"])
    val_start = int(split_cfg["val_start_idx"])
    val_end = int(split_cfg["val_end_idx"])
    test_start = int(split_cfg["test_start_idx"])
    test_end = int(split_cfg.get("test_end_idx") or (df["idx"].max()))

    df_train = df[df["idx"] <= train_end].copy().reset_index(drop=True)
    df_val = df[(df["idx"] >= val_start) & (df["idx"] <= val_end)].copy().reset_index(drop=True)
    df_test = df[(df["idx"] >= test_start) & (df["idx"] <= test_end)].copy().reset_index(drop=True)

    train_path = proc_root / "train" / f"{set_name}_train.csv"
    val_path = proc_root / "val" / f"{set_name}_val.csv"
    test_path = proc_root / "test" / f"{set_name}_test.csv"

    for p, sub_df in [(train_path, df_train), (val_path, df_val), (test_path, df_test)]:
        p.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(p, index=False)

    return df_train, df_val, df_test


def build_windows(
    X: np.ndarray,
    indices: np.ndarray,
    timestamps: pd.Series | None = None,
    window_size: int = 1,
    mode: Literal["window_flat", "window_sequence"] = "window_flat",
    max_gap_minutes: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows of length *window_size* over consecutive snapshots.
    Windows never cross index discontinuities or timestamp gaps > *max_gap_minutes*.

    Returns
    -------
    (X_windowed, window_end_indices)
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    if window_size == 1:
        if mode == "window_sequence":
            return X[:, np.newaxis, :], indices.copy()
        return X.copy(), indices.copy()

    ts_parsed = pd.to_datetime(timestamps) if timestamps is not None else None
    n_samples, n_features = X.shape
    windows = []
    win_indices = []

    for i in range(window_size - 1, n_samples):
        start = i - window_size + 1
        end = i

        # Check index continuity
        idx_slice = indices[start : end + 1]
        if np.any(np.diff(idx_slice) != 1):
            continue

        # Check timestamp gap continuity if provided
        if ts_parsed is not None:
            ts_slice = ts_parsed.iloc[start : end + 1]
            deltas_min = ts_slice.diff().dt.total_seconds().iloc[1:] / 60.0
            if (deltas_min > max_gap_minutes).any():
                continue

        w = X[start : end + 1]  # shape (L, n_features)
        if mode == "window_flat":
            windows.append(w.reshape(-1))
        else:
            windows.append(w)
        win_indices.append(indices[end])

    return np.asarray(windows, dtype=np.float64), np.asarray(win_indices, dtype=np.int64)


def prepare_data(
    set_name: str = "2nd_test",
    channel: str = "b1",
    window_size: int = 1,
    mode: Literal["window_flat", "window_sequence"] = "window_flat",
    processed_dir: str | Path | None = None,
) -> PreparedSplits:
    """
    Full preprocessing pipeline:
      1. Materializes train / val / test CSVs into data/processed/{train,val,test}/.
      2. Selects feature columns for *channel*.
      3. Fits StandardScaler strictly on train.
      4. Transforms train, val, test, and active timeline.
      5. Applies sliding window of size *window_size*.
    """
    proc_root = Path(processed_dir) if processed_dir else (PROJECT_ROOT / "data" / "processed")
    df_train, df_val, df_test = materialize_splits(set_name=set_name, processed_dir=proc_root)

    full_df = pd.read_csv(proc_root / f"{set_name}_features.csv").sort_values("idx").reset_index(drop=True)
    split_cfg = load_labeling_config(set_name)
    test_end = int(split_cfg.get("test_end_idx") or full_df["idx"].max())
    df_timeline = full_df[full_df["idx"] <= test_end].copy().reset_index(drop=True)

    feat_cols = select_feature_columns(df_train, channel=channel)

    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(df_train[feat_cols].to_numpy(dtype=np.float64))
    X_val_raw = scaler.transform(df_val[feat_cols].to_numpy(dtype=np.float64))
    X_test_raw = scaler.transform(df_test[feat_cols].to_numpy(dtype=np.float64))
    X_timeline_raw = scaler.transform(df_timeline[feat_cols].to_numpy(dtype=np.float64))

    X_train, idx_train = build_windows(
        X_train_raw, df_train["idx"].to_numpy(), df_train["timestamp"], window_size=window_size, mode=mode
    )
    X_val, idx_val = build_windows(
        X_val_raw, df_val["idx"].to_numpy(), df_val["timestamp"], window_size=window_size, mode=mode
    )
    X_test, idx_test = build_windows(
        X_test_raw, df_test["idx"].to_numpy(), df_test["timestamp"], window_size=window_size, mode=mode
    )
    X_timeline, idx_timeline = build_windows(
        X_timeline_raw, df_timeline["idx"].to_numpy(), df_timeline["timestamp"], window_size=window_size, mode=mode
    )

    return PreparedSplits(
        set_name=set_name,
        channel=channel,
        feature_names=feat_cols,
        window_size=window_size,
        scaler=scaler,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        df_timeline=df_timeline,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        X_timeline=X_timeline,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        idx_timeline=idx_timeline,
    )

