"""
src/utils/visualization.py

Visualization utilities for NASA IMS Bearing Exploratory Data Analysis (EDA)
and Anomaly Detection reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style() -> None:
    """Configure matplotlib defaults for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    })


def plot_dataset_timeline_zones(
    df_features: pd.DataFrame,
    train_end_idx: int = 400,
    grey_start_idx: int = 401,
    grey_end_idx: int = 450,
    val_start_idx: int = 451,
    val_end_idx: int = 530,
    test_start_idx: int = 531,
    shutdown_idx: int = 982,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the multi-feature timeline for IMS Bearing data showing all four channels
    and highlighting the temporal split zones (Train, Grey Zone, Val, Test, Power Down).

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing extracted features across all snapshots.
    train_end_idx : int
        Last file index of the healthy train set.
    grey_start_idx : int
        First file index of the grey buffer zone.
    grey_end_idx : int
        Last file index of the grey buffer zone.
    val_start_idx : int
        First file index of validation set.
    val_end_idx : int
        Last file index of validation set.
    test_start_idx : int
        First file index of test (anomaly/degradation) set.
    shutdown_idx : int
        File index where experiment was stopped / motor shut off.
    output_path : str | Path | None
        Optional file path to save the figure.

    Returns
    -------
    plt.Figure
    """
    set_plot_style()
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    x = df_features["idx"].values

    # Colors
    c_b1 = "#d62728"  # Red (Fault bearing)
    c_b2 = "#1f77b4"  # Blue (Control)
    c_b3 = "#2ca02c"  # Green (Control)
    c_b4 = "#ff7f0e"  # Orange (Control)

    # 1. RMS
    ax = axes[0]
    ax.plot(x, df_features["b1_rms"], label="Bearing 1 (Outer Race Fault)", color=c_b1, lw=1.6)
    ax.plot(x, df_features["b2_rms"], label="Bearing 2 (Control)", color=c_b2, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b3_rms"], label="Bearing 3 (Control)", color=c_b3, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b4_rms"], label="Bearing 4 (Control)", color=c_b4, lw=0.9, alpha=0.7)
    ax.set_ylabel("RMS Amplitude [g]")
    ax.set_title("Vibration Energy Progression — RMS")
    ax.legend(loc="upper left", framealpha=0.9)

    # 2. Kurtosis
    ax = axes[1]
    ax.plot(x, df_features["b1_kurtosis"], label="Bearing 1 (Outer Race Fault)", color=c_b1, lw=1.6)
    ax.plot(x, df_features["b2_kurtosis"], label="Bearing 2", color=c_b2, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b3_kurtosis"], label="Bearing 3", color=c_b3, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b4_kurtosis"], label="Bearing 4", color=c_b4, lw=0.9, alpha=0.7)
    ax.axhline(3.0, color="gray", linestyle="--", alpha=0.6, label="Normal Gaussian (Kurtosis = 3.0)")
    ax.set_ylabel("Pearson Kurtosis")
    ax.set_title("Impulse Sensitivity — Kurtosis (Spikes indicate incipient micro-impacts)")
    ax.legend(loc="upper left", framealpha=0.9)

    # 3. BPFO Band Energy
    ax = axes[2]
    ax.plot(x, df_features["b1_energy_bpfo"], label="Bearing 1 (Outer Race Defect Band)", color=c_b1, lw=1.6)
    ax.plot(x, df_features["b2_energy_bpfo"], label="Bearing 2", color=c_b2, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b3_energy_bpfo"], label="Bearing 3", color=c_b3, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b4_energy_bpfo"], label="Bearing 4", color=c_b4, lw=0.9, alpha=0.7)
    ax.set_ylabel("BPFO Band Energy")
    ax.set_title("Fault Frequency Verification — FFT Band [216 - 256 Hz] (BPFO ~ 236.4 Hz)")
    ax.legend(loc="upper left", framealpha=0.9)

    # 4. Crest Factor
    ax = axes[3]
    ax.plot(x, df_features["b1_crest_factor"], label="Bearing 1 Crest Factor", color=c_b1, lw=1.6)
    ax.plot(x, df_features["b2_crest_factor"], label="Bearing 2", color=c_b2, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b3_crest_factor"], label="Bearing 3", color=c_b3, lw=0.9, alpha=0.7)
    ax.plot(x, df_features["b4_crest_factor"], label="Bearing 4", color=c_b4, lw=0.9, alpha=0.7)
    ax.set_ylabel("Crest Factor (Peak / RMS)")
    ax.set_xlabel("Snapshot File Index (Chronological Order, 10 min intervals)")
    ax.set_title("Crest Factor — Ratio of Peak to RMS")
    ax.legend(loc="upper left", framealpha=0.9)

    # Shading and zone labels
    zone_patches = [
        (0, train_end_idx, "#2ca02c", 0.15, f"TRAIN (Healthy)\n[0 - {train_end_idx}]"),
        (grey_start_idx, grey_end_idx, "#7f7f7f", 0.22, f"GREY ZONE\n[{grey_start_idx} - {grey_end_idx}]"),
        (val_start_idx, val_end_idx, "#1f77b4", 0.15, f"VAL (Threshold)\n[{val_start_idx} - {val_end_idx}]"),
        (test_start_idx, shutdown_idx - 1, "#d62728", 0.12, f"TEST (Degradation / Failure)\n[{test_start_idx} - {shutdown_idx-1}]"),
        (shutdown_idx, len(x) - 1, "#000000", 0.35, f"POWER OFF\n[{shutdown_idx} - {len(x)-1}]"),
    ]

    for ax in axes:
        for x0, x1, color, alpha, _ in zone_patches:
            ax.axvspan(x0, x1, color=color, alpha=alpha)

        # Mark boundaries with vertical dashed lines
        ax.axvline(train_end_idx, color="#2ca02c", linestyle="--", lw=1.4)
        ax.axvline(grey_end_idx, color="#7f7f7f", linestyle="--", lw=1.4)
        ax.axvline(val_end_idx, color="#1f77b4", linestyle="--", lw=1.4)
        ax.axvline(shutdown_idx, color="#000000", linestyle=":", lw=1.4)

    # Add text banner on top axis
    top_ax = axes[0]
    for x0, x1, _, _, label in zone_patches:
        mid_x = (x0 + x1) / 2.0
        y_pos = top_ax.get_ylim()[1] * 0.88
        top_ax.text(
            mid_x, y_pos, label,
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )

    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")

    return fig


def plot_raw_signal_comparison(
    healthy_signal: np.ndarray,
    degraded_signal: np.ndarray,
    failure_signal: np.ndarray,
    fs: float = 20000.0,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot raw vibration waveforms (amplitude vs time) comparing Healthy, Incipient Fault,
    and Severe Degradation states.
    """
    set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=False)
    t = np.arange(len(healthy_signal)) / fs

    axes[0].plot(t, healthy_signal, color="#2ca02c", lw=0.8)
    axes[0].set_title("Healthy State — Regular low-amplitude vibration")
    axes[0].set_ylabel("Amplitude [g]")

    axes[1].plot(t, degraded_signal, color="#ff7f0e", lw=0.8)
    axes[1].set_title("Incipient Degradation — High-frequency transient impact bursts")
    axes[1].set_ylabel("Amplitude [g]")

    axes[2].plot(t, failure_signal, color="#d62728", lw=0.8)
    axes[2].set_title("Severe Failure — Extreme vibration amplitude and chaotic impacts")
    axes[2].set_ylabel("Amplitude [g]")
    axes[2].set_xlabel("Time [seconds]")

    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")

    return fig


def plot_fft_spectrum_comparison(
    healthy_signal: np.ndarray,
    degraded_signal: np.ndarray,
    failure_signal: np.ndarray,
    fs: float = 20000.0,
    bpfo: float = 236.4,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot FFT amplitude spectrum comparing Healthy, Incipient Fault, and Failure,
    highlighting the outer race defect frequency (BPFO) and its harmonics.
    """
    set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=False)

    n = len(healthy_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    for i, (sig, label, color) in enumerate([
        (healthy_signal, "Healthy State", "#2ca02c"),
        (degraded_signal, "Incipient Degradation", "#ff7f0e"),
        (failure_signal, "Severe Failure", "#d62728"),
    ]):
        ax = axes[i]
        fft_mag = np.abs(np.fft.rfft(sig))
        # Plot up to 2000 Hz where bearing defects and structural resonances dominate
        mask = freqs <= 2000.0
        ax.plot(freqs[mask], fft_mag[mask], color=color, lw=0.9, label=label)

        # Highlight BPFO and harmonics (1x, 2x, 3x)
        for h in [1, 2, 3]:
            f_h = bpfo * h
            if f_h <= 2000.0:
                ax.axvline(f_h, color="purple", linestyle="--", alpha=0.7, lw=1.2,
                           label=f"{h}x BPFO ({f_h:.1f} Hz)" if i == 0 else None)

        ax.set_ylabel("FFT Magnitude")
        ax.set_title(label)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")

    return fig

