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


def plot_reconstruction_error_timeline(
    idx_timeline: np.ndarray,
    scores_timeline: np.ndarray,
    threshold: float,
    model_name: str,
    first_alarm_idx: int | None,
    detection_lead_hours: float,
    detection_lead_files: int,
    split_cfg: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot Anomaly / Reconstruction Error over the full run-to-failure timeline
    alongside the Remaining Useful Life (RUL) countdown curve.
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    train_end = int(split_cfg["train_end_idx"])
    grey_start = int(split_cfg["grey_start_idx"])
    grey_end = int(split_cfg["grey_end_idx"])
    val_start = int(split_cfg["val_start_idx"])
    val_end = int(split_cfg["val_end_idx"])
    test_start = int(split_cfg["test_start_idx"])
    test_end = int(split_cfg.get("test_end_idx", idx_timeline.max()))

    # ── Top Panel: Anomaly Score (Log Scale) ──────────────────────────────
    ax = axes[0]
    ax.plot(idx_timeline, scores_timeline, color="#1f77b4", lw=1.3, label=f"{model_name.upper()} Anomaly Score")
    ax.axhline(threshold, color="#d62728", linestyle="--", lw=1.8, label=f"Threshold (p99 Train) = {threshold:.4f}")

    # Shaded temporal zones
    ax.axvspan(0, train_end, color="#2ca02c", alpha=0.12, label=f"Train [0..{train_end}]")
    ax.axvspan(grey_start, grey_end, color="#7f7f7f", alpha=0.20, label=f"Grey Zone [{grey_start}..{grey_end}]")
    ax.axvspan(val_start, val_end, color="#17becf", alpha=0.15, label=f"Val [{val_start}..{val_end}]")
    ax.axvspan(test_start, test_end, color="#ff7f0e", alpha=0.10, label=f"Test / Degradation [{test_start}..{test_end}]")

    if first_alarm_idx is not None:
        ax.axvline(first_alarm_idx, color="#d62728", linestyle="-.", lw=2.0,
                   label=f"1st Sustained Alarm (Idx {first_alarm_idx}) | RUL = {detection_lead_hours:.1f}h ({detection_lead_files} files)")
        alarm_score = float(scores_timeline[idx_timeline == first_alarm_idx][0]) if np.any(idx_timeline == first_alarm_idx) else threshold
        ax.scatter([first_alarm_idx], [alarm_score], color="#d62728", s=70, zorder=6)

    if np.all(scores_timeline > 0) and (np.max(scores_timeline) / max(np.min(scores_timeline), 1e-9) > 25):
        ax.set_yscale("log")
        ax.set_ylabel("Anomaly Score (log scale)")
    else:
        ax.set_ylabel("Anomaly Score")

    ax.set_title(f"{model_name.upper()} — Anomaly Score Timeline & First Sustained Alarm")
    ax.legend(loc="upper left", framealpha=0.92, ncol=2)

    # ── Bottom Panel: Remaining Useful Life (RUL) Countdown ───────────────
    ax_rul = axes[1]
    rul_hours = (test_end - idx_timeline) * 10.0 / 60.0
    ax_rul.plot(idx_timeline, rul_hours, color="#2ca02c", lw=1.8, label="Remaining Useful Life (RUL) [hours]")
    ax_rul.fill_between(idx_timeline, 0, rul_hours, color="#2ca02c", alpha=0.10)

    if first_alarm_idx is not None:
        ax_rul.axvline(first_alarm_idx, color="#d62728", linestyle="-.", lw=2.0)
        ax_rul.axhline(detection_lead_hours, color="#d62728", linestyle=":", lw=1.4)
        ax_rul.scatter([first_alarm_idx], [detection_lead_hours], color="#d62728", s=70, zorder=6)
        ax_rul.annotate(
            f"Alarme Sustentado (Idx {first_alarm_idx})\nAntecedência (RUL): {detection_lead_hours:.1f} h ({detection_lead_files} arq.)",
            xy=(first_alarm_idx, detection_lead_hours),
            xytext=(max(50, first_alarm_idx - 240), min(145, detection_lead_hours + 25)),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#d62728"),
        )

    ax_rul.set_ylabel("RUL [Hours to Failure]")
    ax_rul.set_xlabel("Snapshot File Index (10 min per snapshot)")
    ax_rul.set_title("Remaining Useful Life (RUL) Countdown at First Sustained Alarm")
    ax_rul.legend(loc="upper right")

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_explainability_fft_rul(
    healthy_raw_sig: np.ndarray,
    alarm_raw_sig: np.ndarray,
    late_raw_sig: np.ndarray,
    healthy_idx: int,
    alarm_idx: int,
    late_idx: int,
    feature_names: Sequence[str],
    alarm_feature_contribs: np.ndarray,
    df_timeline: pd.DataFrame,
    model_name: str,
    detection_lead_hours: float,
    fs: float = 20000.0,
    bpfo: float = 236.4,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    3-panel Physical & Model Explainability figure (as defined in docs/glossario.md):
      1. Physical FFT spectrum comparison (Healthy vs 1st Sustained Alarm vs Late Degradation)
         confirming energy growth at BPFO (~236.4 Hz) and its harmonics (2x, 3x).
      2. Feature attribution bar chart at the First Sustained Alarm window.
      3. Normalized progression of physical indicators (RMS, Kurtosis, BPFO Energy) vs RUL.
    """
    set_plot_style()
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)

    # ── Panel 1 (Top Full Width): Physical FFT Verification around BPFO ───
    ax_fft = fig.add_subplot(gs[0, :])
    n = len(healthy_raw_sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= 20.0) & (freqs <= 1000.0)

    fft_h = np.abs(np.fft.rfft(healthy_raw_sig))[mask]
    fft_a = np.abs(np.fft.rfft(alarm_raw_sig))[mask]
    fft_l = np.abs(np.fft.rfft(late_raw_sig))[mask]
    f_plot = freqs[mask]

    ax_fft.plot(f_plot, fft_l, color="#d62728", alpha=0.45, lw=1.0, label=f"Estágio Avançado (Idx {late_idx})")
    ax_fft.plot(f_plot, fft_a, color="#ff7f0e", lw=1.4, label=f"1º Alarme Sustentado (Idx {alarm_idx} | RUL = {detection_lead_hours:.1f}h)")
    ax_fft.plot(f_plot, fft_h, color="#2ca02c", lw=1.1, alpha=0.85, label=f"Saudável Baseline (Idx {healthy_idx})")

    for h in [1, 2, 3]:
        f_harm = bpfo * h
        ax_fft.axvline(
            f_harm, color="#6a3d9a", linestyle="--", lw=1.5, alpha=0.85,
            label=f"{h}x BPFO ({f_harm:.1f} Hz)" if h == 1 else f"{h}x Harmônico ({f_harm:.1f} Hz)"
        )
        ax_fft.axvspan(f_harm * 0.96, f_harm * 1.04, color="#6a3d9a", alpha=0.08)

    ax_fft.set_title(
        f"{model_name.upper()} — Verificação Física via FFT no 1º Alarme Sustentado (Assinatura Mecânica BPFO = {bpfo} Hz)"
    )
    ax_fft.set_xlabel("Frequência [Hz]")
    ax_fft.set_ylabel("Magnitude FFT |X(f)|")
    ax_fft.legend(loc="upper right", ncol=3, framealpha=0.92)

    # ── Panel 2 (Bottom Left): Per-Feature Attribution at Alarm ───────────
    ax_attr = fig.add_subplot(gs[1, 0])
    total_c = np.sum(alarm_feature_contribs) + 1e-12
    pct_contribs = 100.0 * alarm_feature_contribs / total_c

    # Sort top 10 features
    order = np.argsort(pct_contribs)[-10:]
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = pct_contribs[order]

    bars = ax_attr.barh(sorted_names, sorted_vals, color="#1f77b4", edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, sorted_vals):
        ax_attr.text(val + 0.6, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.5)

    ax_attr.set_xlim(0, max(sorted_vals) * 1.22 if len(sorted_vals) else 100)
    ax_attr.set_xlabel("Contribuição Relativa no Score do Alarme [%]")
    ax_attr.set_title(f"Explicabilidade do Modelo — Top Features no Alarme (Idx {alarm_idx})")

    # ── Panel 3 (Bottom Right): Physical Feature Evolution vs RUL ─────────
    ax_phys = fig.add_subplot(gs[1, 1])
    x_idx = df_timeline["idx"].to_numpy()

    for col, label, color in [
        ("b1_rms", "B1 RMS (Energia Geral)", "#1f77b4"),
        ("b1_kurtosis", "B1 Curtose (Impactos)", "#ff7f0e"),
        ("b1_energy_bpfo", "B1 Energia Banda BPFO", "#6a3d9a"),
    ]:
        if col in df_timeline.columns:
            vals = df_timeline[col].to_numpy(dtype=float)
            # Normalize to [0, 1] up to index 900 so early onset is clearly visible
            ref_max = np.percentile(vals, 98) + 1e-12
            ref_min = np.min(vals)
            norm_vals = np.clip((vals - ref_min) / (ref_max - ref_min), 0, 1.2)
            ax_phys.plot(x_idx, norm_vals, label=label, color=color, lw=1.3, alpha=0.85)

    ax_phys.axvline(alarm_idx, color="#d62728", linestyle="-.", lw=2.0,
                    label=f"1º Alarme ({alarm_idx}) → RUL {detection_lead_hours:.1f}h")
    ax_phys.set_xlabel("Snapshot File Index")
    ax_phys.set_ylabel("Magnitude Normalizada (p98)")
    ax_phys.set_title("Progressão Física (RMS, Curtose e BPFO) vs Instante do Alarme")
    ax_phys.legend(loc="upper left", framealpha=0.9)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_score_distribution(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    model_name: str,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot distribution of anomaly scores across Train, Val, and Test splits."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    use_log = (np.max(test_scores) / max(np.min(train_scores), 1e-9)) > 30
    if use_log:
        t_tr = np.log10(np.maximum(train_scores, 1e-12))
        t_va = np.log10(np.maximum(val_scores, 1e-12))
        t_te = np.log10(np.maximum(test_scores, 1e-12))
        thr_plot = np.log10(max(threshold, 1e-12))
        xlabel = "log10(Anomaly Score)"
    else:
        t_tr, t_va, t_te = train_scores, val_scores, test_scores
        thr_plot = threshold
        xlabel = "Anomaly Score"

    ax.hist(t_tr, bins=30, density=True, alpha=0.55, color="#2ca02c", label="Train (Healthy)")
    ax.hist(t_va, bins=25, density=True, alpha=0.55, color="#17becf", label="Val (Healthy Calibration)")
    ax.hist(t_te, bins=40, density=True, alpha=0.55, color="#d62728", label="Test (Degradation / Failure)")
    ax.axvline(thr_plot, color="black", linestyle="--", lw=2.0, label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name.upper()} — Distribuição dos Scores de Anomalia (Saudável vs Degradação)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_curve(
    train_loss: Sequence[float],
    best_epoch: int,
    model_name: str,
    seed: int,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training MSE loss vs epoch for PyTorch Autoencoders."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    epochs = np.arange(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, color="#1f77b4", lw=1.8, label="Healthy Train MSE Loss")
    if 1 <= best_epoch <= len(train_loss):
        best_val = train_loss[best_epoch - 1]
        ax.axvline(best_epoch, color="#2ca02c", linestyle="--", lw=1.5,
                   label=f"Best Epoch ({best_epoch}) | MSE = {best_val:.5f}")
        ax.scatter([best_epoch], [best_val], color="#2ca02c", s=60, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE Loss")
    ax.set_title(f"{model_name.upper()} Autoencoder — Training Convergence (Seed {seed})")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig



