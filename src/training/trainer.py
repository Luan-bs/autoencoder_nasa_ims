"""
src/training/trainer.py

Experiment runner for Unsupervised Anomaly Detection Baselines (PCA, Isolation Forest)
with strict temporal splitting, seed reproducibility, physical FFT/BPFO + RUL explainability,
and model-segregated artifact storage under experiments/<model>/<run_id>/.

CLI Usage:
    .venv/Scripts/python.exe -m src.training.trainer --model all --allow-dirty
    .venv/Scripts/python.exe -m src.training.trainer --model pca --allow-dirty
    .venv/Scripts/python.exe -m src.training.trainer --model isolation_forest --allow-dirty
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import list_ims_files, load_snapshot
from src.data.preprocessor import load_labeling_config, prepare_data
from src.models.anomaly_detector import build_detector
from src.utils.metrics import compute_phm_metrics, select_threshold
from src.utils.seed import set_seed
from src.utils.visualization import (
    plot_explainability_fft_rul,
    plot_reconstruction_error_timeline,
    plot_score_distribution,
)


def _get_git_info(allow_dirty: bool = False) -> str:
    """Return current git commit hash (with '-dirty' suffix if uncommitted changes exist)."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        is_dirty = bool(status.stdout.strip())
        if is_dirty and not allow_dirty:
            raise RuntimeError(
                "Uncommitted changes detected in git repository! "
                "Commit your changes first or pass --allow-dirty."
            )
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = rev.stdout.strip() or "unknown"
        return f"{commit}-dirty" if is_dirty else commit
    except FileNotFoundError:
        return "no-git"


def _compute_file_sha256(path: Path) -> str:
    """Compute short SHA-256 hash of the features CSV for data lineage."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _compute_bpfo_verification(
    files: list[Path],
    healthy_idx: int,
    alarm_idx: int,
    late_idx: int,
    channel_idx: int = 0,
    fs: float = 20000.0,
    bpfo: float = 236.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load raw vibration waveforms at Healthy, 1st Sustained Alarm, and Late Degradation
    and quantify the FFT peak & band energy around BPFO (~236.4 Hz) for physical verification.
    """
    sig_h = load_snapshot(files[healthy_idx])[:, channel_idx]
    sig_a = load_snapshot(files[alarm_idx])[:, channel_idx]
    sig_l = load_snapshot(files[late_idx])[:, channel_idx]

    freqs = np.fft.rfftfreq(len(sig_h), d=1.0 / fs)
    bpfo_window = (freqs >= bpfo - 10.0) & (freqs <= bpfo + 10.0)

    def _bpfo_stats(sig: np.ndarray) -> dict[str, float]:
        mag = np.abs(np.fft.rfft(sig))
        win_mag = mag[bpfo_window]
        win_freqs = freqs[bpfo_window]
        peak_i = int(np.argmax(win_mag))
        return {
            "peak_freq_hz": round(float(win_freqs[peak_i]), 2),
            "peak_magnitude": round(float(win_mag[peak_i]), 4),
            "band_energy": round(float(np.sum(win_mag**2)), 2),
        }

    h_stat = _bpfo_stats(sig_h)
    a_stat = _bpfo_stats(sig_a)
    l_stat = _bpfo_stats(sig_l)

    summary = {
        "theoretical_bpfo_hz": bpfo,
        "healthy_ref_idx": healthy_idx,
        "alarm_idx": alarm_idx,
        "late_ref_idx": late_idx,
        "healthy_bpfo": h_stat,
        "alarm_bpfo": a_stat,
        "late_bpfo": l_stat,
        "alarm_vs_healthy_energy_ratio": round(
            a_stat["band_energy"] / max(h_stat["band_energy"], 1e-9), 2
        ),
        "late_vs_healthy_energy_ratio": round(
            l_stat["band_energy"] / max(h_stat["band_energy"], 1e-9), 2
        ),
    }
    return sig_h, sig_a, sig_l, summary


def _append_registry(registry_path: Path, columns: list[str], row_dict: dict[str, Any]) -> None:
    """Append a single experiment run row to experiments/registry.csv."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not registry_path.exists() or registry_path.stat().st_size == 0

    with registry_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        filtered_row = {col: row_dict.get(col, "") for col in columns}
        writer.writerow(filtered_row)


def run_single_experiment(
    model_name: str,
    set_name: str = "2nd_test",
    channel: str = "b1",
    seed: int = 42,
    window_size: int = 1,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """
    Execute one full run (train -> threshold selection -> evaluation -> explainability -> save).
    All artefacts are saved in `experiments/<model_name>/<run_id>/`.
    """
    set_seed(seed)

    # Load configs
    with (PROJECT_ROOT / "configs" / "data.yaml").open(encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with (PROJECT_ROOT / "configs" / "labeling.yaml").open(encoding="utf-8") as f:
        labeling_cfg = yaml.safe_load(f)
    with (PROJECT_ROOT / "configs" / "eval.yaml").open(encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    git_commit = _get_git_info(allow_dirty=allow_dirty)

    # Prepare splits & materialize data/processed/{train,val,test}/
    splits = prepare_data(
        set_name=set_name,
        channel=channel,
        window_size=window_size,
        mode="window_flat",
    )

    features_csv_path = PROJECT_ROOT / "data" / "processed" / f"{set_name}_features.csv"
    data_hash = _compute_file_sha256(features_csv_path)

    # Fit detector on healthy training set ONLY
    t0 = time.perf_counter()
    detector = build_detector(model_name, seed=seed)
    detector.fit(splits.X_train)
    train_time_s = round(time.perf_counter() - t0, 4)

    # Score splits
    train_scores = detector.score_samples(splits.X_train)
    val_scores = detector.score_samples(splits.X_val)
    test_scores = detector.score_samples(splits.X_test)
    timeline_scores = detector.score_samples(splits.X_timeline)

    # Threshold chosen strictly without looking at test set
    thr_cfg = eval_cfg["threshold"]
    threshold = select_threshold(
        train_scores=train_scores,
        val_scores=val_scores,
        strategy=thr_cfg["strategy"],
        percentile=float(thr_cfg["percentile"]),
    )

    split_set_cfg = load_labeling_config(set_name)
    end_of_life_idx = int(split_set_cfg.get("test_end_idx", 981))
    sustained_window = int(eval_cfg["metrics"]["sustained_alarm_window_files"])

    metrics = compute_phm_metrics(
        train_scores=train_scores,
        val_scores=val_scores,
        test_scores=test_scores,
        idx_test=splits.idx_test,
        threshold=threshold,
        sustained_window=sustained_window,
        end_of_life_idx=end_of_life_idx,
        interval_minutes=10.0,
    )

    # Explainability: Feature Attribution + FFT / BPFO + RUL at First Sustained Alarm
    alarm_idx = metrics["first_alarm_idx"] or int(splits.idx_test[0])
    alarm_mask = (splits.idx_test >= alarm_idx) & (splits.idx_test < alarm_idx + sustained_window)
    X_alarm = splits.X_test[alarm_mask] if np.any(alarm_mask) else splits.X_test[:sustained_window]
    contribs_alarm = np.mean(detector.feature_contributions(X_alarm), axis=0)

    # Map contributions back to base feature names if window_size > 1
    if window_size > 1:
        contribs_reshaped = contribs_alarm.reshape(window_size, len(splits.feature_names))
        contribs_per_feat = np.mean(contribs_reshaped, axis=0)
    else:
        contribs_per_feat = contribs_alarm

    total_c = float(np.sum(contribs_per_feat)) + 1e-12
    feature_ranking = {
        splits.feature_names[i]: round(float(100.0 * contribs_per_feat[i] / total_c), 2)
        for i in np.argsort(contribs_per_feat)[::-1]
    }

    # Physical FFT check on raw snapshot files
    interim_dir = PROJECT_ROOT / data_cfg["sets"][set_name]["interim_dir"]
    raw_files = list_ims_files(interim_dir)
    sig_h, sig_a, sig_l, bpfo_verification = _compute_bpfo_verification(
        files=raw_files,
        healthy_idx=100,
        alarm_idx=alarm_idx,
        late_idx=951,
        channel_idx=0,
        fs=float(data_cfg["dataset"]["sampling_rate_hz"]),
        bpfo=float(data_cfg["fault_frequencies"]["bpfo"]),
    )

    # Create model-specific run folder: experiments/<model>/<run_id>/
    now_dt = datetime.now()
    ts_str = now_dt.strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts_str}_{detector.name}_{set_name}_seed{seed}"
    run_dir = PROJECT_ROOT / "experiments" / detector.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save config.yaml snapshot
    config_snapshot = {
        "run_id": run_id,
        "model": detector.name,
        "model_metadata": detector.get_metadata(),
        "set_name": set_name,
        "channel": channel,
        "seed": seed,
        "window_size_files": window_size,
        "git_commit": git_commit,
        "data_hash": data_hash,
        "data_config": data_cfg,
        "labeling_config": labeling_cfg,
        "eval_config": eval_cfg,
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_snapshot, fh, sort_keys=False, allow_unicode=True)

    # 2. Save scaler.pkl and model.pkl
    joblib.dump(splits.scaler, run_dir / "scaler.pkl")
    detector.save(run_dir / "model.pkl")

    # 3. Generate and save figures in experiments/<model>/<run_id>/
    plot_reconstruction_error_timeline(
        idx_timeline=splits.idx_timeline,
        scores_timeline=timeline_scores,
        threshold=threshold,
        model_name=detector.name,
        first_alarm_idx=metrics["first_alarm_idx"],
        detection_lead_hours=metrics["detection_lead_hours"],
        detection_lead_files=metrics["detection_lead_files"],
        split_cfg=split_set_cfg,
        output_path=run_dir / "reconstruction_error_timeline.png",
    )

    plot_explainability_fft_rul(
        healthy_raw_sig=sig_h,
        alarm_raw_sig=sig_a,
        late_raw_sig=sig_l,
        healthy_idx=100,
        alarm_idx=alarm_idx,
        late_idx=951,
        feature_names=splits.feature_names,
        alarm_feature_contribs=contribs_per_feat,
        df_timeline=splits.df_timeline,
        model_name=detector.name,
        detection_lead_hours=metrics["detection_lead_hours"],
        fs=float(data_cfg["dataset"]["sampling_rate_hz"]),
        bpfo=float(data_cfg["fault_frequencies"]["bpfo"]),
        output_path=run_dir / "explainability_fft_rul.png",
    )

    plot_score_distribution(
        train_scores=train_scores,
        val_scores=val_scores,
        test_scores=test_scores,
        threshold=threshold,
        model_name=detector.name,
        output_path=run_dir / "score_distribution.png",
    )

    # 4. Save metrics.json with full PHM + RUL + FFT explainability
    lib_versions = f"numpy={np.__version__};pandas={pd.__version__};sklearn={sklearn.__version__}"
    full_metrics_payload = {
        "run_id": run_id,
        "model": detector.name,
        "model_metadata": detector.get_metadata(),
        "test_set": set_name,
        "channel": channel,
        "seed": seed,
        "window_size_files": window_size,
        "threshold_strategy": thr_cfg["strategy"],
        "metrics": metrics,
        "rul_explainability": {
            "end_of_life_idx": end_of_life_idx,
            "first_sustained_alarm_idx": metrics["first_alarm_idx"],
            "confirmed_alarm_idx": metrics["confirmed_alarm_idx"],
            "estimated_rul_files_at_alarm": metrics["detection_lead_files"],
            "estimated_rul_hours_at_alarm": metrics["detection_lead_hours"],
            "estimated_rul_days_at_alarm": round(metrics["detection_lead_hours"] / 24.0, 2),
        },
        "feature_attribution_at_alarm_pct": feature_ranking,
        "fft_bpfo_physical_verification": bpfo_verification,
        "git_commit": git_commit,
        "data_hash": data_hash,
        "lib_versions": lib_versions,
        "device": "cpu",
        "training_time_s": train_time_s,
        "timestamp": now_dt.isoformat(timespec="seconds"),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(full_metrics_payload, fh, indent=2, ensure_ascii=False)

    # 5. Append to experiments/registry.csv
    registry_row = {
        "run_id": run_id,
        "model": detector.name,
        "test_set": set_name,
        "channel": channel,
        "seed": seed,
        "window_size_files": window_size,
        "recall": metrics["recall"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "first_alarm_idx": metrics["first_alarm_idx"],
        "detection_lead_files": metrics["detection_lead_files"],
        "detection_lead_hours": metrics["detection_lead_hours"],
        "separation_ratio": metrics["separation_ratio"],
        "spearman_monotonicity": metrics["spearman_monotonicity"],
        "roc_auc": metrics["roc_auc"],
        "threshold_value": round(metrics["threshold_value"], 6),
        "threshold_strategy": thr_cfg["strategy"],
        "git_commit": git_commit,
        "data_hash": data_hash,
        "lib_versions": lib_versions,
        "device": "cpu",
        "training_time_s": train_time_s,
        "timestamp": now_dt.isoformat(timespec="seconds"),
    }

    registry_path = PROJECT_ROOT / eval_cfg["experiment"]["registry_path"]
    registry_cols = eval_cfg["experiment"]["registry_columns"]
    _append_registry(registry_path, registry_cols, registry_row)

    return full_metrics_payload


def run_model_across_seeds(
    model_name: str,
    set_name: str = "2nd_test",
    channel: str = "b1",
    window_size: int = 1,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """
    Run a baseline model across all seeds in configs/eval.yaml ([42, 123, 456])
    and write an aggregated summary in `experiments/<model>/summary_metrics.json`.
    """
    with (PROJECT_ROOT / "configs" / "eval.yaml").open(encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)
    seeds: list[int] = eval_cfg["seeds"]["list"]

    runs = []
    for seed in seeds:
        print(f"  -> Running {model_name.upper()} | set={set_name} | ch={channel} | seed={seed} ...")
        res = run_single_experiment(
            model_name=model_name,
            set_name=set_name,
            channel=channel,
            seed=seed,
            window_size=window_size,
            allow_dirty=allow_dirty,
        )
        runs.append(res)

    # Aggregate mean +- std across seeds
    metric_keys = [
        "recall",
        "false_alarm_rate",
        "first_alarm_idx",
        "detection_lead_files",
        "detection_lead_hours",
        "separation_ratio",
        "spearman_monotonicity",
        "roc_auc",
        "threshold_value",
    ]
    aggregated = {}
    for k in metric_keys:
        vals = [float(r["metrics"][k]) for r in runs if r["metrics"].get(k) is not None]
        if vals:
            aggregated[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "values": vals,
            }

    model_folder = PROJECT_ROOT / "experiments" / runs[0]["model"]
    summary_payload = {
        "model": runs[0]["model"],
        "test_set": set_name,
        "channel": channel,
        "window_size_files": window_size,
        "seeds": seeds,
        "aggregated_metrics": aggregated,
        "sample_feature_attribution_pct": runs[0]["feature_attribution_at_alarm_pct"],
        "sample_fft_bpfo_verification": runs[0]["fft_bpfo_physical_verification"],
        "run_ids": [r["run_id"] for r in runs],
    }

    summary_path = model_folder / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, ensure_ascii=False)

    print(f"  [OK] Saved {runs[0]['model'].upper()} summary to {summary_path}\n")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Unsupervised Anomaly Detection Baselines.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["pca", "isolation_forest", "all"],
        help="Baseline model to run ('pca', 'isolation_forest', or 'all').",
    )
    parser.add_argument("--set", default="2nd_test", dest="set_name", help="Dataset key in configs/data.yaml.")
    parser.add_argument("--channel", default="b1", help="Channel prefix ('b1', 'b2', 'b3', 'b4', or 'all').")
    parser.add_argument("--window-size", type=int, default=1, help="Window size L in consecutive files.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow running with uncommitted git changes.")
    args = parser.parse_args()

    models_to_run = ["pca", "isolation_forest"] if args.model == "all" else [args.model]
    for m in models_to_run:
        run_model_across_seeds(
            model_name=m,
            set_name=args.set_name,
            channel=args.channel,
            window_size=args.window_size,
            allow_dirty=args.allow_dirty,
        )


if __name__ == "__main__":
    main()

