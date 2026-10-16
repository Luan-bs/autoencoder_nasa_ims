"""
src/models/autoencoder.py

PyTorch Autoencoder architectures (MLP, Conv1D, LSTM) and unified detector wrapper
implementing BaseAnomalyDetector for unsupervised bearing anomaly detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_detector import BaseAnomalyDetector
from src.training.callbacks import TrainConvergenceEarlyStopping, TrainingHistory


def get_device_info() -> tuple[torch.device, str]:
    """Detect runtime compute device (CUDA GPU if available, else CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        desc = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        dev = torch.device("cpu")
        desc = "cpu"
    return dev, desc


# ── 1. MLP Autoencoder (Flat Window Input: [B, L * F]) ───────────────────────


class MLPAutoencoder(nn.Module):
    """Dense (MLP) Autoencoder with symmetric encoder-bottleneck-decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (32, 16),
        latent_dim: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim

        # Encoder
        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, h_dim))
            enc_layers.append(nn.GELU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for h_dim in reversed(self.hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h_dim))
            dec_layers.append(nn.GELU())
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ── 2. Conv1D Autoencoder (Sequence Input: [B, L, F]) ────────────────────────


class Conv1DAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder operating over sequences of L consecutive snapshots
    of shape (batch, L, n_features).
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        channels: list[int] | tuple[int, ...] = (16, 8),
        kernel_size: int = 3,
        latent_dim: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.channels = list(channels)
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        pad = kernel_size // 2
        c0, c1 = self.channels[0], self.channels[1]

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(n_features, c0, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(c0, c1, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
        )
        flat_dim = c1 * seq_len
        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.GELU(),
        )
        self.conv_decoder = nn.Sequential(
            nn.Conv1d(c1, c0, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(c0, n_features, kernel_size=kernel_size, padding=pad),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> (B, F, L)
        h = self.conv_encoder(x.transpose(1, 2))
        h_flat = h.reshape(x.size(0), -1)
        return self.to_latent(h_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        z = self.encode(x)
        h_flat = self.from_latent(z)
        h = h_flat.reshape(b, self.channels[1], self.seq_len)
        out = self.conv_decoder(h)
        # (B, F, L) -> (B, L, F)
        return out.transpose(1, 2)


# ── 3. LSTM Autoencoder (Sequence Input: [B, L, F]) ──────────────────────────


class LSTMAutoencoder(nn.Module):
    """
    Recurrent Sequence-to-Sequence Autoencoder using LSTM layers over L snapshots
    of shape (batch, L, n_features).
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dim: int = 16,
        num_layers: int = 1,
        latent_dim: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        lstm_drop = dropout if num_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_drop,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_drop,
        )
        self.output_proj = nn.Linear(hidden_dim, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        _, (h_n, _) = self.encoder_lstm(x)
        # Take top-layer final hidden state: (B, hidden_dim)
        h_last = h_n[-1]
        return self.to_latent(h_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        h_dec_in = self.from_latent(z)  # (B, hidden_dim)
        # Repeat latent bridge across L time steps: (B, L, hidden_dim)
        seq_in = h_dec_in.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(seq_in)
        return self.output_proj(dec_out)


# ── Unified Detector Wrapper for PyTorch Autoencoders ────────────────────────


class AutoencoderDetector(BaseAnomalyDetector):
    """
    Wraps MLPAutoencoder, Conv1DAutoencoder, or LSTMAutoencoder under the same
    BaseAnomalyDetector interface used by PCA and IsolationForest.
    """

    def __init__(
        self,
        arch: str,
        model_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.name = arch.lower().strip()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.seed = seed

        self.device, self.device_desc = get_device_info()
        self.net: nn.Module | None = None
        self.history: TrainingHistory = TrainingHistory()
        self.input_shape: tuple[int, ...] = ()

    def _build_network(self, sample_shape: tuple[int, ...]) -> nn.Module:
        if self.name == "mlp":
            in_dim = int(np.prod(sample_shape))
            return MLPAutoencoder(
                input_dim=in_dim,
                hidden_dims=self.model_cfg.get("hidden_dims", [32, 16]),
                latent_dim=int(self.model_cfg.get("latent_dim", 4)),
                dropout=float(self.model_cfg.get("dropout", 0.0)),
            )
        elif self.name == "conv1d":
            if len(sample_shape) == 1:
                seq_len, n_feat = 1, sample_shape[0]
            else:
                seq_len, n_feat = sample_shape[0], sample_shape[1]
            return Conv1DAutoencoder(
                seq_len=seq_len,
                n_features=n_feat,
                channels=self.model_cfg.get("channels", [16, 8]),
                kernel_size=int(self.model_cfg.get("kernel_size", 3)),
                latent_dim=int(self.model_cfg.get("latent_dim", 4)),
                dropout=float(self.model_cfg.get("dropout", 0.0)),
            )
        elif self.name == "lstm":
            if len(sample_shape) == 1:
                seq_len, n_feat = 1, sample_shape[0]
            else:
                seq_len, n_feat = sample_shape[0], sample_shape[1]
            return LSTMAutoencoder(
                seq_len=seq_len,
                n_features=n_feat,
                hidden_dim=int(self.model_cfg.get("hidden_dim", 16)),
                num_layers=int(self.model_cfg.get("num_layers", 1)),
                latent_dim=int(self.model_cfg.get("latent_dim", 4)),
                dropout=float(self.model_cfg.get("dropout", 0.0)),
            )
        else:
            raise ValueError(f"Unknown autoencoder architecture: '{self.name}'")

    def _prepare_tensor(self, X: np.ndarray) -> torch.Tensor:
        arr = np.asarray(X, dtype=np.float32)
        if self.name == "mlp" and arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        elif self.name in ("conv1d", "lstm") and arr.ndim == 2:
            arr = arr[:, np.newaxis, :]
        return torch.from_numpy(arr)

    def fit(self, X_train: np.ndarray) -> "AutoencoderDetector":
        """Train autoencoder strictly on healthy training data."""
        torch.manual_seed(self.seed)
        x_tensor = self._prepare_tensor(X_train)
        self.input_shape = tuple(x_tensor.shape[1:])

        self.net = self._build_network(self.input_shape).to(self.device)

        epochs = int(self.train_cfg.get("epochs", 150))
        batch_size = int(self.train_cfg.get("batch_size", 32))
        lr = float(self.train_cfg.get("learning_rate", 1e-3))
        weight_decay = float(self.train_cfg.get("weight_decay", 1e-4))
        es_patience = int(self.train_cfg.get("early_stopping_patience", 30))
        min_delta = float(self.train_cfg.get("min_delta", 1e-5))
        sched_patience = int(self.train_cfg.get("lr_scheduler_patience", 15))
        sched_factor = float(self.train_cfg.get("lr_scheduler_factor", 0.5))

        gen = torch.Generator()
        gen.manual_seed(self.seed)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_factor, patience=sched_patience
        )
        early_stopper = TrainConvergenceEarlyStopping(patience=es_patience, min_delta=min_delta)

        self.history = TrainingHistory()
        self.net.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_total = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

                bs = batch_x.size(0)
                epoch_loss += float(loss.item()) * bs
                n_total += bs

            avg_loss = epoch_loss / max(n_total, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            self.history.train_loss.append(avg_loss)
            self.history.learning_rates.append(current_lr)

            scheduler.step(avg_loss)
            if early_stopper.step(avg_loss, epoch, self.net):
                self.history.stopped_early = True
                break

        early_stopper.restore_best_weights(self.net)
        self.history.best_epoch = early_stopper.best_epoch
        self.history.best_loss = early_stopper.best_loss
        self.net.eval()
        return self

    @torch.no_grad()
    def reconstruct_numpy(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("AutoencoderDetector must be fitted before scoring.")
        self.net.eval()
        x_tensor = self._prepare_tensor(X).to(self.device)
        recon = self.net(x_tensor).cpu().numpy()
        return recon

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        x_in = self._prepare_tensor(X).numpy()
        x_hat = self.reconstruct_numpy(X)
        sq_err = (x_in - x_hat) ** 2
        # Mean over all non-batch dimensions
        axes = tuple(range(1, sq_err.ndim))
        return np.mean(sq_err, axis=axes)

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-feature squared reconstruction error of shape (n_samples, n_features).
        For sequence inputs (n_samples, L, n_features), averages across the L time steps.
        """
        x_in = self._prepare_tensor(X).numpy()
        x_hat = self.reconstruct_numpy(X)
        sq_err = (x_in - x_hat) ** 2
        if sq_err.ndim == 3:
            # Average over time window axis L -> (n_samples, n_features)
            return np.mean(sq_err, axis=1)
        return sq_err

    def get_metadata(self) -> dict[str, Any]:
        n_params = sum(p.numel() for p in self.net.parameters()) if self.net is not None else 0
        return {
            "model_type": f"{self.name.upper()}Autoencoder",
            "input_shape": list(self.input_shape),
            "latent_dim": self.model_cfg.get("latent_dim", 4),
            "n_parameters": int(n_params),
            "epochs_trained": len(self.history.train_loss),
            "best_epoch": int(self.history.best_epoch),
            "best_train_mse": round(float(self.history.best_loss), 6),
            "stopped_early": bool(self.history.stopped_early),
            "device": self.device_desc,
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        """Save PyTorch state_dict and metadata to a .pt file."""
        if self.net is None:
            raise RuntimeError("Cannot save an unfitted AutoencoderDetector.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arch": self.name,
            "input_shape": self.input_shape,
            "model_cfg": self.model_cfg,
            "train_cfg": self.train_cfg,
            "seed": self.seed,
            "state_dict": self.net.state_dict(),
            "history": {
                "train_loss": self.history.train_loss,
                "learning_rates": self.history.learning_rates,
                "best_epoch": self.history.best_epoch,
                "best_loss": self.history.best_loss,
                "stopped_early": self.history.stopped_early,
            },
        }
        torch.save(payload, p)

