"""SNR classification and the SNR-adaptive starting timestep.

The paper's second contribution is to *predict* the noise level of a sample
covariance and to start the reverse chain where the forward process has the same
SNR, which shortens sampling and conditions the prior on the measurement.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from covdiff.diffusion.schedules import GaussianSchedule
from covdiff.utils import hermitian

__all__ = [
    "SNRClassifier",
    "snr_class_index",
    "eigenvalues",
    "train_snr_classifier",
    "predict_snr_db",
    "predict_start_timestep",
]

_EPS = 1e-12


class SNRClassifier(nn.Module):
    """MLP mapping a normalised eigenvalue profile to an SNR class.

    Parameters
    ----------
    n_sensors:
        Length of the eigenvalue vector (``M``).
    n_classes:
        Number of SNR classes (``9`` for the paper's SNR grid).
    hidden:
        Widths of the hidden layers; ReLU activations throughout.

    Notes
    -----
    The input is scaled by its own mean, so the network sees the *shape* of the
    eigenvalue profile rather than the absolute power level.
    """

    def __init__(
        self,
        n_sensors: int,
        n_classes: int = 9,
        hidden: Sequence[int] = (256, 512, 256),
    ):
        super().__init__()
        if n_sensors < 1:
            raise ValueError("n_sensors must be >= 1")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_sensors = int(n_sensors)
        self.n_classes = int(n_classes)

        layers: list[nn.Module] = []
        prev = self.n_sensors
        for width in hidden:
            width = int(width)
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, self.n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, eigvals: Tensor) -> Tensor:
        """Return ``(B, n_classes)`` logits from ``(B, n_sensors)`` eigenvalues."""
        if eigvals.dim() != 2:
            raise ValueError("expected a (B, n_sensors) eigenvalue batch")
        if eigvals.shape[1] != self.n_sensors:
            raise ValueError(
                f"expected {self.n_sensors} eigenvalues, got {eigvals.shape[1]}"
            )
        scale = eigvals.mean(dim=1, keepdim=True).clamp_min(_EPS)
        return self.net(eigvals / scale)


def eigenvalues(R: np.ndarray) -> np.ndarray:
    """Descending real eigenvalues of a Hermitian matrix."""
    matrix = np.asarray(R)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("expected a square matrix")
    values = np.linalg.eigvalsh(hermitian(matrix.astype(np.complex128)))
    return np.ascontiguousarray(values[::-1].real)


def snr_class_index(snr_db: float, snr_grid: Sequence[float]) -> int:
    """Index of the grid point closest to ``snr_db``."""
    grid = np.asarray(snr_grid, dtype=np.float64).reshape(-1)
    if grid.size == 0:
        raise ValueError("snr_grid must be non-empty")
    return int(np.argmin(np.abs(grid - float(snr_db))))


def _features(R_hats: Sequence[np.ndarray]) -> np.ndarray:
    """Stack descending eigenvalue vectors of a sequence of matrices."""
    return np.stack([eigenvalues(R) for R in R_hats]).astype(np.float32)


def train_snr_classifier(
    model: SNRClassifier,
    R_hats: Sequence[np.ndarray],
    snr_db: Sequence[float],
    snr_grid: Sequence[float],
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Any = "cpu",
    seed: int = 0,
) -> SNRClassifier:
    """Fit the SNR classifier on a set of sample covariances.

    Parameters
    ----------
    model:
        Classifier to train (mutated in place and returned).
    R_hats:
        Sequence of ``(M, M)`` covariance matrices.
    snr_db:
        True SNR of each matrix in dB.
    snr_grid:
        The class grid; labels are the nearest grid index.
    epochs, lr, batch_size:
        Plain Adam / cross-entropy training hyper-parameters.
    device:
        Torch device.
    seed:
        Seed for the shuffling generator.

    Returns
    -------
    SNRClassifier
        The trained model, in training mode as left by the last epoch.
    """
    features = _features(R_hats)
    labels = np.asarray(
        [snr_class_index(s, snr_grid) for s in np.asarray(snr_db).reshape(-1)],
        dtype=np.int64,
    )
    if features.shape[0] != labels.size:
        raise ValueError("R_hats and snr_db must have the same length")
    if features.shape[0] == 0:
        raise ValueError("no training samples given")
    grid = np.asarray(snr_grid, dtype=np.float64).reshape(-1)
    if grid.size != model.n_classes:
        raise ValueError(
            f"snr_grid has {grid.size} entries but the model predicts "
            f"{model.n_classes} classes"
        )

    dev = torch.device(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=dev)
    y = torch.as_tensor(labels, dtype=torch.long, device=dev)

    model = model.to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(int(seed))
    n = x.shape[0]
    batch_size = max(1, int(batch_size))

    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=generator).to(dev)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    return model


@torch.no_grad()
def _logits_for(model: SNRClassifier, R: np.ndarray, device: Any) -> Tensor:
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    try:
        feats = _features([R])
        x = torch.as_tensor(feats, dtype=torch.float32, device=dev)
        return model(x)[0]
    finally:
        model.train(was_training)


@torch.no_grad()
def predict_snr_db(
    model: SNRClassifier,
    R: np.ndarray,
    snr_grid: Sequence[float],
    device: Any = "cpu",
) -> float:
    """Posterior-mean SNR estimate in dB for one covariance matrix."""
    grid = torch.as_tensor(
        np.asarray(snr_grid, dtype=np.float64).reshape(-1),
        dtype=torch.float32,
        device=torch.device(device),
    )
    logits = _logits_for(model, R, device)
    probs = torch.softmax(logits, dim=-1)[: grid.numel()]
    return float((probs * grid).sum().item())


@torch.no_grad()
def predict_start_timestep(
    model: SNRClassifier,
    R: np.ndarray,
    schedule: GaussianSchedule,
    snr_grid: Sequence[float],
    device: Any = "cpu",
) -> int:
    """Timestep of ``schedule`` whose forward SNR matches the predicted SNR."""
    snr_db = predict_snr_db(model, R, snr_grid, device=device)
    return int(schedule.timestep_for_snr_db(snr_db))
