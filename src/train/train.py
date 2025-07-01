from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.engine.sim import DenseLayer, run_feedforward_two_layer
from src.models.lif import LIFParameters
from src.models.synapse import ExponentialSynapseParameters


@dataclass
class TwoLayerModel:
    layer1: DenseLayer
    layer2: DenseLayer


def build_default_two_layer(num_input: int, num_hidden: int, num_output: int, rng: np.random.Generator | None = None) -> TwoLayerModel:
    if rng is None:
        rng = np.random.default_rng(0)
    w1 = rng.normal(loc=0.0, scale=0.1, size=(num_input, num_hidden)).astype(np.float64)
    w2 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_output)).astype(np.float64)

    lif1 = LIFParameters(tau_m=0.02, v_rest=-0.065, v_reset=-0.065, v_th=-0.052, r=1.0, t_ref=0.002)
    lif2 = LIFParameters(tau_m=0.02, v_rest=-0.065, v_reset=-0.065, v_th=-0.052, r=1.0, t_ref=0.002)
    syn = ExponentialSynapseParameters(tau_s=0.005)

    layer1 = DenseLayer(weights=w1, lif_params=lif1, syn_params=syn)
    layer2 = DenseLayer(weights=w2, lif_params=lif2, syn_params=syn)
    return TwoLayerModel(layer1=layer1, layer2=layer2)


def simulate_get_counts(
    model: TwoLayerModel,
    batch_spikes: NDArray[np.bool_],  # (T, B, num_input)
    dt: float,
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    # Run two-layer network
    s1, s2 = run_feedforward_two_layer(batch_spikes, model.layer1, model.layer2, dt=dt, batch_size=batch_spikes.shape[1])
    # Spike counts per sample
    counts1 = s1.sum(axis=0).astype(np.int32)  # (B, H)
    counts2 = s2.sum(axis=0).astype(np.int32)  # (B, O)
    return counts1, counts2


def train_softmax(
    features: NDArray[np.float64],  # (N, D)
    labels: NDArray[np.int32],  # (N,)
    num_classes: int,
    lr: float = 0.1,
    epochs: int = 50,
    reg_lambda: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Train a multinomial logistic regression with simple gradient descent.

    Returns (W, b) where W shape (D, C), b shape (C,)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N, D = features.shape
    C = int(num_classes)
    W = rng.normal(0, 0.01, size=(D, C)).astype(np.float64)
    b = np.zeros((C,), dtype=np.float64)

    y_onehot = np.eye(C, dtype=np.float64)[labels]
    for _ in range(epochs):
        logits = features @ W + b  # (N, C)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-9, None)
        # Gradient
        grad_logits = (probs - y_onehot) / N
        grad_W = features.T @ grad_logits + reg_lambda * W
        grad_b = grad_logits.sum(axis=0)
        # Update
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def predict_softmax(features: NDArray[np.float64], W: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.int32]:
    logits = features @ W + b
    return np.argmax(logits, axis=1).astype(np.int32)


def accuracy(pred: NDArray[np.int32], labels: NDArray[np.int32]) -> float:
    return float((pred == labels).mean())


__all__ = [
    "TwoLayerModel",
    "build_default_two_layer",
    "simulate_get_counts",
    "train_softmax",
    "predict_softmax",
    "accuracy",
]


