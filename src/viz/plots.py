from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_spike_raster(spikes: NDArray[np.bool_], title: str = "Spike raster") -> None:
    """Plot raster for (T, N) or (T, B, N) spikes.

    If batch is present, plots the first element.
    """
    if spikes.ndim == 3:
        spikes = spikes[:, 0]
    T, N = spikes.shape
    t_idx, n_idx = np.where(spikes)
    plt.figure(figsize=(8, 4))
    plt.scatter(t_idx, n_idx, s=2)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.tight_layout()


def plot_membrane_trace(v: NDArray[np.float64], title: str = "Membrane potential") -> None:
    """Plot membrane potentials (T, N) or (T, B, N). Uses first neuron/batch."""
    if v.ndim == 3:
        v = v[:, 0, 0]
    elif v.ndim == 2:
        v = v[:, 0]
    plt.figure(figsize=(8, 3))
    plt.plot(v)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("V")
    plt.tight_layout()


def plot_learning_curve(values: list[float], title: str = "Learning curve") -> None:
    plt.figure(figsize=(6, 3))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.tight_layout()


__all__ = [
    "plot_spike_raster",
    "plot_membrane_trace",
    "plot_learning_curve",
]


