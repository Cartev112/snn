from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

try:  # optional performance path
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    njit = None  # type: ignore


@dataclass
class ExponentialSynapseParameters:
    """Parameters for an exponential current-based synapse model.

    tau_s: synaptic time constant (s)
    """

    tau_s: float


@dataclass
class ExponentialSynapseState:
    """State of a batch of synapses feeding a postsynaptic population.

    current: postsynaptic current accumulator with shape (batch_size, num_post)
    """

    current: NDArray[np.float64]


def initialize_exponential_state(
    batch_size: int,
    num_post: int,
) -> ExponentialSynapseState:
    """Initialize synaptic current accumulator to zeros."""
    current = np.zeros((batch_size, num_post), dtype=np.float64)
    return ExponentialSynapseState(current=current)


def synapse_step(
    pre_spikes: NDArray[np.bool_],
    weights: NDArray[np.float64],
    state: ExponentialSynapseState,
    params: ExponentialSynapseParameters,
    dt: float,
) -> Tuple[ExponentialSynapseState, NDArray[np.float64]]:
    """Advance exponential synapse state by one step.

    Implements:
        I[t+1] = I[t] * exp(-dt/tau_s) + S_pre[t] @ W

    Args:
        pre_spikes: (batch_size, num_pre) boolean presynaptic spikes
        weights: (num_pre, num_post) weight matrix
        state: ExponentialSynapseState
        params: ExponentialSynapseParameters
        dt: time step in seconds

    Returns:
        (new_state, post_current) where post_current has shape (batch_size, num_post)
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    batch_size, num_pre = pre_spikes.shape
    w_shape = weights.shape
    if w_shape[0] != num_pre:
        raise ValueError(
            f"weights first dim {w_shape[0]} must equal num_pre {num_pre} (weights shape {w_shape})"
        )

    num_post = w_shape[1]
    if state.current.shape != (batch_size, num_post):
        raise ValueError(
            f"state.current shape {state.current.shape} must be (batch_size, num_post)=({batch_size}, {num_post})"
        )

    alpha = float(np.exp(-dt / params.tau_s)) if params.tau_s > 0 else 0.0
    decayed = state.current * alpha
    drive = pre_spikes.astype(np.float64) @ weights  # (batch, num_post)
    new_current = decayed + drive

    return ExponentialSynapseState(current=new_current.astype(np.float64)), new_current.astype(np.float64)


def _synapse_step_jit(
    pre_spikes: NDArray[np.bool_],
    weights: NDArray[np.float64],
    current: NDArray[np.float64],
    dt: float,
    tau_s: float,
):
    B, P = pre_spikes.shape
    P2, Q = weights.shape
    assert P == P2
    out = current.copy()
    alpha = np.exp(-dt / tau_s) if tau_s > 0 else 0.0
    for b in range(B):
        # compute drive = pre[b] @ W
        for q in range(Q):
            drive = 0.0
            for p in range(P):
                drive += (1.0 if pre_spikes[b, p] else 0.0) * weights[p, q]
            out[b, q] = out[b, q] * alpha + drive
    return out


if njit is not None:  # pragma: no cover
    _synapse_step_jit = njit(_synapse_step_jit)  # type: ignore


def synapse_step_fast(
    pre_spikes: NDArray[np.bool_],
    weights: NDArray[np.float64],
    state: ExponentialSynapseState,
    params: ExponentialSynapseParameters,
    dt: float,
) -> Tuple[ExponentialSynapseState, NDArray[np.float64]]:
    if njit is None:
        return synapse_step(pre_spikes, weights, state, params, dt)
    new_current = _synapse_step_jit(pre_spikes, weights.astype(np.float64), state.current.astype(np.float64), float(dt), float(params.tau_s))
    return ExponentialSynapseState(current=new_current), new_current


__all__ = [
    "ExponentialSynapseParameters",
    "ExponentialSynapseState",
    "initialize_exponential_state",
    "synapse_step",
    "synapse_step_fast",
]


