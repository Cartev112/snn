from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class STDPParameters:
    tau_pre: float = 0.02
    tau_post: float = 0.02
    a_plus: float = 0.01
    a_minus: float = 0.012
    w_min: float = 0.0
    w_max: float = 1.0


@dataclass
class STDPState:
    x_pre: NDArray[np.float64]  # (batch, num_pre)
    x_post: NDArray[np.float64]  # (batch, num_post)


def initialize_stdp_state(batch_size: int, num_pre: int, num_post: int) -> STDPState:
    return STDPState(
        x_pre=np.zeros((batch_size, num_pre), dtype=np.float64),
        x_post=np.zeros((batch_size, num_post), dtype=np.float64),
    )


def stdp_step(
    pre_spikes: NDArray[np.bool_],
    post_spikes: NDArray[np.bool_],
    weights: NDArray[np.float64],
    state: STDPState,
    params: STDPParameters,
    dt: float,
) -> Tuple[STDPState, NDArray[np.float64]]:
    """Pair-based STDP with pre/post traces.

    Weight update (additive):
        Δw += a_plus * (pre_spike)ᵀ @ x_post   (potentiation)
        Δw += -a_minus * x_preᵀ @ (post_spike) (depression)
    Traces decay exponentially and are incremented by spikes.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    # Decay traces
    decay_pre = np.exp(-dt / params.tau_pre)
    decay_post = np.exp(-dt / params.tau_post)
    x_pre = state.x_pre * decay_pre
    x_post = state.x_post * decay_post

    # Increment traces on spikes
    x_pre = x_pre + pre_spikes.astype(np.float64)
    x_post = x_post + post_spikes.astype(np.float64)

    # Compute updates
    # Potentiation: pre at t, recent post traces
    pot = pre_spikes.astype(np.float64).transpose(0, 1)  # (batch, num_pre)
    # For batch-wise outer product, use matrix multiply with broadcasting via batch sum
    # Δw_pot = sum_over_batch (pre[:, :, None] * x_post[:, None, :])
    delta_w_pot = np.einsum("bi,bj->ij", pre_spikes.astype(np.float64), x_post)
    delta_w_dep = np.einsum("bi,bj->ij", x_pre, post_spikes.astype(np.float64))

    dW = params.a_plus * delta_w_pot - params.a_minus * delta_w_dep
    new_weights = np.clip(weights + dW, params.w_min, params.w_max)

    return STDPState(x_pre=x_pre, x_post=x_post), new_weights


__all__ = [
    "STDPParameters",
    "STDPState",
    "initialize_stdp_state",
    "stdp_step",
]


