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
class LIFParameters:
    """Leaky Integrate-and-Fire (current-based) neuron parameters.

    All time quantities are in seconds, voltages in arbitrary units, and current
    in arbitrary units such that R * I has units of voltage.
    """

    tau_m: float  # Membrane time constant (s)
    v_rest: float = -0.065  # Resting potential (V)
    v_reset: float = -0.065  # Reset potential after spike (V)
    v_th: float = -0.052  # Firing threshold (V)
    r: float = 1.0  # Membrane resistance (Ohm, arbitrary scale)
    t_ref: float = 0.002  # Absolute refractory period (s)


@dataclass
class LIFState:
    """State of a batch of LIF neurons.

    v: membrane potentials with shape (batch_size, num_neurons)
    refractory_steps_remaining: integer steps remaining in refractory period per neuron
    """

    v: NDArray[np.float64]
    refractory_steps_remaining: NDArray[np.int32]


def _compute_refractory_steps(params: LIFParameters, dt: float) -> int:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    # Use ceil to guarantee at least 1 step for small t_ref
    return int(np.ceil(params.t_ref / dt)) if params.t_ref > 0 else 0


def initialize_lif_state(
    batch_size: int,
    num_neurons: int,
    params: LIFParameters,
    v_init: float | None = None,
) -> LIFState:
    """Initialize membrane potentials to v_init or v_rest and zero refractory.

    Args:
        batch_size: Number of samples in the batch
        num_neurons: Number of neurons
        params: LIFParameters
        v_init: Optional explicit initial voltage; defaults to params.v_rest
    """
    v0 = params.v_rest if v_init is None else float(v_init)
    v = np.full((batch_size, num_neurons), v0, dtype=np.float64)
    refractory = np.zeros((batch_size, num_neurons), dtype=np.int32)
    return LIFState(v=v, refractory_steps_remaining=refractory)


def lif_step(
    input_current: NDArray[np.float64],
    state: LIFState,
    params: LIFParameters,
    dt: float,
) -> Tuple[LIFState, NDArray[np.bool_]]:
    """Advance LIF state by one time step.

    Implements:
        v[t+1] = v[t] + dt * (-(v[t]-v_rest)/tau_m + r*I[t])
    Spikes occur when v[t+1] >= v_th and neuron is not refractory. On spike,
    v is set to v_reset and a refractory timer is started.

    Args:
        input_current: shape (batch_size, num_neurons)
        state: current LIFState
        params: neuron parameters
        dt: time step in seconds

    Returns:
        (new_state, spikes) where spikes is a bool array of shape like v
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    v = state.v
    refractory = state.refractory_steps_remaining

    if input_current.shape != v.shape:
        raise ValueError(
            f"input_current shape {input_current.shape} must match state.v shape {v.shape}"
        )

    # Neurons currently refractory are held at v_reset and do not integrate
    is_refractory = refractory > 0

    # Euler update for non-refractory neurons
    dv = dt * (-(v - params.v_rest) / params.tau_m + params.r * input_current)
    v_updated = v + dv

    # Enforce refractory hold at v_reset
    v_updated = np.where(is_refractory, params.v_reset, v_updated)

    # Determine spikes for neurons allowed to spike
    can_spike = ~is_refractory
    spikes = np.logical_and(can_spike, v_updated >= params.v_th)

    # Apply reset for spikes
    v_updated = np.where(spikes, params.v_reset, v_updated)

    # Update refractory timers: start timer for spiking neurons, decrement active ones
    refrac_steps = _compute_refractory_steps(params, dt)
    new_refractory = refractory.copy()
    new_refractory = np.where(spikes, refrac_steps, new_refractory)
    # Decrement only those currently in refractory and not just set by spike
    dec_mask = (new_refractory > 0) & ~spikes
    new_refractory = np.where(dec_mask, new_refractory - 1, new_refractory).astype(np.int32)

    return LIFState(v=v_updated.astype(np.float64), refractory_steps_remaining=new_refractory), spikes.astype(np.bool_)


# Optional JIT kernel for batch-wise LIF update
def _lif_step_jit(
    input_current: NDArray[np.float64],
    v: NDArray[np.float64],
    refractory: NDArray[np.int32],
    dt: float,
    tau_m: float,
    v_rest: float,
    v_reset: float,
    v_th: float,
    r: float,
    t_ref_steps: int,
):
    B, N = v.shape
    spikes = np.zeros((B, N), dtype=np.bool_)
    v_out = v.copy()
    refr_out = refractory.copy()
    for b in range(B):
        for n in range(N):
            if refr_out[b, n] > 0:
                v_out[b, n] = v_reset
                refr_out[b, n] -= 1
                continue
            dv = dt * (-(v_out[b, n] - v_rest) / tau_m + r * input_current[b, n])
            vv = v_out[b, n] + dv
            if vv >= v_th:
                spikes[b, n] = True
                v_out[b, n] = v_reset
                refr_out[b, n] = t_ref_steps
            else:
                v_out[b, n] = vv
    return v_out, refr_out, spikes


if njit is not None:  # pragma: no cover
    _lif_step_jit = njit(_lif_step_jit)  # type: ignore


def lif_step_fast(
    input_current: NDArray[np.float64],
    state: LIFState,
    params: LIFParameters,
    dt: float,
) -> Tuple[LIFState, NDArray[np.bool_]]:
    """Numba-accelerated LIF step if numba is available; falls back to lif_step."""
    if njit is None:
        return lif_step(input_current, state, params, dt)
    refr_steps = _compute_refractory_steps(params, dt)
    v_out, refr_out, spikes = _lif_step_jit(
        input_current.astype(np.float64),
        state.v.astype(np.float64),
        state.refractory_steps_remaining.astype(np.int32),
        float(dt),
        float(params.tau_m),
        float(params.v_rest),
        float(params.v_reset),
        float(params.v_th),
        float(params.r),
        int(refr_steps),
    )
    return LIFState(v=v_out, refractory_steps_remaining=refr_out), spikes


__all__ = [
    "LIFParameters",
    "LIFState",
    "initialize_lif_state",
    "lif_step",
    "lif_step_fast",
]


