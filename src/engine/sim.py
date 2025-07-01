from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from src.models.lif import LIFParameters, LIFState, lif_step, lif_step_fast
from src.models.synapse import (
    ExponentialSynapseParameters,
    ExponentialSynapseState,
    synapse_step,
    synapse_step_fast,
)


@dataclass
class DenseLayer:
    """Simple dense feedforward spiking layer: pre -> synapse -> LIF neurons."""

    weights: NDArray[np.float64]  # (num_pre, num_post)
    lif_params: LIFParameters
    syn_params: ExponentialSynapseParameters


@dataclass
class DenseLayerState:
    syn_state: ExponentialSynapseState
    lif_state: LIFState


def initialize_dense_layer_state(
    batch_size: int, num_post: int
) -> DenseLayerState:
    return DenseLayerState(
        syn_state=ExponentialSynapseState(current=np.zeros((batch_size, num_post), dtype=np.float64)),
        lif_state=LIFState(
            v=np.zeros((batch_size, num_post), dtype=np.float64),
            refractory_steps_remaining=np.zeros((batch_size, num_post), dtype=np.int32),
        ),
    )


def dense_layer_step(
    pre_spikes: NDArray[np.bool_],
    state: DenseLayerState,
    layer: DenseLayer,
    dt: float,
) -> Tuple[DenseLayerState, NDArray[np.bool_]]:
    syn_state, i_t = synapse_step_fast(pre_spikes, layer.weights, state.syn_state, layer.syn_params, dt)
    lif_state, spikes = lif_step_fast(i_t, state.lif_state, layer.lif_params, dt)
    return DenseLayerState(syn_state=syn_state, lif_state=lif_state), spikes


def run_feedforward_two_layer(
    input_spikes: NDArray[np.bool_],
    layer1: DenseLayer,
    layer2: DenseLayer,
    dt: float,
    batch_size: int,
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Run a two-layer feedforward SNN over time.

    Args:
        input_spikes: (timesteps, batch, num_input)
    Returns:
        (spikes_l1, spikes_l2) each of shape (timesteps, batch, num_neurons)
    """
    T, B, num_input = input_spikes.shape
    assert B == batch_size

    num_hidden = layer1.weights.shape[1]
    num_output = layer2.weights.shape[1]

    state1 = initialize_dense_layer_state(batch_size, num_hidden)
    state2 = initialize_dense_layer_state(batch_size, num_output)

    # Initialize membrane potentials to v_rest for readability
    state1.lif_state.v.fill(layer1.lif_params.v_rest)
    state2.lif_state.v.fill(layer2.lif_params.v_rest)

    spikes1 = np.zeros((T, B, num_hidden), dtype=np.bool_)
    spikes2 = np.zeros((T, B, num_output), dtype=np.bool_)

    for t in range(T):
        s_in = input_spikes[t]
        state1, s1 = dense_layer_step(s_in, state1, layer1, dt)
        state2, s2 = dense_layer_step(s1, state2, layer2, dt)
        spikes1[t] = s1
        spikes2[t] = s2

    return spikes1, spikes2


__all__ = [
    "DenseLayer",
    "DenseLayerState",
    "initialize_dense_layer_state",
    "dense_layer_step",
    "run_feedforward_two_layer",
]


