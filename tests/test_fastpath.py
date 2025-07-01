from __future__ import annotations

import numpy as np

from src.models.lif import LIFParameters, initialize_lif_state, lif_step, lif_step_fast
from src.models.synapse import ExponentialSynapseParameters, ExponentialSynapseState, synapse_step, synapse_step_fast


def test_fastpath_matches_numpy_lif_and_synapse():
    rng = np.random.default_rng(0)
    B, Npre, Npost = 4, 8, 5
    dt = 0.001

    lifp = LIFParameters(tau_m=0.02, v_rest=-0.065, v_reset=-0.065, v_th=-0.052, r=1.0, t_ref=0.002)
    synp = ExponentialSynapseParameters(tau_s=0.005)

    w = rng.normal(0, 0.1, size=(Npre, Npost)).astype(np.float64)
    syn_state_np = ExponentialSynapseState(current=np.zeros((B, Npost), dtype=np.float64))
    syn_state_jit = ExponentialSynapseState(current=np.zeros((B, Npost), dtype=np.float64))

    lif_state_np = initialize_lif_state(B, Npost, lifp)
    lif_state_jit = initialize_lif_state(B, Npost, lifp)

    for _ in range(10):
        pre = rng.random((B, Npre)) < 0.1
        syn_state_np, I_np = synapse_step(pre, w, syn_state_np, synp, dt)
        syn_state_jit, I_jit = synapse_step_fast(pre, w, syn_state_jit, synp, dt)
        assert np.allclose(I_np, I_jit, atol=1e-8)

        lif_state_np, s_np = lif_step(I_np, lif_state_np, lifp, dt)
        lif_state_jit, s_jit = lif_step_fast(I_jit, lif_state_jit, lifp, dt)
        assert np.array_equal(s_np, s_jit)
        assert np.allclose(lif_state_np.v, lif_state_jit.v, atol=1e-8)


