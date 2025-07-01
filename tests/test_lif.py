from __future__ import annotations

import numpy as np

from src.models.lif import LIFParameters, initialize_lif_state, lif_step


def test_lif_spikes_with_high_current():
    params = LIFParameters(tau_m=0.02, v_rest=-0.065, v_reset=-0.065, v_th=-0.052, r=1.0, t_ref=0.002)
    state = initialize_lif_state(batch_size=1, num_neurons=1, params=params)
    dt = 0.001
    # Constant high current
    spikes_over_time = []
    for _ in range(50):
        I = np.array([[0.8]], dtype=np.float64)
        state, spikes = lif_step(I, state, params, dt)
        spikes_over_time.append(bool(spikes[0, 0]))
    assert any(spikes_over_time)


