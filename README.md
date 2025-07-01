## Spiking Neural Network (SNN) - Keyword Spotting

Minimal, from-scratch SNN for keyword spotting on Google Speech Commands. Includes LIF neurons, exponential synapses, STDP, a NumPy-based simulator, and a CLI for baseline and STDP pipelines.

### Why SNNs?
- **Temporal coding**: Information is carried by discrete spikes over time rather than continuous activations.
- **Sparsity**: Spikes are mostly zeros, making computation/event-driven execution efficient.
- **Biological plausibility**: Local learning rules like STDP rely only on pre/post activity.

### Setup
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### Data
- Download Google Speech Commands (v0.02) and point `--data_root` to its folder. See the dataset page for details.

### Run
- Baseline (readout on hidden spike counts):
```bash
python -m cli.snn_cli --data_root "C:\path\to\speech_commands" --mode baseline
```
- STDP pretraining + readout:
```bash
python -m cli.snn_cli --data_root "C:\path\to\speech_commands" --mode stdp
```

Common flags:
- `--batch_size` (default 32), `--timesteps` (default 100), `--dt` (default 0.01 s), `--hidden` (default 256), `--max_batches` (quick demo cap).

### End-to-End Pipeline
1) **Audio → Features** (`src/encoders/audio_features.py`)
   - Compute 40-bin log-mel spectrogram from 1 s audio windows at 16 kHz.
   - Normalize per utterance to [0,1].
2) **Features → Spikes**
   - Poisson rate coding: for each feature bin with magnitude \(x \in [0,1]\), generate Bernoulli spikes each step with probability \(p = x\, r_{max}\, \Delta t\).
3) **Spiking Network Simulation** (`src/engine/sim.py`)
   - Two dense layers: input→hidden, hidden→output; exponential synapses + LIF neurons.
   - Run for `T` discrete steps of size `dt`.
4) **Readout** (`src/train/train.py`)
   - Accumulate hidden-layer spike counts per sample, train a softmax (multinomial logistic regression) on counts.
5) **(Optional) Unsupervised STDP** (`src/learning/stdp.py`)
   - Pretrain input→hidden weights with STDP on unlabeled (or labeled but unused) batches, then train the readout.

### Mathematical Model
- **Exponential synapse (current-based)**
  \[ I[t+1] = I[t]\,e^{-\Delta t/\tau_s} + S_{pre}[t] \cdot W \]
  where \(S_{pre}[t] \in \{0,1\}^{B\times N_{pre}}\) and \(W \in \mathbb{R}^{N_{pre}\times N_{post}}\).
- **LIF neuron dynamics**
  \[ v[t+1] = v[t] + \Delta t\Big( -\tfrac{v[t]-v_{rest}}{\tau_m} + R\,I[t] \Big) \]
  - Spike if \(v[t+1] \ge v_{th}\) and not refractory; then set \(v[t+1] \leftarrow v_{reset}\), and start refractory timer \(t_{ref}\).
  - Refractory neurons hold \(v = v_{reset}\) and do not integrate.
- **Spike generation**
  \[ S[t] = \mathbb{1}[v[t] \ge v_{th} \land \neg \text{refractory}] \]

All implementations are fully vectorized over batch and neurons in NumPy.

### How Learning Works: STDP (Spike-Timing-Dependent Plasticity)
We implement a pair-based STDP with exponentially decaying traces. Intuition:
- If a presynaptic neuron fires shortly before a postsynaptic spike, strengthen that synapse (causal correlation).
- If presynaptic firing follows a postsynaptic spike, weaken it (anti-causal correlation).

State per batch step:
- Pre-trace \(x_{pre}\) and post-trace \(x_{post}\) that decay exponentially with \(\tau_{pre}, \tau_{post}\), incremented by spikes.

At each time step t:
1) Decay traces: \(x_{pre} \leftarrow x_{pre}\,e^{-\Delta t/\tau_{pre}}\), \(x_{post} \leftarrow x_{post}\,e^{-\Delta t/\tau_{post}}\).
2) Increment on spikes: \(x_{pre} \leftarrow x_{pre} + S_{pre}[t]\), \(x_{post} \leftarrow x_{post} + S_{post}[t]\).
3) Update weights (additive rule, batched over samples and time via einsum):
   - Potentiation: \(\Delta W \mathrel{+}= A_+ \sum_b S_{pre}^{(b)}[t]^\top x_{post}^{(b)}\)
   - Depression: \(\Delta W \mathrel{-}= A_- \sum_b x_{pre}^{(b)\top} S_{post}^{(b)}[t]\)
4) Clip weights to \([w_{min}, w_{max}]\) to ensure stability.

Practical notes:
- Choose \(A_- > A_+\) slightly to avoid runaway potentiation (we default to 0.012 vs 0.01).
- Use weight clipping or normalization; monitor firing rates and sparsity.
- STDP builds feature detectors in the hidden layer; you still train a simple readout on the resulting spike-count features.

### Baseline vs STDP Pipelines
- **Baseline**
  - Randomly initialized weights, run the network, collect hidden spike counts, train softmax readout.
  - Simple and fast; serves as a sanity check.
- **STDP + Readout**
  - Run the first layer in unsupervised mode over batches, apply STDP online per time step, update input→hidden weights.
  - Freeze those weights; run baseline readout training on the improved spike-count features.

### Configuration Tips
- Increase `--timesteps` for more temporal resolution; reduce for speed. Typical: 50–150.
- Tune `--dt` and `r_max` (in `poisson_encode_features`) to balance spike rates and sparsity.
- Hidden size (`--hidden`) around 128–512 is a good starting range.
- For quick experiments, `--max_batches` limits training.

### Visualization
Use helpers in `src/viz/plots.py`:
- `plot_spike_raster(spikes)`: visualize spike rasters for a batch/time window.
- `plot_membrane_trace(v)`: plot a membrane potential trace.
- `plot_learning_curve(values)`: inspect training curves.

### Performance
- Vectorized NumPy operations minimize Python overhead.
- Reduce feature dimensionality or hidden size to fit memory.
- Consider `numba` JIT for hot loops if extending beyond this minimal baseline.

### Troubleshooting
- Ensure Speech Commands audio is 16 kHz. Files with different rates are skipped.
- If feature magnitudes are all near zero, check normalization; log-mel spectrogram returns values normalized per utterance.
- On Windows, avoid piping `pytest` to `cat`; run `python -m pytest -q` directly.
- If accuracy is low: increase `timesteps`, tune `r_max`, adjust thresholds (`v_th`), and verify spike rates are neither saturating nor silent.

### Reproducibility
- We set fixed random seeds for initial weights and Poisson encoding where feasible.
- Data splits follow Speech Commands `validation_list.txt` and `testing_list.txt` when present.

### Extending to Wake-Word Detection (Stretch)
- Add an "unknown" rejection by thresholding softmax confidence.
- For streaming audio, slide a 1 s window and trigger when confidence exceeds a threshold for K consecutive frames.

### Project Structure
```
src/
  encoders/        # audio features, Poisson encoder
  engine/          # simulation loop
  learning/        # STDP
  models/          # LIF neuron, synapse
  network/         # (future) higher-level nets
  train/           # datasets, training utils
  viz/             # plots
cli/
docs/
tests/
```

### License
MIT


