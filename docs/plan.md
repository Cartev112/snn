## Project: Spiking Neural Network (SNN) from Scratch

### 1) Objectives
- **Build from scratch**: Implement a minimal yet complete SNN framework (no deep dependencies) with clear, readable Python code.
- **Deliver an application**: Keyword spotting (KWS) on Google Speech Commands with spike-based audio encoding; stretch: wake-word detection.
- **Learning approaches**: Start with unsupervised STDP for feature formation + simple readout. Stretch: supervised training via surrogate gradients.
- **Quality**: Reproducible experiments, unit-tested core, visualizations (spike rasters, membrane traces), and a CLI/notebook UX.

### 2) Success Criteria
- **KWS accuracy**: >= 90% top-1 on a 10-keyword subset of Speech Commands with <1 s audio window and <= 100 ms decision latency post-window.
- **Runtime**: <= 5 minutes per training epoch on CPU for a 2-layer network with vectorized simulation (10k samples subset acceptable for dev).
- **Explainability**: Plots of spike rasters, membrane traces, weight histograms. Clear documentation describing dynamics and learning rules.
- **Stretch**: Robustness to background noise; false-positive rate < 1% in a wake-word setting.

### 3) Scope
- **In-scope**:
  - LIF neuron model, current-based exponential synapses, refractory handling, resets.
  - Input encoders: audio front-ends (log-mel spectrogram or gammatone filterbank) with Poisson rate or latency coding; optional event-based mic later.
  - Simulation engine: discrete-time, vectorized NumPy implementation.
  - Learning rules: Unsupervised STDP for hidden layers; readout via spike-count linear classifier or logistic regression. Stretch: surrogate gradient.
  - Training/evaluation loop; metrics; visualizations; CLI and notebooks.
- **Out-of-scope (initial)**:
  - Multi-GPU or large-scale distributed training.
  - Complex neuron models (HH, Izhikevich). Plasticity beyond vanilla STDP/weight normalization.
  - Neuromorphic hardware deployment.

### 4) Application Use Case
- **Primary**: Keyword spotting on Google Speech Commands using spike trains derived from audio features.
- **Stretch**: Wake-word detection with background noise and rejection of unknown words.

### 5) System Architecture
- **`src/encoders/`**
  - `audio_features.py`: Compute log-mel or gammatone features from raw audio.
  - `poisson_rate.py`: Convert non-negative feature frames to spike trains over T steps given per-bin rates.
  - `latency.py` (optional): Map feature magnitudes to spike latencies.
- **`src/models/`**
  - `lif.py`: Vectorized LIF neuron update, refractory, threshold/reset.
  - `synapse.py`: Exponential synapse dynamics, weight application, delays (optional minimal support).
- **`src/network/`**
  - `layers.py`: Dense feedforward layers; optional convolutional layer (stretch).
  - `snn.py`: Network container, forward simulation over time.
- **`src/learning/`**
  - `stdp.py`: Pair-based STDP with pre/post traces; weight clamping/normalization.
  - `surrogate.py` (stretch): BPTT with surrogate gradient for spiking nonlinearity.
- **`src/engine/`**
  - `sim.py`: Discrete-time loop, manages state tensors across T.
- **`src/train/`**
  - `train.py`: Training loop, batching, evaluation, checkpointing.
- **`src/viz/`**
  - `plots.py`: Spike rasters, membrane traces, learning curves, confusion matrix.
- **`cli/`**
  - `snn_cli.py`: Run experiments: encode, train, eval, plot.

### 6) Mathematical Model (Discrete-Time)
- **LIF membrane potential** (per neuron):
  \[ v[t+1] = v[t] + \Delta t \big( -\frac{v[t] - v_{rest}}{\tau_m} + R\,I[t] \big) \]
  - If \( v[t+1] \geq v_{th} \) and not refractory: emit spike, set \( v[t+1] \leftarrow v_{reset} \), enter refractory for \( t_{ref} \) steps.
- **Exponential synapse current**:
  \[ I[t+1] = I[t] \cdot e^{-\Delta t/\tau_s} + W \cdot s_{pre}[t] \]
  where \( s_{pre}[t] \in \{0,1\}^{N_{pre}} \) are presynaptic spikes.
- **Spike generation**: \( s[t] = \mathbb{1}[v[t] \geq v_{th} \wedge \neg \text{refractory}] \)
- **Poisson rate encoding**: Given non-negative feature magnitude \(x \in [0, +\infty)\) scaled to [0,1] and max rate \(r_{max}\): \( s[t] \sim \text{Bernoulli}(x\,r_{max}\,\Delta t) \).

### 7) Learning Rules
- **Pair-based STDP (unsupervised)**:
  - Maintain pre/post traces: \( x_{pre}, x_{post} \) with decay constants \( \tau_{pre}, \tau_{post} \).
  - On presynaptic spike: \( x_{pre} \leftarrow x_{pre} + 1 \), weight potentiation on recent post: \( \Delta w \propto A_+ x_{post} \).
  - On postsynaptic spike: \( x_{post} \leftarrow x_{post} + 1 \), weight depression on recent pre: \( \Delta w \propto -A_- x_{pre} \).
  - Clip or normalize weights to maintain stability; optional homeostasis via target firing rates.
- **Readout**:
  - Spike-count features per class window; train linear/logistic classifier on counts (non-spiking readout) or spiking tempotron (optional).
- **Surrogate gradient (stretch)**:
  - Backprop through time with surrogate derivative of Heaviside: e.g., fast-sigmoid \( \sigma'(v) = 1/(1+|v|)^2 \) around threshold.

### 8) Datasets & Preprocessing
- **Google Speech Commands**: 16 kHz audio; trim/pad to 1 s; compute 40 log-mel features (25 ms window, 10 ms hop) or gammatone filterbank; normalize per-utterance. Encode features to spikes via Poisson or latency coding over T=100 steps.
- **Stretch**: Background noise augmentation (speech-noise mix), unknown word rejection via threshold on readout confidence.

### 9) Evaluation & Metrics
- **Accuracy**: Top-1 accuracy on 10-keyword split; unknown/reject rate for out-of-vocabulary (optional).
- **Latency**: Steps to decision; early-stopping once class confidence crosses threshold (optional).
- **Sparsity/Energy proxy**: Average spikes per neuron per sample; peak memory usage.
- **Stability**: Firing rate distributions; weight histograms over training.

### 10) Experiments
- **E1 Baseline**: Audio features → 256 hidden LIF → spike-count readout; no plasticity; train only readout on counts.
- **E2 STDP features**: STDP on input→hidden; freeze; train readout on hidden spike counts.
- **E3 Two-layer SNN**: input→hidden (STDP) → output readout. Compare coding (Poisson vs latency) and window sizes.
- **E4 Surrogate gradient (stretch)**: End-to-end supervised training; compare to E2/E3.
- **E5 Noise robustness**: Evaluate with background noise and unknown words.

### 11) Milestones
- **M1 (Day 1-2)**: Repo scaffold, LIF + synapse, audio feature encoder, basic sim loop, smoke tests.
- **M2 (Day 3-4)**: STDP implementation, training loop for E2, plots (rasters/membrane), baseline KWS result.
- **M3 (Day 5-6)**: Two-layer network, evaluation metrics, CLI + notebook, reach >=90% on 10-keyword task.
- **M4 (Stretch)**: Surrogate gradient or wake-word pipeline; performance tuning (Numba); docs polishing.

### 12) Risks & Mitigations
- **Instability in STDP**: Use weight clipping/normalization; homeostatic targets; tune A+/A- and time constants.
- **Slow simulations**: Vectorize over batch/time; reduce T; optionally enable Numba; profile hotspots.
- **Encoding mismatch**: Calibrate r_max and \(\Delta t\); visualize input spike rates.
- **Reproducibility**: Seed RNG; store configs; log metrics and versions.

### 13) Development Environment
- **Language**: Python 3.10+
- **Core deps**: numpy, matplotlib, scikit-learn (readout), tqdm, jupyter; optional: numba.
- **OS**: Windows 10 (dev), cross-platform assumed.

### 14) Expected Repository Layout
```
docs/
  plan.md
src/
  encoders/
  engine/
  learning/
  models/
  network/
  train/
  viz/
cli/
data/
notebooks/
tests/
```

### 15) Work Plan (Mapping to Tasks)
- **Plan document** → this file.
- **Scaffold structure** → create directories and __init__.py files.
- **Core models** → LIF neuron, synapse dynamics.
- **Encoders** → Audio features; Poisson/latency coding.
- **Sim engine** → Vectorized time stepping.
- **Learning rules** → STDP first; surrogate gradient later.
- **Datasets** → MNIST loader and encoder wrappers.
- **Train/Eval** → Loops, metrics, checkpoints.
- **Viz tools** → Rasters, traces, curves.
- **CLI/Notebooks** → Reproducible experiments.
- **Tests/CI** → Unit tests for dynamics and encoders.
- **Performance** → Profiling, vectorization, optional Numba.
- **Application demo** → Keyword spotting end-to-end; stretch: wake-word detection.

### 16) Open Questions
- Preferred stretch: N-MNIST vs surrogate gradient training?
- Target accuracy/latency tradeoff (T steps, r_max) priorities?
- Any constraints on dependencies or environment?


