from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    sf = None  # type: ignore


def compute_log_mel_spectrogram(
    audio: NDArray[np.float64],
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 40,
    fmin: float = 20.0,
    fmax: float | None = 7600.0,
    eps: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute a simple log-mel spectrogram using NumPy.

    Notes:
        - This is a lightweight implementation to avoid heavy deps; for production
          use, consider librosa/torchaudio.
        - Uses Hann window, magnitude STFT, Mel filterbank, log(1+x).
    Returns:
        (time_frames, n_mels) array normalized to [0,1] via per-utterance max.
    """
    # Window function
    window = np.hanning(win_length)

    # Framing
    num_frames = 1 + (len(audio) - win_length) // hop_length if len(audio) >= win_length else 1
    frames = np.zeros((num_frames, win_length), dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        segment = audio[start:end]
        if len(segment) < win_length:
            padded = np.zeros(win_length, dtype=np.float64)
            padded[: len(segment)] = segment
            segment = padded
        frames[i] = segment * window

    # STFT magnitude
    fft_size = n_fft
    spec = np.fft.rfft(frames, n=fft_size, axis=1)
    mag = np.abs(spec)

    # Mel filterbank
    mel_fb = _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_spec = mag @ mel_fb.T

    # Log compression
    log_mel = np.log1p(np.maximum(mel_spec, 0.0) + eps)

    # Per-utterance normalization to [0,1]
    log_mel -= log_mel.min()
    maxv = log_mel.max() + eps
    log_mel /= maxv
    return log_mel.astype(np.float64)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> NDArray[np.float64]:
    if fmax is None:
        fmax = sr / 2
    # Convert Hz to Mel and back using HTK formula
    def hz_to_mel(f: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m: NDArray[np.float64]) -> NDArray[np.float64]:
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft // 2 + 1) * hz / (sr / 2)).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left or right <= center:
            continue
        fb[i - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        fb[i - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return fb


def poisson_encode_features(
    features: NDArray[np.float64],
    timesteps: int,
    r_max: float = 100.0,
    dt: float = 0.01,
    rng: np.random.Generator | None = None,
) -> NDArray[np.bool_]:
    """Encode non-negative feature matrix (T_frames, F) to spikes over time.

    Args:
        features: 2D array with non-negative values scaled roughly to [0,1]
        timesteps: number of simulation steps to generate
        r_max: maximum firing rate (Hz) corresponding to value 1.0
        dt: time step (s)
        rng: optional numpy Generator

    Returns:
        spikes: (timesteps, F) boolean array
    """
    if rng is None:
        rng = np.random.default_rng()
    features = np.maximum(features, 0.0)
    # Collapse time frames to a feature vector by averaging across time
    feature_vec = features.mean(axis=0)
    lam = feature_vec * r_max * dt  # per-feature Bernoulli probs per step
    lam = np.clip(lam, 0.0, 0.95)  # numerical safety
    spikes = rng.random((timesteps, feature_vec.shape[0])) < lam
    return spikes


__all__ = [
    "compute_log_mel_spectrogram",
    "poisson_encode_features",
]


