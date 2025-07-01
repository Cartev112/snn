from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.encoders.audio_features import (
    compute_log_mel_spectrogram,
    poisson_encode_features,
)


DEFAULT_KEYWORDS: Tuple[str, ...] = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)


def _read_wav(path: str) -> Tuple[int, NDArray[np.float64]]:
    """Read WAV file as mono float64 in [-1,1] and return (sample_rate, audio).

    Tries soundfile, then scipy, then stdlib wave. Assumes 16kHz for Speech Commands.
    """
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(path, dtype="float64")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return sr, audio.astype(np.float64)
    except Exception:
        pass

    try:  # pragma: no cover
        from scipy.io import wavfile  # type: ignore

        sr, audio_i = wavfile.read(path)
        if audio_i.ndim > 1:
            audio_i = np.mean(audio_i, axis=1)
        # Normalize depending on dtype
        if np.issubdtype(audio_i.dtype, np.integer):
            maxv = np.iinfo(audio_i.dtype).max
            audio = audio_i.astype(np.float64) / maxv
        else:
            audio = audio_i.astype(np.float64)
        return int(sr), audio
    except Exception:
        pass

    import wave

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n)
    # Convert bytes to np array
    if sampwidth == 2:
        audio_i = np.frombuffer(raw, dtype=np.int16)
        maxv = np.iinfo(np.int16).max
        audio = audio_i.astype(np.float64) / maxv
    elif sampwidth == 1:
        audio_i = np.frombuffer(raw, dtype=np.uint8)
        audio = (audio_i.astype(np.float64) - 128.0) / 128.0
    else:
        # Fallback generic conversion
        audio = np.frombuffer(raw, dtype=np.float64)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return int(sr), audio.astype(np.float64)


@dataclass
class SpeechCommandsSample:
    filepath: str
    label: str


def index_speech_commands(
    root: str, keywords: Sequence[str] = DEFAULT_KEYWORDS
) -> Tuple[List[SpeechCommandsSample], List[SpeechCommandsSample], List[SpeechCommandsSample]]:
    """Index dataset into train/val/test splits using official list files if present.

    Returns: (train, val, test) lists of samples.
    """
    keywords_set = set(keywords)
    all_samples: List[SpeechCommandsSample] = []
    for entry in os.listdir(root):
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir):
            continue
        if entry.startswith("_"):
            continue
        for fname in os.listdir(subdir):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(subdir, fname)
            label = entry if entry in keywords_set else "_unknown_"
            all_samples.append(SpeechCommandsSample(filepath=path, label=label))

    # Use official lists if available
    def _read_list_file(name: str) -> List[str]:
        fpath = os.path.join(root, name)
        if not os.path.exists(fpath):
            return []
        with open(fpath, "r", encoding="utf-8") as f:
            return [line.strip().replace("/", os.sep) for line in f if line.strip()]

    val_list = set(_read_list_file("validation_list.txt"))
    test_list = set(_read_list_file("testing_list.txt"))

    def _rel_path(p: str) -> str:
        return os.path.relpath(p, root)

    train: List[SpeechCommandsSample] = []
    val: List[SpeechCommandsSample] = []
    test: List[SpeechCommandsSample] = []
    if val_list or test_list:
        for s in all_samples:
            rel = _rel_path(s.filepath)
            if rel in test_list:
                test.append(s)
            elif rel in val_list:
                val.append(s)
            else:
                train.append(s)
    else:
        # Simple random split if lists missing
        rng = random.Random(1337)
        rng.shuffle(all_samples)
        n = len(all_samples)
        n_val = int(0.1 * n)
        n_test = int(0.1 * n)
        val = all_samples[:n_val]
        test = all_samples[n_val : n_val + n_test]
        train = all_samples[n_val + n_test :]

    return train, val, test


class SpeechCommandsDataLoader:
    def __init__(
        self,
        root: str,
        keywords: Sequence[str] = DEFAULT_KEYWORDS,
        batch_size: int = 32,
        timesteps: int = 100,
        dt: float = 0.01,
        r_max: float = 100.0,
        n_mels: int = 40,
    ) -> None:
        self.root = root
        self.keywords = list(keywords)
        self.label_to_index: Dict[str, int] = {kw: i for i, kw in enumerate(self.keywords)}
        self.label_to_index["_unknown_"] = len(self.keywords)
        self.num_classes = len(self.keywords) + 1
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.dt = dt
        self.r_max = r_max
        self.n_mels = n_mels
        self.rng = np.random.default_rng(42)

        self.train, self.val, self.test = index_speech_commands(root, self.keywords)

    def _load_and_encode(self, sample: SpeechCommandsSample) -> Tuple[NDArray[np.bool_], int]:
        sr, audio = _read_wav(sample.filepath)
        if sr != 16000:
            # Basic guard: many Speech Commands files are 16kHz; skip mismatched
            raise ValueError(f"Expected 16kHz audio, got {sr} for {sample.filepath}")
        # Trim/pad to 1s
        target_len = 16000
        if len(audio) < target_len:
            pad = np.zeros(target_len, dtype=np.float64)
            pad[: len(audio)] = audio
            audio = pad
        else:
            audio = audio[:target_len]

        feats = compute_log_mel_spectrogram(audio, sample_rate=sr, n_mels=self.n_mels)
        spikes = poisson_encode_features(feats, timesteps=self.timesteps, r_max=self.r_max, dt=self.dt, rng=self.rng)
        label_idx = self.label_to_index.get(sample.label, self.label_to_index["_unknown_"])
        return spikes, label_idx

    def iter_split(self, split: str) -> Iterable[Tuple[NDArray[np.bool_], NDArray[np.int32]]]:
        if split == "train":
            samples = self.train
        elif split == "val":
            samples = self.val
        elif split == "test":
            samples = self.test
        else:
            raise ValueError("split must be one of {train,val,test}")

        batch_spikes: List[NDArray[np.bool_]] = []
        batch_labels: List[int] = []
        for s in samples:
            try:
                spikes, y = self._load_and_encode(s)
                batch_spikes.append(spikes)
                batch_labels.append(y)
            except Exception:
                continue
            if len(batch_spikes) == self.batch_size:
                # Stack to (T, B, F)
                stacked = np.stack(batch_spikes, axis=1)
                labels = np.array(batch_labels, dtype=np.int32)
                yield stacked, labels
                batch_spikes.clear()
                batch_labels.clear()
        if batch_spikes:
            stacked = np.stack(batch_spikes, axis=1)
            labels = np.array(batch_labels, dtype=np.int32)
            yield stacked, labels


__all__ = [
    "DEFAULT_KEYWORDS",
    "SpeechCommandsDataLoader",
    "index_speech_commands",
]


