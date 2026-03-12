"""Lightweight single-speaker matcher using MFCC-like features."""

from __future__ import annotations

import logging
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerConfig:
    voices_dir: Path
    threshold: float = 0.80
    scan_interval_s: int = 10
    sample_rate: int = 16000


class SpeakerRecognizer:
    """Loads reference voices and matches full utterance to nearest known speaker."""

    def __init__(self, config: SpeakerConfig) -> None:
        self._config = config
        self._reference_embeddings: Dict[str, np.ndarray] = {}
        self._last_scan_epoch = 0.0
        self._last_dir_mtime = -1.0
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        voices_dir = self._config.voices_dir
        if not voices_dir.exists():
            self._reference_embeddings = {}
            return

        try:
            dir_mtime = voices_dir.stat().st_mtime
        except FileNotFoundError:
            self._reference_embeddings = {}
            return

        now = time.monotonic()
        if not force:
            interval_ok = (now - self._last_scan_epoch) >= self._config.scan_interval_s
            if (not interval_ok) and (dir_mtime == self._last_dir_mtime):
                return

        self._last_scan_epoch = now
        self._last_dir_mtime = dir_mtime

        refs: Dict[str, np.ndarray] = {}
        for wav_path in sorted(voices_dir.glob("*.wav")):
            name = wav_path.stem.strip()
            if not name:
                continue

            try:
                samples, sr = _read_wav_mono_float32(wav_path)
                samples = _resample_linear(samples, sr, self._config.sample_rate)
                refs[name] = _embed(samples, self._config.sample_rate)
            except Exception as err:
                _LOGGER.warning("Skipping invalid reference %s: %s", wav_path, err)

        self._reference_embeddings = refs
        _LOGGER.info("Loaded %d speaker reference(s)", len(self._reference_embeddings))

    def identify(self, wav_path: str) -> Optional[str]:
        """Returns best matching speaker name or None if unknown."""
        self.refresh()
        if not self._reference_embeddings:
            return None

        try:
            samples, sr = _read_wav_mono_float32(Path(wav_path))
            samples = _resample_linear(samples, sr, self._config.sample_rate)
            emb = _embed(samples, self._config.sample_rate)
        except Exception as err:
            _LOGGER.debug("Speaker match failed: %s", err)
            return None

        best_name: Optional[str] = None
        best_score = float("-inf")

        for name, ref_emb in self._reference_embeddings.items():
            score = float(np.dot(emb, ref_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None or best_score < self._config.threshold:
            return None

        _LOGGER.debug("Matched speaker=%s score=%.4f", best_name, best_score)
        return best_name


def _read_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        comp_type = wav_file.getcomptype()
        frames = wav_file.getnframes()

        if comp_type != "NONE":
            raise ValueError(f"Compressed WAV is not supported: {comp_type}")

        raw = wav_file.readframes(frames)

    if sample_width == 1:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        pcm = (pcm - 128.0) / 128.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    return pcm.astype(np.float32, copy=False), sample_rate


def _resample_linear(
    samples: np.ndarray, src_rate: int, target_rate: int
) -> np.ndarray:
    if src_rate == target_rate:
        return samples

    if samples.size < 2:
        return samples

    target_len = max(1, int(round(samples.size * target_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    resampled = np.interp(dst_x, src_x, samples)
    return resampled.astype(np.float32, copy=False)


def _embed(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if samples.size == 0:
        raise ValueError("Empty audio")

    frame_len = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    n_fft = 512

    if samples.size < frame_len:
        pad = np.zeros(frame_len - samples.size, dtype=np.float32)
        samples = np.concatenate([samples, pad])

    frames = []
    for start in range(0, samples.size - frame_len + 1, hop):
        frames.append(samples[start : start + frame_len])

    if not frames:
        frames = [samples[:frame_len]]

    frame_mat = np.stack(frames).astype(np.float32)
    frame_mat *= np.hamming(frame_len).astype(np.float32)

    spectrum = np.fft.rfft(frame_mat, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)

    n_mels = 26
    n_mfcc = 13
    fbanks = _mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)

    mel_energy = np.maximum(power @ fbanks.T, 1e-10)
    log_mel = np.log(mel_energy)

    dct = _dct_matrix(n_mfcc=n_mfcc, n_mels=n_mels)
    mfcc = log_mel @ dct.T

    emb = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)], axis=0)
    norm = np.linalg.norm(emb)
    if norm <= 0:
        raise ValueError("Zero-norm embedding")

    return (emb / norm).astype(np.float32, copy=False)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_min = 0.0
    f_max = sample_rate / 2.0

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, num=n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points], dtype=np.float32)

    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    n_freqs = n_fft // 2 + 1

    fbanks = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        left = max(0, left)
        right = min(n_freqs - 1, right)
        center = min(right - 1, center)

        for k in range(left, center):
            fbanks[m - 1, k] = (k - left) / max(1, (center - left))
        for k in range(center, right):
            fbanks[m - 1, k] = (right - k) / max(1, (right - center))

    return fbanks


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    m = np.arange(n_mels, dtype=np.float32)
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    return np.cos((np.pi / n_mels) * (m + 0.5) * k).astype(np.float32)
