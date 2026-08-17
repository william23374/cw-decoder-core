#!/usr/bin/env python3
"""
envelope.py — Envelope detection for CW signals.

Provides multiple detection methods aligned with international conventions
(FLDIGI, CW Skimmer, morse-audio-decoder):
- Square-law: fast, simple, good for clean signals
- RMS: more robust in noise, used by CW Skimmer
- Hann-RMS: better frequency selectivity via Hann windowing
"""

import numpy as np


def square_law_envelope(sig: np.ndarray, fs: int, smooth_ms: float = 12.0) -> np.ndarray:
    """Square-law detection + moving average smoothing."""
    env = sig.astype(np.float64) ** 2
    win = max(int(smooth_ms / 1000 * fs), 1)
    env = np.convolve(env, np.ones(win) / win, mode='same')
    env = np.sqrt(np.maximum(env, 0))
    if env.max() > 0:
        env = env / env.max()
    return env.astype(np.float32)


def rms_envelope(sig: np.ndarray, fs: int, window_ms: float = 10.0) -> np.ndarray:
    """
    RMS moving-window envelope extraction (used by CW Skimmer & morse-audio-decoder).
    More robust in noise than square-law detection.
    """
    win = max(int(window_ms / 1000 * fs), 1)
    sq = sig.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode='same'))
    if rms.max() > 0:
        rms = rms / rms.max()
    return rms.astype(np.float32)


def hann_rms_envelope(sig: np.ndarray, fs: int, window_ms: float = 10.0) -> np.ndarray:
    """
    Hann-windowed RMS envelope (better frequency selectivity, reduces spectral leakage).
    Used by international CW decoders for cleaner envelope in noisy conditions.
    """
    win = max(int(window_ms / 1000 * fs), 3)
    sq = sig.astype(np.float64) ** 2
    hann = np.hanning(win)
    hann = hann / hann.sum()
    rms = np.sqrt(np.convolve(sq, hann, mode='same'))
    if rms.max() > 0:
        rms = rms / rms.max()
    return rms.astype(np.float32)


def onset_enhanced_envelope(sig: np.ndarray, fs: int,
                            smooth_ms: float = 8.0,
                            onset_ms: float = 3.0) -> np.ndarray:
    """
    Square-law envelope blended with a positive energy derivative (onset).

    Classical onset detection sharpens mark attacks that global thresholding
    smears — reduces truncated leading dits that become '?' / wrong letters.
    """
    base = square_law_envelope(sig, fs, smooth_ms=smooth_ms).astype(np.float64)
    # Energy derivative on lightly smoothed power
    win = max(int(onset_ms / 1000.0 * fs), 1)
    power = sig.astype(np.float64) ** 2
    power = np.convolve(power, np.ones(win) / win, mode='same')
    d = np.diff(power, prepend=power[0])
    onset = np.maximum(d, 0.0)
    if onset.max() > 0:
        onset = onset / onset.max()
    # Blend: keep amplitude info, boost rising edges
    env = 0.75 * base + 0.25 * onset
    if env.max() > 0:
        env = env / env.max()
    return env.astype(np.float32)


def normalize_envelope(env: np.ndarray, fs: int, window_ms: float = 500) -> np.ndarray:
    """
    Envelope normalization: divide by local mean to eliminate QSB amplitude variation.
    Keeps envelope at ~1.0 amplitude locally for global thresholding.
    """
    win = max(int(window_ms / 1000 * fs), 1)
    kernel = np.ones(win) / win
    local_mean = np.convolve(env, kernel, mode='same')
    local_mean = np.maximum(local_mean, 1e-6)
    env_norm = env / local_mean
    if env_norm.max() > 0:
        env_norm = env_norm / env_norm.max()
    return env_norm.astype(np.float32)
