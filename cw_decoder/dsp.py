#!/usr/bin/env python3
"""
dsp.py — Digital Signal Processing utilities for CW decoding.

Shared filter, SNR estimation, and audio I/O functions.
Aligned with international ham decoder conventions (FLDIGI, CW Skimmer).
"""

import numpy as np
from typing import Optional
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, iirnotch


# ============================================================
# Audio I/O
# ============================================================

def load_wav(path: str):
    """Load audio file (WAV/MP3/MP4/FLAC/M4A/OGG) as mono float32 normalized.

    Uses scipy.io.wavfile for WAV; falls back to ffmpeg for other formats.
    Returns (sample_rate, signal).
    """
    # Try scipy wavfile first (fast path, no subprocess)
    try:
        fs, sig = wavfile.read(path)
        if sig.dtype != np.float32:
            sig = sig.astype(np.float32)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        return fs, sig / (np.max(np.abs(sig)) + 1e-8)
    except ValueError:
        pass  # Not a WAV/RIFF file, fall through to ffmpeg

    # ── ffmpeg fallback for MP3, MP4, FLAC, M4A, OGG, etc. ──
    import subprocess

    def _ffprobe_get(key: str) -> str:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', f'stream={key}',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise ValueError(f"ffprobe failed for: {path}")
        return result.stdout.strip()

    try:
        sr_str = _ffprobe_get('sample_rate')
        if not sr_str:
            raise ValueError(f"No audio stream in: {path}")
        fs = int(sr_str)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg/ffprobe not found — install ffmpeg to decode non-WAV audio")

    try:
        result = subprocess.run(
            ['ffmpeg', '-i', path, '-f', 'f32le', '-acodec', 'pcm_f32le',
             '-ac', '1', '-', '-loglevel', 'error'],
            capture_output=True, timeout=120)
        if result.returncode != 0:
            raise ValueError(f"ffmpeg decode failed: {result.stderr.decode()}")
        sig = np.frombuffer(result.stdout, dtype=np.float32)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found — install ffmpeg to decode non-WAV audio")

    if len(sig) == 0:
        raise ValueError(f"No audio data decoded from: {path}")

    return fs, sig / (np.max(np.abs(sig)) + 1e-8)


# ============================================================
# Filters
# ============================================================

def bandpass(sig: np.ndarray, fs: int, low: float, high: float,
             order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [max(low, 1) / nyq,
                          min(high, fs // 2 - 1) / nyq], btype='band')
    return filtfilt(b, a, sig)


def notch_filter_hum(sig: np.ndarray, fs: int, freq: float = 60.0,
                     Q: float = 30.0) -> np.ndarray:
    """Notch filter: remove specific frequency interference (e.g. 60Hz hum)."""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig).astype(np.float32)


# ============================================================
# Goertzel tone detection / AFC (NUE-PSK / FLDIGI-style)
# ============================================================

def goertzel_power(block: np.ndarray, fs: float, freq: float) -> float:
    """Single-frequency DFT power at exact `freq` (not bin-quantized)."""
    n = len(block)
    if n < 4:
        return 0.0
    # Exact ω — critical for AFC fine search (±0.5 Hz steps)
    w = 2.0 * np.pi * freq / fs
    n_idx = np.arange(n, dtype=np.float64)
    x = np.asarray(block, dtype=np.float64)
    re = float(np.dot(x, np.cos(w * n_idx)))
    im = float(-np.dot(x, np.sin(w * n_idx)))
    return re * re + im * im


def goertzel_envelope(sig: np.ndarray, fs: int, freq: float,
                      block_ms: float = 2.5) -> np.ndarray:
    """
    Block-wise Goertzel magnitude → CW keying envelope.

    ~2.5 ms blocks give ~12 updates per dit at 40 WPM (NUE-PSK / Fallows).
    Output is normalized float32, length == len(sig).
    """
    n_block = max(int(block_ms / 1000.0 * fs), 8)
    n = len(sig)
    if n < n_block * 2:
        env = np.abs(sig.astype(np.float64))
        if env.max() > 0:
            env = env / env.max()
        return env.astype(np.float32)

    n_blocks = n // n_block
    # Exact ω for target tone (sub-bin AFC / narrow CW pitch)
    w = 2.0 * np.pi * freq / fs
    n_idx = np.arange(n_block, dtype=np.float64)
    cos_t = np.cos(w * n_idx)
    sin_t = np.sin(w * n_idx)
    win = np.hanning(n_block)

    shaped = sig[:n_blocks * n_block].astype(np.float64).reshape(n_blocks, n_block) * win
    re = shaped @ cos_t
    im = -(shaped @ sin_t)
    mags = np.sqrt(np.maximum(re * re + im * im, 0.0))

    if n_blocks >= 5:
        mags = np.convolve(mags, np.array([0.15, 0.7, 0.15]), mode='same')

    env = np.repeat(mags, n_block)
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)), mode='edge')
    else:
        env = env[:n]
    env = np.convolve(env, np.ones(n_block) / n_block, mode='same')
    if env.max() > 0:
        env = env / env.max()
    return env.astype(np.float32)


def refine_cw_freq_afc(sig: np.ndarray, fs: int, freq0: float,
                       search_hz: float = 12.0, step_hz: float = 0.5,
                       block_ms: float = 40.0) -> float:
    """
    AFC: refine CW tone by maximizing Goertzel power near `freq0`.

    Uses a mid-clip of the audio so silence / QRM at the edges don't pull.
    Returns best frequency in Hz (unchanged if search is empty).
    """
    if freq0 <= 0 or len(sig) < fs // 4:
        return float(freq0)

    # Mid 1–2 s window
    n_win = min(len(sig), max(int(1.0 * fs), int(block_ms / 1000.0 * fs) * 8))
    mid = len(sig) // 2
    half = n_win // 2
    clip = sig[max(0, mid - half):mid + half]
    n_block = max(int(block_ms / 1000.0 * fs), 32)
    if len(clip) < n_block:
        return float(freq0)
    # Average a few blocks in the clip
    n_use = (len(clip) // n_block) * n_block
    clip = clip[:n_use]
    win = np.hanning(n_block)

    best_f = float(freq0)
    best_p = -1.0
    f = freq0 - search_hz
    while f <= freq0 + search_hz + 1e-9:
        if 250.0 <= f <= 1600.0:
            power = 0.0
            n_b = 0
            for i in range(0, len(clip) - n_block + 1, n_block):
                block = clip[i:i + n_block].astype(np.float64) * win
                power += goertzel_power(block, fs, f)
                n_b += 1
                if n_b >= 6:
                    break
            if power > best_p:
                best_p = power
                best_f = float(f)
        f += step_hz
    return best_f


# ============================================================
# SNR estimation
# ============================================================

def estimate_snr_db(sig: np.ndarray, fs: int,
                    f_center: Optional[float] = None) -> float:
    """Estimate signal SNR in dB using power spectrum method."""
    spectrum = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)

    if f_center:
        mask = (freqs > f_center - 100) & (freqs < f_center + 100)
    else:
        mask = (freqs > 300) & (freqs < 1500)

    if not np.any(mask):
        return 0.0

    peak_idx = np.argmax(spectrum[mask])
    signal_power = spectrum[mask][peak_idx] ** 2
    noise_power = np.median(spectrum[mask] ** 2)

    if noise_power < 1e-10:
        return 30.0

    return float(10 * np.log10(signal_power / noise_power))


# ============================================================
# STFT peak tracker (MorseNet-style detection without NN)
# ============================================================

def track_cw_peaks_stft(sig: np.ndarray, fs: int,
                        fmin: float = 300.0, fmax: float = 1500.0,
                        top_k: int = 8,
                        max_seconds: float = 4.0) -> list:
    """
    Spectrogram peak tracker for CW tones.

    Inspired by MorseNet's spectrogram detection idea, but classical:
    STFT → per-frame local maxima → score by presence × intermittency
    × mean power. Returns list of ``{'freq', 'score', ...}`` compatible
    with ``scan_cw_frequencies`` candidates.
    """
    from scipy.signal import stft

    n = min(len(sig), int(max_seconds * fs))
    if n < fs // 4:
        return []
    clip = np.asarray(sig[:n], dtype=np.float64)
    nperseg = max(64, int(0.04 * fs))  # ~40 ms
    noverlap = nperseg // 2
    freqs, _times, Zxx = stft(
        clip, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Zxx)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return []
    f_band = freqs[band]
    m_band = mag[band, :]
    if m_band.size == 0 or m_band.shape[1] < 4:
        return []

    # Frame-wise peak picking (local max along frequency)
    peak_hits = np.zeros(m_band.shape[0], dtype=np.float64)
    peak_power = np.zeros(m_band.shape[0], dtype=np.float64)
    for t in range(m_band.shape[1]):
        col = m_band[:, t]
        thr = np.median(col) * 6.0
        for i in range(1, len(col) - 1):
            if col[i] >= col[i - 1] and col[i] >= col[i + 1] and col[i] > thr:
                peak_hits[i] += 1.0
                peak_power[i] += col[i]

    presence = peak_hits / max(m_band.shape[1], 1)
    mean_p = peak_power / np.maximum(peak_hits, 1.0)
    # Intermittency: CW should not be present every frame
    intermittency = 1.0 - np.abs(presence - 0.35)
    intermittency = np.clip(intermittency, 0.05, 1.0)
    energy = mean_p / (mean_p.max() + 1e-12)
    scores = energy * (0.4 + 0.6 * presence) * intermittency

    order = np.argsort(scores)[::-1]
    out = []
    used = []
    for idx in order:
        if scores[idx] <= 0:
            break
        f = float(f_band[idx])
        if any(abs(f - u) < 5.0 for u in used):
            continue
        used.append(f)
        out.append({
            'freq': f,
            'score': float(scores[idx]),
            'energy': float(energy[idx]),
            'inter': float(intermittency[idx]),
            'duty_cycle': float(presence[idx]),
            'source': 'stft',
        })
        if len(out) >= top_k:
            break
    return out


def merge_freq_candidates(primary: list, extra: list,
                          min_sep_hz: float = 3.0) -> list:
    """Merge STFT / FFT candidate lists; keep highest score per neighborhood."""
    merged = list(primary or []) + list(extra or [])
    merged.sort(key=lambda c: -c.get('score', 0.0))
    kept = []
    for c in merged:
        f = float(c['freq'])
        if any(abs(f - float(k['freq'])) < min_sep_hz for k in kept):
            continue
        kept.append(c)
    return kept
