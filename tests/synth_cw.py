"""Minimal CW synthesizer for CI tests (no external fixtures)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from cw_decoder.morse import MORSE_REV


def _events(text: str, wpm: float, dash_dot: float = 3.0):
    dit_ms = 1200.0 / wpm
    dah_ms = dit_ms * dash_dot
    events = []
    for ch in text.upper():
        if ch == " ":
            events.append(("off", dit_ms * 4))
            continue
        code = MORSE_REV.get(ch)
        if not code:
            continue
        for i, sym in enumerate(code):
            events.append(("on", dit_ms if sym == "." else dah_ms))
            if i < len(code) - 1:
                events.append(("off", dit_ms))
        events.append(("off", dit_ms * 3))
    return events


def synthesize_cw(
    text: str,
    *,
    fs: int = 16000,
    tone: float = 750.0,
    wpm: float = 20.0,
    snr_db: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    events = _events(text, wpm)
    chunks = []
    phase = 0.0
    for state, dur_ms in events:
        n = max(1, int(dur_ms / 1000.0 * fs))
        if state == "on":
            t = np.arange(n) / fs
            chunk = np.sin(2 * np.pi * tone * t + phase).astype(np.float32)
            phase = (phase + 2 * np.pi * tone * n / fs) % (2 * np.pi)
            ramp = max(1, int(fs * 0.003))
            env = np.ones(n, dtype=np.float32)
            if n > 2 * ramp:
                env[:ramp] = np.linspace(0, 1, ramp, dtype=np.float32)
                env[-ramp:] = np.linspace(1, 0, ramp, dtype=np.float32)
            chunk *= env
        else:
            chunk = np.zeros(n, dtype=np.float32)
        chunks.append(chunk)
    sig = np.concatenate(chunks) if chunks else np.zeros(fs, dtype=np.float32)
    peak = float(np.max(np.abs(sig)) + 1e-8)
    sig = sig / peak
    if snr_db is not None:
        power = float(np.mean(sig**2) + 1e-10)
        noise_power = power / (10 ** (snr_db / 10.0))
        sig = sig + rng.normal(0.0, np.sqrt(noise_power), size=sig.shape).astype(
            np.float32
        )
        peak = float(np.max(np.abs(sig)) + 1e-8)
        sig = sig / peak
    return sig.astype(np.float32)


def write_wav(path: Path | str, sig: np.ndarray, fs: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(sig, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())
    return path
