#!/usr/bin/env python3
"""
realtime.py — Low-latency multi-stream CW decoder (CW-Skimmer direction).

Isolated from decode_v13 file/batch path so offline sample accuracy is unchanged.

Architecture (Skimmer-like):
  rolling FFT peak tracker → N parallel narrowband channels →
  short sliding-window envelope decode → per-stream text with overlap merge

Latency target: ~0.5–1.5 s (window), not full-file multi-config search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, lfilter

from cw_decoder.envelope import square_law_envelope
from cw_decoder.decoder import _decode_from_envelope


@dataclass
class StreamHit:
    """One decode emission from a tracked tone."""
    freq_hz: float
    text: str
    score: float
    wpm: float
    stream_id: int


@dataclass
class _Track:
    freq: float
    energy: float = 0.0
    last_seen: float = 0.0
    text: str = ""
    miss: int = 0
    # IIR bandpass state (b, a, zi)
    iir: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    # Rolling audio for this channel (baseband-ish filtered)
    buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class MultiStreamRealtimeDecoder:
    """
    Parallel tone tracker + per-stream short-window CW decode.

    Does NOT call decode_v13 / multi_config / contest stitch — those stay
    on the offline path for sample accuracy.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_streams: int = 6,
        bw_hz: float = 40.0,
        window_sec: float = 1.2,
        hop_sec: float = 0.35,
        scan_sec: float = 0.5,
        min_freq: float = 300.0,
        max_freq: float = 1500.0,
        min_sep_hz: float = 12.0,
        force_freqs: Optional[List[float]] = None,
    ):
        self.fs = int(sample_rate)
        self.max_streams = int(max_streams)
        self.bw_hz = float(bw_hz)
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.scan_sec = float(scan_sec)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.min_sep_hz = float(min_sep_hz)
        self.force_freqs = list(force_freqs) if force_freqs else None

        self._ring = np.zeros(0, dtype=np.float32)
        self._t = 0.0  # seconds of audio consumed
        self._next_scan_t = 0.0
        self._next_decode_t = 0.0
        self._tracks: List[_Track] = []
        self._stream_id_seq = 0
        self._track_ids: Dict[int, int] = {}  # id(track) -> stream_id

        self._win_n = max(int(self.window_sec * self.fs), self.fs // 2)
        self._ring_max = max(int(3.0 * self.fs), self._win_n * 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_chunk(self, audio: np.ndarray) -> List[StreamHit]:
        """
        Feed mono float32 audio in [-1, 1]. Returns new/updated stream hits
        (may be empty if not enough audio yet).
        """
        if audio is None or len(audio) == 0:
            return []
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._ring = np.concatenate([self._ring, x])
        if len(self._ring) > self._ring_max:
            drop = len(self._ring) - self._ring_max
            self._ring = self._ring[drop:]
        self._t += len(x) / self.fs

        hits: List[StreamHit] = []

        if self._t >= self._next_scan_t:
            self._rescan_peaks()
            self._next_scan_t = self._t + self.scan_sec

        # Filter new samples into each track buffer
        self._filter_into_tracks(x)

        if self._t >= self._next_decode_t and len(self._ring) >= self._win_n:
            hits = self._decode_tracks()
            self._next_decode_t = self._t + self.hop_sec

        return hits

    def snapshot_texts(self) -> Dict[float, str]:
        """Current best text per tracked frequency."""
        return {round(tr.freq, 1): tr.text for tr in self._tracks if tr.text}

    def reset(self) -> None:
        self._ring = np.zeros(0, dtype=np.float32)
        self._t = 0.0
        self._next_scan_t = 0.0
        self._next_decode_t = 0.0
        self._tracks = []
        self._track_ids = {}

    # ------------------------------------------------------------------
    # Peak tracking
    # ------------------------------------------------------------------
    def _rescan_peaks(self) -> None:
        n = min(len(self._ring), max(int(self.scan_sec * self.fs), self.fs // 4))
        if n < self.fs // 8:
            return
        seg = self._ring[-n:]

        if self.force_freqs:
            peaks = [(float(f), 1.0) for f in self.force_freqs]
        else:
            peaks = self._fft_peaks(seg)

        # Match existing tracks / spawn new
        assigned = set()
        for tr in self._tracks:
            tr.miss += 1

        for freq, energy in peaks:
            best_i, best_df = None, 1e9
            for i, tr in enumerate(self._tracks):
                if i in assigned:
                    continue
                df = abs(tr.freq - freq)
                if df < best_df:
                    best_df, best_i = df, i
            if best_i is not None and best_df <= self.min_sep_hz:
                tr = self._tracks[best_i]
                old_f = tr.freq
                # Slow pull toward measured peak (frequency lock)
                tr.freq = 0.7 * tr.freq + 0.3 * freq
                tr.energy = energy
                tr.last_seen = self._t
                tr.miss = 0
                assigned.add(best_i)
                if tr.iir is None or abs(tr.freq - old_f) > 3.0:
                    self._ensure_iir(tr)
            elif len(self._tracks) < self.max_streams:
                tr = _Track(freq=freq, energy=energy, last_seen=self._t, miss=0)
                self._ensure_iir(tr)
                self._tracks.append(tr)
                self._stream_id_seq += 1
                self._track_ids[id(tr)] = self._stream_id_seq

        # Drop stale tracks (keep force-locked tones)
        kept = []
        for tr in self._tracks:
            forced = self.force_freqs and any(
                abs(tr.freq - f) < self.min_sep_hz for f in self.force_freqs)
            if tr.miss <= 4 or forced:
                kept.append(tr)
        kept.sort(key=lambda t: -t.energy)
        self._tracks = kept[: self.max_streams]

    def _fft_peaks(self, seg: np.ndarray) -> List[Tuple[float, float]]:
        n = len(seg)
        # Hann to reduce leakage between close CW tones
        windowed = seg * np.hanning(n)
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        f2 = freqs[mask]
        p2 = spec[mask]
        if len(p2) < 8:
            return []
        noise = float(np.median(p2)) + 1e-12
        # Local maxima
        peaks: List[Tuple[float, float]] = []
        for i in range(2, len(p2) - 2):
            if p2[i] >= p2[i - 1] and p2[i] >= p2[i + 1] and p2[i] > noise * 8.0:
                peaks.append((float(f2[i]), float(p2[i] / noise)))
        peaks.sort(key=lambda x: -x[1])

        # Enforce min separation
        selected: List[Tuple[float, float]] = []
        for f, e in peaks:
            if all(abs(f - sf) >= self.min_sep_hz for sf, _ in selected):
                selected.append((f, e))
            if len(selected) >= self.max_streams:
                break
        return selected

    def _ensure_iir(self, tr: _Track) -> None:
        from scipy.signal import lfilter_zi
        low = max(50.0, tr.freq - self.bw_hz)
        high = min(self.fs * 0.45, tr.freq + self.bw_hz)
        if high <= low + 5:
            return
        b, a = butter(2, [low / (self.fs * 0.5), high / (self.fs * 0.5)], btype='band')
        zi = lfilter_zi(b, a) * 0.0
        tr.iir = (b, a, zi)

    # ------------------------------------------------------------------
    # Per-stream filter + decode
    # ------------------------------------------------------------------
    def _filter_into_tracks(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return
        for tr in self._tracks:
            if tr.iir is None:
                self._ensure_iir(tr)
            if tr.iir is None:
                continue
            b, a, zi = tr.iir
            y, zf = lfilter(b, a, x.astype(np.float64), zi=zi)
            tr.iir = (b, a, zf)
            y32 = y.astype(np.float32)
            tr.buf = np.concatenate([tr.buf, y32])
            if len(tr.buf) > self._win_n * 2:
                tr.buf = tr.buf[-self._win_n * 2:]

    def _decode_tracks(self) -> List[StreamHit]:
        hits: List[StreamHit] = []
        for tr in self._tracks:
            if len(tr.buf) < self._win_n // 2:
                continue
            seg = tr.buf[-self._win_n:]
            # Energy gate — skip silent channels
            if float(np.std(seg)) < 1e-4:
                continue
            env = square_law_envelope(seg, self.fs, smooth_ms=8.0)
            result = _decode_from_envelope(env, self.fs, threshold_mult=1.0)
            if not result:
                continue
            text_raw, score, meta = result
            text = (text_raw or "").strip()
            if len(text) < 2 or score < 0.25:
                continue
            # Overlap-merge into track transcript
            merged = _merge_overlap(tr.text, text)
            if merged != tr.text and merged.strip():
                tr.text = merged
                sid = self._track_ids.get(id(tr), 0)
                hits.append(StreamHit(
                    freq_hz=float(tr.freq),
                    text=merged,
                    score=float(score),
                    wpm=float(meta.get('wpm', 0) or 0),
                    stream_id=sid,
                ))
        return hits


def _merge_overlap(prev: str, cur: str) -> str:
    """Append only the novel suffix of cur relative to prev."""
    if not prev:
        return cur
    if not cur:
        return prev
    if cur in prev:
        return prev
    if prev in cur:
        return cur
    max_ov = min(len(prev), len(cur), 40)
    for ov in range(max_ov, 0, -1):
        if prev[-ov:] == cur[:ov]:
            return prev + cur[ov:]
    # Soft: if tails share alnum run
    return prev + cur


# Backward-compatible single-stream helper (README-shaped API)
class CWDecoderRealtime:
    """
    Single- or multi-stream realtime decoder.

    Example:
        dec = CWDecoderRealtime(sample_rate=16000, target_freq=750.0)
        hits = dec.process_chunk(audio)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_freq: Optional[float] = None,
        max_streams: int = 6,
        **kwargs,
    ):
        force = [float(target_freq)] if target_freq is not None else None
        self._dec = MultiStreamRealtimeDecoder(
            sample_rate=sample_rate,
            max_streams=1 if target_freq is not None and max_streams <= 1 else max_streams,
            force_freqs=force,
            **kwargs,
        )
        self._last_text = ""

    def process_chunk(self, audio: np.ndarray, samplerate: Optional[int] = None) -> str:
        """
        Process audio; return combined text across streams (Skimmer-style
        concatenated view). Empty string if nothing new.
        """
        if samplerate is not None and int(samplerate) != self._dec.fs:
            # Simple ignore mismatch — caller should open stream at configured rate
            pass
        hits = self._dec.process_chunk(audio)
        if not hits:
            return ""
        # Prefer loudest / first stream's full transcript for simple API
        snap = self._dec.snapshot_texts()
        if not snap:
            return ""
        # Join streams as [f=xxx] text
        parts = [f"[{f:.0f}] {t}" for f, t in sorted(snap.items())]
        combined = " | ".join(parts)
        if combined == self._last_text:
            return ""
        self._last_text = combined
        return combined

    @property
    def streams(self) -> Dict[float, str]:
        return self._dec.snapshot_texts()
