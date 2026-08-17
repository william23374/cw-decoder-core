#!/usr/bin/env python3
"""
pll.py — Second-order Costas PLL (Phase-Locked Loop) for CW frequency drift tracking.

Implements Roadmap Phase 3.1

Principle:
- Costas loop performs carrier tracking on CW signals
- Second-order loop filter smooths phase error
- Can track frequency drift up to ±5Hz/s

Applications:
- Replace fixed frequency estimation with dynamic tracking
- Output I/Q demodulated baseband signal for envelope extraction
"""

import numpy as np
from typing import Tuple, Dict, Optional


class CostasPLL:
    """
    Second-order Costas PLL

    Tracks CW signal frequency drift (typical causes: VFO drift, Doppler effect)

    Parameters:
    - loop_bw: loop bandwidth (Hz), lower = slower but more stable
      - Narrow (1-3 Hz): very stable, but can't track fast drift
      - Medium (5-10 Hz): balanced, suits most scenarios
      - Wide (15-30 Hz): fast tracking, but may lose lock
    - damping: damping ratio, 0.707 = critical damping
      - < 0.707: underdamped, oscillates
      - = 0.707: critically damped, fastest settle without oscillation
      - > 0.707: overdamped, stable but slow response
    """

    def __init__(self, fs: int, f0: float,
                 loop_bw: float = 5.0,
                 damping: float = 0.707,
                 freq_limit: float = 20.0):
        """
        Initialize PLL

        Args:
            fs: sample rate (Hz)
            f0: initial center frequency (Hz)
            loop_bw: loop bandwidth (Hz)
            damping: damping ratio
            freq_limit: frequency deviation limit (Hz), prevents runaway
        """
        self.fs = fs
        self.f0 = f0
        self.loop_bw = loop_bw
        self.damping = damping
        self.freq_limit = freq_limit

        # Loop state
        self.phase = 0.0
        self.freq = f0
        self.phase_error = 0.0

        # Compute loop coefficients
        self._compute_coefficients()

        # History
        self.freq_history = []
        self.phase_history = []
        self.error_history = []
        self.locked = False
        self.lock_time = 0

    def _compute_coefficients(self):
        """Compute second-order loop filter coefficients."""
        dt = 1.0 / self.fs
        # Natural frequency
        wn = 2 * np.pi * self.loop_bw / (self.damping + 1 / (4 * self.damping))
        # Discretization coefficients
        denom = 1 + 2 * self.damping * wn * dt + (wn * dt) ** 2
        self.alpha = 2 * self.damping * wn * dt / denom
        self.beta = (wn * dt) ** 2 / denom

    def process(self, signal: np.ndarray,
                return_full: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process signal, output I-channel demodulation and frequency track.

        Args:
            signal: input signal
            return_full: return full data (I, Q, freq, phase)

        Returns:
            I: I-channel demodulated signal (baseband)
            freq_track: locked frequency track

            If return_full=True:
            I, Q, freq_track, phase_track
        """
        n = len(signal)
        I = np.zeros(n)
        Q = np.zeros(n)
        freq_out = np.zeros(n)
        phase_out = np.zeros(n)

        # Pre-compute time vector
        t = np.arange(n) / self.fs

        for i in range(n):
            # Mixing (digital downconversion)
            cos_val = np.cos(self.phase)
            sin_val = np.sin(self.phase)

            I[i] = signal[i] * cos_val
            Q[i] = signal[i] * sin_val

            # Costas phase detector
            # For CW, I-channel polarity determines phase error direction
            sign_I = 1.0 if I[i] >= 0 else -1.0
            self.phase_error = sign_I * Q[i]

            # Loop filter (second-order)
            self.freq += self.beta * self.phase_error * self.fs
            # Frequency clamping
            self.freq = np.clip(self.freq, self.f0 - self.freq_limit,
                                self.f0 + self.freq_limit)
            # Phase update
            self.phase += 2 * np.pi * self.freq / self.fs + self.alpha * self.phase_error
            # Phase wrap to [-pi, pi]
            self.phase = np.arctan2(np.sin(self.phase), np.cos(self.phase))

            freq_out[i] = self.freq
            phase_out[i] = self.phase

        # Save history
        self.freq_history.extend(freq_out.tolist())
        self.phase_history.extend(phase_out.tolist())
        self.error_history.append(self.phase_error)

        # Lock detection
        self._check_lock(freq_out)

        if return_full:
            return I, Q, freq_out, phase_out
        return I, freq_out

    def _check_lock(self, freq_track: np.ndarray):
        """Detect lock status."""
        if len(self.freq_history) > 100:
            recent = np.array(self.freq_history[-100:])
            freq_std = np.std(recent)
            # Frequency std < 1Hz considered locked
            if freq_std < 1.0 and not self.locked:
                self.locked = True
                self.lock_time = len(self.freq_history)
            elif freq_std > 3.0:
                self.locked = False

    def process_block(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Block processing mode (faster, uses vectorization).
        Suitable for offline processing.
        """
        n = len(signal)

        # Initial frequency track estimate (based on instantaneous frequency)
        analytic = signal * np.exp(-1j * 2 * np.pi * self.f0 * np.arange(n) / self.fs)
        instant_freq = np.unwrap(np.angle(analytic))
        instant_freq = np.diff(instant_freq) * self.fs / (2 * np.pi)
        instant_freq = np.concatenate([[self.f0], instant_freq + self.f0])

        # Low-pass filter frequency track (simulate loop filter)
        from scipy.signal import butter, filtfilt
        cutoff = min(self.loop_bw * 2, self.fs / 4)
        b, a = butter(2, cutoff / (self.fs / 2), btype='low')
        freq_smooth = filtfilt(b, a, instant_freq)

        # Use smoothed frequency for downconversion
        phase_track = 2 * np.pi * np.cumsum(freq_smooth) / self.fs
        I = signal * np.cos(phase_track)
        Q = signal * np.sin(phase_track)

        self.freq_history.extend(freq_smooth.tolist())
        self.phase_history.extend(phase_track.tolist())

        return I, freq_smooth

    def get_lock_status(self) -> Dict:
        """Get lock status."""
        current_freq = self.freq
        freq_drift = current_freq - self.f0
        freq_drift_ppm = freq_drift / self.f0 * 1e6 if self.f0 > 0 else 0

        return {
            "locked": self.locked,
            "current_freq": current_freq,
            "initial_freq": self.f0,
            "freq_drift_hz": freq_drift,
            "freq_drift_ppm": freq_drift_ppm,
            "n_samples": len(self.freq_history),
            "lock_time_samples": self.lock_time if self.locked else None,
        }

    def reset(self, f0: Optional[float] = None):
        """Reset PLL state."""
        if f0 is not None:
            self.f0 = f0
        self.phase = 0.0
        self.freq = self.f0
        self.phase_error = 0.0
        self.freq_history = []
        self.phase_history = []
        self.error_history = []
        self.locked = False
        self.lock_time = 0


class FrequencyTracker:
    """
    Frequency tracker (high-level PLL wrapper).
    Provides simpler interface for CW decoder integration.
    """

    def __init__(self, fs: int, initial_freq: float = 750.0):
        self.fs = fs
        self.pll = CostasPLL(fs, initial_freq, loop_bw=5.0)

    def track(self, signal: np.ndarray) -> Dict:
        """
        Track signal frequency and return demodulation result.

        Returns:
            dict with keys:
            - I: I-channel signal
            - freq_track: frequency track
            - locked: lock status
            - avg_freq: average frequency
        """
        I, freq_track = self.pll.process(signal)

        return {
            "I": I,
            "freq_track": freq_track,
            "locked": self.pll.locked,
            "avg_freq": np.mean(freq_track),
            "freq_std": np.std(freq_track),
            "status": self.pll.get_lock_status(),
        }


def estimate_freq_with_pll(signal: np.ndarray, fs: int,
                           initial_freq: float = None) -> Tuple[float, np.ndarray]:
    """
    Estimate CW signal frequency using PLL.

    Args:
        signal: input signal
        fs: sample rate
        initial_freq: initial frequency estimate (if None, use FFT estimate)

    Returns:
        freq: locked frequency
        freq_track: frequency track
    """
    # If no initial frequency, use FFT coarse estimate
    if initial_freq is None:
        spec = np.abs(np.fft.rfft(signal[:fs * 2]))
        freqs = np.fft.rfftfreq(min(len(signal), fs * 2), 1 / fs)
        mask = (freqs >= 300) & (freqs <= 1500)
        if np.any(mask):
            peak_idx = np.argmax(spec[mask])
            initial_freq = freqs[mask][peak_idx]
        else:
            initial_freq = 750.0

    # Refine with PLL
    pll = CostasPLL(fs, initial_freq, loop_bw=3.0)
    I, freq_track = pll.process(signal)

    # Return average frequency after lock
    if pll.locked:
        locked_freq = np.mean(freq_track[pll.lock_time:])
    else:
        locked_freq = np.mean(freq_track[-100:])

    return locked_freq, freq_track


if __name__ == "__main__":
    # Test PLL
    print("=" * 60)
    print("PLL Tracker Test")
    print("=" * 60)

    fs = 16000
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs

    # Generate test signal: 750Hz CW with frequency drift
    f0 = 750.0
    drift_rate = 2.0  # Hz/s
    freq_true = f0 + drift_rate * t
    phase_true = 2 * np.pi * np.cumsum(freq_true) / fs
    signal = np.sin(phase_true)

    # Add noise
    noise = np.random.randn(len(signal)) * 0.1
    signal_noisy = signal + noise

    print(f"\nTest signal:")
    print(f"  Initial freq: {f0} Hz")
    print(f"  Drift rate: {drift_rate} Hz/s")
    print(f"  Duration: {duration} s")
    print(f"  SNR: ~20 dB")

    # PLL tracking
    pll = CostasPLL(fs, f0, loop_bw=5.0)
    I, freq_track = pll.process(signal_noisy)

    status = pll.get_lock_status()
    print(f"\nPLL results:")
    print(f"  Locked: {status['locked']}")
    print(f"  Final freq: {status['current_freq']:.2f} Hz")
    print(f"  Freq drift: {status['freq_drift_hz']:.2f} Hz")
    print(f"  Freq std: {np.std(freq_track):.3f} Hz")

    # Compare true vs tracked frequency
    final_true_freq = freq_true[-1]
    final_tracked_freq = freq_track[-1]
    error = abs(final_true_freq - final_tracked_freq)
    print(f"\nTracking error:")
    print(f"  True final freq: {final_true_freq:.2f} Hz")
    print(f"  Tracked final freq: {final_tracked_freq:.2f} Hz")
    print(f"  Error: {error:.2f} Hz")
