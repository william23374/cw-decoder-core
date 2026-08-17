#!/usr/bin/env python3
"""
integrator.py — Coherent integrator for extracting CW signals at extremely low SNR.

Implements Roadmap Phase 3.3

Principle:
- Signal is mixed to baseband (I/Q) with local reference oscillator
- Long rectangular-window integration on I/Q
- Composite magnitude: sqrt(I² + Q²)

SNR gain analysis:
- For pure sine signal: post-integration signal amplitude ∝ N
- For Gaussian white noise: post-integration noise amplitude ∝ sqrt(N)
- Net SNR gain = N / sqrt(N)

Example: 100ms integration @ 16kHz → N=1600 → SNR gain = 40 = 16dB

Applications:
- Extreme weak signals at SNR < -5dB
- When normal square-law detection cannot extract envelope
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.signal import firwin, lfilter, butter, filtfilt


class CoherentIntegrator:
    """
    Coherent integrator (for extreme weak signals at SNR < -5dB)

    Parameters:
    - integ_ms: integration time (milliseconds)
      - Shorter (50-100ms): preserves time resolution, lower SNR gain
      - Longer (150-300ms): maximum SNR gain, but may blur fast signals
      - Recommendation: choose based on WPM, WPM < 15 can use longer integration
    - cw_freq: CW signal frequency (Hz)
      - Must be accurate, otherwise integration cancels the signal
    """

    def __init__(self, sample_rate: int, cw_freq: float, integ_ms: int = 100):
        """
        Initialize coherent integrator.

        Args:
            sample_rate: sample rate (Hz)
            cw_freq: CW signal frequency (Hz)
            integ_ms: integration time (ms)
        """
        self.sample_rate = sample_rate
        self.cw_freq = cw_freq
        self.integ_ms = integ_ms
        self.win_size = int(integ_ms / 1000.0 * sample_rate)

        # Pre-compute reference oscillator (one window length)
        t = np.arange(self.win_size) / sample_rate
        self.ref_cos = np.cos(2 * np.pi * cw_freq * t)
        self.ref_sin = np.sin(2 * np.pi * cw_freq * t)

        # FIR integration kernel (rectangular window)
        self.kernel = np.ones(self.win_size) / self.win_size

    def process_block(self, signal: np.ndarray) -> np.ndarray:
        """
        Block processing for entire signal (non-real-time).

        Uses FIR filter to implement sliding window integration, fast.

        Args:
            signal: input signal

        Returns:
            envelope: integrated envelope signal
        """
        n = len(signal)
        t = np.arange(n) / self.sample_rate

        # Mixing (downconvert to baseband)
        I = signal * np.cos(2 * np.pi * self.cw_freq * t)
        Q = signal * np.sin(2 * np.pi * self.cw_freq * t)

        # FIR integration (rectangular window sliding average)
        I_int = lfilter(self.kernel, 1.0, I)
        Q_int = lfilter(self.kernel, 1.0, Q)

        # Composite envelope
        env = np.sqrt(I_int ** 2 + Q_int ** 2)

        # Normalize
        env = env / (np.max(env) + 1e-8)

        return env

    def process_block_hann(self, signal: np.ndarray) -> np.ndarray:
        """
        Weighted integration using Hann window (reduces spectral leakage).

        Better frequency selectivity than rectangular window.
        """
        n = len(signal)
        t = np.arange(n) / self.sample_rate

        # Mixing
        I = signal * np.cos(2 * np.pi * self.cw_freq * t)
        Q = signal * np.sin(2 * np.pi * self.cw_freq * t)

        # Hann window integration kernel
        hann_kernel = np.hanning(self.win_size)
        hann_kernel = hann_kernel / np.sum(hann_kernel)

        I_int = np.convolve(I, hann_kernel, mode='same')
        Q_int = np.convolve(Q, hann_kernel, mode='same')

        env = np.sqrt(I_int ** 2 + Q_int ** 2)
        env = env / (np.max(env) + 1e-8)

        return env

    def process_streaming(self, sample: float) -> float:
        """
        Real-time streaming processing (single sample).

        For SDR / microphone real-time input.

        Note: This method is slow, only suitable for real-time scenarios.
        """
        # Update buffer
        if not hasattr(self, '_buf'):
            self._buf = np.zeros(self.win_size)
            self._buf_idx = 0

        self._buf[self._buf_idx] = sample
        self._buf_idx = (self._buf_idx + 1) % self.win_size

        # Mixing (current buffer)
        t = np.arange(self.win_size) / self.sample_rate
        ref_cos = np.cos(2 * np.pi * self.cw_freq * t)
        ref_sin = np.sin(2 * np.pi * self.cw_freq * t)

        I = np.sum(self._buf * ref_cos)
        Q = np.sum(self._buf * ref_sin)

        return np.sqrt(I * I + Q * Q)

    def get_snr_gain_db(self) -> float:
        """Compute theoretical SNR gain (dB)."""
        # SNR gain = sqrt(N), where N = integration window samples
        n = self.win_size
        gain_linear = np.sqrt(n)
        gain_db = 20 * np.log10(gain_linear)
        return gain_db


class AdaptiveIntegrator:
    """
    Adaptive coherent integrator.

    Automatically adjusts integration time based on signal SNR.
    """

    def __init__(self, sample_rate: int, cw_freq: float,
                 min_integ_ms: int = 50,
                 max_integ_ms: int = 200):
        self.sample_rate = sample_rate
        self.cw_freq = cw_freq
        self.min_integ_ms = min_integ_ms
        self.max_integ_ms = max_integ_ms

        # Create integrators with different integration times
        self.integrators = {}
        for ms in range(min_integ_ms, max_integ_ms + 1, 10):
            self.integrators[ms] = CoherentIntegrator(sample_rate, cw_freq, ms)

    def process(self, signal: np.ndarray,
                target_snr: float = 10.0) -> Tuple[np.ndarray, int]:
        """
        Adaptive integration processing.

        Args:
            signal: input signal
            target_snr: target output SNR (dB)

        Returns:
            envelope: integrated envelope
            used_integ_ms: integration time used
        """
        # Estimate input SNR first
        input_snr = self._estimate_snr(signal)

        # Compute required SNR gain
        needed_gain = target_snr - input_snr

        # Select integration time based on gain
        # SNR gain ≈ 10 * log10(integ_ms * fs / 1000)
        if needed_gain <= 0:
            used_ms = self.min_integ_ms
        else:
            # Required samples N = 10^(gain/10)^2
            needed_n = (10 ** (needed_gain / 10)) ** 2
            needed_ms = needed_n / self.sample_rate * 1000
            used_ms = int(np.clip(needed_ms, self.min_integ_ms, self.max_integ_ms))
            # Align to 10ms steps
            used_ms = (used_ms // 10) * 10

        # Use corresponding integrator
        integ = self.integrators.get(used_ms,
                                     CoherentIntegrator(self.sample_rate, self.cw_freq, used_ms))
        envelope = integ.process_block(signal)

        return envelope, used_ms

    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate signal SNR (simplified version)."""
        # Use power spectrum method
        spec = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / self.sample_rate)

        # Find signal peak
        mask = (freqs > 100) & (freqs < 2000)
        if not np.any(mask):
            return 0.0

        peak_idx = np.argmax(spec[mask])
        signal_power = spec[mask][peak_idx] ** 2

        # Estimate noise power (median)
        noise_power = np.median(spec[mask] ** 2)

        if noise_power < 1e-10:
            return 30.0  # Very clean

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)


def integrate_coherently(signal: np.ndarray, sample_rate: int, cw_freq: float,
                         integ_ms: int = 100) -> np.ndarray:
    """
    Convenience function: perform coherent integration on signal.

    Args:
        signal: input signal
        sample_rate: sample rate (Hz)
        cw_freq: CW frequency (Hz)
        integ_ms: integration time (ms)

    Returns:
        envelope: integrated envelope
    """
    integrator = CoherentIntegrator(sample_rate, cw_freq, integ_ms)
    return integrator.process_block(signal)


if __name__ == "__main__":
    # Test coherent integrator
    print("=" * 60)
    print("Coherent Integrator Test")
    print("=" * 60)

    sample_rate = 16000
    duration = 2.0
    t = np.arange(int(sample_rate * duration)) / sample_rate

    # Generate weak CW signal
    cw_freq = 750.0

    # Simple on/off keying
    key_shape = np.zeros_like(t)
    # Create some dots and dashes
    dot_samples = int(0.06 * sample_rate)  # 60ms dot
    dash_samples = int(0.18 * sample_rate)  # 180ms dash
    gap_samples = int(0.06 * sample_rate)

    # Manually build keying waveform
    pos = 0
    segments = [
        (1, dot_samples),   # .
        (0, gap_samples),
        (1, dot_samples),   # .
        (0, gap_samples),
        (1, dot_samples),   # .
        (0, gap_samples * 3),  # character gap
        (1, dash_samples),  # -
        (0, gap_samples),
        (1, dash_samples),  # -
        (0, gap_samples),
        (1, dash_samples),  # -
    ]

    for state, length in segments:
        if pos + length <= len(key_shape):
            key_shape[pos:pos + length] = state
            pos += length

    # Generate CW signal
    cw_signal = np.sin(2 * np.pi * cw_freq * t) * key_shape

    # Add noise (SNR = -10dB)
    signal_power = np.mean(cw_signal ** 2)
    noise_power = signal_power / (10 ** (-10 / 10))  # -10dB SNR
    noise = np.random.randn(len(t)) * np.sqrt(noise_power)
    signal_noisy = cw_signal + noise

    print(f"\nTest signal:")
    print(f"  CW frequency: {cw_freq} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Input SNR: -10 dB")

    # Test different integration times
    print(f"\nCoherent integration results:")
    for integ_ms in [50, 100, 150, 200]:
        integrator = CoherentIntegrator(sample_rate, cw_freq, integ_ms)
        env = integrator.process_block(signal_noisy)

        # Compute output SNR (simplified estimate)
        env_signal = np.max(env)
        env_noise = np.median(env)
        if env_noise > 1e-6:
            out_snr = 20 * np.log10(env_signal / env_noise)
        else:
            out_snr = 30.0

        theoretical_gain = integrator.get_snr_gain_db()
        print(f"  Integration {integ_ms}ms: theoretical gain={theoretical_gain:.1f}dB, "
              f"output SNR≈{out_snr:.1f}dB")

    # Test adaptive integrator
    print(f"\nAdaptive integrator:")
    adaptive = AdaptiveIntegrator(sample_rate, cw_freq, min_integ_ms=50, max_integ_ms=200)
    env_adaptive, used_ms = adaptive.process(signal_noisy, target_snr=15.0)
    print(f"  Auto-selected integration time: {used_ms}ms")
