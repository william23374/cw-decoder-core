#!/usr/bin/env python3
"""
agc.py — Slow AGC (Automatic Gain Control) for QSB fading compensation.

Implements Roadmap Phase 3.2

Principle:
- QSB fading is slow periodic variation of signal amplitude (typical rate 0.5-2 Hz)
- Slow AGC adjusts gain to compensate for fading
- Key: time constants must be slow enough not to destroy dot/dash amplitude differences

Design points:
- attack_ms: response time when signal suddenly increases (fast, to prevent overload)
- release_ms: recovery time when signal fades (slow, to smooth QSB)
- Asymmetric time constants are the core of AGC-based QSB mitigation
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.signal import lfilter


class SlowAGC:
    """
    Slow AGC: time constants 1.5-2.5 seconds

    Design goals:
    - Slow enough to smooth QSB fading (typical rate 0.5-2 Hz)
    - Fast enough not to flatten dot/dash amplitude differences

    Parameters:
    - attack_ms: attack time (gain reduction response speed)
      - Typical: 20-100 ms
      - Shorter: fast response to strong signals, prevents overload
      - Longer: smoother, but may temporarily overload
    - release_ms: release time (gain increase recovery speed)
      - Typical: 1000-3000 ms
      - Shorter: fast recovery of weak signals, but may introduce noise modulation
      - Longer: smoother gain changes, but slower recovery
    - target_level: target output level (0.0-1.0)
      - Typical: 0.5-0.8
    - max_gain: maximum gain limit
      - Prevents excessive noise amplification
    """

    def __init__(self, sample_rate: int,
                 attack_ms: float = 50.0,
                 release_ms: float = 2000.0,
                 target_level: float = 0.7,
                 max_gain: float = 50.0,
                 min_gain: float = 0.1):
        """
        Initialize slow AGC.

        Args:
            sample_rate: sample rate (Hz)
            attack_ms: attack time (ms)
            release_ms: release time (ms)
            target_level: target output level
            max_gain: maximum gain
            min_gain: minimum gain
        """
        self.sample_rate = sample_rate
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.target_level = target_level
        self.max_gain = max_gain
        self.min_gain = min_gain

        # Compute smoothing coefficients
        self.attack_alpha = 1.0 - np.exp(-1000.0 / (sample_rate * attack_ms / 1000.0))
        self.release_alpha = 1.0 - np.exp(-1000.0 / (sample_rate * release_ms / 1000.0))

        # State
        self.gain = 1.0
        self.envelope = None
        self.gain_history = None

    def process(self, signal: np.ndarray,
                return_details: bool = False) -> np.ndarray:
        """
        Process signal, output gain-normalized signal.

        Args:
            signal: input signal
            return_details: whether to return envelope and gain curve

        Returns:
            output: normalized signal

            If return_details=True:
            output, envelope, gain_curve
        """
        # Rectification (absolute value)
        rect = np.abs(signal)

        # Envelope tracking (slow low-pass filter)
        # Use first-order IIR filter to emulate analog AGC response
        env = np.zeros_like(rect, dtype=np.float64)
        env[0] = rect[0]

        # Time constant: ~0.5 seconds for envelope tracking
        alpha_slow = 1.0 - np.exp(-1.0 / (self.sample_rate * 0.5))

        for i in range(1, len(rect)):
            env[i] = (1 - alpha_slow) * env[i - 1] + alpha_slow * rect[i]

        # Compute gain
        gain_curve = np.zeros_like(env)
        for i in range(len(env)):
            # Desired gain
            desired_gain = self.target_level / (env[i] + 1e-6)
            # Clamp
            desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)

            # Asymmetric time constant tracking
            if desired_gain > self.gain:
                # Signal weaker, need to increase gain (release)
                self.gain += self.release_alpha * (desired_gain - self.gain)
            else:
                # Signal stronger, need to decrease gain (attack)
                self.gain += self.attack_alpha * (desired_gain - self.gain)

            gain_curve[i] = self.gain

        # Apply gain
        output = signal * gain_curve

        # Clamp to prevent clipping
        output = np.clip(output, -1.0, 1.0)

        # Save state
        self.envelope = env
        self.gain_history = gain_curve

        if return_details:
            return output, env, gain_curve
        return output

    def process_fast(self, signal: np.ndarray) -> np.ndarray:
        """
        Fast processing mode (uses vectorized operations).

        Suitable for long signal processing, ~10x faster than process().
        """
        # Rectification
        rect = np.abs(signal)

        # Envelope tracking (using scipy lfilter)
        alpha_slow = 1.0 - np.exp(-1.0 / (self.sample_rate * 0.5))
        b = [alpha_slow]
        a = [1.0, -(1 - alpha_slow)]
        env = lfilter(b, a, rect)

        # Compute target gain
        target_gain = self.target_level / (env + 1e-6)
        target_gain = np.clip(target_gain, self.min_gain, self.max_gain)

        # Gain smoothing (using asymmetric time constants)
        # Simplified: use single time constant (average of attack and release)
        alpha_avg = (self.attack_alpha + self.release_alpha) / 2
        b_gain = [alpha_avg]
        a_gain = [1.0, -(1 - alpha_avg)]
        gain_curve = lfilter(b_gain, a_gain, target_gain)

        # Clamp
        gain_curve = np.clip(gain_curve, self.min_gain, self.max_gain)

        # Apply gain
        output = signal * gain_curve
        output = np.clip(output, -1.0, 1.0)

        self.envelope = env
        self.gain_history = gain_curve

        return output

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        if self.gain_history is None:
            return {"error": "No signal processed"}

        return {
            "avg_gain": float(np.mean(self.gain_history)),
            "min_gain": float(np.min(self.gain_history)),
            "max_gain": float(np.max(self.gain_history)),
            "gain_range_db": float(20 * np.log10(np.max(self.gain_history) /
                                                  (np.min(self.gain_history) + 1e-6))),
            "envelope_mean": float(np.mean(self.envelope)),
            "envelope_std": float(np.std(self.envelope)),
        }

    def reset(self):
        """Reset AGC state."""
        self.gain = 1.0
        self.envelope = None
        self.gain_history = None


class QSBCompensator:
    """
    QSB fading compensator (high-level AGC wrapper).

    Specifically optimized for QSB fading scenarios.
    """

    def __init__(self, sample_rate: int, qsb_rate: float = 1.0):
        """
        Initialize QSB compensator.

        Args:
            sample_rate: sample rate (Hz)
            qsb_rate: expected QSB fading rate (Hz), typical 0.5-2.0 Hz
        """
        self.sample_rate = sample_rate
        self.qsb_rate = qsb_rate

        # Set AGC parameters based on QSB rate
        # Release time should be longer than QSB period
        qsb_period_ms = 1000.0 / qsb_rate
        release_ms = qsb_period_ms * 2  # release time = 2x QSB period

        self.agc = SlowAGC(
            sample_rate=sample_rate,
            attack_ms=50.0,
            release_ms=release_ms,
            target_level=0.7,
        )

    def compensate(self, signal: np.ndarray) -> np.ndarray:
        """Compensate for QSB fading."""
        return self.agc.process_fast(signal)


def apply_agc(signal: np.ndarray, sample_rate: int,
              mode: str = "slow") -> np.ndarray:
    """
    Convenience function: apply AGC to a signal.

    Args:
        signal: input signal
        sample_rate: sample rate (Hz)
        mode: AGC mode
            - "slow": slow AGC (suitable for QSB)
            - "fast": fast AGC (suitable for general scenarios)
            - "none": no AGC

    Returns:
        Normalized signal
    """
    if mode == "none":
        return signal

    if mode == "fast":
        agc = SlowAGC(sample_rate, attack_ms=20.0, release_ms=500.0)
    else:  # slow
        agc = SlowAGC(sample_rate, attack_ms=50.0, release_ms=2000.0)

    return agc.process_fast(signal)


if __name__ == "__main__":
    # Test AGC
    print("=" * 60)
    print("Slow AGC Test")
    print("=" * 60)

    sample_rate = 16000
    duration = 5.0
    t = np.arange(int(sample_rate * duration)) / sample_rate

    # Generate test signal: CW signal + QSB fading
    cw_freq = 750.0
    qsb_rate = 0.8  # Hz
    qsb_depth = 0.7  # fading depth

    # Basic CW signal (simple on/off keying)
    cw_signal = np.sin(2 * np.pi * cw_freq * t)

    # QSB fading (amplitude modulation)
    qsb_envelope = 1.0 - qsb_depth * (1 + np.sin(2 * np.pi * qsb_rate * t)) / 2
    signal_faded = cw_signal * qsb_envelope

    # Add noise
    noise = np.random.randn(len(signal_faded)) * 0.05
    signal_noisy = signal_faded + noise

    print(f"\nTest signal:")
    print(f"  CW frequency: {cw_freq} Hz")
    print(f"  QSB rate: {qsb_rate} Hz")
    print(f"  QSB depth: {qsb_depth * 100:.0f}%")
    print(f"  Duration: {duration} s")

    # Apply AGC
    agc = SlowAGC(sample_rate, attack_ms=50.0, release_ms=2000.0)
    output, env, gain = agc.process(signal_noisy, return_details=True)

    # Statistics
    input_range = np.max(np.abs(signal_noisy)) - np.min(np.abs(signal_noisy))
    output_range = np.max(np.abs(output)) - np.min(np.abs(output))

    print(f"\nAGC results:")
    print(f"  Input amplitude range: {input_range:.3f}")
    print(f"  Output amplitude range: {output_range:.3f}")
    print(f"  Average gain: {np.mean(gain):.2f}")
    print(f"  Gain range: {np.min(gain):.2f} - {np.max(gain):.2f}")

    diag = agc.get_diagnostics()
    print(f"  Gain dynamic range: {diag['gain_range_db']:.1f} dB")

    # Evaluate QSB compensation effectiveness
    # Compute output amplitude variation (should be more stable than input)
    input_std = np.std(np.abs(signal_noisy))
    output_std = np.std(np.abs(output))
    reduction = (1 - output_std / input_std) * 100

    print(f"\nQSB compensation effectiveness:")
    print(f"  Input amplitude std: {input_std:.4f}")
    print(f"  Output amplitude std: {output_std:.4f}")
    print(f"  Amplitude variation reduced: {reduction:.1f}%")
