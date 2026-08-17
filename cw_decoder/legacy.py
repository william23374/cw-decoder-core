#!/usr/bin/env python3
"""
CW Decoder V12.7 — Legacy Edition

Based on V12.6 fixes:
  1. Weak signal frequency estimation: frequency scan + envelope intermittency
     validation when FFT peak is unreliable
  2. Scoring function: decoded character count weight + non-'?' character ratio
  3. Bug key prior: use known ratio when mark segments too few
  4. Full test suite validation + complete reporting
"""
import sys, os, argparse, json, warnings, time
from typing import Optional, List, Tuple

import numpy as np
warnings.filterwarnings("ignore")

from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from cw_decoder.morse import MORSE, levenshtein, soft_decode
from cw_decoder.dsp import load_wav, bandpass
from cw_decoder.envelope import square_law_envelope
from cw_decoder.threshold import otsu_threshold, auto_threshold
from cw_decoder.segmentation import extract_segments


# Aliases for backward compatibility (V13 decoder.py imports these)
lev = levenshtein

# ============================================================
# Frequency scan + intermittency validation (robust for weak signals)
# ============================================================
def scan_cw_frequencies(signal, sample_rate, verbose=False):
    """
    Scan 300-1500Hz range, apply narrowband detection at each frequency,
    compute envelope 'intermittency score' (higher variance = more CW-like),
    return candidate frequencies sorted by score.
    """
    N = min(len(signal), sample_rate * 2)
    # FFT coarse scan
    spectrum = np.abs(np.fft.rfft(signal[:N]))
    freqs_fft = np.fft.rfftfreq(N, 1 / sample_rate)
    mask_range = (freqs_fft >= 300) & (freqs_fft <= 1500)
    coarse = freqs_fft[mask_range]
    coarse_spec = spectrum[mask_range]

    # Top 30 energy bins as candidates
    top_n = min(30, len(coarse))
    top_idx = np.argsort(coarse_spec)[-top_n:][::-1]
    candidates = []
    for idx in top_idx:
        freq = float(coarse[idx])
        # Intermittency test at each candidate frequency
        signal_bp = bandpass(signal, sample_rate, freq - 40, freq + 40)
        envelope = np.abs(signal_bp)
        if envelope.max() < 1e-6:
            continue
        envelope_norm = envelope / (envelope.max() + 1e-8)

        # Intermittency = envelope variance (CW has high variance, continuous wave low)
        intermittency = float(np.var(envelope_norm))

        # Smoothed mark ratio
        from scipy.ndimage import uniform_filter1d
        env_smooth = uniform_filter1d(envelope_norm.astype(np.float64), size=sample_rate // 20)
        duty_cycle = float((env_smooth > 0.3).mean())

        # Composite score
        energy_norm = float(coarse_spec[idx] / (coarse_spec.max() + 1e-8))
        inter_weight = min(intermittency * 10, 1.0)
        
        # Duty cycle penalty: CW signals typically 20-60% duty cycle
        # Continuous signals (100% duty cycle) are likely interference
        duty_penalty = 1.0
        if duty_cycle > 0.9:
            duty_penalty = 0.1  # Very strong penalty: almost certainly interference
        elif duty_cycle > 0.8:
            duty_penalty = 0.3  # Strong penalty for continuous signals
        elif duty_cycle > 0.6:
            duty_penalty = 0.7  # Moderate penalty
        elif duty_cycle < 0.1:
            duty_penalty = 0.5  # Penalty for very low duty cycle (noise)
        
        score = energy_norm * (0.5 + 0.5 * inter_weight) * duty_penalty

        candidates.append({
            'freq': freq, 'score': score,
            'energy': energy_norm, 'inter': intermittency,
            'duty_cycle': duty_cycle
        })

    candidates.sort(key=lambda x: -x['score'])

    if verbose and candidates:
        print(f"[*] Frequency scan TOP5:")
        for c in candidates[:5]:
            print(f"    {c['freq']:7.0f}Hz  E={c['energy']:.2f}  I={c['inter']:.3f}  "
                  f"Mark={c['duty_cycle'] * 100:.0f}%  score={c['score']:.3f}")

    return candidates

# ============================================================
# Dot/dash classification (with prior)
# ============================================================
def classify_on_off(mark_durations, expected_ratio=3.0):
    """
    Dot/dash classification with prior ratio support.
    Uses expected_ratio as prior when too few segments or atypical ratio.
    """
    arr = np.array(mark_durations, float)
    if len(arr) < 2:
        d = arr[0] if len(arr) else 60.0
        return ['dot'] * len(mark_durations), d, d * expected_ratio

    # Quantile method (more robust than K-means for small datasets)
    sorted_d = np.sort(arr)
    n = len(sorted_d)
    # Split using lower half and median
    lower = sorted_d[:n // 2]
    upper = sorted_d[n // 2:]
    dit_est = np.median(lower) * 1.05  # slight compensation
    dash_est = np.median(upper) * 0.95
    ratio = dash_est / max(dit_est, 1)

    # If ratio atypical, apply prior correction
    if ratio < 1.8 or ratio > 5.0:
        # Use median of all segments as dit estimate, multiply by prior for dah
        dit_est = np.median(arr) * 0.7
        dash_est = dit_est * expected_ratio
        ratio = expected_ratio

    # Classify each segment
    labels = []
    for dur in mark_durations:
        boundary = (dit_est + dash_est) / 2
        labels.append('dash' if dur > boundary else 'dot')

    return labels, dit_est, dash_est

# ============================================================
# Gap classification (3-cluster K-means)
# ============================================================
def classify_gaps(space_durations, dit):
    """Classify off-durations into in_char / char_gap / word_gap."""
    if len(space_durations) < 3:
        return ['char_gap' if d > dit * 2 else 'in_char' for d in space_durations]
    X = np.array(space_durations).reshape(-1, 1)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(3, len(space_durations)), n_init=5, random_state=0).fit(X)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    rank_map = {order[0]: 'in_char'}
    if len(order) > 1: rank_map[order[1]] = 'char_gap'
    if len(order) > 2: rank_map[order[2]] = 'word_gap'
    return [rank_map.get(c, 'char_gap') for c in km.predict(X)]

# ============================================================
# Complete decode at single frequency
# ============================================================
def decode_at_freq(signal, sample_rate, cw_freq, verbose=False):
    """Full decode at cw_freq. Returns (text, score, meta)."""
    signal_bp = bandpass(signal, sample_rate, cw_freq - 60, cw_freq + 60)
    envelope = square_law_envelope(signal_bp, sample_rate, smooth_ms=12.0)
    threshold = auto_threshold(envelope)
    key_states = (envelope > threshold).astype(np.int8)

    # Fallback
    duty_cycle = key_states.mean()
    if duty_cycle < 0.05 or duty_cycle > 0.97:
        threshold2 = 0.35 * envelope.max()
        states2 = (envelope > threshold2).astype(np.int8)
        if 0.08 < states2.mean() < 0.92: key_states = states2; threshold = threshold2

    segments = extract_segments(key_states, sample_rate, min_ms=3.0)
    mark_segs = [(s, d) for s, d in segments if s == 1]
    space_segs = [(s, d) for s, d in segments if s == 0]

    if len(mark_segs) < 3:
        return "", 0.0, {}

    mark_durations = [d for s, d in mark_segs]
    space_durations = [d for s, d in space_segs]

    # Dot/dash classification
    labels, dit, dah = classify_on_off(mark_durations, expected_ratio=3.0)
    if dit < 5: dit = 20
    wpm = 1200.0 / dit if dit > 0 else 0

    # Gap classification
    gap_labels = classify_gaps(space_durations, dit)

    # Assemble morse
    morse_chars = []; current = ""; mark_idx = space_idx = 0
    for state, dur in segments:
        if state == 1:
            if mark_idx < len(labels): current += '.' if labels[mark_idx] == 'dot' else '-'
            mark_idx += 1
        else:
            if space_idx < len(gap_labels): gap_label = gap_labels[space_idx]
            else: gap_label = 'char_gap'
            space_idx += 1
            if gap_label == 'in_char': continue
            if current: morse_chars.append(current); current = ""
            if gap_label == 'word_gap': morse_chars.append(' ')
    if current: morse_chars.append(current)

    # Decode
    decoded = []
    for mc in morse_chars:
        if mc == ' ': decoded.append(' ')
        else: decoded.append(soft_decode(mc))
    text = ''.join(decoded)

    # Scoring: valid character count + mark ratio plausibility
    n_valid = sum(1 for ch in text if ch != '?')
    n_total = max(len(text.replace(' ', '')) if text else 1, 1)
    valid_ratio = n_valid / n_total
    # Mark ratio penalty (too full or too empty both penalized)
    duty_cycle_penalty = 1.0 - abs(duty_cycle - 0.4) * 1.5
    duty_cycle_penalty = max(0.3, min(1.0, duty_cycle_penalty))
    score = valid_ratio * duty_cycle_penalty

    if verbose:
        ratio = dah / max(dit, 1)
        print(f"    {cw_freq:7.0f}Hz → '{text[:40]:40s}' score={score:.2f} "
              f"dit={dit:.0f}ms WPM={wpm:.0f} ratio={ratio:.1f}")

    meta = {'dit': dit, 'dah': dah, 'wpm': wpm, 'duty_cycle': duty_cycle, 'n_chars': n_valid}
    return text, score, meta

# ============================================================
# Main decoder
# ============================================================
def decode(path, ref_text=None, verbose=False):
    sample_rate, signal = load_wav(path)

    # 1. Frequency scan
    candidates = scan_cw_frequencies(signal, sample_rate, verbose=verbose)
    if not candidates:
        return {"text": "", "error": "No candidate frequency", "method": "V12.7"}

    # 2. Decode top 3 candidates
    top_n = min(3, len(candidates))
    results = []
    for c in candidates[:top_n]:
        freq = c['freq']
        text, score, meta = decode_at_freq(signal, sample_rate, freq, verbose=False)
        results.append((text, score, freq, meta))

    # 3. Select best
    results.sort(key=lambda x: -x[1])
    best_text, best_score, best_freq, best_meta = results[0]

    if verbose:
        print(f"[*] Best: {best_freq:.0f}Hz score={best_score:.2f}")
        for t, sc, f, m in results[:3]:
            print(f"    {f:7.0f}Hz → '{t[:40]:40s}' sc={sc:.2f}")

    # 4. Accuracy
    accuracy = None
    if ref_text:
        ref = ''.join(ch.upper() for ch in ref_text if ch.isalnum() or ch.isspace())
        got = ''.join(ch.upper() for ch in best_text if ch.isalnum() or ch.isspace())
        if ref and got:
            from difflib import SequenceMatcher
            accuracy = SequenceMatcher(None, got, ref).ratio()

    return {
        "text": best_text, "score": best_score, "freq": best_freq,
        "accuracy": accuracy, "method": "V12.7-FreqScan",
        "wpm": round(best_meta.get('wpm', 0), 1),
        "dit_ms": round(best_meta.get('dit', 0), 1),
    }

# ============================================================
# CLI + self-test
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CW Decoder V12.7")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--ref")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test: _self_test()
    elif args.input:
        r = decode(args.input, ref_text=args.ref, verbose=args.verbose)
        print(f"Frequency: {r.get('freq', '?'):.0f}Hz")
        print(f"Text:      {r.get('text', '')}")
        if r.get('accuracy') is not None: print(f"Accuracy:  {r['accuracy'] * 100:.0f}%")

def _self_test():
    import pycw, wave
    sample_rate = 16000
    print("=" * 60); print("CW Decoder V12.7 — Self-test"); print("=" * 60)
    results = {}

    def save(name, signal):
        with wave.open(name, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
            wf.writeframes((signal * 32767).astype(np.int16).tobytes())

    # 1. Clean
    print("\n[1] Clean CQ 20WPM")
    raw = pycw.generate("CQ CQ CQ DE W1AW K", wpm=20, tone=750, sample_rate=sample_rate)
    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    save("/tmp/v127_1.wav", signal)
    r = decode("/tmp/v127_1.wav", ref_text="CQ CQ CQ DE W1AW K", verbose=True)
    results['clean'] = r.get('accuracy', 0)

    # 2. QRM
    print("\n[2] QRM +8dB 820Hz 35WPM")
    raw2 = pycw.generate("CQ TEST DE K3EST K", wpm=35, tone=700, sample_rate=sample_rate)
    s2 = np.frombuffer(raw2, dtype=np.int16).astype(np.float32) / 32768.0
    t = np.arange(len(s2)) / sample_rate
    interference = np.sin(2 * np.pi * 820 * t) * 0.3
    mod = np.where(np.sin(2 * np.pi * 2 * t) > 0, 1.0, 0.3)
    sp = np.mean(s2 ** 2)
    s2q = s2 + interference * mod * np.sqrt(sp * 10 ** (8 / 10))
    save("/tmp/v127_2.wav", s2q)
    r = decode("/tmp/v127_2.wav", ref_text="CQ TEST DE K3EST K", verbose=True)
    results['qrm'] = r.get('accuracy', 0)

    # 3. Weak signal -5dB
    print("\n[3] -5dB weak signal 20WPM")
    np.random.seed(42)
    sp = np.mean(signal ** 2)
    noise = np.random.normal(0, np.sqrt(sp / 10 ** (-5 / 10)), len(signal))
    save("/tmp/v127_3.wav", signal + noise)
    r = decode("/tmp/v127_3.wav", ref_text="CQ CQ CQ DE W1AW K", verbose=True)
    results['n5'] = r.get('accuracy', 0)

    # 4. Weak signal -10dB
    print("\n[4] -10dB weak signal 20WPM")
    noise10 = np.random.normal(0, np.sqrt(sp / 10 ** (-10 / 10)), len(signal))
    save("/tmp/v127_4.wav", signal + noise10)
    r = decode("/tmp/v127_4.wav", ref_text="CQ CQ CQ DE W1AW K", verbose=True)
    results['n10'] = r.get('accuracy', 0)

    # 5. Bug Key
    print("\n[5] Bug Key 2.2:1")
    def tone(ms, f=720): return np.sin(2 * np.pi * f * np.arange(int(ms / 1000 * sample_rate)) / sample_rate).astype(np.float32)
    def silence(ms): return np.zeros(int(ms / 1000 * sample_rate), dtype=np.float32)
    d = 80; ah = int(d * 2.2); g = d; cg = d * 3
    bug = np.concatenate([tone(ah), silence(g), tone(d), silence(g), tone(ah), silence(g), tone(d),
                          silence(cg), tone(ah), silence(g), tone(d), silence(g), tone(ah), silence(g), tone(d), silence(cg)])
    bug = bug / (np.max(np.abs(bug)) + 1e-8)
    save("/tmp/v127_5.wav", bug)
    r = decode("/tmp/v127_5.wav", ref_text="C C C", verbose=True)
    results['bug'] = r.get('accuracy', 0)

    # 6. 35WPM high speed
    print("\n[6] 35WPM high speed")
    raw6 = pycw.generate("CQ TEST DE K3EST K", wpm=35, tone=700, sample_rate=sample_rate)
    s6 = np.frombuffer(raw6, dtype=np.int16).astype(np.float32) / 32768.0
    save("/tmp/v127_6.wav", s6)
    r = decode("/tmp/v127_6.wav", ref_text="CQ TEST DE K3EST K", verbose=True)
    results['qrq'] = r.get('accuracy', 0)

    # 7. Dual frequency
    print("\n[7] Dual frequency 700+850Hz")
    raw_a = pycw.generate("CQ CQ CQ", wpm=20, tone=700, sample_rate=sample_rate)
    sa = np.frombuffer(raw_a, dtype=np.int16).astype(np.float32) / 32768.0
    raw_b = pycw.generate("SOS SOS", wpm=15, tone=850, sample_rate=sample_rate)
    sb = np.frombuffer(raw_b, dtype=np.int16).astype(np.float32) / 32768.0
    mn = min(len(sa), len(sb))
    dual = sa[:mn] * 0.7 + sb[:mn] * 0.5
    save("/tmp/v127_7.wav", dual)
    r = decode("/tmp/v127_7.wav", verbose=True)
    results['dual'] = r.get('score', 0)

    # Summary
    print("\n" + "=" * 60)
    print("V12.7 Self-test Summary")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:8s}: {v * 100:.0f}% {'✅' if v >= 0.8 else ('⚠️' if v >= 0.3 else '❌')}")
    n = sum(1 for v in results.values() if v >= 0.8)
    print(f"\n  {n}/{len(results)} passed")

if __name__ == "__main__":
    main()
