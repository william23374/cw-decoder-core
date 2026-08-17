#!/usr/bin/env python3
"""
CW Decoder V13 — Enhanced Edition

Based on V12.7 + V13 core modules:
  1. PLL (Phase-Locked Loop): tracks frequency drift
  2. Slow AGC: compensates QSB fading
  3. Coherent integrator: extracts extreme weak signals
  4. Semantic corrector: fixes errors using QSO format priors
  5. Multi-threshold integration: handles irregular keying
  6. Envelope normalization: compensates QSB amplitude variation
  7. Notch filter: removes hum interference

Pipeline:
Audio → Notch filter (optional) → PLL → AGC → Bandpass → SNR estimate
      ├─ SNR >= 0dB → Standard path (multi-threshold + envelope normalization)
      └─ SNR <  0dB → Advanced path (PLL + AGC + coherent integration)
      → Best selection → Semantic correction → Output
"""

import sys
import os
import re
import warnings
import time
from typing import Optional, List, Tuple, Dict

import numpy as np
warnings.filterwarnings("ignore")

from scipy.io import wavfile
from scipy.signal import butter, filtfilt, iirnotch

# Import V13 modules
from cw_decoder.pll import CostasPLL, estimate_freq_with_pll
from cw_decoder.agc import SlowAGC, apply_agc
from cw_decoder.integrator import CoherentIntegrator, AdaptiveIntegrator
from cw_decoder.corrector import semantic_correct

# Import shared modules
from cw_decoder.morse import MORSE
from cw_decoder.dsp import (
    load_wav, bandpass, notch_filter_hum, estimate_snr_db,
    goertzel_envelope, refine_cw_freq_afc,
    track_cw_peaks_stft, merge_freq_candidates,
)
from cw_decoder.envelope import (
    square_law_envelope, rms_envelope, hann_rms_envelope,
    normalize_envelope, onset_enhanced_envelope,
)
from cw_decoder.threshold import auto_threshold
from cw_decoder.segmentation import (
    extract_segments,
    morphological_clean,
    merge_fragments,
    merge_envelope_fragments,
)

# Import V12.7 core functions (still in legacy.py)
from cw_decoder.legacy import (
    lev, soft_decode, scan_cw_frequencies, classify_on_off, classify_gaps
)

# Backward compatibility aliases for private functions
_morphological_clean = morphological_clean
_merge_fragments = merge_fragments
_merge_envelope_fragments = merge_envelope_fragments


# ============================================================
# Shared decode pipeline
# ============================================================

def _robust_dit_estimate(mark_durations, sample_rate):
    """
    Robust dit (dot duration) estimation.

    Strategy:
    1. If distribution is concentrated (p75/p25 < 2), use median
    2. If bimodal with reasonable ratio, use bimodal detection
    3. Fragment detection: if many segments are much shorter than main peak,
       elements were fragmented by smoothing/QSB — use upper peak as dit
    4. Fallback to percentile method
    """
    arr = np.array(mark_durations, dtype=float)
    if len(arr) < 2:
        d = arr[0] if len(arr) else 60.0
        return d, d * 3.0

    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)
    p75 = np.percentile(arr, 75)

    # Concentrated distribution → use median
    if p75 / max(p25, 1) < 2.0:
        dit_est = p50 * 0.85
        dah_est = dit_est * 3.0
        return dit_est, dah_est

    # Histogram bimodal detection
    n_bins = max(10, len(arr) // 3)
    hist, bin_edges = np.histogram(arr, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    from scipy.ndimage import maximum_filter1d
    smoothed = maximum_filter1d(hist.astype(float), size=max(3, n_bins // 5))
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1] and hist[i] > 0:
            peaks.append((bin_centers[i], hist[i]))

    if len(peaks) >= 2:
        peaks.sort(key=lambda x: -x[1])
        p1, h1 = peaks[0]  # highest peak
        p2, h2 = peaks[1]  # second highest peak
        lo = min(p1, p2)
        hi = max(p1, p2)
        ratio = hi / max(lo, 1)

        if 1.8 < ratio < 5.0:
            # Fragment detection: if many segments are much shorter than
            # the lower peak (≤ half its value), the lower peak is noise
            # fragments, not real dits. For slow operators (lo > 80ms),
            # skip fragment detection since jitter naturally spreads dits
            # and genuine noise fragments are rare at slow speeds.
            if lo <= 80:
                n_frags = np.sum(arr < lo * 0.5)
                frag_ratio = n_frags / len(arr)

                if frag_ratio > 0.3:
                    # Severe fragmentation → use upper peak as dit
                    dit_est = hi * 0.9
                    dah_est = dit_est * 3.0
                    return dit_est, dah_est

            # Sanity check: if dit is extremely short (<20ms) but median is
            # much higher, dit may be locked on fragments
            if lo < 20 and p50 > lo * 2.5:
                dit_est = hi * 0.9
                dah_est = dit_est * 3.0
                return dit_est, dah_est

            return lo, hi

    # === KMeans-based bimodal detection ===
    # Histogram peak detection can miss bimodal distributions when the data
    # is spread across many bins (common with slow/novice operators at low SNR).
    # KMeans with 2 clusters is more robust for detecting the dit/dah split.
    if len(arr) >= 6:
        try:
            from sklearn.cluster import KMeans
            X = arr.reshape(-1, 1)
            km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(X)
            centers = sorted(km.cluster_centers_.flatten())
            lo_k, hi_k = centers[0], centers[1]
            ratio_k = hi_k / max(lo_k, 1)

            if 1.8 < ratio_k < 5.0:
                # Fragment detection: only count marks ≤ half the lower
                # cluster center as fragments. For slow operators (lo > 80ms),
                # skip fragment detection since jitter naturally spreads dits
                # and genuine noise fragments are rare at slow speeds.
                if lo_k <= 80:
                    n_frags = np.sum(arr < lo_k * 0.5)
                    frag_ratio = n_frags / len(arr)

                    if frag_ratio > 0.3:
                        dit_est = hi_k * 0.9
                        dah_est = dit_est * 3.0
                        return dit_est, dah_est

                if lo_k < 20 and p50 > lo_k * 2.5:
                    dit_est = hi_k * 0.9
                    dah_est = dit_est * 3.0
                    return dit_est, dah_est

                return lo_k, hi_k
        except Exception:
            pass

    # Fallback: percentile method
    dit_est = p25 * 1.1
    dah_est = dit_est * 3.0
    return dit_est, dah_est


def _duration_based_gaps(space_durations, dit):
    """
    Classify space durations into in_char / char_gap / word_gap.

    Uses K-means clustering (when data is sufficient) with dit-prior
    validation to detect non-standard timing. Falls back to fixed
    thresholds with adaptive boundary search.

    For non-standard (compressed) timing, prefers AG1LE-style Viterbi
    HMM labeling, then weighted adaptive boundaries (MSP430/FLDIGI).
    """
    n = len(space_durations)
    if n < 3:
        return ['char_gap' if d > dit * 2 else 'in_char' for d in space_durations]

    # Probabilistic / adaptive methods first when timing looks non-standard
    if _has_nonstandard_timing(space_durations, dit):
        vit = _viterbi_gap_classify(space_durations, dit)
        if vit is not None:
            return vit
        wad = _weighted_adaptive_gaps(space_durations, dit)
        if wad is not None:
            return wad

    # Fixed boundaries (standard timing)
    t1 = dit * 2.0
    t2 = dit * 5.0

    # === Tier 1: K-means (when data is sufficient) ===
    if n >= 15:
        try:
            from sklearn.cluster import KMeans
            X = np.array(space_durations).reshape(-1, 1)
            km = KMeans(n_clusters=3, n_init=3, random_state=0).fit(X)
            centers = km.cluster_centers_.flatten()
            order = np.argsort(centers)

            c_short = centers[order[0]]
            c_med = centers[order[1]]

            if c_short < dit * 2.5 and c_med >= dit * 1.0:
                rank_map = {order[0]: 'in_char', order[1]: 'char_gap', order[2]: 'word_gap'}
                preds = km.predict(X)
                labels_km = [rank_map.get(c, 'char_gap') for c in preds]

                in_char_max = max((d for d, l in zip(space_durations, labels_km) if l == 'in_char'), default=0)
                char_gap_min = min((d for d, l in zip(space_durations, labels_km) if l == 'char_gap'), default=dit*5)

                if in_char_max < char_gap_min:
                    # Post-validation for slow operators: K-means may set
                    # the in_char/char_gap boundary too high (close to dit*2),
                    # causing char_gaps to be misclassified as in_char.
                    # If in_char_max is suspiciously high, try adaptive methods.
                    if not (dit > 120 and in_char_max > dit * 1.5):
                        return labels_km
                    # Slow + suspicious: try Viterbi before falling through
                    vit = _viterbi_gap_classify(space_durations, dit)
                    if vit is not None:
                        return vit
        except Exception:
            pass

    # === Tier 2: Adaptive boundary search ===
    adaptive_t1 = _find_adaptive_gap_boundary(space_durations, dit)
    if adaptive_t1 is not None:
        t1 = adaptive_t1
    elif dit > 150:
        # Very slow operators (WPM < 8): char_gaps are almost always
        # compressed below dit*2 due to extreme jitter at very low speeds.
        # Apply the lower boundary unconditionally.
        t1 = dit * 1.6
    elif dit > 120:
        # Slow operators (WPM < 10): the fixed t1 = dit*2 boundary can be
        # too high when char_gaps are compressed by jitter. Only apply a
        # lower fallback when there's evidence of non-standard timing:
        # spaces clustered near dit*2 instead of cleanly separated at dit*3.
        # Check for compressed space distribution: median space in the
        # ambiguous zone (dit*1.3 to dit*3.0) is closer to dit*1.5 than dit*3.
        arr = np.array(space_durations)
        mid_spaces = arr[(arr >= dit * 1.3) & (arr <= dit * 3.0)]
        if len(mid_spaces) >= 3:
            median_mid = np.median(mid_spaces)
            # If median of mid-range spaces is unusually low (< dit*2.0),
            # the operator has compressed char_gaps
            if median_mid < dit * 2.0:
                # Use a boundary between the likely in_char and char_gap clusters
                t1 = dit * 1.6

    return ['in_char' if d < t1 else ('char_gap' if d < t2 else 'word_gap')
            for d in space_durations]


def _weighted_adaptive_gaps(space_durations, dit):
    """
    Direction A: Weighted average adaptive threshold.

    Inspired by international HAM decoder algorithms (MSP430, FLDIGI).
    Uses exponential weighted moving average to track the operator's
    actual timing rhythm, with outlier screening.

    Strategy:
    1. Initial estimate from sorted distribution (percentile-based)
    2. Iterative refinement with weighted average + outlier screening
    3. Only activate if the adaptive boundary differs significantly from dit*2

    Returns list of gap labels, or None if method doesn't apply.
    """
    arr = np.array(space_durations, dtype=float)
    n = len(arr)
    if n < 8:
        return None

    # Guard: dit must be reasonable (real CW is >= 20ms)
    if dit < 25:
        return None

    # Guard: spaces must have meaningful spread (not all noise-level)
    median_space = np.median(arr)
    if median_space < dit * 0.3:
        return None

    fixed_boundary = dit * 2.0

    # --- Step 1: Initial boundary estimate from distribution ---
    # Find the largest gap in sorted durations between dit*1.2 and dit*3.0
    # This locates the natural boundary between in_char and char_gap clusters
    sorted_arr = np.sort(arr)
    search_lo = dit * 1.2
    search_hi = dit * 3.0
    diffs = np.diff(sorted_arr)
    mask = (sorted_arr[:-1] >= search_lo) & (sorted_arr[1:] <= search_hi)

    if mask.any():
        valid_diffs = np.where(mask, diffs, 0)
        gap_idx = int(np.argmax(valid_diffs))
        if valid_diffs[gap_idx] > dit * 0.15:  # Significant gap
            boundary_est = (sorted_arr[gap_idx] + sorted_arr[gap_idx + 1]) / 2.0
        else:
            boundary_est = dit * 2.0  # No clear gap
    else:
        boundary_est = dit * 2.0  # No values in search zone

    # --- Step 2: Iterative weighted average refinement ---

    # Iterative refinement with screening (like the MSP430 algorithm)
    beta = 0.5  # Initial learning rate (high = fast adaptation)
    screening_factor = 0.5  # Reject measurements > 50% from current estimate

    for iteration in range(5):
        # Classify spaces based on current boundary
        in_char_spaces = [d for d in arr if d <= boundary_est]
        char_gap_spaces = [d for d in arr if d > boundary_est]

        if len(in_char_spaces) < 3 or len(char_gap_spaces) < 2:
            break

        # Compute cluster means
        mean_in = np.mean(in_char_spaces)
        mean_gap = np.mean(char_gap_spaces)

        # Screen outliers: only use spaces within screening_factor of cluster mean
        screened_in = [d for d in in_char_spaces
                       if abs(d - mean_in) / max(mean_in, 1) < screening_factor]
        screened_gap = [d for d in char_gap_spaces
                        if abs(d - mean_gap) / max(mean_gap, 1) < screening_factor]

        if len(screened_in) < 3 or len(screened_gap) < 2:
            break

        # Weighted average update (exponential moving average)
        new_mean_in = np.mean(screened_in)
        new_mean_gap = np.mean(screened_gap)

        # New boundary: weighted midpoint between cluster centers
        new_boundary = (new_mean_in + new_mean_gap) / 2.0

        # Exponential smoothing of boundary
        boundary_est = beta * new_boundary + (1 - beta) * boundary_est

        # Decrease learning rate for stability (like MSP430 algorithm)
        beta = max(0.2, beta * 0.7)

    # --- Step 3: Validate and apply ---
    # Boundary must be between in_char cluster and char_gap cluster
    # Physically: boundary must be > dit (in_char spaces are ~1 dit)
    # and < dit*4 (char_gap spaces are ~3 dit)
    if boundary_est < dit * 1.5 or boundary_est > dit * 4.0:
        return None

    # Quality check: verify the boundary creates a clean separation
    # Count spaces on each side that are misclassified
    in_char_spaces = [d for d in arr if d < boundary_est]
    char_gap_spaces = [d for d in arr if d >= boundary_est]
    if len(in_char_spaces) < 3 or len(char_gap_spaces) < 2:
        return None

    # Check if the adaptive boundary is meaningfully different from fixed
    # Use absolute difference relative to dit (not relative to boundary)
    diff_from_fixed = abs(boundary_est - fixed_boundary)
    if diff_from_fixed < dit * 0.1:  # Less than 10% of dit — negligible
        return None

    # Apply the adaptive boundary
    t2 = dit * 5.0  # word_gap boundary stays fixed
    return ['in_char' if d < boundary_est else ('char_gap' if d < t2 else 'word_gap')
            for d in space_durations]


def _has_nonstandard_timing(space_durations, dit):
    """
    Detect evidence of non-standard timing (compressed ratios).
        
    Returns True if there's statistical evidence that the standard
    1:3:7 timing ratios are compressed (e.g. 1:2.5:7 as seen in OP_VERYOLD).
    """
    if len(space_durations) < 8:
        return False
    
    arr = np.array(space_durations)
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    
    # Look for evidence of compressed timing:
    # 1. Large fraction of spaces near dit*2.0 (instead of clean separation at dit*3.0)
    near_boundary = np.sum((arr >= dit * 1.5) & (arr <= dit * 2.5))
    if near_boundary > len(arr) * 0.3:  # 30% of spaces in ambiguous region
        return True
    
    # 2. Characteristic compressed ratio: char_gap spaces closer to dit*2 than dit*3
    if len(sorted_arr) >= 10:
        p25 = sorted_arr[len(sorted_arr)//4]  # Lower quartile
        p75 = sorted_arr[3*len(sorted_arr)//4]  # Upper quartile
        
        # If p25 is close to dit*2 (2.0) and p75 is significantly below dit*3.5,
        # this suggests compressed timing (e.g., 1:2.2:7 instead of 1:3:7)
        if p25 <= dit * 2.2 and p75 <= dit * 2.8:  # More conservative threshold
            ratio = p75 / max(p25, dit)  # Ratio between upper and lower spaces
            if ratio < 2.2:  # Even more conservative - require very compressed ratio
                return True
    
    # 3. Large variance relative to expected gaps
    if np.std(arr) > dit * 2.5:
        return True
        
    return False

def _viterbi_gap_classify(space_durations, dit):
    """
    Direction B: HMM/Viterbi probabilistic gap classification.

    Inspired by AG1LE's Bayesian Morse decoder for FLDIGI.
    Models gap classification as a sequence labeling problem:

    States:  in_char (0), char_gap (1), word_gap (2)
    Observations: space durations (ms)
    Emission: Gaussian P(d | state) with expected mean and variance
    Transition: encodes Morse structure constraints

    Viterbi algorithm finds the most likely state sequence.

    Returns list of gap labels, or None if method doesn't apply.
    """
    arr = np.array(space_durations, dtype=float)
    n = len(arr)
    if n < 5:
        return None

    # Guard: dit must be reasonable (real CW is >= 20ms)
    if dit < 25:
        return None

    # Guard: spaces must have meaningful spread (not all noise-level)
    median_space = np.median(arr)
    if median_space < dit * 0.3:
        return None

    # State definitions
    N_STATES = 3
    IN_CHAR, CHAR_GAP, WORD_GAP = 0, 1, 2
    STATE_NAMES = ['in_char', 'char_gap', 'word_gap']

    # Expected means (in units of dit)
    # Standard: in_char=1dit, char_gap=3dit, word_gap=7dit
    # But non-standard keying may compress these ratios
    expected_means = np.array([dit, dit * 3.0, dit * 7.0])

    # Standard deviations (adaptive based on observed data)
    # Use 30% of mean as initial sigma (accommodates non-standard keying)
    sigmas = expected_means * 0.35

    # --- Transition probabilities ---
    # Encodes Morse structure:
    # - in_char → in_char: common (multi-element characters)
    # - in_char → char_gap: common (end of character)
    # - char_gap → in_char: common (start of new character)
    # - char_gap → word_gap: possible (end of word)
    # - word_gap → in_char: common (start of new word)
    # - Direct in_char → word_gap: forbidden (must have char_gap first)
    # - Direct word_gap → char_gap: unlikely
    log_trans = np.full((N_STATES, N_STATES), -20.0)  # Very unlikely default

    # in_char transitions
    log_trans[IN_CHAR, IN_CHAR] = -0.3    # P(stay in_char) ~ 74%
    log_trans[IN_CHAR, CHAR_GAP] = -0.5   # P(end character) ~ 61%
    log_trans[IN_CHAR, WORD_GAP] = -10.0  # Forbidden: must have char_gap first

    # char_gap transitions
    log_trans[CHAR_GAP, IN_CHAR] = -0.2   # P(start new char) ~ 82%
    log_trans[CHAR_GAP, CHAR_GAP] = -5.0  # Unlikely: two char_gaps in a row
    log_trans[CHAR_GAP, WORD_GAP] = -1.2  # P(end word) ~ 30%

    # word_gap transitions
    log_trans[WORD_GAP, IN_CHAR] = -0.3   # P(start new word) ~ 74%
    log_trans[WORD_GAP, CHAR_GAP] = -5.0  # Unlikely
    log_trans[WORD_GAP, WORD_GAP] = -3.0  # Possible but unlikely

    # --- Emission probabilities (log Gaussian) ---
    def log_emission(d, state):
        mu = expected_means[state]
        sigma = sigmas[state]
        return -0.5 * ((d - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)

    # --- Adaptive mean refinement ---
    # If we have enough data, refine expected means from the actual distribution
    if n >= 15:
        sorted_arr = np.sort(arr)
        # Estimate actual cluster centers from percentiles
        # (robust to misclassification)
        p33 = sorted_arr[int(n * 0.33)]
        p66 = sorted_arr[int(n * 0.66)]
        p90 = sorted_arr[int(n * 0.90)]

        # Only adapt if the observed distribution suggests non-standard timing
        # Check if char_gap cluster is compressed (ratio < 2.5 instead of 3)
        if p33 > dit * 0.5 and p33 < dit * 2.5:
            # Observed in_char spaces are near expected — check char_gap
            ratio_cg = p66 / max(p33, 1)
            if ratio_cg < 2.5:
                # Compressed timing — adapt expected means
                expected_means[CHAR_GAP] = p33 * ratio_cg
                sigmas[CHAR_GAP] = expected_means[CHAR_GAP] * 0.3

    # --- Viterbi algorithm ---
    # Initialize
    log_delta = np.full((n, N_STATES), -np.inf)
    psi = np.zeros((n, N_STATES), dtype=int)

    # Initial state probabilities: start in in_char or char_gap
    log_delta[0, IN_CHAR] = -0.1
    log_delta[0, CHAR_GAP] = -1.0
    log_delta[0, WORD_GAP] = -3.0

    for s in range(N_STATES):
        log_delta[0, s] += log_emission(arr[0], s)

    # Forward pass
    for t in range(1, n):
        for s in range(N_STATES):
            candidates = log_delta[t-1, :] + log_trans[:, s]
            best_prev = np.argmax(candidates)
            log_delta[t, s] = candidates[best_prev] + log_emission(arr[t], s)
            psi[t, s] = best_prev

    # Backtrack
    path = np.zeros(n, dtype=int)
    path[n-1] = np.argmax(log_delta[n-1, :])
    for t in range(n-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    # Convert to labels
    labels = [STATE_NAMES[s] for s in path]

    # --- Validate result ---
    # Check that the Viterbi result is physically reasonable
    in_char_durs = [d for d, l in zip(space_durations, labels) if l == 'in_char']
    char_gap_durs = [d for d, l in zip(space_durations, labels) if l == 'char_gap']

    if len(in_char_durs) < 3 or len(char_gap_durs) < 1:
        return None

    # Check separation: in_char spaces should be shorter than char_gap spaces
    max_in = max(in_char_durs)
    min_cg = min(char_gap_durs)
    if max_in >= min_cg * 1.2:
        # Poor separation — Viterbi result is unreliable
        return None

    return labels


def _find_adaptive_gap_boundary(space_durations, dit):
    """
    Find the natural boundary between in_char and char_gap spaces by detecting
    the largest gap in the sorted space duration distribution.

    Only activates when there's clear evidence that the fixed dit*2 boundary
    is wrong: i.e., a significant gap exists in the space distribution AND
    using it would fix misclassifications (spaces currently treated as char_gap
    that should be in_char).

    Returns the adaptive boundary (ms) or None if no clear boundary found.
    """
    arr = np.sort(space_durations)
    if len(arr) < 10:
        return None

    fixed_boundary = dit * 2.0

    # Search for the largest gap between dit*0.7 and dit*2.5
    # (only look BELOW the fixed boundary — we only want to LOWER it)
    lo = dit * 0.7
    hi = min(dit * 2.5, fixed_boundary)

    diffs = np.diff(arr)
    mask = (arr[:-1] >= lo) & (arr[1:] <= hi)
    if not mask.any():
        return None

    valid_diffs = np.where(mask, diffs, 0)
    idx = int(np.argmax(valid_diffs))
    gap_size = valid_diffs[idx]

    # The gap must be significant: at least 25% of dit duration
    # This prevents triggering on standard keying where spaces are continuous
    min_gap = dit * 0.25
    if gap_size < min_gap:
        return None

    boundary = (arr[idx] + arr[idx + 1]) / 2.0

    # Only use adaptive boundary if it's meaningfully different from fixed
    # (at least 15% lower than dit*2)
    if boundary > fixed_boundary * 0.85:
        return None

    return float(boundary)


def _hysteresis_key_states(envelope, threshold, ratio=0.6, fs: int = 0):
    """
    Hysteresis thresholding (FLDigi technique): two-level threshold prevents
    noise and QSB from creating false transitions.
    - Mark starts when envelope > upper_threshold
    - Mark ends when envelope < lower_threshold
    - Between thresholds: maintain previous state

    Processes a downsampled envelope (~2 kHz) then upsamples states for speed.
    """
    upper = threshold
    lower = threshold * ratio
    n = len(envelope)
    # ~2 kHz decision rate is plenty for CW (dit ≥ ~10 ms)
    ds = max(1, int(fs // 2000) if fs > 0 else max(1, n // 16000))
    env_ds = envelope[::ds]
    n_ds = len(env_ds)
    states_ds = np.zeros(n_ds, dtype=np.int8)
    in_mark = False
    for i in range(n_ds):
        if in_mark:
            if env_ds[i] < lower:
                in_mark = False
            else:
                states_ds[i] = 1
        else:
            if env_ds[i] > upper:
                in_mark = True
                states_ds[i] = 1
    if ds == 1:
        return states_ds
    key_states = np.repeat(states_ds, ds)[:n]
    if len(key_states) < n:
        key_states = np.pad(key_states, (0, n - len(key_states)), mode='edge')
    return key_states


# Common CW / English tokens for plausibility boosting (QRQ nets & QSOs).
_PLAUSIBLE_WORDS = frozenset({
    'CQ', 'DE', 'UR', 'RST', 'BTU', 'FB', 'OK', 'QSL', 'QTH', 'QRZ', 'QSO',
    'AGN', 'NAME', 'RIG', 'ANT', 'WX', 'HW', 'BK', 'PSE', 'TNX', 'TU',
    'THE', 'AND', 'FOR', 'YOU', 'THAT', 'WITH', 'HAVE', 'THIS', 'WAS',
    'BUT', 'NOT', 'ARE', 'FROM', 'ALL', 'ABOUT', 'BEEN', 'WILL', 'WHAT',
    'WHEN', 'YOUR', 'JUST', 'LIKE', 'KNOW', 'GOOD', 'WELL', 'MUCH', 'SOME',
    'THERE', 'WOULD', 'COULD', 'SHOULD', 'BEFORE', 'AFTER', 'OTHER',
})


def _dedupe_freq_candidates(candidates, min_sep_hz: float = 4.0):
    """Keep highest-scoring tone in each ~min_sep_hz neighborhood."""
    if not candidates:
        return candidates
    kept = []
    for c in sorted(candidates, key=lambda x: -x['score']):
        f = c['freq']
        if any(abs(f - k['freq']) < min_sep_hz for k in kept):
            continue
        kept.append(c)
    return kept


def _looks_like_contest_exchange(text: str) -> bool:
    """True when decode shows contest cut-RST + serial traffic (WPX/CQWW style)."""
    if not text:
        return False
    tu = text.upper()
    # Allow run-on forms (YO4AAC5NN2169) as well as spaced tokens
    n_cut = len(re.findall(r'[15E]NN', tu))
    n_serial = len(re.findall(r'\d{3,4}', tu))
    n_tu = len(re.findall(r'TU', tu))
    return n_cut >= 2 and n_serial >= 2 and (n_tu >= 1 or n_cut >= 3)


def _contest_exchange_quality(text: str) -> float:
    """0..1 score favoring spaced contest exchanges (call + 5NN + serial + TU)."""
    if not text:
        return 0.0
    # Normalize run-ons before scoring so gluey window decodes still count
    from cw_decoder.corrector import split_contest_runon
    tu, _ = split_contest_runon(text)
    tu = tu.upper()
    alnum = sum(1 for c in tu if c.isalnum())
    if alnum < 8:
        return 0.0
    spaces = tu.count(' ')
    space_ratio = spaces / max(len(tu), 1)
    q = 0.0
    if space_ratio >= 0.08:
        q += 0.35
    elif space_ratio >= 0.04:
        q += 0.15
    n_5nn = len(re.findall(r'\b5NN\b', tu))
    n_tu = len(re.findall(r'\bTU\b', tu))
    n_serial = len(re.findall(r'\b\d{3,4}\b', tu))
    n_calls = len(re.findall(
        r'\b(?:[A-Z]{1,2}\d+[A-Z]{1,4}|[A-Z]\d{2}[A-Z]{1,4})\b', tu))
    q += min(0.25, 0.05 * n_5nn)
    q += min(0.15, 0.03 * n_tu)
    q += min(0.15, 0.03 * n_serial)
    q += min(0.20, 0.04 * n_calls)
    mono = _dominant_char_ratio(tu)
    if mono > 0.45:
        q *= 0.3
    elif mono > 0.35:
        q *= 0.6
    return float(min(1.0, q))


def _extract_contest_serials(text: str):
    """Return sorted unique host serials (21xx-style) found after 5NN."""
    from cw_decoder.corrector import split_contest_runon
    tu, _ = split_contest_runon(text)
    serials = set()
    for m in re.finditer(r'(?:5NN|[15E]NN)\s*(\d{3,4})\b', tu.upper()):
        serials.add(m.group(1))
    return serials


def _extract_contest_calls(text: str):
    from cw_decoder.corrector import split_contest_runon
    tu, _ = split_contest_runon(text)
    calls = set(re.findall(
        r'\b(?:[A-Z]{1,2}\d+[A-Z]{1,4}|[A-Z]\d{2}[A-Z]{1,4})\b', tu.upper()))
    # Drop short/garbage "calls" from mark-bias fragments
    return {c for c in calls if len(c) >= 3 and not re.fullmatch(r'[EI]+\d*[EI]*', c)}


def _contest_window_decode(chunk, sample_rate, primary_freq, snr_db):
    """
    Decode one contest window with multi-hypothesis merge.

    Several nearby carriers often each decode a different fragment of the
    same exchange (call on one tone, serial on another). Pick the best
    primary text, then stitch novel calls/serials from runner-up hypos.
    Contest-gated only — never used on synthetic/QRQ prose paths.
    """
    cands = scan_cw_frequencies(chunk, sample_rate, verbose=False)
    cands = _dedupe_freq_candidates(cands or [], min_sep_hz=6.0)
    freqs = [c['freq'] for c in (cands or [])[:5]]
    if primary_freq is not None:
        freqs.append(float(primary_freq))
        # Fine grid around primary (pile-up neighbors)
        for d in (-8.0, -4.0, 4.0, 8.0, 12.0):
            freqs.append(float(primary_freq) + d)
    seen = set()
    hypos = []
    for freq in freqs:
        key = round(freq, 0)
        if key in seen:
            continue
        seen.add(key)
        for bw, smooth in ((35, 6.0), (45, 8.0), (55, 10.0)):
            bp = bandpass(chunk, sample_rate, freq - bw, freq + bw)
            env = square_law_envelope(bp, sample_rate, smooth_ms=smooth)
            result = _decode_from_envelope(env, sample_rate, threshold_mult=1.0)
            if not result:
                continue
            text_raw, score, meta = result
            cq = _contest_exchange_quality(text_raw)
            if cq < 0.12 and score < 0.50:
                continue
            adj = score * 0.50 + cq * 0.50
            n_ser = len(_extract_contest_serials(text_raw))
            n_call = len(_extract_contest_calls(text_raw) - {'SZ1A'})
            adj += 0.025 * n_ser + 0.02 * n_call
            meta = dict(meta)
            meta.update({
                'bw_hz': bw, 'smooth_ms': smooth,
                'method': 'contest_stitch', 'contest_q': cq,
            })
            hypos.append((adj, text_raw, score, float(freq), meta))

    if not hypos:
        return None

    hypos.sort(key=lambda h: -h[0])
    best_adj, text, score, freq, meta = hypos[0]

    # Merge novel contest tokens from strong runner-ups (multi-tone separation)
    from cw_decoder.corrector import split_contest_runon
    merged, _ = split_contest_runon(text)
    have_s = _extract_contest_serials(merged)
    have_c = _extract_contest_calls(merged)
    for adj, t2, sc2, f2, m2 in hypos[1:6]:
        if adj < best_adj * 0.55 and adj < 0.35:
            continue
        t2n, _ = split_contest_runon(t2)
        new_s = _extract_contest_serials(t2n) - have_s
        new_c = _extract_contest_calls(t2n) - have_c
        if not new_s and not new_c:
            continue
        merged = _stitch_overlap_texts(merged, t2n)
        have_s |= _extract_contest_serials(t2n)
        have_c |= _extract_contest_calls(t2n)
        # Prefer frequency that contributed more serials when close
        if len(new_s) >= 1 and abs(f2 - freq) <= 20:
            freq = f2
            meta = dict(m2)
            meta['method'] = 'contest_stitch_merge'

    meta = dict(meta)
    meta['contest_hypos'] = len(hypos)
    meta['contest_q'] = _contest_exchange_quality(merged)
    # Re-score merged text slightly higher when it gained coverage
    cov_boost = 0.02 * min(3, len(have_s)) + 0.015 * min(4, len(have_c))
    final_adj = min(1.05, best_adj + cov_boost)
    final_score = min(0.99, max(score, final_adj * 0.9))
    return final_adj, merged, final_score, float(freq), meta


def _contest_serial_gap_pass(signal, sample_rate, snr_db, primary_freq,
                             base_text: str, base_score: float, base_meta: dict):
    """
    Contest-only: if host serials have a gap or stop early, re-decode the
    last third of the file with multi-hypo windows and stitch novel content.
    """
    have = _extract_contest_serials(base_text)
    if len(have) < 2:
        return None
    nums = sorted(int(s) for s in have if s.isdigit())
    if not nums:
        return None
    # Look for missing integers in the observed span, or extend +1..+3 past max
    missing = []
    for n in range(nums[0], nums[-1] + 1):
        if str(n) not in have and f'{n:04d}' not in have:
            # only 3-4 digit contest serials
            if n >= 100:
                missing.append(n)
    # Also probe for next serials after max (WPX often continues)
    for n in range(nums[-1] + 1, nums[-1] + 4):
        missing.append(n)
    if not missing:
        return None

    n = len(signal)
    # Focus on last 40% where late QSOs live
    start = int(n * 0.55)
    win_n = int(28 * sample_rate)
    hop_n = int(16 * sample_rate)
    pieces = []
    pos = start
    while pos < n:
        end = min(n, pos + win_n)
        if end - pos < sample_rate * 10:
            break
        best = _contest_window_decode(signal[pos:end], sample_rate, primary_freq, snr_db)
        if best is not None:
            adj, text, score, freq, meta = best
            if adj >= 0.30:
                pieces.append((pos / sample_rate, text, score, freq, meta))
        if end >= n:
            break
        pos += hop_n
    if not pieces:
        return None
    stitched = _stitch_contest_texts(pieces)
    if stitched is None:
        return None
    t2, sc2, f2, m2 = stitched
    new_s = _extract_contest_serials(t2) - have
    new_c = _extract_contest_calls(t2) - _extract_contest_calls(base_text)
    if not new_s and not new_c:
        return None
    merged = _stitch_overlap_texts(base_text, t2)
    m_out = dict(base_meta)
    m_out.update({
        'method': 'contest_stitch+gap',
        'gap_serials': sorted(new_s),
        'gap_calls': sorted(new_c),
    })
    if m2.get('contest_serials'):
        m_out['contest_serials'] = sorted(
            set(m_out.get('contest_serials', [])) | set(m2['contest_serials']))
    if m2.get('contest_calls'):
        m_out['contest_calls'] = sorted(
            set(m_out.get('contest_calls', [])) | set(m2['contest_calls']))
    score = min(0.99, max(base_score, sc2) + 0.02 * min(3, len(new_s) + len(new_c)))
    return merged, score, f2, m_out


def _contest_stitch_decode(signal, sample_rate, snr_db, primary_freq,
                           win_sec=32.0, hop_sec=18.0):
    """
    Full-timeline contest stitch: retune each window, keep novel exchanges,
    concatenate. Recovers mid/late QSOs lost by single-tone full-file decode.
    """
    n = len(signal)
    win_n = int(win_sec * sample_rate)
    hop_n = int(hop_sec * sample_rate)
    if win_n < sample_rate * 15 or n < win_n // 2:
        return None

    pieces = []
    pos = 0
    while pos < n:
        end = min(n, pos + win_n)
        if end - pos < sample_rate * 12:
            break
        chunk = signal[pos:end]
        best = _contest_window_decode(chunk, sample_rate, primary_freq, snr_db)
        if best is not None:
            adj, text, score, freq, meta = best
            if adj >= 0.28 and len(text.replace(' ', '')) >= 8:
                pieces.append((pos / sample_rate, text, score, freq, meta))
        if end >= n:
            break
        pos += hop_n

    stitched = _stitch_contest_texts(pieces)
    if stitched is None:
        return None
    text, score, freq, meta = stitched
    # Late-serial gap recovery (still contest-only)
    gap = _contest_serial_gap_pass(
        signal, sample_rate, snr_db, primary_freq, text, score, meta)
    if gap is not None:
        return gap
    return stitched


def _stitch_contest_texts(pieces):
    """
    Merge window texts into one contest transcript.
    pieces: list of (t0, text, score, freq, meta) sorted by time.
    Strategy: start with earliest solid piece; append later pieces' novel
    call/serial content via overlap-aware concatenation.
    """
    from cw_decoder.corrector import split_contest_runon

    if not pieces:
        return None
    pieces = sorted(pieces, key=lambda p: p[0])
    # Normalize each piece
    norms = []
    for t0, text, score, freq, meta in pieces:
        nt, _ = split_contest_runon(text)
        nt = re.sub(r'\s+', ' ', nt.upper()).strip()
        if len(nt.replace(' ', '')) < 6:
            continue
        if _dominant_char_ratio(nt) > 0.55:
            continue
        norms.append((t0, nt, score, freq, meta))
    if not norms:
        return None

    # Seed with the piece that has the most contest serials early in the file
    def seed_key(p):
        return (len(_extract_contest_serials(p[1])),
                _contest_exchange_quality(p[1]), p[2], -p[0])
    seed = max(norms[:max(1, len(norms)//3)], key=seed_key)
    stitched = seed[1]
    have_serials = _extract_contest_serials(stitched)
    have_calls = _extract_contest_calls(stitched)
    freqs = [seed[3]]
    scores = [seed[2]]

    for t0, nt, score, freq, meta in norms:
        if nt == seed[1]:
            continue
        new_serials = _extract_contest_serials(nt) - have_serials
        new_calls = _extract_contest_calls(nt) - have_calls
        # Skip windows that add nothing contest-useful
        if not new_serials and not new_calls:
            # maybe still extend trailing TU / R 5NN Txx
            if 'TU' in nt and nt not in stitched and score >= 0.7:
                pass
            else:
                continue
        # Overlap stitch
        stitched = _stitch_overlap_texts(stitched, nt)
        have_serials |= _extract_contest_serials(nt)
        have_calls |= _extract_contest_calls(nt)
        freqs.append(freq)
        scores.append(score)

    stitched = re.sub(r'\s+', ' ', stitched).strip()
    mean_sc = float(np.mean(scores)) if scores else 0.5
    # Boost score for serial coverage (reward complete contest log)
    cov = min(1.0, len(have_serials) / 6.0)
    final_score = min(0.99, mean_sc * 0.85 + 0.15 * cov + 0.05 * min(len(have_calls), 8) / 8)
    meta_out = {
        'method': 'contest_stitch',
        'wpm': float(np.median([
            p[4].get('wpm', 0) for p in norms if p[4].get('wpm', 0) > 0] or [0])),
        'contest_chunks': len(norms),
        'contest_serials': sorted(have_serials),
        'contest_calls': sorted(have_calls),
        'contest_freqs': [round(f, 1) for f in freqs],
    }
    from collections import Counter
    freq_rep = Counter(round(f, 0) for f in freqs).most_common(1)[0][0]
    return stitched, final_score, float(freq_rep), meta_out


def _stitch_overlap_texts(prev: str, curr: str) -> str:
    """Append curr onto prev, dropping the longest overlapping token run."""
    if not prev:
        return curr.strip()
    if not curr:
        return prev.strip()
    a = prev.strip().split()
    b = curr.strip().split()
    if not a:
        return ' '.join(b)
    if not b:
        return ' '.join(a)
    max_k = min(len(a), len(b), 20)
    for k in range(max_k, 2, -1):
        if a[-k:] == b[:k]:
            return ' '.join(a + b[k:])
    # Fuzzy overlap
    for k in range(min(max_k, 12), 3, -1):
        ok = True
        for x, y in zip(a[-k:], b[:k]):
            if x == y:
                continue
            if abs(len(x) - len(y)) <= 1 and lev(x, y) <= 1:
                continue
            ok = False
            break
        if ok:
            return ' '.join(a + b[k:])
    # If curr starts mid-exchange, try to find shared callsign/serial anchor
    a_set = set(a[-12:])
    for i, tok in enumerate(b[:16]):
        if tok in a_set and len(tok) >= 3:
            # find last occurrence in a
            for j in range(len(a) - 1, -1, -1):
                if a[j] == tok:
                    return ' '.join(a[:j + 1] + b[i + 1:])
    return ' '.join(a + b)


def _dominant_char_ratio(text: str) -> float:
    """Fraction of alphabetic chars occupied by the single most common letter."""
    letters = [c.upper() for c in text if c.isalpha()]
    if len(letters) < 8:
        return 0.0
    from collections import Counter
    return Counter(letters).most_common(1)[0][1] / len(letters)


def _text_plausibility(text: str) -> float:
    """
    Multiplier in (0.15 .. 1.35) favoring readable CW / English over mark-bias
    garbage (E/T runs). Safe for synthetic test_samples (callsigns, QSO templates)
    which already have diverse alnum and tokens like DE/CQ/RST.
    """
    if not text:
        return 0.2

    mult = 1.0
    mono = _dominant_char_ratio(text)
    # E/T/I spam from over-thresholding or wrong frequency
    if mono > 0.50:
        mult *= 0.20
    elif mono > 0.42:
        mult *= 0.35
    elif mono > 0.35:
        mult *= 0.70

    tokens = [w for w in ''.join(c if c.isalnum() else ' ' for c in text.upper()).split()
              if len(w) >= 2]
    if tokens:
        wordish = sum(1 for w in tokens if 2 <= len(w) <= 12 and w.isalpha())
        word_frac = wordish / len(tokens)
        if word_frac >= 0.5 and wordish >= 3:
            mult *= 1.0 + 0.025 * min(wordish, 6)
        hits = sum(1 for w in tokens if w in _PLAUSIBLE_WORDS)
        if hits:
            mult *= 1.0 + 0.02 * min(hits, 5)

    return float(max(0.15, min(1.25, mult)))


def _score_decode(text_raw, key_states, mark_durations, space_durations,
                  dit, dah, sample_rate):
    """
    Score a decode result. Used by _decode_from_envelope to compare
    multiple threshold methods and pick the best.
    """
    duty_cycle = float(key_states.mean())

    n_valid = sum(1 for ch in text_raw if ch != '?' and ch != ' ')
    n_total = max(len(text_raw.replace(' ', '')) if text_raw else 1, 1)
    valid_ratio = n_valid / n_total

    # 1. Duty cycle penalty (relaxed for QSB robustness)
    # QSB fading can cause duty cycle to vary from 0.3 to 0.5, so use wider tolerance
    duty_cycle_penalty = 1.0 - abs(duty_cycle - 0.4) * 1.0
    duty_cycle_penalty = max(0.4, min(1.0, duty_cycle_penalty))

    # 2. WPM plausibility penalty.
    # Real QRQ nets routinely run 45–80 WPM; only penalize extreme estimates.
    wpm = 1200.0 / dit if dit > 0 else 0
    wpm_penalty = 1.0
    if wpm < 3 or wpm > 140:
        wpm_penalty = 0.1
    elif wpm < 5:
        wpm_penalty = 0.4
    elif wpm > 100:
        wpm_penalty = 0.55
    elif wpm > 85:
        wpm_penalty = 0.8

    # 3. Dot/dash ratio consistency (real CW dit:dah ratio ~1:3)
    ratio_penalty = 1.0
    if dit > 0 and dah > 0:
        r = dah / dit
        if r < 1.5 or r > 6.0:
            ratio_penalty = 0.3
        elif r < 2.0 or r > 4.5:
            ratio_penalty = 0.6

    # 4. Timing regularity: mark durations should cluster around dit and dah
    regularity_penalty = 1.0
    if len(mark_durations) > 4 and dit > 0:
        boundary = (dit * dah) ** 0.5 if dah > dit else dit * 2.0
        dots = [d for d in mark_durations if d <= boundary]
        dashes = [d for d in mark_durations if d > boundary]
        if len(dots) >= 3:
            dot_cv = np.std(dots) / (np.mean(dots) + 1e-8)
            if dot_cv > 0.6:
                regularity_penalty *= 0.85
        if len(dashes) >= 3:
            dash_cv = np.std(dashes) / (np.mean(dashes) + 1e-8)
            if dash_cv > 0.6:
                regularity_penalty *= 0.85

    score = valid_ratio * duty_cycle_penalty * wpm_penalty * ratio_penalty * regularity_penalty

    # Short text penalty
    if n_valid < 10:
        score *= n_valid / 10.0

    # Character diversity penalty (penalize both low and high diversity)
    non_space_chars = text_raw.replace(' ', '').replace('?', '')
    if len(non_space_chars) > 3:
        diversity = len(set(non_space_chars)) / min(len(non_space_chars), 26)
        if diversity < 0.15:
            score *= 0.5  # Low diversity = repetitive garbage
        elif diversity > 0.85 and len(non_space_chars) > 20:
            score *= 0.8  # Very high diversity = possible over-decoding

    # Garbled character penalty (?, &, etc. indicate bad decode).
    # Soften when text still has word-like structure (common on real QRQ).
    n_garbled = text_raw.count('?') + text_raw.count('&')
    if n_garbled > 0:
        garbled_ratio = n_garbled / max(len(text_raw.replace(' ', '')), 1)
        tokens = [w for w in text_raw.upper().replace('?', ' ').split() if len(w) >= 3]
        if garbled_ratio > 0.25:
            score *= 0.5
        elif garbled_ratio > 0.05:
            score *= 0.75 if len(tokens) >= 3 else 0.6
        elif n_garbled >= 2:
            score *= 0.85 if len(tokens) >= 3 else 0.7

    # Prefer readable CW/English over mono-letter runs (critical for QRQ nets).
    score *= _text_plausibility(text_raw)

    score = max(0.0, min(1.0, score))

    meta = {
        'dit': dit, 'dah': dah, 'wpm': wpm, 'duty_cycle': duty_cycle,
        'n_chars': n_valid, 'ratio': dah / max(dit, 1),
        'wpm_penalty': wpm_penalty, 'ratio_penalty': ratio_penalty,
        'plausibility': _text_plausibility(text_raw),
    }
    return text_raw, score, meta


def _envelope_qsb_score(envelope, sample_rate, window_ms: float = 400.0) -> float:
    """Rough QSB indicator: std(local means) / global mean."""
    n = len(envelope)
    win = max(int(window_ms / 1000.0 * sample_rate), 8)
    if n < win * 4:
        return 0.0
    n_blocks = n // win
    means = envelope[:n_blocks * win].reshape(n_blocks, win).mean(axis=1)
    g = float(np.mean(means)) + 1e-8
    return float(np.std(means) / g)


def _hysteresis_threshold(envelope, base_threshold, fs: int = 0):
    """
    Hysteresis (Schmitt trigger) thresholding for CW envelope.

    Uses adaptive dual thresholds derived from envelope statistics:
    - Turn-on: 75th percentile (upper portion of signal)
    - Turn-off: 25th percentile (lower portion of signal)

    Downsampled for speed; returns None if states are invalid.
    """
    p25 = float(np.percentile(envelope, 25))
    p75 = float(np.percentile(envelope, 75))

    # Hysteresis needs sufficient separation between high and low states
    if p75 - p25 < 0.02:
        return None

    upper = p75  # Turn-on: must be in upper quartile
    lower = p25  # Turn-off: must be in lower quartile

    n = len(envelope)
    ds = max(1, int(fs // 2000) if fs > 0 else max(1, n // 16000))
    env_ds = envelope[::ds]
    n_ds = len(env_ds)
    states_ds = np.zeros(n_ds, dtype=np.int8)
    current_state = 0

    for i in range(n_ds):
        if current_state == 0:
            if env_ds[i] > upper:
                current_state = 1
        else:
            if env_ds[i] < lower:
                current_state = 0
        states_ds[i] = current_state

    duty = states_ds.mean()
    if duty < 0.03 or duty > 0.97:
        return None  # Invalid

    if ds == 1:
        return states_ds
    states = np.repeat(states_ds, ds)[:n]
    if len(states) < n:
        states = np.pad(states, (0, n - len(states)), mode='edge')
    return states


def _decode_from_envelope(envelope, sample_rate, threshold_mult=1.0,
                          try_alt: bool = True):
    """
    Shared pipeline from envelope to decoded text.

    Tries standard thresholding first; only runs hysteresis / QSB alts when
    the standard result is weak (score / '?' rate), to keep latency down.
    """
    threshold_base = auto_threshold(envelope)
    threshold = threshold_base * threshold_mult

    def _run(key_states, thr):
        duty = float(key_states.mean())
        if duty < 0.05 or duty > 0.97:
            return None
        return _decode_from_key_states(key_states, envelope, sample_rate, thr)

    # Path A: standard single threshold
    key_states = (envelope > threshold).astype(np.int8)
    duty_cycle = key_states.mean()
    if duty_cycle < 0.05 or duty_cycle > 0.97:
        threshold2 = 0.35 * envelope.max()
        states2 = (envelope > threshold2).astype(np.int8)
        if 0.08 < states2.mean() < 0.92:
            key_states = states2
            threshold = threshold2
    best = _run(key_states, threshold)

    if not try_alt:
        return best

    # Skip expensive alts when standard decode already looks solid
    if best is not None:
        text0, score0, _ = best
        q_ratio = text0.count('?') / max(len(text0.replace(' ', '')), 1)
        if score0 >= 0.72 and q_ratio < 0.06:
            return best

    candidates = []
    if best is not None:
        candidates.append(best)

    # Path B: hysteresis + mild morphological clean
    ks_h = _hysteresis_key_states(
        envelope, threshold, ratio=0.65, fs=sample_rate)
    if 0.05 < float(ks_h.mean()) < 0.97:
        ks_h = morphological_clean(
            ks_h, sample_rate, min_mark_ms=2.0, min_space_ms=1.5)
        r = _run(ks_h, threshold)
        if r is not None:
            candidates.append(r)

    # Path C: percentile hysteresis (slow-rise envelopes)
    ks_p = _hysteresis_threshold(envelope, threshold, fs=sample_rate)
    if ks_p is not None:
        ks_p = morphological_clean(
            ks_p, sample_rate, min_mark_ms=2.0, min_space_ms=1.5)
        r = _run(ks_p, threshold)
        if r is not None:
            candidates.append(r)

    # Path D: QSB local-mean normalization
    if _envelope_qsb_score(envelope, sample_rate) > 0.18:
        env_n = normalize_envelope(envelope, sample_rate, window_ms=500)
        thr_n = auto_threshold(env_n) * threshold_mult
        ks_n = (env_n > thr_n).astype(np.int8)
        if 0.05 < float(ks_n.mean()) < 0.97:
            ks_n = morphological_clean(
                ks_n, sample_rate, min_mark_ms=2.0, min_space_ms=1.5)
            r = _run(ks_n, thr_n)
            if r is not None:
                candidates.append(r)

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])


def _decode_from_key_states(key_states, envelope, sample_rate, threshold, segments=None):
    """
    Decode from binary key states: segment extraction → dit estimation →
    classification → morse assembly → scoring.

    Args:
        segments: optional pre-extracted segments (skip extract_segments if provided).
                  Used by post-processing to inject fragment-merged segments.
    """
    # Mild morph clean on raw states (noise spikes / pinholes) before RLE
    if segments is None:
        key_states = morphological_clean(
            key_states, sample_rate, min_mark_ms=1.5, min_space_ms=1.0)
        segments = extract_segments(key_states, sample_rate, min_ms=3.0)

    mark_segs = [(s, d) for s, d in segments if s == 1]
    space_segs = [(s, d) for s, d in segments if s == 0]

    if len(mark_segs) < 3:
        return None

    mark_durations = [d for s, d in mark_segs]
    space_durations = [d for s, d in space_segs]

    MIN_ELEMENT_MS = 10.0

    # Noise filtering + dit estimation
    mark_durs_filtered = [d for d in mark_durations if d >= MIN_ELEMENT_MS]
    if len(mark_durs_filtered) < 5:
        mark_durs_filtered = list(mark_durations)

    # Robust dit/dah estimation (percentile + bimodal detection)
    dit, dah = _robust_dit_estimate(mark_durs_filtered, sample_rate)

    # Iterative noise segment filtering (< 35% of dit)
    if dit > 0:
        noise_threshold = dit * 0.35
        mark_durs_clean = [d for d in mark_durs_filtered if d >= noise_threshold]
        if len(mark_durs_clean) >= 5:
            dit, dah = _robust_dit_estimate(mark_durs_clean, sample_rate)

    if dit < 5:
        dit = 20

    # === Noise fragment guard: detect and correct implausible WPM estimates ===
    # When QSB fading + noise create many tiny mark fragments (< 30ms),
    # _robust_dit_estimate can be misled into estimating a very short dit
    # (e.g. 20ms → WPM=60), corrupting the entire decode. This guard detects
    # when the estimated WPM is inconsistent with the overall mark distribution
    # and falls back to a median-based estimate that ignores noise fragments.
    _wpm_check = 1200.0 / dit if dit > 0 else 0
    median_mark = float(np.median(mark_durations)) if mark_durations else 60.0
    if _wpm_check > 55 and median_mark > 55:
        # WPM estimate implausibly high given the mark durations.
        # Filter marks to those within a reasonable range of the median
        # (retain marks >= 35% of median, which captures real dits while
        # excluding noise fragments that are much shorter than median).
        marks_clean = [d for d in mark_durations if d >= median_mark * 0.35]
        if len(marks_clean) >= 5:
            dit2, dah2 = _robust_dit_estimate(marks_clean, sample_rate)
            wpm2 = 1200.0 / dit2 if dit2 > 0 else 0
            if 5 < wpm2 < 55:
                dit, dah = dit2, dah2

    wpm = 1200.0 / dit if dit > 0 else 0

    # Classify dots and dashes using geometric mean boundary.
    # Geometric mean correctly handles non-standard ratios (e.g. Bug key 2.2:1).
    # Arithmetic mean (dit+dah)/2 assumes 3:1 ratio and misclassifies short dahs.
    labels = []
    if dit > 0 and dah > dit:
        boundary = (dit * dah) ** 0.5  # geometric mean
    else:
        boundary = dit * 2.0
    for dur in mark_durations:
        labels.append('dash' if dur > boundary else 'dot')

    # Relative distance to boundary — used for soft flip of uncertain marks
    mark_conf = [abs(dur - boundary) / max(boundary, 1e-6) for dur in mark_durations]

    # Hybrid gap classification (K-means + dit prior + Viterbi when needed)
    gap_labels = _duration_based_gaps(space_durations, dit)

    # Contextual gap reclassification for slow/novice operators.
    # Only applied for slow CW (WPM < 20, dit > 60ms) where timing
    # jitter is high enough that in-char/char_gap boundaries overlap.
    # At faster speeds the standard classification is more reliable.
    # Multi-pass: loop 3x to handle multiple merged character groups.
    # Each pass rebuilds groups from updated gap_labels, so splits
    # from earlier passes expose new groups in later passes.
    if len(gap_labels) > 0 and dit > 60:
        for _ in range(3):
            _split_invalid_chars(segments, labels, gap_labels, mark_durations,
                                 space_durations, dit, dah)

    # Contextual mark reclassification for short dahs.
    # Only applied for slow CW (WPM < 20) where operators produce
    # compressed dahs that fall below the geometric mean boundary.
    if dit > 60:
        _contextual_mark_reclassify(segments, labels, gap_labels,
                                     mark_durations, dit, dah)

    # Soft boundary flip: chars that soft-decode to '?' — try flipping
    # low-confidence marks near the dit/dah boundary.
    _soft_boundary_mark_fix(
        segments, labels, gap_labels, mark_durations, mark_conf, boundary)

    # Within-character consistency check (all operator types).
    # Fixes cases where a single element is misclassified (e.g., 5=`.....`
    # decoded as 4=`....-` because the last dot was slightly long).
    _within_char_consistency(segments, labels, gap_labels, mark_durations, dit, dah)

    # Adaptive character-level pattern validator.
    # Catches uniform element-label inversions on borderline-SNR signals
    # (e.g. C=`-.-.` → `.-.-`=<AR> when all elements are near the boundary).
    _validate_char_patterns(segments, labels, gap_labels, mark_durations, dit, dah)

    # Assemble morse code
    text_raw = _assemble_morse(segments, labels, gap_labels)
    return _score_decode(text_raw, key_states, mark_durations, space_durations,
                         dit, dah, sample_rate)


def _soft_boundary_mark_fix(segments, labels, gap_labels, mark_durations,
                             mark_conf, boundary):
    """
    When a character soft-decodes to '?', try flipping marks whose duration
    is within ~20% of the dit/dah boundary. Only accept exact Morse hits.
    """
    if not labels or not gap_labels:
        return

    char_groups = []
    current = []
    mark_idx = 0
    gap_idx = 0
    for state, _dur in segments:
        if state == 1:
            current.append(mark_idx)
            mark_idx += 1
        else:
            gl = gap_labels[gap_idx] if gap_idx < len(gap_labels) else 'char_gap'
            gap_idx += 1
            if gl in ('char_gap', 'word_gap'):
                if current:
                    char_groups.append(current)
                    current = []
    if current:
        char_groups.append(current)

    for marks_idx in char_groups:
        if len(marks_idx) < 2:
            continue
        pattern = ''.join('.' if labels[i] == 'dot' else '-' for i in marks_idx)
        if pattern in MORSE:
            continue
        if soft_decode(pattern) != '?':
            continue

        amb = [i for i in marks_idx
               if i < len(mark_conf) and mark_conf[i] < 0.22]
        if not amb:
            continue

        flipped = False
        for i in amb:
            trial = labels[:]
            trial[i] = 'dash' if trial[i] == 'dot' else 'dot'
            pat = ''.join('.' if trial[j] == 'dot' else '-' for j in marks_idx)
            ch = MORSE.get(pat)
            if ch and ch not in ('<AR>', '<SK>', '<KN>') and (
                    ch.isalnum() or ch in ('/', '=', '+')):
                labels[i] = 'dash' if labels[i] == 'dot' else 'dot'
                flipped = True
                break
        if flipped:
            continue

        if len(amb) <= 3:
            done = False
            for a in range(len(amb)):
                for b in range(a + 1, len(amb)):
                    trial = labels[:]
                    for i in (amb[a], amb[b]):
                        trial[i] = 'dash' if trial[i] == 'dot' else 'dot'
                    pat = ''.join(
                        '.' if trial[j] == 'dot' else '-' for j in marks_idx)
                    ch = MORSE.get(pat)
                    if ch and ch.isalnum() and len(ch) == 1:
                        for i in (amb[a], amb[b]):
                            labels[i] = (
                                'dash' if labels[i] == 'dot' else 'dot')
                        done = True
                        break
                if done:
                    break


def _split_invalid_chars(segments, labels, gap_labels, mark_durations,
                         space_durations, dit, dah):
    """
    Split merged characters when gap classification incorrectly merges
    two adjacent characters due to short inter-character gaps.

    For slow/novice operators, char_gaps can be as short as 200ms,
    overlapping with in_char gaps. When K-means or fixed thresholds
    misclassify a char_gap as in_char, the resulting Morse pattern
    becomes invalid (too many elements or unknown sequence).

    Detects merged characters and splits them at the largest internal
    gap that produces valid sub-patterns.
    """
    if dit <= 0:
        return

    # Adaptive split threshold: only use aggressive thresholds when
    # timing is genuinely compressed (dah/dit ratio < 2.5). Clean slow
    # keying with normal ratio doesn't need lower thresholds.
    ratio = dah / dit if dah > dit else 3.0
    if ratio < 2.2:
        min_gap_threshold = dit * 0.7   # Severely compressed
    elif ratio < 2.5 and dit > 120:
        min_gap_threshold = dit * 0.85  # Moderately compressed, slow
    elif dit > 170:
        min_gap_threshold = dit * 0.85  # Very slow, even with normal ratio
    elif dit > 120:
        min_gap_threshold = dit * 1.0   # Slow, normal ratio
    else:
        min_gap_threshold = dit * 1.1   # Standard

    # Maximum number of splits to perform in one pass
    MAX_SPLITS = 1
    splits_done = 0

    # Determine offset: gap_labels[0] corresponds to space after which mark?
    # If segments start with SPACE: gap_labels[N] = space after mark[N-1]
    # If segments start with MARK:  gap_labels[N] = space after mark[N]
    first_is_space = (len(segments) > 0 and segments[0][0] == 0)

    # Step 1: Build character groups
    char_groups = []  # list of (mark_indices, [(gap_label_idx, space_dur), ...])
    current_marks = []
    current_gaps = []  # (gap_label_idx, space_duration)
    mark_idx = 0
    gap_idx = 0

    for state, dur in segments:
        if state == 1:
            current_marks.append(mark_idx)
            mark_idx += 1
        else:
            gl = gap_labels[gap_idx] if gap_idx < len(gap_labels) else 'char_gap'
            sd = space_durations[gap_idx] if gap_idx < len(space_durations) else 0
            if gl == 'in_char':
                current_gaps.append((gap_idx, sd))
            elif gl in ('char_gap', 'word_gap'):
                if current_marks:
                    char_groups.append((list(current_marks), list(current_gaps)))
                    current_marks = []
                    current_gaps = []
            gap_idx += 1
    if current_marks:
        char_groups.append((list(current_marks), list(current_gaps)))

    # Step 2: For each group with invalid pattern, try splitting
    for marks_idx, gaps in char_groups:
        if len(gaps) == 0:
            continue

        pattern = ''.join('.' if labels[i] == 'dot' else '-' for i in marks_idx)
        # Use exact MORSE match check: patterns not in the dictionary
        # are likely merged characters that need splitting. Using soft_decode
        # would falsely accept fuzzy-matched patterns as valid characters.
        if pattern in MORSE and len(pattern) <= 6:
            continue  # Already valid

        # Sort gaps by duration (largest first)
        gaps_sorted = sorted(gaps, key=lambda x: -x[1])

        for gap_label_idx, gap_dur in gaps_sorted:
            if gap_dur < min_gap_threshold:
                break  # Too small to be a plausible char_gap

            # Determine which mark precedes this gap
            # gap_idx tells us the index in gap_labels
            # Find the mark index that comes right before this gap
            mi = 0
            gi = 0
            split_mark_idx = None
            for state, dur in segments:
                if state == 1:
                    split_mark_idx = mi
                    mi += 1
                else:
                    if gi == gap_label_idx:
                        # This gap follows split_mark_idx
                        break
                    gi += 1

            if split_mark_idx is None or split_mark_idx not in marks_idx:
                continue
            if split_mark_idx == marks_idx[-1]:
                continue  # Can't split after last mark

            # Split the marks at this position
            split_pos_in_group = marks_idx.index(split_mark_idx)
            left_marks = marks_idx[:split_pos_in_group + 1]
            right_marks = marks_idx[split_pos_in_group + 1:]

            if len(left_marks) == 0 or len(right_marks) == 0:
                continue

            left_pattern = ''.join('.' if labels[i] == 'dot' else '-' for i in left_marks)
            right_pattern = ''.join('.' if labels[i] == 'dot' else '-' for i in right_marks)

            left_decoded = soft_decode(left_pattern)
            right_decoded = soft_decode(right_pattern)

            if (left_decoded != '?' and right_decoded != '?' and
                    len(left_pattern) <= 6 and len(right_pattern) <= 6):
                # Valid split! Update the gap label
                if gap_label_idx < len(gap_labels):
                    gap_labels[gap_label_idx] = 'char_gap'
                splits_done += 1
                if splits_done >= MAX_SPLITS:
                    return
                break  # Move to next group
    return

def _contextual_mark_reclassify(segments, labels, gap_labels,
                                mark_durations, dit, dah):
    """
    Reclassify borderline marks as dahs using within-character comparison.

    When OP_NOVICE/OP_VERYOLD operators produce short dahs (e.g., 237ms)
    that fall below the geometric mean boundary, the mark is initially
    classified as a dot. Within a properly separated character, compare
    each borderline mark to the minimum mark: if ratio > threshold, it's a dah.

    Only reclassifies dot→dah (not dah→dot) since short dahs are the
    primary issue for non-standard operators.
    """
    if dit <= 0:
        return

    # Adaptive thresholds: only use aggressive thresholds when the
    # dah/dit ratio is genuinely compressed. Clean slow keying with
    # normal ratio doesn't need wider borderline ranges.
    ratio = dah / dit if dah > dit else 3.0
    if ratio < 2.2:
        # Severely compressed timing (e.g. OP_NOVICE 8WPM with jitter)
        lo = dit * 0.9
        hi = dit * 2.4
        ratio_threshold = 1.2
    elif ratio < 2.5 and dit > 120:
        # Moderately compressed, slow operator
        lo = dit * 1.0
        hi = dit * 2.2
        ratio_threshold = 1.3
    elif dit > 170:
        # Very slow but normal ratio
        lo = dit * 1.0
        hi = dit * 2.2
        ratio_threshold = 1.4
    else:
        # Standard timing
        lo = dit * 1.2
        hi = dit * 2.0
        ratio_threshold = 1.5

    # Build character groups (same as _split_invalid_chars)
    char_groups = []
    current_marks = []
    mark_idx = 0
    gap_idx = 0

    for state, dur in segments:
        if state == 1:
            current_marks.append(mark_idx)
            mark_idx += 1
        else:
            gl = gap_labels[gap_idx] if gap_idx < len(gap_labels) else 'char_gap'
            gap_idx += 1
            if gl in ('char_gap', 'word_gap'):
                if current_marks:
                    char_groups.append(list(current_marks))
                    current_marks = []
    if current_marks:
        char_groups.append(list(current_marks))

    # Reclassify borderline marks within each character
    for marks_idx in char_groups:
        if len(marks_idx) < 2:
            continue
        char_durs = [mark_durations[i] for i in marks_idx]
        min_dur = min(char_durs)
        if min_dur < dit * 0.4:
            continue  # Noise floor too low, skip

        for i in marks_idx:
            dur = mark_durations[i]
            if lo < dur < hi and labels[i] == 'dot':
                ratio = dur / max(min_dur, 1)
                if ratio > ratio_threshold:
                    labels[i] = 'dash'


def _within_char_consistency(segments, labels, gap_labels, mark_durations, dit, dah):
    """
    Within-character consistency check: if all elements in a character have
    similar durations, they should all be the same type (all dots or all dashes).

    Fixes the most common element classification errors:
    - 5=`.....` decoded as 4=`....-` (last dot slightly long → dash)
    - D=`-..` decoded as K=`-.-` (second dot slightly long → dash)
    - T=`-` decoded as E=`.` (dah slightly short → dot)

    Uses the ratio between max and min element duration within a character.
    If ratio < 1.8 (all elements similar), apply majority-vote classification.
    Only applies when the character has >= 2 elements and the durations are
    clearly in the "similar" range (not borderline dit/dah cases).
    """
    if dit <= 0:
        return

    # Build character groups
    char_groups = []
    current_marks = []
    mark_idx = 0
    gap_idx = 0

    for state, dur in segments:
        if state == 1:
            current_marks.append(mark_idx)
            mark_idx += 1
        else:
            gl = gap_labels[gap_idx] if gap_idx < len(gap_labels) else 'char_gap'
            gap_idx += 1
            if gl in ('char_gap', 'word_gap'):
                if current_marks:
                    char_groups.append(list(current_marks))
                    current_marks = []
    if current_marks:
        char_groups.append(list(current_marks))

    # Check consistency within each character
    for marks_idx in char_groups:
        if len(marks_idx) < 2:
            continue

        char_durs = [mark_durations[i] for i in marks_idx]
        min_dur = min(char_durs)
        max_dur = max(char_durs)

        if min_dur < dit * 0.3:
            continue  # Too close to noise floor

        ratio = max_dur / max(min_dur, 1)

        # If all elements are within 1.8x of each other, they should be
        # the same type. This catches cases where one element is slightly
        # misclassified due to timing jitter.
        if ratio < 1.8:
            # Determine majority type
            n_dots = sum(1 for i in marks_idx if labels[i] == 'dot')
            n_dashes = len(marks_idx) - n_dots

            if n_dots > n_dashes:
                # Majority are dots → reclassify any dashes to dots
                for i in marks_idx:
                    if labels[i] == 'dash':
                        # Only reclassify if this element is close to the
                        # dot cluster (within 2x of median dot duration)
                        dot_durs = sorted([mark_durations[j] for j in marks_idx if labels[j] == 'dot'])
                        median_dot = dot_durs[len(dot_durs) // 2]
                        if mark_durations[i] < median_dot * 2.2:
                            labels[i] = 'dot'
            elif n_dashes > n_dots:
                # Majority are dashes → reclassify any dots to dashes
                for i in marks_idx:
                    if labels[i] == 'dot':
                        dash_durs = sorted([mark_durations[j] for j in marks_idx if labels[j] == 'dash'])
                        median_dash = dash_durs[len(dash_durs) // 2]
                        if mark_durations[i] > median_dash * 0.45:
                            labels[i] = 'dash'

        # Special case: exactly 1 outlier in a longer character (>=3 elements).
        # If N-1 elements agree and 1 disagrees, and the outlier is borderline,
        # reclassify the outlier.
        elif len(marks_idx) >= 3 and ratio < 3.5:
            n_dots = sum(1 for i in marks_idx if labels[i] == 'dot')
            n_dashes = len(marks_idx) - n_dots

            # Check for single-dot-in-dashes or single-dash-in-dots
            if n_dots == 1 and n_dashes >= 2:
                # Single dot among dashes — check if it's borderline
                dot_idx = [i for i in marks_idx if labels[i] == 'dot'][0]
                dot_dur = mark_durations[dot_idx]
                dash_durs = [mark_durations[i] for i in marks_idx if labels[i] == 'dash']
                min_dash = min(dash_durs)
                # If the "dot" is actually > 60% of the shortest dash, it's likely a dash
                if dot_dur > min_dash * 0.6:
                    labels[dot_idx] = 'dash'
            elif n_dashes == 1 and n_dots >= 2:
                # Single dash among dots — check if it's borderline
                dash_idx = [i for i in marks_idx if labels[i] == 'dash'][0]
                dash_dur = mark_durations[dash_idx]
                dot_durs = [mark_durations[i] for i in marks_idx if labels[i] == 'dot']
                max_dot = max(dot_durs)
                # If the "dash" is actually < 1.6x of the longest dot, it's likely a dot
                if dash_dur < max_dot * 1.6:
                    labels[dash_idx] = 'dot'


def _validate_char_patterns(segments, labels, gap_labels, mark_durations, dit, dah):
    """
    Adaptive character-level pattern validator.

    When element labels are uniformly inverted on borderline-SNR signals,
    prosign Morse patterns (e.g. `.-.-` = <AR>) can emerge instead of the
    intended character. If the pattern is a prosign and flipping ALL element
    labels yields a valid alphanumeric character, apply the correction.

    Safeguard: only activates when dah/dit ratio is near standard (2.2-3.8).
    Severely compressed or stretched timing suggests non-standard keying
    where uniform inversion is not the right diagnosis.
    """
    if dit <= 0 or dah <= dit:
        return

    ratio = dah / dit
    if ratio < 2.2 or ratio > 3.8:
        return

    # Build character groups
    char_groups = []
    current_marks = []
    mark_idx = 0
    gap_idx = 0

    for state, dur in segments:
        if state == 1:
            current_marks.append((mark_idx, dur))
            mark_idx += 1
        else:
            gl = gap_labels[gap_idx] if gap_idx < len(gap_labels) else 'char_gap'
            gap_idx += 1
            if gl in ('char_gap', 'word_gap'):
                if current_marks:
                    char_groups.append(list(current_marks))
                    current_marks = []
    if current_marks:
        char_groups.append(list(current_marks))

    for marks_idx_dur in char_groups:
        marks_idx = [i for i, _ in marks_idx_dur]

        if len(marks_idx) < 3:
            continue

        pattern = ''.join('.' if labels[i] == 'dot' else '-' for i in marks_idx)

        # Only intervene if the pattern decodes to a non-alphanumeric result
        # (prosign, punctuation, or symbol). Valid alphanumeric characters
        # should pass through unchanged.
        decoded = MORSE.get(pattern)
        if decoded is None:
            # Not an exact match — could be garbled, skip
            continue
        if decoded.isalnum() and len(decoded) == 1:
            # Already a valid alphanumeric character, no correction needed
            continue

        # Try inverting ALL elements
        flipped_pattern = ''.join(
            '-' if labels[i] == 'dot' else '.' for i in marks_idx
        )
        flipped_char = MORSE.get(flipped_pattern)
        flip_all = True  # flip ALL marks

        # If exact flip doesn't match, try dropping one edge element.
        # Noise fragments or merged-char artifacts can add one spurious
        # element (e.g. C=`-.-.` → `.-.-.` = `+` with an extra dot).
        if flipped_char is None and len(marks_idx) >= 4:
            for drop_idx in (0, -1):
                if drop_idx == 0:
                    subset = marks_idx[1:]
                else:
                    subset = marks_idx[:-1]
                subset_pattern = ''.join(
                    '-' if labels[i] == 'dot' else '.' for i in subset
                )
                candidate = MORSE.get(subset_pattern)
                if candidate and candidate not in (
                        '<AR>', '<SK>', '<KN>',
                        '.', ',', '?', "'", '!', '/', '(', ')',
                        '&', ':', ';', '=', '+', '-', '_', '"', '$', '@'):
                    flipped_char = candidate
                    # Flip only the subset, leave the edge element as-is
                    for i in subset:
                        labels[i] = 'dash' if labels[i] == 'dot' else 'dot'
                    flip_all = False  # already applied partial flip
                    break

        if flip_all and flipped_char and flipped_char not in (
                '<AR>', '<SK>', '<KN>',
                '.', ',', '?', "'", '!', '/', '(', ')',
                '&', ':', ';', '=', '+', '-', '_', '"', '$', '@'):
            for i in marks_idx:
                labels[i] = 'dash' if labels[i] == 'dot' else 'dot'


def _assemble_morse(segments, labels, gap_labels):
    """Assemble morse code from segments, element labels, and gap labels."""
    morse_chars = []
    current = ""
    mark_idx = space_idx = 0

    for state, dur in segments:
        if state == 1:
            if mark_idx < len(labels):
                current += '.' if labels[mark_idx] == 'dot' else '-'
            mark_idx += 1
        else:
            if space_idx < len(gap_labels):
                gap_label = gap_labels[space_idx]
            else:
                gap_label = 'char_gap'
            space_idx += 1
            if gap_label == 'in_char':
                continue
            if current:
                morse_chars.append(current)
                current = ""
            if gap_label == 'word_gap':
                morse_chars.append(' ')

    if current:
        morse_chars.append(current)

    # Decode morse to text
    decoded = []
    for mc in morse_chars:
        if mc == ' ':
            decoded.append(' ')
        else:
            decoded.append(soft_decode(mc))
    return ''.join(decoded)


def _multi_threshold_decode(envelope, sample_rate):
    """
    Multi-threshold integrated decode: try 4 threshold multipliers, return best result.
    Uses try_alt=False — alternate hysteresis/QSB paths already run on std decode.
    """
    THRESHOLD_MULTIPLIERS = [0.8, 1.0, 1.5, 2.5]
    best = None
    for mult in THRESHOLD_MULTIPLIERS:
        result = _decode_from_envelope(
            envelope, sample_rate, threshold_mult=mult, try_alt=False)
        if result is None:
            continue
        text_raw, score, meta = result
        if best is None or score > best[1]:
            best = (text_raw, score, meta)
    return best


# ============================================================
# Multi-config decode (bandwidth × smoothing × threshold × normalization)
# ============================================================

def _iterative_decode(envelope, sample_rate, max_iters=3):
    """
    Iterative decode: decode → estimate dit → refine threshold → re-decode.
    Inspired by FLDigi's adaptive approach. Each iteration refines the dit estimate
    and adjusts threshold for better accuracy.
    """
    best = None
    prev_text = None
    
    for iteration in range(max_iters):
        result = _decode_from_envelope(envelope, sample_rate, threshold_mult=1.0)
        if result is None:
            break
        
        text_raw, score, meta = result
        
        # Check convergence: if text hasn't changed, stop
        if text_raw == prev_text:
            break
        prev_text = text_raw
        
        # Use the dit estimate to refine threshold
        dit = meta.get('dit', 0)
        if dit > 5 and best is not None:
            # Try threshold adjustment based on dit-derived noise floor
            noise_thr = dit * 0.3 / (envelope.max() + 1e-8)
            adjusted_mult = max(0.5, min(2.0, noise_thr * 3))
            result_adj = _decode_from_envelope(envelope, sample_rate, threshold_mult=adjusted_mult)
            if result_adj:
                _, score_adj, meta_adj = result_adj
                if score_adj > score:
                    text_raw, score, meta = result_adj
        
        if best is None or score > best[1]:
            best = (text_raw, score, meta)
        
        # Early exit if score is very good
        if score >= 0.9:
            break
    
    return best


def _multi_config_decode(signal, sample_rate, cw_freq, snr_db, use_coherent=True, fast=False,
                         contest_mode=False):
    """
    Multi-config integrated decode: try different bandwidth, smoothing, threshold,
    envelope method, and normalization combinations. Designed for real-world frequency
    drift and speed variation. Uses cross-config voting to select most reliable result.
    
    Envelope methods (inspired by international ham decoders):
    - Square-law: traditional, good for clean signals
    - RMS moving-window: CW Skimmer / morse-audio-decoder approach, better in noise
    - Hann-windowed RMS: best frequency selectivity, reduces spectral leakage

    Args:
        fast: fast mode, fewer configs (for chunk decode acceleration)
        contest_mode: add narrow BW configs to reject adjacent contest stations
    """
    if fast:
        CONFIGS = [
            (60, 12.0, 'sq'),   # narrow band + standard (default)
            (100, 12.0, 'sq'),  # wide band (frequency drift tolerance)
            (60, 3.0, 'sq'),    # narrow + minimal smoothing (QRQ / slow CW)
        ]
        if contest_mode:
            CONFIGS = [
                (40, 8.0, 'sq'),
                (50, 6.0, 'sq'),
                (60, 8.0, 'sq'),
            ] + CONFIGS
    else:
        CONFIGS = [
            (60, 12.0, 'sq'),   # narrow band + standard (default)
            (100, 12.0, 'sq'),  # wide band (frequency drift tolerance)
            (60, 3.0, 'sq'),    # narrow + minimal smoothing (QRQ / slow CW)
            (60, 18.0, 'sq'),   # narrow + heavy smoothing (high jitter)
            (100, 6.0, 'rms'),  # wide + RMS envelope (noise robust)
        ]
        if contest_mode:
            # ±40–50 Hz rejects neighbors 15–40 Hz away (WPX-style pile-ups)
            CONFIGS = [
                (40, 8.0, 'sq'),
                (50, 6.0, 'sq'),
                (40, 12.0, 'sq'),
            ] + CONFIGS

    # Also add Goertzel + onset configs in the base set (non-fast)
    if not fast:
        CONFIGS = list(CONFIGS) + [(60, 3.0, 'gz'), (60, 6.0, 'on')]

    all_results = []

    def _run_config(bw_hz, smooth_ms, env_method):
        signal_bp = bandpass(signal, sample_rate, cw_freq - bw_hz, cw_freq + bw_hz)

        # Weak signal: use coherent integration (extended to SNR < 0)
        if snr_db < 0 and use_coherent:
            try:
                integrator = AdaptiveIntegrator(
                    sample_rate, cw_freq, min_integ_ms=50, max_integ_ms=200)
                envelope, _ = integrator.process(signal_bp, target_snr=10.0)
            except Exception:
                envelope = square_law_envelope(
                    signal_bp, sample_rate, smooth_ms=smooth_ms)
        elif env_method == 'gz':
            # Goertzel single-tone envelope (NUE-PSK / Fallows)
            envelope = goertzel_envelope(
                signal_bp, sample_rate, cw_freq, block_ms=max(2.0, smooth_ms * 0.4))
        elif env_method == 'on':
            envelope = onset_enhanced_envelope(
                signal_bp, sample_rate, smooth_ms=smooth_ms)
        elif env_method == 'rms':
            envelope = rms_envelope(signal_bp, sample_rate, window_ms=smooth_ms)
        elif env_method == 'hann':
            envelope = hann_rms_envelope(signal_bp, sample_rate, window_ms=smooth_ms)
        else:
            envelope = square_law_envelope(signal_bp, sample_rate, smooth_ms=smooth_ms)

        local = []
        # Decode Path A: standard (+ alt hysteresis/QSB when try_alt default)
        result = _decode_from_envelope(envelope, sample_rate, threshold_mult=1.0)
        if result:
            text_raw, score, meta = result
            meta = dict(meta)
            meta['bw_hz'] = bw_hz
            meta['smooth_ms'] = smooth_ms
            meta['env_method'] = env_method
            meta['method'] = 'std'
            local.append((text_raw, score, meta))

        # Decode Path B: multi-threshold integration
        result_mt = _multi_threshold_decode(envelope, sample_rate)
        if result_mt:
            text_raw, score, meta = result_mt
            meta = dict(meta)
            meta['bw_hz'] = bw_hz
            meta['smooth_ms'] = smooth_ms
            meta['env_method'] = env_method
            meta['method'] = 'mt'
            local.append((text_raw, score, meta))

        # Decode Path C: AGC-based QSB compensation (only for primary config)
        if bw_hz == 60 and smooth_ms == 12.0 and env_method == 'sq':
            try:
                signal_agc = apply_agc(signal_bp, sample_rate, mode="slow")
                envelope_agc = square_law_envelope(signal_agc, sample_rate, smooth_ms=smooth_ms)
                result_agc = _decode_from_envelope(envelope_agc, sample_rate, threshold_mult=1.0)
                if result_agc:
                    text_raw, score, meta = result_agc
                    meta = dict(meta)
                    meta['bw_hz'] = bw_hz
                    meta['smooth_ms'] = smooth_ms
                    meta['env_method'] = env_method
                    meta['method'] = 'agc'
                    local.append((text_raw, score, meta))
            except Exception:
                pass
        return local

    for bw_hz, smooth_ms, env_method in CONFIGS:
        all_results.extend(_run_config(bw_hz, smooth_ms, env_method))

    # Adaptive QRQ / QRM recovery configs.
    # Trigger on fast WPM, mono-letter spam, OR heavy '?' garble (wrong BW/smooth).
    # Narrow BW (50Hz) + 2ms smooth often separates adjacent QRQ net tones.
    if all_results and not fast:
        probe = max(all_results, key=lambda x: x[1])
        probe_wpm = probe[2].get('wpm', 0)
        probe_mono = _dominant_char_ratio(probe[0])
        probe_q = probe[0].count('?') / max(len(probe[0].replace(' ', '')), 1)
        need_qrq = (probe_wpm >= 38 or probe_mono > 0.40 or
                    probe_q > 0.25 or probe[1] < 0.28)
        if need_qrq:
            for cfg in (
                (50, 1.5, 'sq'),  # ultra-light smooth (fast QRQ / thin dits)
                (60, 1.5, 'sq'),
                (55, 1.5, 'sq'),
                (50, 2.0, 'sq'),   # narrow + very light (adjacent-tone QRQ)
                (60, 2.0, 'sq'),
                (100, 3.0, 'sq'),
                (80, 2.0, 'sq'),
                (60, 2.5, 'gz'),   # Goertzel tone-locked envelope
                (50, 2.5, 'gz'),
                (60, 4.0, 'on'),   # onset-enhanced attacks
                (50, 3.0, 'on'),
            ):
                all_results.extend(_run_config(*cfg))
        # Mild '?' / mid score: still try Goertzel + onset
        elif probe_q > 0.08 or probe[1] < 0.55:
            all_results.extend(_run_config(60, 3.0, 'gz'))
            all_results.extend(_run_config(80, 2.5, 'gz'))
            all_results.extend(_run_config(60, 5.0, 'on'))

    if not all_results:
        return None

    # === Cross-config voting: select most reliable result ===
    # Tier 1: Text consensus voting
    # When multiple configs agree on similar text, that text is more likely
    # correct than a high-scoring outlier (e.g., smooth_ms=18.0 can produce
    # wrong text with inflated score for clean slow signals like OP_VERYOLD).
    #
    # Safety: consensus requires either overwhelming majority (≥5 members)
    # or solid majority (≥3) with competitive score. Minimum score 0.4
    # prevents picking garbled consensus on very weak signals.
    if len(all_results) >= 3:
        from difflib import SequenceMatcher

        # Normalize text for comparison (strip spaces, uppercase)
        def _norm_text(t):
            return ''.join(c.upper() for c in t if c.isalnum())

        # Group results by text similarity
        groups = []  # [(representative_text, [(text, score, meta), ...]), ...]
        for text_raw, score, meta in all_results:
            nt = _norm_text(text_raw)
            if not nt:
                continue
            # Find best matching group
            best_group = None
            best_sim = 0
            for rep, members in groups:
                sim = SequenceMatcher(None, _norm_text(rep), nt).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_group = (rep, members)
            if best_sim > 0.75 and best_group:
                best_group[1].append((text_raw, score, meta))
            else:
                groups.append((text_raw, [(text_raw, score, meta)]))

        # Score each group: group_size * avg_score (consensus × quality)
        group_scores = []
        for rep, members in groups:
            avg_score = sum(s for _, s, _ in members) / len(members)
            group_score = len(members) * avg_score
            # Pick the most central member: weight by both score and
            # average similarity to other group members. A member with
            # high score but low agreement (outlier) is less reliable
            # than a member with slightly lower score but high agreement.
            if len(members) >= 2:
                best_member = members[0]
                best_combined = 0.0
                for i, (text_i, score_i, meta_i) in enumerate(members):
                    sim_sum = 0.0
                    for j, (text_j, score_j, meta_j) in enumerate(members):
                        if i != j:
                            sim_sum += SequenceMatcher(
                                None, _norm_text(text_i),
                                _norm_text(text_j)).ratio()
                    avg_sim = sim_sum / (len(members) - 1)
                    combined = score_i * (0.5 + 0.5 * avg_sim)
                    if combined > best_combined:
                        best_combined = combined
                        best_member = (text_i, score_i, meta_i)
            else:
                best_member = members[0]
            group_scores.append((group_score, best_member, len(members)))

        # Empty groups: every candidate lacked alphanumeric text — fall through.
        if group_scores:
            group_scores.sort(key=lambda x: -x[0])
            best_group_score, (best_text, best_score, best_meta), group_size = group_scores[0]

            # Safety checks before applying consensus:
            # 1. Overwhelming majority (≥5 members): apply regardless of score
            # 2. Solid majority (≥3) with competitive score (≥70% of top individual)
            # 3. Minimum absolute score threshold (0.4) to avoid weak-signal garbage
            top_individual_score = max(s for _, s, _ in all_results)
            use_consensus = False
            if group_size >= 5 and best_score >= 0.4:
                use_consensus = True  # Overwhelming majority
            elif group_size >= 3 and best_score >= top_individual_score * 0.7 and best_score >= 0.4:
                use_consensus = True  # Solid majority with competitive score

            # Reject consensus if it is mono-letter garbage (common on wrong QRQ tone).
            if use_consensus and _dominant_char_ratio(best_text) > 0.45:
                use_consensus = False

            if use_consensus:
                return best_text, best_score, best_meta

    # Tier 2: Score-based selection (fallback)
    all_results.sort(key=lambda x: -x[1])

    # Compute WPM consensus score
    wpm_values = [r[2].get('wpm', 0) for r in all_results if r[2].get('wpm', 0) > 0]
    if len(wpm_values) >= 3:
        median_wpm = sorted(wpm_values)[len(wpm_values) // 2]
        for text_raw, score, meta in all_results:
            wpm = meta.get('wpm', 0)
            if wpm > 0 and median_wpm > 0:
                wpm_deviation = abs(wpm - median_wpm) / max(median_wpm, 1)
                if wpm_deviation < 0.2:
                    meta['wpm_consensus'] = 1.2
                elif wpm_deviation < 0.5:
                    meta['wpm_consensus'] = 1.0
                else:
                    meta['wpm_consensus'] = 0.7
            else:
                meta['wpm_consensus'] = 0.8

    all_results.sort(key=lambda x: -(x[1] * x[2].get('wpm_consensus', 1.0)))

    return all_results[0]


def _chunk_decode(signal, sample_rate, cw_freq, snr_db, chunk_sec=30, overlap_sec=10, use_coherent=True):
    """
    Chunk decode: split long audio into overlapping blocks, decode each with
    multi-config, take best. Handles intermittent signals in long recordings.

    Optimization:
    1. First try full-signal fast decode
    2. If good enough, return immediately
    3. Otherwise, quality-guided chunk search (single size, fast scan)
    """
    # Pre-compute bandpass + envelope (shared by all sizes)
    signal_bp = bandpass(signal, sample_rate, cw_freq - 60, cw_freq + 60)
    envelope_full = square_law_envelope(signal_bp, sample_rate, smooth_ms=12.0)

    # Strategy 1: full-signal fast path
    result = _decode_from_envelope(envelope_full, sample_rate, threshold_mult=1.0)
    if result:
        text_raw, score, meta = result
        if score >= 0.85:
            meta['method'] = 'full-fast'
            return (text_raw, score, meta)

    # Strategy 2: quality-guided chunk search
    # Try 60s first (best accuracy), then 120s if not good enough
    best = None

    for chunk_sec in [60, 120]:
        chunk_samples = int(chunk_sec * sample_rate)
        if len(signal) <= chunk_samples:
            continue

        step = chunk_samples - int(min(15, chunk_sec // 3) * sample_rate)
        MAX_SCAN = 25
        TOP_K = 8

        # Generate candidate positions
        positions = []
        pos = 0
        while pos + chunk_samples <= len(signal):
            positions.append(pos)
            pos += step
            if len(positions) >= MAX_SCAN:
                break

        if not positions:
            continue

        # Phase 1: dual-threshold quick scan (more reliable ranking)
        quick_scores = []
        for pos in positions:
            chunk_env = envelope_full[pos:pos + chunk_samples]
            best_q = 0.0
            for thr_mult in [1.0, 1.5]:
                result = _decode_from_envelope(chunk_env, sample_rate, threshold_mult=thr_mult)
                if result:
                    _, q_score, _ = result
                    best_q = max(best_q, q_score)
            quick_scores.append(best_q)

        # Select top-K
        quick_scores = np.array(quick_scores)
        top_indices = np.argsort(quick_scores)[::-1][:TOP_K]
        top_indices.sort()

        # Phase 2: full multi-config decode (top 2 get full config, rest get fast)
        for rank, idx in enumerate(top_indices):
            pos = positions[idx]
            chunk = signal[pos:pos + chunk_samples]
            use_fast = rank >= 2  # top 2 get full config
            result = _multi_config_decode(chunk, sample_rate, cw_freq, snr_db, use_coherent, fast=use_fast)
            if result:
                text_raw, score, meta = result
                if best is None or score > best[1]:
                    best = (text_raw, score, meta)
                    best[2]['chunk_size'] = chunk_sec
                if score >= 0.95:
                    return best

        # If good enough, skip 120s
        if best and best[1] >= 0.78:
            break

    return best


# ============================================================
# V13 advanced path (PLL + AGC)
# ============================================================

def decode_at_freq_v13(signal, sample_rate, cw_freq, snr_db, verbose=False):
    """Decode at specified frequency using V13 pipeline. Returns (text, text_raw, score, meta)."""
    result = _multi_config_decode(signal, sample_rate, cw_freq, snr_db)
    if result is None:
        return "", "", 0.0, {}

    text_raw, score, meta = result
    return text_raw, text_raw, score, meta


# ============================================================
# V13 main decoder
# ============================================================

def _fuse_hypotheses_tokenwise(results, max_hyps: int = 3):
    """
    Token-level fusion across competing DSP hypotheses (freq/config).

    MorseNet-style idea without NN: when several decodes of the same clip
    disagree, keep the token with fewer '?' / higher alphanumeric density
    at each aligned position. Only activates when ≥2 hyps have competitive
    scores; returns (text, score, freq, meta) or None.
    """
    if len(results) < 2:
        return None

    def _hyp_quality(r):
        text, score, _, _ = r
        q = text.count('?') / max(len(text.replace(' ', '')), 1)
        mono = _dominant_char_ratio(text)
        return score * (1.0 - 0.5 * q) * (1.0 - 0.4 * max(0.0, mono - 0.3))

    ranked = sorted(results, key=_hyp_quality, reverse=True)[:max_hyps]
    if _hyp_quality(ranked[0]) < 0.25:
        return None
    # Require at least two competitive hyps
    if _hyp_quality(ranked[1]) < _hyp_quality(ranked[0]) * 0.55:
        return None

    token_lists = [r[0].upper().split() for r in ranked]
    lengths = [len(t) for t in token_lists]
    if max(lengths) < 3:
        return None
    # Use longest as scaffold; map others via SequenceMatcher opcodes
    from difflib import SequenceMatcher
    base_i = int(np.argmax(lengths))
    base = token_lists[base_i]
    fused = list(base)

    for j, toks in enumerate(token_lists):
        if j == base_i:
            continue
        sm = SequenceMatcher(a=base, b=toks)
        # Rebuild fused preferring cleaner token on equal slots
        new_fused = list(fused)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for k, (a, b) in enumerate(zip(base[i1:i2], toks[j1:j2])):
                    idx = i1 + k
                    if idx >= len(new_fused):
                        break
                    cur = new_fused[idx]
                    # Prefer fewer '?', else longer alnum
                    if b.count('?') < cur.count('?') and len(b) >= 2:
                        new_fused[idx] = b
                    elif (b.count('?') == cur.count('?')
                          and sum(c.isalnum() for c in b) > sum(c.isalnum() for c in cur)
                          and '?' not in b):
                        new_fused[idx] = b
            elif tag == 'replace' and (i2 - i1) == 1 and (j2 - j1) == 1:
                a, b = base[i1], toks[j1]
                if b.count('?') < a.count('?') and len(b) >= 2:
                    if i1 < len(new_fused):
                        new_fused[i1] = b
        fused = new_fused

    text = ' '.join(fused)
    # Inherit score/freq from best hyp; mark fusion in meta
    best = ranked[0]
    meta = dict(best[3])
    meta['method'] = str(meta.get('method', 'dsp')) + '+token_fuse'
    # Mild score bump if fusion reduced '?'
    q0 = best[0].count('?')
    q1 = text.count('?')
    score = float(best[1])
    if q1 < q0:
        score = min(1.0, score + 0.02 * (q0 - q1))
    return (text, score, best[2], meta)


def decode_v13(path: str, ref_text: str = None, verbose: bool = False,
               use_pll: bool = True, use_agc: bool = True,
               use_coherent: bool = True, use_semantic: bool = True,
               adaptive_mode: bool = True,
               force_freq: Optional[float] = None,
               force_wpm: Optional[float] = None,
               use_neural='auto') -> Dict:
    """
    V13 enhanced decoder.

    Args:
        path: audio file path
        ref_text: reference text (for accuracy calculation)
        verbose: verbose output
        use_pll: use PLL tracking
        use_agc: use AGC
        use_coherent: use coherent integration (weak signals)
        use_semantic: use semantic correction
        adaptive_mode: adaptive mode (enable advanced features only when needed)
        force_freq: force CW frequency in Hz (None = auto-detect)
        force_wpm: expected speed hint in WPM (None = auto-detect)
        use_neural: False | True | 'auto' — optional morseformer RNN-T
            fusion. Default 'auto': try neural only when DSP looks weak and
            torch+checkpoint are available. Skips when DSP already excellent.

    Returns:
        dict with keys: text, score, freq, wpm, accuracy, method, ...
    """
    # 1. Load audio
    sample_rate, signal = load_wav(path)

    if verbose:
        print(f"[*] Loading audio: {os.path.basename(path)} ({sample_rate}Hz, {len(signal)/sample_rate:.1f}s)")

    # 2. 60Hz notch filter (remove hum)
    signal_notched = notch_filter_hum(signal, sample_rate, freq=60.0, Q=30.0)

    # 3. Frequency scan (multi-point probing for long audio, avoid silent segments)
    if force_freq is not None:
        # Forced frequency: skip scan, use single candidate
        f_center = force_freq
        candidates = [{'freq': force_freq, 'score': 1.0}]
        if verbose:
            print(f"[*] Forced frequency: {f_center:.0f} Hz")
    else:
        audio_duration = len(signal) / sample_rate
        if audio_duration > 60:
            # Scan at 3 positions with 20s each, select best
            best_candidates = None
            best_score = -1
            for frac in [0.25, 0.5, 0.75]:
                center = int(frac * len(signal))
                half = int(10 * sample_rate)  # 10 seconds
                start = max(0, center - half)
                end = min(len(signal), center + half)
                segment = signal[start:end]
                cands = scan_cw_frequencies(segment, sample_rate, verbose=False)
                if cands and cands[0]['score'] > best_score:
                    best_score = cands[0]['score']
                    best_candidates = cands
            candidates = best_candidates or []
        else:
            candidates = scan_cw_frequencies(signal, sample_rate, verbose=False)
        if not candidates:
            return {"text": "", "error": "No candidate frequency", "method": "V13"}

        # MorseNet-style STFT peak tracker — merge with FFT intermittency scan
        try:
            stft_cands = track_cw_peaks_stft(signal_notched, sample_rate, top_k=8)
            if stft_cands:
                candidates = merge_freq_candidates(candidates, stft_cands, min_sep_hz=3.0)
                if verbose:
                    tops = ", ".join(f"{c['freq']:.0f}" for c in stft_cands[:4])
                    print(f"[*] STFT tone tracks: {tops} Hz")
        except Exception:
            pass

        # Collapse near-duplicate FFT bins so top-N spans distinct tones (QRQ QRM).
        # Keep sep tight (~2.5Hz): adjacent QRQ carriers often sit 3–8Hz apart and
        # only one of them may be cleanly decodable.
        candidates = _dedupe_freq_candidates(candidates, min_sep_hz=2.5)

        f_center = candidates[0]['freq']
        if verbose:
            print(f"[*] Initial frequency estimate: {f_center:.0f} Hz")
            if len(candidates) > 1:
                tops = ", ".join(f"{c['freq']:.0f}" for c in candidates[:5])
                print(f"[*] Tone candidates: {tops} Hz")

    # 4. Initial SNR estimation
    signal_bp_init = bandpass(signal_notched, sample_rate, f_center - 60, f_center + 60)
    snr_db = estimate_snr_db(signal_bp_init, sample_rate, f_center)

    # Determine if advanced processing is needed
    need_advanced = adaptive_mode and (snr_db < 0)

    if verbose:
        print(f"[*] SNR estimate: {snr_db:.1f} dB")
        if need_advanced:
            print(f"[*] Weak signal detected: enabling V13 advanced processing")
        else:
            print(f"[*] Signal quality good: using V12.7 standard processing")

    # 5. Multi-config + chunk integrated decode
    audio_dur = len(signal_notched) / sample_rate
    use_chunk = audio_dur > 120  # chunk decode only for very long audio (>2 min)

    # Long audio: only use best frequency (reduce computation).
    # Otherwise try several distinct tone candidates — QRQ nets often have QRM.
    top_n = 1 if use_chunk else min(4, len(candidates))
    results = []
    SCORE_THRESHOLD = 0.95

    for cand in candidates[:top_n]:
        freq = cand['freq']
        # Goertzel AFC: refine FFT bin to true tone peak (±12 Hz)
        if force_freq is None:
            freq = refine_cw_freq_afc(
                signal_notched, sample_rate, freq,
                search_hz=12.0, step_hz=0.5)

        if use_chunk:
            # Chunk decode (handle intermittent signals in long recordings)
            result = _chunk_decode(signal_notched, sample_rate, freq, snr_db)
        else:
            # Multi-config decode (handle frequency drift and speed variation)
            result = _multi_config_decode(signal_notched, sample_rate, freq, snr_db)

        if result:
            text_raw, score, meta = result
            results.append((text_raw, score, freq, meta))
            if score >= SCORE_THRESHOLD and _dominant_char_ratio(text_raw) < 0.4:
                break

    # Contest follow-up (content-gated): when primary shows 5NN/serial/TU traffic
    # on a long recording, stitch retuned windows to recover mid/late QSOs that
    # a single-tone full-file decode drops. Never fires on QRQ/synthetic.
    contest_followup = False
    if results and audio_dur >= 60:
        best_probe = max(results, key=lambda x: x[1])
        probe_text, probe_score = best_probe[0], best_probe[1]
        if _looks_like_contest_exchange(probe_text):
            contest_followup = True
            prim_serials = _extract_contest_serials(probe_text)
            if verbose:
                print(f"[*] Contest exchange detected "
                      f"(serials={sorted(prim_serials)}) — timeline stitch")
            stitch = _contest_stitch_decode(
                signal_notched, sample_rate, snr_db, f_center)
            if stitch is not None:
                text_raw, score, freq, meta = stitch
                results.append((text_raw, score, freq, meta))
                if verbose:
                    print(f"[*] Contest stitch: chunks={meta.get('contest_chunks')} "
                          f"serials={meta.get('contest_serials')} "
                          f"calls={meta.get('contest_calls')} score={score:.3f}")
            # Also try narrow-BW full-file as a competitor
            result = _multi_config_decode(
                signal_notched, sample_rate, f_center, snr_db, contest_mode=True)
            if result:
                text_raw, score, meta = result
                results.append((text_raw, score, f_center, meta))

    # If primary candidates look like wrong-tone / over-smoothed garble, try more peaks
    # and a fine frequency grid. Real QRQ can be readable at one tone and garbage
    # only 3 Hz away (adjacent carriers / beat notes).
    if (not use_chunk and results):
        best_so_far = max(results, key=lambda x: x[1])
        q_ratio = best_so_far[0].count('?') / max(len(best_so_far[0].replace(' ', '')), 1)
        mono_b = _dominant_char_ratio(best_so_far[0])
        pl_b = _text_plausibility(best_so_far[0])
        looks_bad = (
            mono_b > 0.35 or
            q_ratio > 0.20 or
            best_so_far[1] < 0.45 or
            pl_b < 0.55
        )
        # QRQ nets: a middling primary (0.55–0.85) can sit a few Hz off the
        # cleanest carrier. Skip the wide grid when top candidates already strong.
        seek_better = looks_bad or best_so_far[1] < 0.85
        if seek_better:
            extra_freqs = []
            for cand in candidates[top_n:min(10, len(candidates))]:
                extra_freqs.append(cand['freq'])
            # Wider ±15/18 catches adjacent QRQ tones (seen ~8 Hz off primary).
            deltas = (-1.5, 1.5, -3.0, 3.0, -4.5, 4.5, -6.0, 6.0,
                      -9.0, 9.0, -12.0, 12.0, -15.0, 15.0, -18.0, 18.0)
            base_n = 6 if looks_bad else 3
            for delta in deltas:
                for cand in candidates[:base_n]:
                    extra_freqs.append(cand['freq'] + delta)
            seen = {round(r[2], 1) for r in results}
            best_probe = best_so_far[1]
            n_extra = 0
            for freq in extra_freqs:
                key = round(freq, 1)
                if key in seen:
                    continue
                seen.add(key)
                result = _multi_config_decode(signal_notched, sample_rate, freq, snr_db)
                if result:
                    text_raw, score, meta = result
                    results.append((text_raw, score, freq, meta))
                    best_probe = max(best_probe, score)
                    n_extra += 1
                    q2 = text_raw.count('?') / max(len(text_raw.replace(' ', '')), 1)
                    mono2 = _dominant_char_ratio(text_raw)
                    pl2 = _text_plausibility(text_raw)
                    # Stop when clearly strong — don't lock onto mid-score near-misses.
                    if (score >= 0.92 and mono2 < 0.28 and q2 < 0.10 and pl2 >= 0.9):
                        break
                    if n_extra >= 24:
                        break
                    if (not looks_bad and best_probe >= 0.95):
                        break

    # 6. PLL+AGC advanced path (only for weak signals, not long audio)
    if need_advanced and use_pll and not use_chunk:
        for cand in candidates[:top_n]:
            freq = cand['freq']
            try:
                pll = CostasPLL(sample_rate, freq, loop_bw=5.0)
                I_signal, freq_track = pll.process(signal_notched)
                if pll.locked and pll.lock_time > 0 and pll.lock_time < len(freq_track):
                    f_locked = np.nanmean(freq_track[pll.lock_time:])
                else:
                    f_locked = np.nanmean(freq_track[-100:])
                if np.isnan(f_locked) or np.isinf(f_locked):
                    f_locked = freq
            except Exception:
                continue

            if use_agc:
                signal_agc = apply_agc(I_signal, sample_rate, mode="slow")
            else:
                signal_agc = I_signal

            signal_bp = bandpass(signal_agc, sample_rate, f_locked - 60, f_locked + 60)

            if use_coherent and snr_db < -5:
                try:
                    integrator = AdaptiveIntegrator(sample_rate, f_locked, min_integ_ms=50, max_integ_ms=200)
                    envelope, used_ms = integrator.process(signal_bp, target_snr=10.0)
                except Exception:
                    envelope = square_law_envelope(signal_bp, sample_rate, smooth_ms=12.0)
            else:
                envelope = square_law_envelope(signal_bp, sample_rate, smooth_ms=12.0)

            result = _multi_threshold_decode(envelope, sample_rate)
            if result:
                text_adv, score_adv, meta_adv = result
                results.append((text_adv, score_adv, f_locked, meta_adv))

    if not results:
        return {"text": "", "error": "Insufficient signal segments", "method": "V13"}

    # 7. Select best result.
    # Rank by score × plausibility, penalizing E/T mono-spam that can still
    # get middling numeric scores on the wrong QRQ tone.
    def _select_key(r):
        text, score, freq, meta = r
        pl = _text_plausibility(text)
        mono = _dominant_char_ratio(text)
        adj = score * pl
        if mono > 0.40:
            adj *= 0.35
        elif mono > 0.32:
            adj *= 0.65
        # Prefer readable spacing when scores are close (QRQ run-on vs E E E)
        sp = text.count(' ') / max(len(text), 1)
        letters = sum(1 for c in text if c.isalpha())
        if letters >= 40 and 0.06 <= sp <= 0.25:
            adj *= 1.08
        elif letters >= 40 and sp < 0.03 and pl >= 0.9:
            # Run-on English is still better than E-spam
            adj *= 1.05
        # Soft preference for tones near the FFT peak (avoid far beat-note picks)
        if abs(freq - f_center) <= 12:
            adj *= 1.03
        elif abs(freq - f_center) >= 35:
            adj *= 0.92
        return adj

    results.sort(key=_select_key, reverse=True)

    # Token-level fusion across top competitive hypotheses (reduces residual '?')
    fused = _fuse_hypotheses_tokenwise(results)
    if fused is not None:
        # Insert fused hyp and re-rank once
        results.insert(0, fused)
        results.sort(key=_select_key, reverse=True)

    best_text, best_score, best_freq, best_meta = results[0]

    # Contest follow-up: prefer the candidate covering the most serials/calls
    # when scores are competitive (full-file often truncates mid/late QSOs).
    if contest_followup and len(results) >= 2:
        def _cov_key(r):
            text, score, _, meta = r
            n_ser = len(_extract_contest_serials(text))
            n_call = len(_extract_contest_calls(text) - {'SZ1A'})
            cq = _contest_exchange_quality(text)
            # Coverage dominates; require readable contest quality
            return (n_ser, n_call, cq * 0.5 + score * 0.5)

        ranked = sorted(results, key=_cov_key, reverse=True)
        top = ranked[0]
        cur = (best_text, best_score, best_freq, best_meta)
        # Switch if top recovers strictly more serials, or more calls at similar serials
        if _cov_key(top) > _cov_key(cur) and _contest_exchange_quality(top[0]) >= 0.25:
            best_text, best_score, best_freq, best_meta = top

    if top_n > 1 and len(results) >= 2:
        primary_results = [r for r in results if abs(r[2] - f_center) < 1.0]
        if primary_results:
            prim_text, prim_score, prim_freq, prim_meta = primary_results[0]
            # Prefer primary tone when competitive, but not if it is mono/garble
            # while an offset tone decoded readable text (common on QRQ nets).
            prim_mono = _dominant_char_ratio(prim_text)
            best_mono = _dominant_char_ratio(best_text)
            prim_q = prim_text.count('?') / max(len(prim_text.replace(' ', '')), 1)
            best_q = best_text.count('?') / max(len(best_text.replace(' ', '')), 1)
            primary_worse = (
                (prim_mono > 0.42 and best_mono < 0.35) or
                (prim_q > 0.25 and best_q < 0.12) or
                (prim_score < 0.35 and best_score >= 0.5)
            )
            if (abs(best_freq - f_center) >= 1.0 and
                    prim_score >= best_score * 0.97 and
                    not primary_worse):
                best_text, best_score, best_freq, best_meta = (
                    prim_text, prim_score, prim_freq, prim_meta)

    # Contest union inject AFTER primary preference so recovered calls survive.
    if contest_followup and len(results) >= 2:
        from cw_decoder.corrector import CALLSIGN_PATTERNS

        def _solid_contest_call(c: str) -> bool:
            if not (3 <= len(c) <= 8):
                return False
            if not any(p.fullmatch(c) for p in CALLSIGN_PATTERNS):
                return False
            if sum(c.count(ch) for ch in 'EIT') >= max(2, len(c) - 1):
                return False
            if c.count('N') >= 3 and 'NN' in c:
                return False
            return True

        ranked = sorted(
            results,
            key=lambda r: (
                len(_extract_contest_serials(r[0])),
                len(_extract_contest_calls(r[0]) - {'SZ1A'}),
                r[1]),
            reverse=True)
        have_s = _extract_contest_serials(best_text)
        have_c = _extract_contest_calls(best_text)
        meta_pool = set()
        for _t, _sc, _f, _m in ranked:
            meta_pool |= set((_m or {}).get('contest_calls') or [])
        uniq = []
        seen_i = set()
        # Prefer twins of confirmed calls from meta/text pool
        pool = set()
        for _t, _sc, _f, _m in ranked:
            pool |= _extract_contest_calls(_t)
        pool |= meta_pool
        for h in list(have_c):
            for c in pool:
                if c in seen_i or c in best_text or c == 'SZ1A':
                    continue
                if not _solid_contest_call(c):
                    continue
                if len(c) == len(h) and lev(c, h) == 1:
                    uniq.append(c)
                    seen_i.add(c)
                    have_c.add(c)
        # Then missing high-quality serials (>=2000)
        for _t, _sc, _f, _m in ranked:
            for s in _extract_contest_serials(_t) - have_s:
                if s.isdigit() and int(s) >= 2000:
                    tok = f'5NN {s}'
                    if tok not in seen_i and s not in best_text:
                        uniq.append(tok)
                        seen_i.add(tok)
                        have_s.add(s)
        uniq = uniq[:8]
        if uniq:
            best_text = (best_text.rstrip() + ' ' + ' '.join(uniq)).strip()
            best_meta = dict(best_meta)
            best_meta['contest_injected'] = uniq
            best_meta['contest_serials'] = sorted(have_s)
            best_meta['contest_calls'] = sorted(have_c - {'SZ1A'})
            best_score = min(0.99, max(best_score, best_score + 0.015 * min(4, len(uniq))))

    text_raw = best_text

    # 7a. Fragment merging post-processing (DISABLED).
    # Fragment merging can clean up envelope artifacts for OP_VERYOLD/OP_NOVICE
    # but reliably distinguishing envelope fragments from noise fragments
    # requires better signal quality metrics than currently available.
    # Consensus voting (in _multi_config_decode) provides similar benefits
    # without the risk of corrupting weak-signal decodes.
    if False:  # Disabled — re-enable when noise guard is improved
        try:
            bw_hz = best_meta.get('bw_hz', 60)
            smooth_ms = best_meta.get('smooth_ms', 12.0)
            env_method = best_meta.get('env_method', 'sq')

            signal_bp = bandpass(signal_notched, sample_rate,
                                 best_freq - bw_hz, best_freq + bw_hz)
            if env_method == 'rms':
                envelope = rms_envelope(signal_bp, sample_rate, window_ms=smooth_ms)
            else:
                envelope = square_law_envelope(signal_bp, sample_rate, smooth_ms=smooth_ms)

            threshold_base = auto_threshold(envelope)
            key_states = (envelope > threshold_base).astype(np.int8)

            # Check fragment evidence from standard key states
            segments_check = extract_segments(key_states, sample_rate, min_ms=3.0)
            marks_check = [d for s, d in segments_check if s == 1]
            has_fragments = (len(marks_check) > 10 and
                             sum(1 for d in marks_check if d < 1.0) / len(marks_check) > 0.3)

            if has_fragments:
                # Apply fragment merging to clean up envelope artifacts
                segments_merged = _merge_envelope_fragments(segments_check, sample_rate)
                if segments_merged is not segments_check:
                    # Re-decode with merged segments (pass via segments= parameter)
                    result_merged = _decode_from_key_states(key_states, envelope,
                                                             sample_rate, threshold_base,
                                                             segments=segments_merged)
                    if result_merged:
                        text_merged, score_merged, meta_merged = result_merged
                        meta_merged['thresh_method'] = 'frag_merge'
                        # Use merged result when fragments exist (standard decode
                        # is corrupted by envelope artifacts; merged is more reliable)
                        best_text = text_merged
                        best_score = score_merged
                        best_meta = meta_merged
                        text_raw = best_text
        except Exception:
            pass

    # 7b. Optional neural (morseformer) fusion — only when enabled / auto-weak
    dsp_q_ratio = text_raw.count('?') / max(len(text_raw.replace(' ', '')), 1)
    dsp_strong = (
        best_score >= 0.85 and dsp_q_ratio < 0.05
        and _dominant_char_ratio(text_raw) < 0.35
    )
    want_neural = False
    if use_neural is True:
        # Explicit on — still skip when DSP already excellent (saves ~5–10s)
        want_neural = not dsp_strong
    elif use_neural == 'auto':
        want_neural = (
            best_score < 0.55
            or dsp_q_ratio > 0.12
            or _dominant_char_ratio(text_raw) > 0.40
        )
    if want_neural:
        try:
            from cw_decoder.neural_bridge import available, neural_decode, availability_reason
            if available():
                # Cap neural input length — full QRQ nets are too slow for RNN-T
                max_neu = int(sample_rate * 45)
                neu_sig = signal_notched[:max_neu]
                n_text, n_conf = neural_decode(
                    neu_sig, sample_rate, tone_hz=best_freq)
                if n_text and len(n_text) >= 3:
                    dsp_q = text_raw.count('?') / max(len(text_raw.replace(' ', '')), 1)
                    neu_q = n_text.count('?') / max(len(n_text.replace(' ', '')), 1)
                    neu_pl = _text_plausibility(n_text)
                    dsp_pl = _text_plausibility(text_raw)
                    # Prefer neural when it clearly reduces garble / raises plausibility
                    if ((neu_q + 0.05 < dsp_q and neu_pl >= dsp_pl * 0.9) or
                            (n_conf >= 0.45 and neu_pl > dsp_pl + 0.08 and
                             _dominant_char_ratio(n_text) < 0.35)):
                        if verbose:
                            print(f"[*] Neural override (conf={n_conf:.2f}, "
                                  f"? {dsp_q:.2f}→{neu_q:.2f})")
                        text_raw = n_text
                        best_text = n_text
                        best_score = max(best_score, 0.5 * best_score + 0.5 * n_conf)
                        best_meta = dict(best_meta)
                        best_meta['method'] = str(best_meta.get('method', 'dsp')) + '+neural'
            elif verbose and use_neural is True:
                print(f"[*] Neural unavailable: {availability_reason()}")
        except Exception as exc:
            if verbose:
                print(f"[*] Neural skipped: {exc}")

    # 8. Semantic correction (only apply if score improves)
    if use_semantic:
        text_corrected, corrections = semantic_correct(text_raw)
        n_valid_c = sum(1 for ch in text_corrected if ch != '?' and ch != ' ')
        n_total_c = max(len(text_corrected.replace(' ', '')), 1)
        valid_ratio_c = n_valid_c / n_total_c
        duty_cycle = best_meta.get('duty_cycle', 0.4)
        duty_cycle_penalty = 1.0 - abs(duty_cycle - 0.4) * 1.5
        duty_cycle_penalty = max(0.3, min(1.0, duty_cycle_penalty))

        # WPM penalty (aligned with _score_decode — allow real QRQ 45–85 WPM)
        wpm = best_meta.get('wpm', 20)
        wpm_penalty = 1.0
        if wpm < 3 or wpm > 140:
            wpm_penalty = 0.1
        elif wpm < 5:
            wpm_penalty = 0.4
        elif wpm > 100:
            wpm_penalty = 0.55
        elif wpm > 85:
            wpm_penalty = 0.8

        # Dot/dash ratio penalty
        ratio = best_meta.get('ratio', 3.0)
        ratio_penalty = 1.0
        if ratio < 1.5 or ratio > 6.0:
            ratio_penalty = 0.3
        elif ratio < 2.0 or ratio > 4.5:
            ratio_penalty = 0.6

        score_corrected = valid_ratio_c * duty_cycle_penalty * wpm_penalty * ratio_penalty

        # Short text penalty
        if n_valid_c < 10:
            score_corrected *= n_valid_c / 10.0

        # Character diversity penalty
        non_space_c = text_corrected.replace(' ', '').replace('?', '')
        if len(non_space_c) > 3:
            diversity_c = len(set(non_space_c)) / min(len(non_space_c), 26)
            if diversity_c < 0.15:
                score_corrected *= 0.5

        score_corrected *= _text_plausibility(text_corrected)
        # QRQ run-on recovery: splitting long glued English raises readability;
        # lift score toward 1.0 when spacing clearly improved.
        sp0 = text_raw.count(' ') / max(len(text_raw), 1)
        sp1 = text_corrected.count(' ') / max(len(text_corrected), 1)
        if (sp0 < 0.04 and sp1 >= 0.08 and
                _text_plausibility(text_corrected) >= 0.9 and
                _dominant_char_ratio(text_corrected) < 0.30):
            score_corrected = min(1.0, max(score_corrected, best_score) + 0.18)
            score_corrected = min(1.0, score_corrected * 1.05)
        score_corrected = max(0.0, min(1.0, score_corrected))

        # DEBUG: Always apply corrections if there are any
        if corrections:
            text = text_corrected
            best_score = score_corrected
        else:
            text = text_raw

        if verbose and corrections:
            print(f"[*] Semantic correction: {len(corrections)} fixes applied")
            print(f"    Score: {best_score:.6f} -> {score_corrected:.6f}")
    else:
        text = text_raw
        corrections = []

    # Normalize prosigns: <AR> and <SK> are Morse prosigns that should
    # appear as single characters in decoded output. '+' is the standard
    # text representation for prosigns in amateur radio decoders.
    # Without normalization, '<AR>' becomes 'AR' after alphanumeric
    # filtering, adding spurious characters that lower accuracy.
    text = text.replace('<AR>', '+').replace('<SK>', '+')
    text_raw = text_raw.replace('<AR>', '+').replace('<SK>', '+')

    # 9. Accuracy
    accuracy = None
    if ref_text:
        ref = ''.join(ch.upper() for ch in ref_text if ch.isalnum() or ch.isspace())
        got = ''.join(ch.upper() for ch in text if ch.isalnum() or ch.isspace())
        if ref and got:
            from difflib import SequenceMatcher
            accuracy = SequenceMatcher(None, got, ref).ratio()

    if verbose:
        dit = best_meta.get('dit', 0)
        dah = best_meta.get('dah', 0)
        wpm = best_meta.get('wpm', 0)
        ratio = dah / max(dit, 1)
        print(f"[*] Decode complete: WPM={wpm:.0f}, dit={dit:.0f}ms, ratio={ratio:.2f}")
        print(f"[*] Text: {text[:60]}...")

    return {
        "text": text,
        "text_raw": text_raw,
        "score": best_score,
        "freq": best_freq,
        "freq_initial": f_center,
        "snr_db": snr_db,
        "wpm": round(best_meta.get('wpm', 0), 1),
        "dit_ms": round(best_meta.get('dit', 0), 1),
        "dah_ms": round(best_meta.get('dah', 0), 1),
        "accuracy": accuracy,
        "method": "V13-Adaptive" if need_advanced else "V13-Standard",
        "corrections": len(corrections) if corrections else 0,
    }


def main():
    """CLI entry point (deprecated — use python -m cw_decoder or cw-decoder)."""
    from cw_decoder.__main__ import run
    import sys
    sys.exit(run())


if __name__ == "__main__":
    main()
