#!/usr/bin/env python3
"""
segmentation.py — Segment extraction, morphological cleaning, fragment merging.

Aligned with international CW decoder conventions (FLDIGI, CW Skimmer).
Provides vectorized RLE-based segment extraction and morphological operations
for cleaning binary key-state signals.
"""

import numpy as np
from typing import List, Tuple


def extract_segments(states: np.ndarray, fs: int,
                     min_ms: float = 3.0) -> List[Tuple[int, float]]:
    """Vectorized segment extraction using np.diff (~100x faster than Python loop)."""
    min_samples = max(int(min_ms / 1000 * fs), 1)
    states = np.asarray(states, dtype=np.int8)
    n = len(states)
    if n == 0:
        return []

    # Find transition points (vectorized RLE)
    diffs = np.diff(states)
    change_idx = np.flatnonzero(diffs) + 1
    starts = np.empty(len(change_idx) + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = change_idx
    run_lengths = np.diff(np.append(starts, n))
    run_states = states[starts]

    # Mark short segments
    short = run_lengths < min_samples
    ms_per_sample = 1000.0 / fs

    # Merge short segments into previous segment
    for i in range(len(run_lengths)):
        if short[i]:
            if i > 0:
                run_lengths[i - 1] += run_lengths[i]
                run_lengths[i] = 0

    keep = run_lengths > 0
    return [(int(run_states[i]), float(run_lengths[i] * ms_per_sample))
            for i in range(len(run_lengths)) if keep[i]]


def morphological_clean(states: np.ndarray, fs: int,
                        min_mark_ms: float = 0,
                        min_space_ms: float = 0) -> np.ndarray:
    """
    Morphological cleaning: opening (remove noise spikes) and closing (fill small gaps)
    on the binary key signal. Efficient segment-scan implementation.

    Args:
        states: binary signal (0=space/1=mark)
        fs: sampling rate (Hz)
        min_mark_ms: minimum mark (ON) segment width (shorter ones removed)
        min_space_ms: minimum space (OFF) segment width (shorter ones filled)

    Returns:
        Cleaned binary signal
    """
    if min_mark_ms <= 0 and min_space_ms <= 0:
        return states.copy()

    cleaned = states.copy()

    # Find all segment boundaries
    diffs = np.diff(cleaned)
    changes = np.where(diffs != 0)[0] + 1
    if len(changes) == 0:
        return cleaned

    # Build segment table
    boundaries = np.concatenate([[0], changes, [len(cleaned)]])
    seg_starts = boundaries[:-1]
    seg_lengths = np.diff(boundaries)
    seg_states = cleaned[seg_starts]

    # Opening: remove short mark segments
    if min_mark_ms > 0:
        min_mark_samples = min_mark_ms / 1000 * fs
        for i in range(len(seg_lengths)):
            if seg_states[i] == 1 and seg_lengths[i] < min_mark_samples:
                s, e = seg_starts[i], seg_starts[i] + int(seg_lengths[i])
                cleaned[s:e] = 0
                seg_states[i] = 0

    # Closing: fill short space segments
    if min_space_ms > 0:
        min_space_samples = min_space_ms / 1000 * fs
        diffs2 = np.diff(cleaned)
        changes2 = np.where(diffs2 != 0)[0] + 1
        if len(changes2) > 0:
            bounds2 = np.concatenate([[0], changes2, [len(cleaned)]])
            for i in range(len(bounds2) - 1):
                s = bounds2[i]
                e = bounds2[i + 1]
                dur_ms = (e - s) / fs * 1000
                if cleaned[s] == 0 and dur_ms < min_space_ms:
                    cleaned[s:e] = 1

    return cleaned


def merge_fragments(segs: List[Tuple[int, float]],
                    gap_threshold_ms: float) -> List[Tuple[int, float]]:
    """
    Merge elements fragmented by QSB/smoothing.
    When the space between two mark segments < gap_threshold_ms, merge into one.
    """
    if len(segs) < 3:
        return segs
    result = []
    pending_space = None
    for state, dur in segs:
        if state == 1:
            if (pending_space is not None and pending_space[1] < gap_threshold_ms
                    and result and result[-1][0] == 1):
                prev_mark = result.pop()
                result.append((1, prev_mark[1] + pending_space[1] + dur))
                pending_space = None
            else:
                if pending_space is not None:
                    result.append(pending_space)
                result.append((1, dur))
                pending_space = None
        else:
            if pending_space is not None:
                result.append(pending_space)
            pending_space = (0, dur)
    if pending_space is not None:
        result.append(pending_space)
    return result


def merge_envelope_fragments(segs: List[Tuple[int, float]],
                             fs: int) -> List[Tuple[int, float]]:
    """
    Merge tiny mark-space fragments caused by slow envelope rise/fall times.

    Only activates when there is clear evidence of envelope fragmentation
    (many sub-1ms marks). This prevents accidentally merging real short
    elements (e.g. fast dots at 40WPM).
    """
    if len(segs) < 3:
        return segs

    # Pass 1: merge adjacent same-type segments
    merged = []
    for seg in segs:
        if merged and merged[-1][0] == seg[0]:
            merged[-1] = (seg[0], merged[-1][1] + seg[1])
        else:
            merged.append(seg)

    # Check if there are actual envelope fragments (sub-1ms marks)
    all_marks = [d for s, d in merged if s == 1]
    if len(all_marks) < 5:
        return segs
    tiny_marks = sum(1 for d in all_marks if d < 1.0)
    fragment_ratio = tiny_marks / len(all_marks)
    if fragment_ratio < 0.3:
        return segs

    # Pass 2: rough dit estimate from marks > 50ms
    mark_durs = [d for s, d in merged if s == 1 and d > 50]
    if len(mark_durs) < 3:
        return segs
    rough_dit = float(np.median(mark_durs))

    # Merge threshold: 15% of rough dit, capped at 15ms
    merge_thr = min(rough_dit * 0.15, 15.0)
    if merge_thr < 3.0:
        return segs

    # Convert to lists for modification
    for i in range(len(merged)):
        merged[i] = list(merged[i])

    # Pass 3: iteratively merge tiny triplets
    for _ in range(20):
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            # mark-space-mark: absorb space into mark
            if (i + 2 < len(merged) and
                    merged[i][0] == 1 and merged[i][1] < merge_thr and
                    merged[i + 1][0] == 0 and merged[i + 1][1] < merge_thr and
                    merged[i + 2][0] == 1 and merged[i + 2][1] < merge_thr):
                dur = merged[i][1] + merged[i + 1][1] + merged[i + 2][1]
                new_merged.append([1, dur])
                i += 3
                changed = True
            # space-mark-space: absorb mark into space
            elif (i + 2 < len(merged) and
                  merged[i][0] == 0 and merged[i + 1][0] == 1 and
                  merged[i + 1][1] < merge_thr and
                  merged[i + 2][0] == 0 and merged[i + 2][1] < merge_thr):
                dur = merged[i][1] + merged[i + 1][1] + merged[i + 2][1]
                new_merged.append([0, dur])
                i += 3
                changed = True
            else:
                new_merged.append(merged[i])
                i += 1
        merged = new_merged
        if not changed:
            break

    # Pass 4: final merge of adjacent same-type
    final = []
    for seg in merged:
        if final and final[-1][0] == seg[0]:
            final[-1][1] += seg[1]
        else:
            final.append(seg)

    return [(s, d) for s, d in final]
