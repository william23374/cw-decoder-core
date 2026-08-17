#!/usr/bin/env python3
"""
threshold.py — Adaptive thresholding for CW envelope binarization.

Aligned with FLDIGI and CW Skimmer thresholding conventions:
- Otsu's method: optimal binary threshold via inter-class variance
- Auto-threshold: percentile-based adaptive method for varying SNR
"""

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    """Otsu's method for optimal binary threshold."""
    hist, edges = np.histogram(values, bins=bins, range=(0, values.max() + 1e-8))
    centers = (edges[:-1] + edges[1:]) / 2
    total = len(values)
    s_total = values.sum()
    s_bg = 0
    w_bg = 0
    best = 0
    thr = 0.3 * values.max()
    for i in range(len(hist)):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        s_bg += centers[i] * hist[i]
        mb = s_bg / w_bg
        mf = (s_total - s_bg) / w_fg
        v = w_bg * w_fg * (mb - mf) ** 2
        if v > best:
            best = v
            thr = centers[i]
    return float(thr)


def auto_threshold(env: np.ndarray) -> float:
    """Automatic threshold selection based on envelope statistics."""
    p10, p25, p50, p75, p90 = np.percentile(env, [10, 25, 50, 75, 90])
    if p10 < 0.02 and (p90 - p50) > 0.3:
        thr = (p25 + p75) / 2
        thr = max(thr, 0.15)
    elif p90 - p50 > 0.3 * env.max():
        thr = p50 + 0.3 * (p90 - p50)
    else:
        thr = otsu_threshold(env)
    return float(thr)
