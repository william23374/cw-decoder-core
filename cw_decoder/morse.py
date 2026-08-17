#!/usr/bin/env python3
"""
morse.py — Morse code table, Levenshtein distance, soft decode.

Shared by all decoder versions. Aligned with international conventions
(FLDIGI, CW Skimmer, morse-audio-decoder).
"""

# ============================================================
# Morse code table (ITU-R M.1677-1)
# ============================================================
MORSE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'",
    '-.-.--': '!', '-..-.': '/', '-.--.': '(', '-.--.-': ')',
    '.-...': '&', '---...': ':', '-.-.-.': ';', '-...-': '=',
    '.-.-.': '+', '-....-': '-', '..--.-': '_', '.-..-.': '"',
    '...-..-': '$', '.--.-.': '@', '.-.-': '<AR>', '...-.': '<SK>',
    '-.--.': '<KN>',
}

# Reverse lookup: character → Morse code
MORSE_REV = {v: k for k, v in MORSE.items()}


def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[-1] + 1,
                           prev[j] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def soft_decode(code: str) -> str:
    """
    Soft Morse decode: exact match or closest by edit distance.

    Prosigns (<AR>, <SK>, <KN>) require exact match only — no fuzzy matching
    because their patterns (.-.-, ...-., -.--.) are short and easily confused
    with merged-character artifacts.

    Fuzzy matching is conservative: max distance = 1, only for patterns
    with >= 3 elements. Shorter patterns are too ambiguous for fuzzy matching.
    """
    if code in MORSE:
        return MORSE[code]

    # Conservative fuzzy matching: only for patterns with >= 3 elements,
    # and only max distance = 1 (not 2). This prevents incorrect fuzzy
    # matches for short/noisy patterns while still recovering from single
    # element errors in longer patterns.
    if len(code) < 3:
        return '?'

    best, bd = '?', 1
    for k, v in MORSE.items():
        # Prosigns: exact match only (prevent merged chars from being
        # fuzzy-matched to <AR>/<SK>/<KN> via edit distance)
        if v in ('<AR>', '<SK>', '<KN>'):
            continue
        # Length guard: don't match patterns with very different lengths
        if abs(len(k) - len(code)) > 1:
            continue
        d = levenshtein(code, k)
        if d < bd:
            bd, best = d, v
    return best
