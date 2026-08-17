"""Unit tests for Morse table helpers."""

from cw_decoder.morse import MORSE, MORSE_REV, levenshtein, soft_decode


def test_morse_roundtrip_letters():
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        code = MORSE_REV[ch]
        assert MORSE[code] == ch


def test_soft_decode_exact():
    assert soft_decode(".-") == "A"
    assert soft_decode("---") == "O"
    assert soft_decode("...") == "S"


def test_levenshtein():
    assert levenshtein("CQ", "CQ") == 0
    assert levenshtein("CQ", "CK") == 1
    assert levenshtein("", "ABC") == 3
