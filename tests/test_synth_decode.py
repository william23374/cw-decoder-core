"""End-to-end decode tests using synthesized CW (no fixtures)."""

from __future__ import annotations

import re

import pytest

from cw_decoder import decode_v13
from tests.synth_cw import synthesize_cw, write_wav


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


@pytest.mark.parametrize(
    "text,wpm,tone",
    [
        ("CQ CQ DE W1AW K", 20, 750),
        ("W1AW DE N0CALL", 25, 700),
        ("CQ TEST DE K3EST K", 22, 720),
    ],
)
def test_synth_clean_decode(tmp_path, text, wpm, tone):
    wav = tmp_path / "cw.wav"
    sig = synthesize_cw(text, fs=16000, tone=tone, wpm=wpm, snr_db=None, seed=42)
    write_wav(wav, sig, fs=16000)

    result = decode_v13(str(wav), ref_text=text, verbose=False, use_neural=False)
    assert result is not None
    got = _alnum(result.get("text", ""))
    ref = _alnum(text)
    # Character accuracy from decoder when ref provided
    acc = float(result.get("accuracy") or 0.0)
    assert acc >= 0.70 or (
        len(got) > 0 and sum(a == b for a, b in zip(got, ref)) / max(len(ref), 1) >= 0.7
    ), f"low accuracy={acc:.2f} got={got!r} ref={ref!r}"


def test_synth_noisy_still_decodes(tmp_path):
    text = "CQ DE W1AW K"
    wav = tmp_path / "noisy.wav"
    sig = synthesize_cw(text, fs=16000, tone=750, wpm=20, snr_db=15, seed=7)
    write_wav(wav, sig, fs=16000)

    result = decode_v13(str(wav), ref_text=text, verbose=False, use_neural=False)
    assert result is not None
    assert _alnum(result.get("text", ""))
    acc = float(result.get("accuracy") or 0.0)
    assert acc >= 0.40


def test_package_version():
    import cw_decoder

    assert hasattr(cw_decoder, "__version__")
    assert cw_decoder.__version__
