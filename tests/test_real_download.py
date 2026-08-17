"""Download real HF QRQ sample from GitHub Releases and decode it."""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

import pytest

from cw_decoder import decode_v13

REPO = os.environ.get("CW_FIXTURE_REPO", "william23374/cw-decoder-core")
TAG = os.environ.get("CW_FIXTURE_TAG", "fixtures-v1")
# Prefer short WAV for CI speed; allow override to full m4a.
ASSET = os.environ.get("CW_FIXTURE_ASSET", "w5uxh_k6kx_hf_qrq_45s.wav")

EXPECTED_MARKERS = ("POWER", "SUPPLY", "KEYER")


def _asset_url(asset: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{TAG}/{asset}"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    print(f"[*] Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "cw-decoder-core-ci"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    print(f"[*] Saved {dest} ({dest.stat().st_size} bytes)")
    return dest


@pytest.mark.real_audio
def test_download_and_decode_w5uxh_k6kx(tmp_path, monkeypatch):
    """GitHub Release → decode LIVE HF QRQ QSO (W5UXH & K6KX)."""
    cache = Path(os.environ.get("CW_FIXTURE_CACHE", tmp_path / "fixtures"))
    path = _download(_asset_url(ASSET), cache / ASSET)

    # Full m4a needs ffmpeg; wav does not.
    if path.suffix.lower() in {".m4a", ".mp3", ".mp4"}:
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg required for m4a fixtures")

    result = decode_v13(str(path), verbose=True, use_neural=False)
    assert result is not None, "decode returned None"

    text = (result.get("text") or "").upper()
    alnum = re.sub(r"[^A-Z0-9]", "", text)
    wpm = float(result.get("wpm") or 0)

    print(f"[*] Decoded WPM={wpm:.1f} chars={len(alnum)}")
    print(f"[*] Text preview: {text[:180]}")

    assert len(alnum) >= 20, f"too little decoded text: {text!r}"
    assert wpm >= 25, f"expected QRQ-ish speed, got WPM={wpm}"
    # Soft content check: at least one known phrase fragment from this clip
    assert any(m in text for m in EXPECTED_MARKERS), (
        f"expected one of {EXPECTED_MARKERS} in decoded text, got: {text[:240]!r}"
    )
