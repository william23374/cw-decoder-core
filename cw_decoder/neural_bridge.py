#!/usr/bin/env python3
"""
Optional morseformer (RNN-T / CTC) bridge for V13.

Lazy-loads torch + morseformer only when requested. If dependencies or
checkpoints are missing, callers get ``available() == False`` and a no-op
decode — DSP path remains the default and is never broken.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RNNT = _ROOT / "morseformer" / "checkpoints" / "rnnt_phase11b.pt"
_DEFAULT_LM = _ROOT / "morseformer" / "checkpoints" / "lm_phase5_2.pt"

_STATE: Dict = {
    "checked": False,
    "ok": False,
    "reason": "",
    "model": None,
    "lm": None,
    "device": None,
}


def available(ckpt: Optional[str] = None) -> bool:
    """True if torch + morseformer + checkpoint can be used."""
    _ensure_loaded(ckpt)
    return bool(_STATE["ok"])


def availability_reason() -> str:
    _ensure_loaded(None)
    return str(_STATE.get("reason") or "")


def _ensure_loaded(ckpt: Optional[str]) -> None:
    if _STATE["checked"] and ckpt is None:
        return
    _STATE["checked"] = True
    ckpt_path = Path(ckpt) if ckpt else _DEFAULT_RNNT
    if not ckpt_path.exists():
        _STATE["ok"] = False
        _STATE["reason"] = f"checkpoint missing: {ckpt_path}"
        return
    try:
        import torch
    except ImportError:
        _STATE["ok"] = False
        _STATE["reason"] = "torch not installed"
        return

    # Prefer morseformer package on the sibling path
    mf_root = str(_ROOT / "morseformer")
    import sys
    if mf_root not in sys.path:
        sys.path.insert(0, mf_root)

    try:
        from scripts.decode_audio import (  # type: ignore
            _acoustic_cfg_from_state,
            _auto_device,
            _is_rnnt_checkpoint,
            _rnnt_cfg_from_state,
        )
        from morseformer.decoding.streaming import StreamingConfig, decode_offline
        from morseformer.models.lm import GptLM, LmConfig
        from morseformer.models.rnnt import RnntModel
        from morseformer.models.acoustic import AcousticModel
    except Exception as exc:
        _STATE["ok"] = False
        _STATE["reason"] = f"morseformer import failed: {exc}"
        return

    try:
        device = torch.device(_auto_device())
        blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        is_rnnt = _is_rnnt_checkpoint(blob)
        state = dict(blob["model"])
        if blob.get("ema"):
            for k, v in blob["ema"].items():
                if k in state:
                    state[k] = v
        if is_rnnt:
            model = RnntModel(_rnnt_cfg_from_state(state)).to(device).eval()
        else:
            model = AcousticModel(_acoustic_cfg_from_state(state)).to(device).eval()
        model.load_state_dict(state)

        lm = None
        if is_rnnt and _DEFAULT_LM.exists():
            try:
                lm_blob = torch.load(
                    str(_DEFAULT_LM), map_location="cpu", weights_only=False)
                mcfg = lm_blob["config"]["model"]
                lm = GptLM(LmConfig(
                    vocab_size=mcfg["vocab_size"],
                    d_model=mcfg["d_model"],
                    n_heads=mcfg["n_heads"],
                    n_layers=mcfg["n_layers"],
                    dropout=mcfg.get("dropout", 0.1),
                )).to(device).eval()
                lm_state = dict(lm_blob["model"])
                if lm_blob.get("ema"):
                    for k, v in lm_blob["ema"].items():
                        if k in lm_state:
                            lm_state[k] = v
                lm.load_state_dict(lm_state)
            except Exception:
                lm = None

        _STATE.update({
            "ok": True,
            "reason": "ready",
            "model": model,
            "lm": lm,
            "device": device,
            "is_rnnt": is_rnnt,
            "decode_offline": decode_offline,
            "StreamingConfig": StreamingConfig,
            "torch": torch,
        })
    except Exception as exc:
        _STATE["ok"] = False
        _STATE["reason"] = f"load failed: {exc}"


def neural_decode(
    audio: np.ndarray,
    sample_rate: int,
    tone_hz: float = 600.0,
    fusion_weight: float = 0.7,
    ckpt: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Decode mono float audio with morseformer when available.

    Returns (text, confidence_proxy). confidence_proxy is a crude
    length/charset heuristic in [0, 1] for ranking vs DSP — not a
    calibrated posterior.
    """
    _ensure_loaded(ckpt)
    if not _STATE["ok"]:
        return "", 0.0

    torch = _STATE["torch"]
    model = _STATE["model"]
    device = _STATE["device"]
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 8000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sample_rate), 8000)
        audio = resample_poly(audio, 8000 // g, int(sample_rate) // g).astype(np.float32)
        sample_rate = 8000

    if not _STATE.get("is_rnnt"):
        # CTC-only path: simple feature → greedy (minimal support)
        return "", 0.0

    StreamingConfig = _STATE["StreamingConfig"]
    decode_offline = _STATE["decode_offline"]
    cfg = StreamingConfig(
        window_seconds=6.0,
        hop_seconds=2.0,
        sample_rate=8000,
        carrier_hz=float(tone_hz),
        bandwidth_hz=200.0,
        confidence_threshold=0.35,
    )
    lm = _STATE.get("lm")
    fw = float(fusion_weight) if lm is not None else 0.0
    with torch.inference_mode():
        text = decode_offline(
            model, audio, cfg, device, lm=lm, fusion_weight=fw)
    text = (text or "").strip().upper()
    if not text:
        return "", 0.0
    # Proxy confidence: penalize very short / digit-soup outputs
    alnum = sum(ch.isalnum() for ch in text)
    q = text.count("?")
    conf = min(1.0, alnum / 40.0) * (1.0 - min(0.8, q / max(alnum, 1)))
    return text, float(conf)
