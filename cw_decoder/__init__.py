"""
CW Decoder — Adaptive Morse code decoder for noisy audio.

Public API:
    decode_v13(path, ...)          — V13 offline decoder (recommended)
    decode(path, ...)              — V12.7 legacy decoder
    MultiStreamRealtimeDecoder     — low-latency multi-tone live path
    CWDecoderRealtime              — simple realtime wrapper
"""

__version__ = "13.3"

from cw_decoder.decoder import decode_v13
from cw_decoder.legacy import decode
from cw_decoder.realtime import MultiStreamRealtimeDecoder, CWDecoderRealtime

__all__ = [
    "decode_v13",
    "decode",
    "MultiStreamRealtimeDecoder",
    "CWDecoderRealtime",
    "__version__",
]
