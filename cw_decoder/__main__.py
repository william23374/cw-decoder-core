#!/usr/bin/env python3
"""
CW Decoder CLI — Aligned with international ham radio decoder conventions
(FLDIGI, cwget, morse2txt, CW Skimmer).

Usage:
    python -m cw_decoder -i audio.wav              # basic decode
    python -m cw_decoder -i audio.wav -f 700       # force frequency
    python -m cw_decoder -i audio.wav -w 25        # speed hint
    python -m cw_decoder -i audio.wav -v           # verbose
    python -m cw_decoder -i audio.wav -o out.txt   # save to file
    python -m cw_decoder -i audio.wav -q           # quiet (text only)
    python -m cw_decoder -i audio.wav --json        # machine-readable JSON
    python -m cw_decoder -V                          # show version
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional

from cw_decoder import __version__, decode_v13


def _format_single_result(result: Dict, ns: argparse.Namespace) -> str:
    """Format a single decode result for output."""
    if ns.json_output:
        json_out = {
            "text": result.get("text", ""),
            "text_raw": result.get("text_raw", ""),
            "freq_hz": result.get("freq"),
            "freq_initial_hz": result.get("freq_initial"),
            "snr_db": result.get("snr_db"),
            "wpm": result.get("wpm"),
            "dit_ms": result.get("dit_ms"),
            "dah_ms": result.get("dah_ms"),
            "score": result.get("score"),
            "method": result.get("method"),
            "corrections": result.get("corrections", 0),
        }
        if result.get("accuracy") is not None:
            json_out["accuracy"] = round(result["accuracy"], 4)
        return json.dumps(json_out, indent=2, ensure_ascii=False)
    elif ns.quiet:
        return result.get("text", "")
    else:
        lines = []
        freq = result.get("freq", 0)
        snr = result.get("snr_db", 0)
        wpm = result.get("wpm", 0)
        text = result.get("text", "")
        acc = result.get("accuracy")
        method = result.get("method", "")

        if ns.verbose:
            dit = result.get("dit_ms", 0)
            dah = result.get("dah_ms", 0)
            corr = result.get("corrections", 0)
            lines.append(f"Method:     {method}")
            lines.append(f"Frequency:  {freq:.0f} Hz")
            lines.append(f"SNR:        {snr:.1f} dB")
            lines.append(f"WPM:        {wpm:.0f}")
            lines.append(f"Dit/Dah:    {dit:.0f}/{dah:.0f} ms")
            if corr > 0:
                lines.append(f"Fixes:      {corr} semantic corrections")
        else:
            lines.append(f"Frequency: {freq:.0f} Hz  |  SNR: {snr:.1f} dB  |  WPM: {wpm:.0f}")

        lines.append(f"Text:       {text}")
        if acc is not None:
            lines.append(f"Accuracy:   {acc * 100:.0f}%")
        return "\n".join(lines)


# ================================================================
#  Single-file mode
# ================================================================

def _run_single(ns: argparse.Namespace, input_path: str) -> int:
    """Decode a single audio file."""
    if not Path(input_path).exists():
        print(f"cw-decoder: error: file not found: {input_path}", file=sys.stderr)
        return 1

    if ns.verbose:
        mode_parts = []
        if ns.freq:
            mode_parts.append(f"f={ns.freq:.0f}Hz")
        if ns.wpm > 0:
            mode_parts.append(f"wpm={ns.wpm:.0f}")
        mode_str = " [" + ", ".join(mode_parts) + "]" if mode_parts else ""
        print(f"[*] Loading: {Path(input_path).name}{mode_str}")

    result = decode_v13(
        input_path,
        verbose=ns.verbose,
        force_freq=ns.freq,
        force_wpm=ns.wpm if ns.wpm > 0 else None,
    )

    if "error" in result:
        print(f"cw-decoder: error: {result['error']}", file=sys.stderr)
        return 1

    output_text = _format_single_result(result, ns)

    if ns.output:
        with open(ns.output, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")
        if not ns.quiet and not ns.json_output:
            print(f"Output saved to: {ns.output}")
    else:
        print(output_text)

    return 0


# ================================================================
#  Parser
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser aligned with international CLI conventions."""

    prog = os.environ.get("CW_DECODER_PROG", "cw-decoder")
    parser = argparse.ArgumentParser(
        prog=prog,
        description="CW Decoder — Adaptive Morse code decoder for noisy audio.",
        epilog="Examples:\n"
               "  %(prog)s -i recording.wav\n"
               "  %(prog)s -i recording.wav -f 700\n"
               "  %(prog)s -i recording.wav -w 25 -v\n"
               "  %(prog)s -i recording.wav -o out.txt\n"
               "  %(prog)s -i recording.wav -q\n"
               "  %(prog)s -i recording.wav --json\n"
               "  %(prog)s -V",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ───────────────────────────────────────────────────
    io_group = parser.add_argument_group("I/O options")
    io_group.add_argument(
        "-i", "--input", dest="input_path", required=True,
        help="Input audio file (WAV, MP3, MP4, M4A, FLAC, OGG)",
    )
    io_group.add_argument(
        "-o", "--output",
        help="Output text file (default: stdout)",
    )

    # ── Signal ────────────────────────────────────────────────
    sig_group = parser.add_argument_group("Signal options")
    sig_group.add_argument(
        "-f", "--freq", type=float, default=None,
        help="Force CW frequency in Hz (default: auto-detect)",
    )
    sig_group.add_argument(
        "-w", "--wpm", type=float, default=0,
        help="Expected speed hint in WPM, 0=auto-detect (default: 0)",
    )

    # ── Output format ─────────────────────────────────────────
    fmt_group = parser.add_argument_group("Output format")
    fmt_group.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output (show SNR, WPM, dit/dah, corrections)",
    )
    fmt_group.add_argument(
        "-q", "--quiet", action="store_true",
        help="Quiet mode: output decoded text only (no metadata)",
    )
    fmt_group.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Machine-readable JSON output",
    )

    # ── Meta ──────────────────────────────────────────────────
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


# ================================================================
#  Entry point
# ================================================================

def run(args: Optional[list] = None) -> int:
    """Run the decoder. Returns 0 on success, 1 on error."""
    parser = build_parser()
    ns = parser.parse_args(args)
    return _run_single(ns, ns.input_path)


if __name__ == "__main__":
    sys.exit(run())
