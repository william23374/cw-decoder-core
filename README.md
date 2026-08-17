# CW Decoder Core

Offline Morse code (CW) audio decoder — core DSP package extracted from V13.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Decode

```bash
python3 -m cw_decoder audio.wav -v
python3 -m cw_decoder audio.wav --ref "CQ CQ DE W1AW K" -v
```

```python
from cw_decoder import decode_v13

result = decode_v13("audio.wav", verbose=True, use_neural=False)
print(result["text"], result["wpm"])
```

## Tests (CI)

```bash
pytest
```

CI synthesizes clean CW audio on the fly — no large fixtures required.

## Layout

```
cw_decoder/          # decoder package
  decoder.py         # V13 offline path
  dsp.py / envelope / threshold / segmentation
  pll.py / agc.py / integrator.py
  corrector.py / morse.py / realtime.py
tests/               # self-contained synth + unit tests
.github/workflows/   # GitHub Actions
```
