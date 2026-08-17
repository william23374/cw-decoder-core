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
# Unit + synthetic CW (no network)
pytest -m "not real_audio"

# Download LIVE HF QRQ sample from GitHub Releases and decode
pytest -v -m real_audio
```

**Real-audio fixture** (Release `fixtures-v1`):

- Source: *HF QRQ CW QSO between W5UXH & K6KX - recorded LIVE off the air.m4a*
- Assets: https://github.com/william23374/cw-decoder-core/releases/tag/fixtures-v1
- CI job **Real HF QRQ (download + decode)** downloads the 45 s WAV excerpt and runs `decode_v13`

From the parent monorepo:

```bash
./run-tests.sh          # synth/unit
./run-tests.sh real     # download + decode
./run-tests.sh ci       # push and wait for Actions
```

## Layout

```
cw_decoder/          # decoder package
tests/               # synth unit tests + real_audio download test
fixtures/README.md   # points at GitHub Release assets (binaries not in git)
.github/workflows/   # unit matrix + real-audio job
```
