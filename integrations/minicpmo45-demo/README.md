# MiniCPM-o 4.5 Demo integration for TFD-STAR 2.0

This directory is a **thin, reviewable integration delta** for the upstream
[`OpenBMB/MiniCPM-o-Demo`](https://github.com/OpenBMB/MiniCPM-o-Demo) runtime.
It contains a Git patch for changed upstream files and an overlay containing
only files authored by this project. The full upstream repository is
intentionally not vendored here.

## What the overlay adds

- loads the released GRPO PEFT adapter into `model.llm` without replacing the
  MiniCPM-o model wrapper;
- runs the trained 10-32-3 turn-taking MLP on real 100 ms audio frames;
- adds the optional Faster-Whisper-small ASR safety monitor and routes the
  transcript through TrustGate;
- keeps low-risk ASR uncertainty on the base-model path instead of turning every
  sound into a canned clarification;
- uses a voiced-frame watermark so the same rolling ASR window is not evaluated
  repeatedly during silence;
- returns deterministic `clarify` / `stop` speech in one consistent voice;
- implements local-immediate barge-in, AEC-aware client VAD, one-shot
  `force_listen`, reasoned `listen` messages, and adaptive jitter buffering;
- compiles the TTS path **after** LoRA attachment; full LLM compilation is an
  opt-in because it was not cost-effective on the measured 32 GB 4080 runtime;
- replaces the upstream product page with the TFD-STAR acoustic console and adds
  Python/JavaScript regression tests.

## Apply the integration

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git MiniCPM-o-Demo
git -C MiniCPM-o-Demo checkout <the-upstream-revision-you-have-validated>
python /path/to/trusted-full-duplex-agent/integrations/minicpmo45-demo/apply_integration.py MiniCPM-o-Demo
```

Copy `tfd-runtime.env.example` to a file outside Git, adjust the absolute paths,
then export it before starting the upstream backend, worker and gateway.

```bash
set -a
. ./tfd-runtime.env
set +a
```

The tested topology is:

```text
Browser :18006 (SSH forward)
  -> Gateway :8016 / registry :8017
  -> Worker :22640
  -> Python backend :22600
  -> MiniCPM-o 4.5 + GRPO LoRA + turn MLP + ASR/TrustGate
```

For exact launch flags, use the upstream Demo documentation corresponding to the
checked-out revision. TFD-specific configuration is entirely environment driven
and listed in `tfd-runtime.env.example`.

## Verify before a live session

```bash
python -m pytest tests/test_tfd_runtime.py -q
npx vitest run \
  tests/js/audio-duplex-branding-contract.test.js \
  tests/js/backend-compile-keepalive-contract.test.js \
  tests/js/realtime-bargein-contract.test.js
```

The release gate used for v2.0 is 10 Python runtime tests and 10 JavaScript
contract tests. A live GPU probe must additionally show that the adapter, turn
MLP and TrustGate are loaded in the backend log.

## Attribution and boundary

MiniCPM-o, the upstream Demo server and the base model remain OpenBMB work. The
patch and project-owned overlay are the TFD-STAR system integration. Base weights
and the upstream source tree are not redistributed by this repository. Review
the upstream model and code terms before deploying or redistributing a combined
package.
