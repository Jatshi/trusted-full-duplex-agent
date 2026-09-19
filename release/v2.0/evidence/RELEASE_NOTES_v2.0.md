# TFD-STAR 2.0 release notes

Release date: 2026-09-19

## Release thesis

Version 2.0 turns the earlier research components into one real-time system. The
MiniCPM-o 4.5 Demo can now load the project GRPO adapter, run the trained
turn-taking MLP on live audio, transcribe the user turn through an ASR safety
path, apply TrustGate, emit deterministic guardrail speech and expose all of
those decisions in logs/metrics.

## New in 2.0

### Real runtime integration

- Added a reviewable MiniCPM-o-Demo overlay under
  `integrations/minicpmo45-demo/overlay/`.
- Added explicit runtime switches for the GRPO adapter, turn-taking MLP and
  TrustGate. The base model remains runnable when they are disabled.
- GRPO LoRA is attached to the internal LLM without replacing the multimodal
  MiniCPM-o wrapper.

### ASR-backed TrustGate

- Added Faster-Whisper-small as a CPU-int8 safety monitor.
- Added Mandarin decoding, duration-weighted average log-probability,
  no-speech confidence and Traditional-to-Simplified Chinese normalization.
- Added noise-only `listen` behavior and low-risk uncertainty fallback to the
  base audio model, fixing the “any sound becomes clarify” failure.
- Added a voiced-frame watermark and a 12-second rolling window, preventing the
  same speech from being re-evaluated repeatedly during silence.

### Turn-taking in the live path

- Runs the released 10-32-3 MLP on real 100 ms frames.
- Requires minimum voiced evidence, trailing silence and four consecutive take
  confirmations before it releases model speech.
- Publishes hold/take/backchannel probabilities and frame counters in metrics.

### Barge-in and playback protocol

- Added AEC-aware browser VAD and immediate local playback cancellation.
- Sends one-shot `force_listen` so the server state catches up without creating
  a sticky listen mode.
- Added `turn_end`, `model_listen` and `force_listen` reasons. Normal completion
  no longer truncates queued tail audio.
- Added adaptive localhost/public-relay jitter buffering. The validated local
  tunnel default is 700 ms in the final release.

### Voice and interface

- Re-generated clarify/stop guardrail speech in one female voice, matching the
  normal assistant persona.
- Rebuilt the page as the TFD-STAR acoustic console; removed upstream branding
  and the unrelated FAQ.
- Added local fonts and input-device selection support.

### Performance

- Adapter attachment now occurs before optional compilation.
- TTS-only compile is the default; full LLM compilation is opt-in after a
  greater-than-12-minute compile attempt proved unsuitable for this runtime.
- One-time TTS warm-up measured 132.6 seconds.
- Latest short live probe: backend median 0.742 s / mean 0.794 s; client median
  0.824 s / mean 0.897 s.

### Documentation and packaging

- Added the standalone Chinese deep-learning manual:
  `docs/TFD_STAR_2.0_全双工可信语音智能体_深度学习手册.html`.
- Added a compact capability-ownership and evidence note.
- Added a full-source F-drive archive manifest and checksums for locally owned
  artifacts.
- Refreshed the Hugging Face model card and grouped v2.0 assets under `v2.0/`.

## Compatibility and migration

The upstream Demo is not vendored. Apply the overlay to the validated upstream
revision and export `tfd-runtime.env.example`. Existing research scripts remain
compatible. The base models and Faster-Whisper weights must be downloaded from
their original repositories.

## Verification

- main research tests: 49 passed
- integration runtime tests: 10 Python tests
- browser/protocol contracts: 10 JavaScript tests
- full live chain: RTX 4080 SUPER 32 GB, one session, local SSH tunnel

Exact post-release test counts and hashes are recorded in `release/v2.0/`.

## Known limitations

- Turn-taking results are from parameterized synthetic audio and need real
  conversational-domain validation.
- The 9B GRPO improvement is small and close to sampling noise; negative sweeps
  are intentionally retained.
- WebSocket Float32 PCM is still used. Opus/WebRTC and sub-model-unit streaming
  are future work.
- The direct tunnel is a research deployment. Public production serving still
  requires TLS, authentication, abuse controls and concurrency testing.
