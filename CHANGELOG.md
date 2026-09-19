# Changelog

## [2.0.0] - 2026-09-19

### Added

- MiniCPM-o-Demo integration overlay for the released GRPO adapter, live
  turn-taking MLP and ASR-backed TrustGate.
- Faster-Whisper-small safety path with Chinese normalization, confidence and
  no-speech handling.
- Client-immediate barge-in, reasoned listen protocol and adaptive jitter buffer.
- Unified guardrail voice, TFD-STAR acoustic-console UI and integration tests.
- Standalone full-project Chinese learning manual and v2.0 evidence note.

### Changed

- Irreversible-risk floor and ambiguity handling no longer allow clear-speech
  delete/transfer commands to bypass confirmation.
- Adapter is attached before TTS-only compilation.
- Live Gate evaluation uses a voiced-frame watermark and a 12-second rolling
  ASR window.

### Preserved

- All GRPO sweep results, including negative MiniCPM-o 9B runs.
- Baseline mode when TFD runtime environment variables are unset.

## [1.0.0] - 2026-09-17

- Initial reproducible research package: TrustGate, synthetic turn-taking MLP,
  barge-in mechanism/A-B experiments, GRPO LoRA training and evidence artifacts.
