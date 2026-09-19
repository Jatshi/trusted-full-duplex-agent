# TFD-STAR 2.0 architecture and evidence note

This is the compact, diff-friendly companion to the standalone HTML learning
manual. It distinguishes project work from upstream capability and records the
evidence used for the v2.0 release.

## Capability ownership

| Layer | Origin | What v2.0 actually does |
|---|---|---|
| speech-to-speech generation, audio tokenizer, token2wav, KV state | MiniCPM-o 4.5 | used as the multimodal base model |
| Gateway / Worker / Python backend protocol | OpenBMB MiniCPM-o-Demo | extended through the overlay in `integrations/` |
| GRPO LoRA | this project | PEFT adapter attached to the internal LLM before serving |
| 10-feature turn-taking MLP | this project | real 100 ms audio frames to hold/take/backchannel |
| TrustGate | this project | execute/clarify/stop policy with risk and ambiguity paths |
| Faster-Whisper safety transcript | **new in 2.0** | grounds TrustGate on live waveform instead of hand-provided text |
| local-immediate barge-in and reasoned listen protocol | **new in 2.0** | stops local playback before the server round trip |
| consistent guardrail voice | **new in 2.0** | deterministic clarify/stop WAVs use the same female voice |
| TTS-only compile-after-adapter | **new in 2.0** | retains LoRA while recovering real-time TTS margin |
| TFD acoustic console | **new in 2.0** | branded UI, device selector, metrics and adaptive buffer |

## Measured release evidence

- GPU: RTX 4080 SUPER 32 GB, one interactive session.
- LoRA status: the backend log reports the `tfd_grpo` adapter active.
- TTS compile warm-up: 132.6 s once; TTS graph compiled, LLM graph deliberately
  skipped by default.
- ASR: Faster-Whisper-small, CPU int8, Mandarin, beam size 1, 12 s rolling window.
- Latest short live probe: backend median 0.742 s / mean 0.794 s; client-observed
  median 0.824 s / mean 0.897 s.
- Previous direct-tunnel probe: spoken chunks 856-946 ms end to end; the Gradio
  relay expanded similar generation work to roughly 2.3-4.2 s.
- Turn MLP: validation frame accuracy 96.46%; held-out simulated event set 0%
  false takeover and 0% miss at four confirmation frames, median take latency
  553.9 ms. This is synthetic-data evidence, not a spontaneous-dialog corpus.
- Python integration tests: 10 passed.
- JavaScript contract tests: 10 passed.

## Known limitations

1. The released GRPO experiment is small. MiniCPM-o 9B gains on the selected
   20-step run are near the measured sampling-noise band; other 9B settings
   regress. The negative results remain in the release.
2. The turn MLP was trained on parameterized synthetic acoustics. It is a
   deployable controller and regression target, not a claim of SOTA turn-taking.
3. TrustGate uses ASR as a safety monitor. It does not replace the base model's
   audio understanding. Low-risk ASR uncertainty is deferred to the base model.
4. The browser transport is still Float32 PCM over WebSocket. Opus/WebRTC and
   model-native 250 ms units are future work, not v2.0 claims.
5. The direct SSH tunnel is the validated low-latency path. A public production
   deployment still needs HTTPS/WSS, authentication, abuse controls and metrics.
