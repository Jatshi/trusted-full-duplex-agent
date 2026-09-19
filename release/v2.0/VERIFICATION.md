# TFD-STAR 2.0 verification record

Date: 2026-09-19

## Automated gates

| Gate | Result |
|---|---:|
| Main research/unit suite | 49 passed in 6.12s |
| Demo Python runtime integration | 10 passed in 3.68s |
| Demo browser/protocol contracts | 10 passed |
| Learning HTML parser/link check | pass, 58,426 bytes, 26 IDs, no broken anchors |
| Integration patch reverse-check against validated working tree | pass |
| `git diff --check` | pass (line-ending notices only) |

The main suite is run with single-threaded BLAS on Windows. During release
validation, NumPy `polyfit` entered a Windows MKL `lstsq` process abort for a
five-frame linear slope. The featurizer now uses the algebraically equivalent
closed-form one-variable least-squares slope. This removes an unnecessary LAPACK
call from the 100 ms online path and all 49 tests pass.

## Visual gate

The standalone learning manual was rendered in headless Chrome at 1440×1100.
Navigation, Chinese text, offline CSS, 2.0 badges and the first content section
render correctly. Preview: `docs/assets/learning-manual-preview.png`.

## Live GPU evidence

- RTX 4080 SUPER 32 GB, one session;
- GRPO adapter active in backend log;
- turn-taking MLP active with 100 ms frames and four-frame confirmation;
- Faster-Whisper-small CPU-int8 and TrustGate active;
- one-time TTS compile warm-up 132.6 s;
- latest short probe: backend median/mean 0.742/0.794 s, client
  median/mean 0.824/0.897 s.

This is a release smoke/latency probe, not a production concurrency or long-soak
benchmark.
