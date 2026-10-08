"""E42 wait-mode gate: an instruction-conditioned semantic completion gate.

Why this module exists
----------------------
The CFDC ``WAIT_RESUME`` axis measures 0/94 on both stacks.  Two independent
causes were isolated before this module was written (see
``experiments/E42_WAIT_RESUME_PLAN.md``):

1. the 0/94 itself is a scoring artifact (oracle cancel at ``pause_end_s``
   zeroes the ``[final+0.1, final+3.0]`` scoring window), and
2. the real behavioural defect is ``gap_talk``: the model speaks inside the
   mid-story pause even though the system instruction says "stay silent
   through such a pause".  The failure is structural -- the 100 ms
   ``StreamingTurnController`` that decides ``allow_speak`` never receives the
   system prompt, so instruction following cannot happen at that layer.

This module adds the missing instruction conditioning *above* the frame-level
controller: when a session's system prompt carries WAIT semantics, the gate
reuses the ASR transcript that TrustGate already computes and holds the turn
while the transcript does not end at a sentence boundary.

Design constraints honoured here
--------------------------------
* **Default off.** ``TFD_TURN_WAIT_MODE`` defaults to ``off``; with the default
  this module is never consulted and runtime behaviour is unchanged.
* **ACK path untouched.** Wait mode is only active when the prompt carries WAIT
  markers and *no* ACK markers, so the ``ACK_IN_GAP`` axis (the measured
  early-talk advantage) is unaffected.
* **Zero extra inference.** The transcript is the one TrustGate already
  produces; no second ASR pass, no LLM call.
* **Fail-closed and conservative.** When the transcript is missing or clearly
  truncated the gate holds.  Holding is consistent with the WAIT partial
  order ("do not talk over them"); the cost is a possible ``no_service``, never
  a barge-in.
* **Holding must be reversible.** The caller (``server.py``) must *not* advance
  the TrustGate release watermark when this gate holds, and must *not* reset the
  accumulated ASR audio.  Otherwise a hold on one chunk could become permanent,
  because the runner pushes no audio after the last frame.

E43 extension (2026-10-02)
--------------------------
E42a proved the gate alone cannot fix ``WAIT_RESUME`` (3/94, unchanged): the
model already *is* released mid-pause and still over-suppresses after the
caller finishes.  E43 attacks the resume side with two independent, env-gated
levers that both live in or behind this module:

* **E43a (runner-side tail silence).**  ``155`` gains ``--tail-silence-frames``
  and pushes real silence frames after the manifest audio, giving the
  controller its ``min_silence_frames`` confirmation *after* ``final`` so the
  gate can actually release once and the model gets one free generate at
  ``final + tail``.  No server-side change beyond the existing gate.
* **E43b (prompt append).**  ``TFD_TURN_WAIT_PROMPT_APPEND`` appends one
  clarifying clause to wait-mode prompts only, directly counteracting the
  measured over-suppression ("stay silent" is being obeyed past its scope).

This module deliberately contains **no** timing/physical measurement and makes
no claim about audible stop latency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

#: Substrings that mark a WAIT-semantics system instruction.  Matched
#: case-insensitively against the session prompt.  The WAIT instruction in the
#: frozen v3 intent manifest (``cfdc_intent_v3_manifest.jsonl``, 94 rows) is the
#: reference positive sample.
WAIT_MARKERS: tuple[str, ...] = (
    "stay silent",
    "mid-story pause",
    "thinking time",
)

#: Substrings that mark an ACK-semantics instruction.  Their presence vetoes
#: wait mode so the ACK axis can never be gated by this module.
ACK_MARKERS: tuple[str, ...] = (
    "briefly acknowledge",
)

#: Characters treated as sentence-final punctuation.  Faster-Whisper emits
#: Latin punctuation for English audio; the CJK set is included because the
#: ASR language is configurable and the same gate may be reused for Chinese.
SENTENCE_TERMINALS: str = ".!?。！？；;…"

#: Closing characters stripped before the terminal test so a quoted or
#: parenthesised sentence still counts as complete.
_TRAILING_CLOSERS: str = "\"')]}》」』”’"

#: Reason codes, kept stable so metrics and tests can assert on them.
REASON_INACTIVE = "inactive"
REASON_RELEASE_COMPLETE = "release_complete_sentence"
REASON_HOLD_NO_TRANSCRIPT = "hold_no_transcript"
REASON_HOLD_INCOMPLETE = "hold_incomplete_sentence"
REASON_HOLD_SHORT_TAIL = "hold_short_tail"
REASON_HOLD_INSUFFICIENT_QUIET = "hold_insufficient_quiet"

ENV_WAIT_MODE = "TFD_TURN_WAIT_MODE"
ENV_MIN_SILENCE_FRAMES = "TFD_TURN_WAIT_MIN_SILENCE_FRAMES"
ENV_TAIL_WORD_MIN = "TFD_TURN_WAIT_TAIL_WORD_MIN"
#: E43b: optional clause appended to a wait-mode session's system prompt before
#: ``duplex_prepare``.  Empty (default) means the manifest instruction is passed
#: through untouched, i.e. the frozen eval condition.  Only ever applied to
#: sessions already detected as wait-mode, so the ACK axis is structurally
#: unreachable from this knob.
ENV_WAIT_PROMPT_APPEND = "TFD_TURN_WAIT_PROMPT_APPEND"


def wait_mode_enabled() -> bool:
    """Return True only when the operator explicitly opts into ``auto``.

    ``off`` (and any unset/empty/unknown value) keeps the legacy behaviour so
    this gate can never silently change an existing condition.
    """
    return os.environ.get(ENV_WAIT_MODE, "off").strip().lower() == "auto"


def wait_prompt_append() -> str:
    """Return the E43b prompt-append clause, or "" when the knob is off."""
    return os.environ.get(ENV_WAIT_PROMPT_APPEND, "").strip()


def apply_prompt_append(system_prompt: Optional[str], append: Optional[str]) -> str:
    """Append ``append`` to ``system_prompt`` as one extra sentence.

    Empty/whitespace ``append`` (or an empty prompt) is a no-op returning the
    original prompt unchanged, so callers do not need their own guard.
    """
    prompt_text = (system_prompt or "").strip()
    clause = (append or "").strip()
    if not prompt_text or not clause:
        return prompt_text
    return f"{prompt_text} {clause}"


def detect_wait_mode(system_prompt: Optional[str]) -> bool:
    """Decide whether a session prompt carries WAIT semantics.

    An empty prompt yields False: without an instruction there is nothing to
    condition on, and guessing would risk gating a normal conversation.
    """
    lowered = (system_prompt or "").strip().lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in ACK_MARKERS):
        return False
    return any(marker in lowered for marker in WAIT_MARKERS)


def transcript_ends_sentence(text: Optional[str]) -> bool:
    """True when the transcript ends at a sentence boundary."""
    stripped = (text or "").strip().rstrip(_TRAILING_CLOSERS).strip()
    return bool(stripped) and stripped[-1] in SENTENCE_TERMINALS


def _tail_word_count(text: Optional[str]) -> int:
    """Word count of the last clause/sentence of ``text``.

    Chinese transcripts have no spaces; a CJK character counts as one word so
    the threshold behaves consistently across languages.
    """
    stripped = (text or "").strip().rstrip(_TRAILING_CLOSERS).strip()
    if not stripped:
        return 0
    tail = stripped
    for splitter in SENTENCE_TERMINALS + ",，、":
        if splitter in tail:
            tail = tail.rsplit(splitter, 1)[-1]
    latin_words = [token for token in tail.split() if any(char.isalnum() for char in token)]
    if latin_words:
        return len(latin_words)
    return sum(1 for char in tail if char.isalnum())


@dataclass(frozen=True)
class WaitGateDecision:
    active: bool
    hold: bool
    reason: str

    def metrics(self) -> dict[str, object]:
        return {
            "tfd_wait_gate_active": self.active,
            "tfd_wait_gate_hold": self.hold,
            "tfd_wait_gate_reason": self.reason,
        }


@dataclass(frozen=True)
class WaitGateConfig:
    """Frame-level thresholds for the gate.

    ``min_silence_frames`` defaults to **0**, i.e. no additional quiet
    requirement.  This is a deliberate, evidence-driven choice: the CFDC
    runner (``114_dsb_stereo_capture.record_episode``) stops pushing audio after
    the last frame and only *sleeps* through ``drain_seconds``.  The server
    therefore has no input event after the final chunk, so the final chunk is
    the **only** opportunity to release the turn.  Requiring extra quiet frames
    on top of the controller's existing ``min_silence_frames`` (4) would risk
    never releasing at all and would turn currently-passing cases into
    ``no_service``.  Raise it only with a runner that pushes tail silence.

    ``tail_word_min`` is an *additional* hold condition and defaults to 0,
    i.e. disabled.  It exists so a future run can require a minimum tail length
    without editing code; it is not part of the E42a pre-registered behaviour.
    """

    min_silence_frames: int = 0
    tail_word_min: int = 0

    @classmethod
    def from_env(cls) -> "WaitGateConfig":
        return cls(
            min_silence_frames=_read_int(ENV_MIN_SILENCE_FRAMES, 0),
            tail_word_min=_read_int(ENV_TAIL_WORD_MIN, 0),
        )


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def evaluate_wait_gate(
    *,
    wait_mode: bool,
    transcript: Optional[str],
    trailing_silence_frames: int,
    config: Optional[WaitGateConfig] = None,
) -> WaitGateDecision:
    """Hold the turn while the caller's utterance is not semantically complete.

    ``hold=True`` means "keep listening, do not emit audio now".  The gate is
    only consulted when ``wait_mode`` is True; otherwise it returns an inactive
    decision that must be ignored by the caller.
    """
    cfg = config or WaitGateConfig()
    if not wait_mode:
        return WaitGateDecision(active=False, hold=False, reason=REASON_INACTIVE)

    quiet_frames = int(trailing_silence_frames)
    if quiet_frames < cfg.min_silence_frames:
        return WaitGateDecision(
            active=True, hold=True, reason=REASON_HOLD_INSUFFICIENT_QUIET
        )

    if not (transcript or "").strip():
        # No usable transcript: either ASR found no speech or it failed.  Under
        # WAIT semantics the safe action is to keep listening.
        return WaitGateDecision(
            active=True, hold=True, reason=REASON_HOLD_NO_TRANSCRIPT
        )

    if not transcript_ends_sentence(transcript):
        return WaitGateDecision(
            active=True, hold=True, reason=REASON_HOLD_INCOMPLETE
        )

    if cfg.tail_word_min > 0 and _tail_word_count(transcript) < cfg.tail_word_min:
        return WaitGateDecision(active=True, hold=True, reason=REASON_HOLD_SHORT_TAIL)

    return WaitGateDecision(active=True, hold=False, reason=REASON_RELEASE_COMPLETE)


def markers_found(system_prompt: Optional[str], markers: Iterable[str]) -> tuple[str, ...]:
    """Diagnostic helper: which markers occur in ``system_prompt``."""
    lowered = (system_prompt or "").lower()
    return tuple(marker for marker in markers if marker in lowered)
