"""E37a: rule-based reward for the DSB turn-action GRPO family.

The frozen 2.0 GRPO LoRA was trained on handwritten text flows with a
BLEU-to-reference reward (33_prep_bargein_rl_data.py). DSB gap cases have no
human reference ("what should the assistant say in the gap" is open), so the
E37a reward is rule-based and pre-registered here:

  action_fit   the core gap_talk signal.
                 speak-expected rows (TAKE_TURN/READBACK/BACKCHANNEL/
                 INTERRUPT): 1.0 if the generation is non-empty else 0.0.
                 silence-expected rows (LISTEN/NO_BACKCHANNEL): 1.0 if the
                 generation is empty/whitespace else 0.0.
  brevity      a gap ack must be short. speak rows: linear ramp from the
                 per-action char limit to zero at 2x the limit. silence
                 rows: 1.0 when empty (redundant with action_fit but keeps
                 the weight semantics uniform).
  salience     READBACK rows only: character-bigram Jaccard between the
                 generation and the salience_text (last 8 words of the
                 caller's pre-trigger speech -- the detail to confirm).
                 Reuses tfd.eval.metrics.bigram_overlap, the same metric
                 family as the barge-in context_recall closure. Empty
                 salience_text contributes 0.0 (and callers may drop the
                 weight; the default weights below keep it at 0 for rows
                 without salience by construction).

Weights (pre-registered defaults): action_fit 1.0, brevity 0.3, salience 0.3.
The total is a weighted mean, so it stays in [0, 1].
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {"action_fit": 1.0, "brevity": 0.3, "salience": 0.3}

SPEAK_EXPECTED = ("TAKE_TURN", "READBACK", "BACKCHANNEL", "INTERRUPT")
SILENCE_EXPECTED = ("LISTEN", "NO_BACKCHANNEL")

# per-action brevity char limits (gap utterances are one short sentence)
BREVITY_LIMITS = {
    "BACKCHANNEL": 40,
    "READBACK": 120,
    "TAKE_TURN": 160,
    "INTERRUPT": 160,
}


def dsb_turn_reward(gen_text: str, row: dict,
                    weights: dict = None) -> float:
    """Reward a generation for one dsb_turn_action row (value in [0, 1])."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()
                  if k in DEFAULT_WEIGHTS})
    gen = (gen_text or "").strip()
    action = row.get("expected_action", "")
    speak_expected = action in SPEAK_EXPECTED
    spoke = 1.0 if gen else 0.0

    action_fit = spoke if speak_expected else (1.0 - spoke)

    if speak_expected:
        limit = BREVITY_LIMITS.get(action, 160)
        overshoot = max(0, len(gen) - limit)
        brevity = max(0.0, 1.0 - overshoot / max(limit, 1))
    else:
        brevity = 1.0 if not gen else 0.0

    salience_text = (row.get("salience_text") or "").strip()
    salience_applies = (speak_expected and action == "READBACK"
                        and salience_text)
    if salience_applies:
        from tfd.eval.metrics import bigram_overlap
        salience = bigram_overlap(gen, salience_text)
    else:
        salience = 0.0

    # only weights whose term is actually in effect enter the denominator,
    # so a clean silence row (no salience possible) can still reach 1.0
    total_w = w["action_fit"] + w["brevity"] + (w["salience"] if salience_applies
                                                else 0.0)
    if total_w <= 0:
        return 0.0
    return (w["action_fit"] * action_fit
            + w["brevity"] * brevity
            + (w["salience"] * salience if salience_applies else 0.0)) / total_w
