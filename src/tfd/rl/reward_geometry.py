"""E46 reward diagnostics; a hand-built group never proves model exploration.

wait_v2 is a text-content proxy candidate, not a duplex timing reward. It keeps
WAIT_HOLD binary so stylistic differences in spoken failures cannot masquerade
as progress on the silence action. GAP_CARRY uses the historical reward.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from .dsb_turn_reward import dsb_turn_reward
from .rewards import grpo_advantage

PROFILES = ("legacy", "wait_v2")
REQUIRED_FAMILIES = ("WAIT_HOLD", "RESUME_ANSWER")
MIN_REWARD_STD = 0.02
MIN_SIGNAL_FRACTION = 0.5
STOPWORDS = frozenset("a an the i you my your is are was were be and or to of in on it that this have has will would can could please me let sure ok okay got noted thanks thank".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 1 and w not in STOPWORDS}


def reward_components(text: str, row: dict) -> dict:
    """Continuous relevance is a disclosed lexical proxy, not semantic truth."""
    gen = text.strip()
    legacy = dsb_turn_reward(gen, row)
    salience = (row.get("salience_text") or "").strip()
    context = " ".join(str(m.get("content", "")) for m in row.get("context", [])
                       if m.get("role") == "user").strip()
    norm = lambda t: " ".join(re.findall(r"\w+", t.lower()))
    copied = bool(gen) and any(norm(gen) == norm(t) for t in (context, salience) if t)
    target, generated = _words(salience), _words(gen)
    quality = len(target & generated) / len(target | generated) if target and generated else 0.0
    if copied:
        quality = 0.0
    return {"legacy": legacy, "context_quality": quality,
            "spoke": bool(gen), "verbatim_copy": copied}


def score_text(text: str, row: dict, profile: str = "legacy") -> float:
    if profile not in PROFILES:
        raise ValueError(f"unknown reward profile: {profile}")
    if not isinstance(text, str):
        raise ValueError("completion must be a string")
    if profile == "legacy" or row.get("row_kind") != "RESUME_ANSWER":
        return dsb_turn_reward(text, row)
    parts = reward_components(text, row)
    if not parts["spoke"]:
        return 0.0
    return 0.7 * parts["legacy"] + 0.3 * parts["context_quality"]


def group_summary(row: dict, texts: list[str], profile: str = "legacy") -> dict:
    if len(texts) < 2 or any(not isinstance(t, str) for t in texts):
        raise ValueError("group needs at least two string completions")
    rewards = [score_text(t, row, profile) for t in texts]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    advantages = grpo_advantage(rewards)
    speak_count = sum(bool(t.strip()) for t in texts)
    return {"reward_mean": mean, "reward_std": std,
            "reward_spread": max(rewards) - min(rewards),
            "rewards": rewards, "nonzero_adv": sum(abs(a) > 1e-8 for a in advantages),
            "informative": std >= MIN_REWARD_STD,
            "behavior_mixed": 0 < speak_count < len(texts),
            "speak_count": speak_count, "group_size": len(texts),
            "unique_texts": len(set(t.strip() for t in texts)),
            "flat_at_zero": max(rewards) == 0,
            "flat_at_one": min(rewards) == 1}


def audit_rollouts(rows: list[dict], records: list[dict], profile: str = "legacy",
                   sampling: dict | None = None, min_groups: int = 12) -> dict:
    if min_groups < 1:
        raise ValueError("min_groups must be positive")
    empirical = bool(sampling and sampling.get("model_run_performed") is True)
    if empirical:
        if (sampling.get("mode") != "collect_only" or not sampling.get("model_identity")
                or not re.fullmatch(r"[0-9a-f]{64}", sampling.get("source_sha256", ""))
                or not isinstance(sampling.get("seed"), int)
                or not isinstance(sampling.get("group_size"), int)
                or sampling["group_size"] < 2
                or not math.isfinite(float(sampling.get("temperature", 0)))
                or float(sampling.get("temperature", 0)) <= 0):
            raise ValueError("real rollout sampling provenance is incomplete")
    groups, seen, by_family = [], set(), defaultdict(list)
    for record in records:
        index = record.get("row_index")
        if type(index) is not int or not 0 <= index < len(rows) or index in seen:
            raise ValueError("invalid/duplicate row_index in rollout record")
        seen.add(index)
        texts = record.get("texts")
        if not isinstance(texts, list) or (empirical and len(texts) != sampling["group_size"]):
            raise ValueError("rollout group size does not match provenance")
        row = rows[index]
        if (record.get("row_id", row.get("id")) != row.get("id")
                or record.get("row_kind", row.get("row_kind")) != row.get("row_kind")):
            raise ValueError("rollout row identity does not match source row")
        stat = group_summary(row, texts, profile)
        family = row.get("row_kind", "GAP_CARRY")
        stat.update(row_index=index, row_id=row.get("id"), row_kind=family)
        groups.append(stat)
        by_family[family].append(stat)
    families = {}
    for family in sorted(set(by_family) | set(REQUIRED_FAMILIES)):
        stats = by_family[family]
        n = len(stats)
        informative = sum(s["informative"] for s in stats)
        behavior = sum(s["behavior_mixed"] for s in stats)
        families[family] = {"groups": n, "signal_groups": informative,
                            "signal_fraction": informative / n if n else None,
                            "behavior_mixed_groups": behavior,
                            "behavior_mixed_fraction": behavior / n if n else None,
                            "flat_zero_groups": sum(s["flat_at_zero"] for s in stats),
                            "flat_one_groups": sum(s["flat_at_one"] for s in stats)}
    fraction = sum(s["informative"] for s in groups) / len(groups) if groups else None
    passed = empirical and all(
        families[f]["groups"] >= min_groups
        and families[f]["signal_fraction"] >= MIN_SIGNAL_FRACTION
        and families[f]["behavior_mixed_fraction"] >= MIN_SIGNAL_FRACTION
        for f in REQUIRED_FAMILIES)
    return {"schema": "tfd.e46.reward_geometry.v1", "reward_profile": profile,
            "groups": groups, "families": families, "model_run_performed": empirical,
            "empirical_signal_fraction": fraction if empirical else None,
            "synthetic_probe_signal_fraction": fraction if not empirical else None,
            "sampling_gate_passed": bool(passed), "sampling": sampling,
            "criteria": {"min_groups_per_family": min_groups,
                         "min_reward_std": MIN_REWARD_STD,
                         "min_signal_fraction_per_family": MIN_SIGNAL_FRACTION,
                         "min_behavior_mixed_fraction_per_family": MIN_SIGNAL_FRACTION},
            "training_objective_ready": False, "gpu_training_ready": False,
            "blockers": (["real_model_rollouts_missing"] if not empirical else [])
                        + (["sampling_gate_failed"] if not passed else [])
                        + ["legacy_signed_kl_and_eos_mask_require_separate_validation"],
            "claim_scope": "reward feasibility only; no audio timing/WAIT improvement claim"}


def regularizer_identity_probe() -> dict:
    """A tiny fixed-sample gradient check; no trained model or GPU needed."""
    import torch

    logits = torch.tensor([0.8, -0.2], requires_grad=True)
    logp = logits.log_softmax(dim=0)[0]
    ref = logp.detach()
    legacy = logp - ref
    legacy_grad = torch.autograd.grad(legacy, logits, retain_graph=True)[0]
    delta = ref - logp
    token_k3 = torch.exp(delta) - delta - 1
    k3_grad = torch.autograd.grad(token_k3, logits)[0]
    return {"policy_equals_reference": True, "legacy_value": legacy.item(),
            "legacy_gradient_norm": legacy_grad.norm().item(),
            "token_k3_value": token_k3.item(), "token_k3_gradient_norm": k3_grad.norm().item(),
            "claim_scope": "fixed-sample numerical counterexample; not a new training result"}
