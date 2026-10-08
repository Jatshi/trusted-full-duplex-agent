"""Metrics that expose safety/latency trade-offs instead of one accuracy."""
from __future__ import annotations

import numpy as np


def risk_coverage_curve(confidence: np.ndarray, errors: np.ndarray) -> list[tuple[float, float]]:
    """Return selective risk at every retained-coverage operating point."""
    conf = np.asarray(confidence, dtype=np.float64)
    err = np.asarray(errors, dtype=np.float64)
    if conf.ndim != 1 or err.shape != conf.shape or len(conf) == 0:
        raise ValueError("confidence and errors must be non-empty equal-length vectors")
    if not np.isfinite(conf).all() or not np.isfinite(err).all():
        raise ValueError("confidence/errors must be finite")
    order = np.argsort(-conf, kind="stable")
    ordered_errors = err[order]
    cumulative_risk = np.cumsum(ordered_errors) / np.arange(1, len(err) + 1)
    coverage = np.arange(1, len(err) + 1) / len(err)
    return [(float(c), float(r)) for c, r in zip(coverage, cumulative_risk)]


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or y.shape != (p.shape[0],) or bins <= 0:
        raise ValueError("invalid probabilities, labels or bin count")
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence > lo) & (confidence <= hi)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or y.shape != (p.shape[0],):
        raise ValueError("invalid probabilities or labels")
    target = np.eye(p.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum((p - target) ** 2, axis=1)))


def event_timing_metrics(
    predicted_event_index: np.ndarray,
    target_event_index: np.ndarray,
    *,
    frame_ms: int,
    tolerance_ms: float = 320.0,
) -> dict[str, float | int | None]:
    """Score time-to-event predictions without hiding abstention failures.

    ``-1`` denotes a right-censored sequence (no event). Latency is signed:
    negative values are early actions and positive values are late actions.
    False-event and missed-event rates use their respective eligible subsets as
    denominators, so a model cannot improve latency by firing on everything.
    """
    predicted = np.asarray(predicted_event_index, dtype=np.int64)
    target = np.asarray(target_event_index, dtype=np.int64)
    if predicted.ndim != 1 or predicted.shape != target.shape or not len(predicted):
        raise ValueError("event indices must be non-empty equal-length vectors")
    if frame_ms <= 0 or tolerance_ms < 0:
        raise ValueError("frame_ms must be positive and tolerance_ms non-negative")
    if (predicted < -1).any() or (target < -1).any():
        raise ValueError("event indices must be -1 or non-negative")

    predicted_event = predicted >= 0
    target_event = target >= 0
    matched = predicted_event & target_event
    false_events = predicted_event & ~target_event
    missed_events = ~predicted_event & target_event
    target_absent = ~target_event
    latency = (predicted[matched] - target[matched]).astype(np.float64) * frame_ms
    absolute = np.abs(latency)

    def _rate(mask: np.ndarray, eligible: np.ndarray) -> float:
        denominator = int(eligible.sum())
        return float(mask.sum() / denominator) if denominator else 0.0

    return {
        "num_sequences": int(len(target)),
        "target_events": int(target_event.sum()),
        "matched_events": int(matched.sum()),
        "false_events": int(false_events.sum()),
        "missed_events": int(missed_events.sum()),
        "false_event_rate": _rate(false_events, target_absent),
        "missed_event_rate": _rate(missed_events, target_event),
        "within_tolerance_rate": (
            float(np.mean(absolute <= tolerance_ms)) if len(absolute) else 0.0
        ),
        "latency_ms_mean": float(np.mean(latency)) if len(latency) else None,
        "latency_ms_p50": float(np.median(latency)) if len(latency) else None,
        "absolute_latency_ms_p95": (
            float(np.quantile(absolute, 0.95)) if len(absolute) else None
        ),
    }
