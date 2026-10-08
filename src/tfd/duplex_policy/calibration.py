"""Post-hoc calibration tools kept separate from policy training data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_probabilities(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C] with N>0 and C>=2")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = p.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("each probability row must have positive mass")
    return p / sums


@dataclass
class SplitConformalClassifier:
    """Simple finite-sample split conformal classifier.

    Non-conformity is ``1 - p(true_class)``. With ``group_ids``, each dialogue
    contributes its maximum frame score as one exchangeable calibration unit,
    yielding a conservative simultaneous-within-group target. The group should
    combine speaker, conversation and augmentation family as appropriate.
    """

    alpha: float = 0.1
    qhat: float | None = None
    num_classes: int | None = None
    calibration_unit: str = "frame"
    num_calibration_units: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")

    def fit(self, probabilities: np.ndarray, labels: np.ndarray,
            group_ids: list[str] | np.ndarray | None = None) -> "SplitConformalClassifier":
        p = _validate_probabilities(probabilities)
        y = np.asarray(labels, dtype=np.int64)
        if y.shape != (p.shape[0],) or (y < 0).any() or (y >= p.shape[1]).any():
            raise ValueError("labels must be valid class indices with shape [N]")
        scores = 1.0 - p[np.arange(len(y)), y]
        if group_ids is not None:
            groups = np.asarray(group_ids, dtype=str)
            if groups.shape != (len(scores),) or any(not group for group in groups):
                raise ValueError("group_ids must be nonempty IDs aligned with frames")
            scores = np.asarray([scores[groups == group].max() for group in np.unique(groups)])
            self.calibration_unit = "leakage_group_max"
        else:
            self.calibration_unit = "frame"
        self.num_calibration_units = int(len(scores))
        rank = int(np.ceil((len(scores) + 1) * (1.0 - self.alpha)))
        # If rank exceeds n, the finite-sample conformal threshold is +inf;
        # scores live in [0,1], so qhat=1 includes all classes.
        self.qhat = 1.0 if rank > len(scores) else float(np.sort(scores)[max(rank, 1) - 1])
        self.num_classes = int(p.shape[1])
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[set[int]]:
        if self.qhat is None or self.num_classes is None:
            raise RuntimeError("conformal classifier must be fitted first")
        p = _validate_probabilities(probabilities)
        if p.shape[1] != self.num_classes:
            raise ValueError("class dimension differs from calibration data")
        out: list[set[int]] = []
        for row in p:
            members = set(np.flatnonzero((1.0 - row) <= self.qhat).tolist())
            if not members:
                members.add(int(np.argmax(row)))
            out.append(members)
        return out

    def state_dict(self) -> dict:
        if self.qhat is None or self.num_classes is None:
            raise RuntimeError("cannot serialize an unfitted calibrator")
        return {
            "method": "split_conformal_threshold",
            "alpha": self.alpha,
            "qhat": self.qhat,
            "num_classes": self.num_classes,
            "calibration_unit": self.calibration_unit,
            "num_calibration_units": self.num_calibration_units,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "SplitConformalClassifier":
        obj = cls(alpha=float(payload["alpha"]))
        obj.qhat = float(payload["qhat"])
        obj.num_classes = int(payload["num_classes"])
        obj.calibration_unit = str(payload.get("calibration_unit", "frame"))
        obj.num_calibration_units = payload.get("num_calibration_units")
        return obj
