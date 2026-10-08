"""TFD-STAR 3.0 causal duplex-policy tests.

These tests deliberately use tiny tensors so the core research claims can be
checked on CPU before any AutoDL time is purchased.
"""
from __future__ import annotations

import json
import math
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_model_shapes_and_strict_causality():
    from tfd.duplex_policy.model import CausalDuplexPolicy, DuplexPolicyConfig

    torch.manual_seed(7)
    cfg = DuplexPolicyConfig(
        acoustic_dim=4,
        semantic_dim=6,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        num_actions=7,
        num_risk_levels=3,
        dropout=0.0,
    )
    model = CausalDuplexPolicy(cfg).eval()
    acoustic = torch.randn(2, 8, 4)
    semantic = torch.randn(2, 8, 6)
    valid = torch.ones(2, 8, dtype=torch.bool)

    with torch.no_grad():
        full = model(acoustic, semantic, valid)
        changed_acoustic = acoustic.clone()
        changed_semantic = semantic.clone()
        changed_acoustic[:, 5:] = torch.randn_like(changed_acoustic[:, 5:]) * 20
        changed_semantic[:, 5:] = torch.randn_like(changed_semantic[:, 5:]) * 20
        changed = model(changed_acoustic, changed_semantic, valid)

    assert full.action_logits.shape == (2, 8, 7)
    assert full.take_hazard_logits.shape == (2, 8)
    assert full.yield_hazard_logits.shape == (2, 8)
    assert full.risk_alpha.shape == (2, 8, 3)
    assert torch.all(full.risk_alpha > 1.0)
    # Changing future frames must not alter any prefix representation/output.
    assert torch.allclose(full.action_logits[:, :5], changed.action_logits[:, :5], atol=1e-6)
    assert torch.allclose(full.hidden[:, :5], changed.hidden[:, :5], atol=1e-6)


def test_discrete_hazard_nll_prefers_correct_event_time_and_handles_censoring():
    from tfd.duplex_policy.losses import discrete_time_hazard_nll

    valid = torch.ones(2, 5, dtype=torch.bool)
    event_index = torch.tensor([2, -1])  # second sequence is right-censored
    good = torch.tensor([
        [-5.0, -5.0, 5.0, -5.0, -5.0],
        [-5.0, -5.0, -5.0, -5.0, -5.0],
    ])
    bad = torch.tensor([
        [5.0, -5.0, -5.0, -5.0, -5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
    ])
    assert discrete_time_hazard_nll(good, event_index, valid) < 0.05
    assert discrete_time_hazard_nll(good, event_index, valid) < discrete_time_hazard_nll(
        bad, event_index, valid
    )


def test_prefix_consistency_is_zero_for_unchanged_beliefs():
    from tfd.duplex_policy.losses import prefix_consistency_kl

    logits = torch.tensor([[[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    assert prefix_consistency_kl(logits, valid).item() == pytest.approx(0.0, abs=1e-7)


def test_split_conformal_prediction_sets_are_nonempty_and_cover_calibration_labels():
    from tfd.duplex_policy.calibration import SplitConformalClassifier

    probs = np.asarray(
        [
            [0.90, 0.08, 0.02],
            [0.15, 0.75, 0.10],
            [0.05, 0.10, 0.85],
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 2, 0, 1])
    cal = SplitConformalClassifier(alpha=0.2).fit(probs, labels)
    sets = cal.predict_sets(probs)
    assert all(len(s) >= 1 for s in sets)
    empirical = np.mean([int(y in s) for y, s in zip(labels, sets)])
    assert empirical >= 0.8
    payload = cal.state_dict()
    restored = SplitConformalClassifier.from_state_dict(payload)
    assert restored.predict_sets(probs) == sets


def test_conformal_uses_independent_dialogues_not_correlated_frames():
    from tfd.duplex_policy.calibration import SplitConformalClassifier

    scores = np.asarray([0.1] * 99 + [0.9, 0.2])
    probs = np.stack([1.0 - scores, scores], axis=1)
    labels = np.zeros(len(scores), dtype=np.int64)
    per_frame = SplitConformalClassifier(alpha=0.5).fit(probs, labels)
    grouped = SplitConformalClassifier(alpha=0.5).fit(
        probs, labels, group_ids=["a"] * 100 + ["b"]
    )
    assert per_frame.qhat == pytest.approx(0.1)
    assert grouped.qhat == pytest.approx(0.9)
    assert grouped.num_calibration_units == 2
    assert grouped.state_dict()["calibration_unit"] == "leakage_group_max"
    # At alpha below 1/(n+1), a finite-sample threshold cannot exclude classes.
    conservative = SplitConformalClassifier(alpha=0.1).fit(
        probs, labels, group_ids=["a"] * 100 + ["b"]
    )
    assert conservative.qhat == 1.0


def test_safety_shield_is_selective_under_risk_and_ood():
    from tfd.duplex_policy.actions import Action
    from tfd.duplex_policy.shield import RiskAwareActionShield

    shield = RiskAwareActionShield(high_risk=0.7, ood_threshold=0.8)
    assert shield.decide({Action.EXECUTE}, risk=0.1, ood_score=0.0) == Action.EXECUTE
    assert shield.decide({Action.EXECUTE, Action.CLARIFY}, risk=0.3, ood_score=0.0) == Action.CLARIFY
    assert shield.decide({Action.EXECUTE}, risk=0.9, ood_score=0.0) == Action.CLARIFY
    assert shield.decide({Action.EXECUTE, Action.STOP}, risk=0.9, ood_score=0.0) == Action.STOP
    assert shield.decide({Action.TAKE}, risk=0.0, ood_score=0.95) == Action.LISTEN


def _record(dialogue_id: str, n: int = 4) -> dict:
    return {
        "schema_version": "3.0",
        "dialogue_id": dialogue_id,
        "frame_ms": 160,
        "source": "unit-test",
        "license": "CC-BY-4.0",
        "acoustic": [[float(i), 0.0] for i in range(n)],
        "semantic": [[0.0, float(i), 1.0] for i in range(n)],
        "actions": ["listen"] * (n - 1) + ["take"],
        "risk_labels": [0] * n,
        "take_event_index": n - 1,
        "yield_event_index": None,
        "condition": {"snr_db": 20.0, "overlap": False},
    }


def test_jsonl_dataset_validation_padding_and_group_split(tmp_path: Path):
    from tfd.duplex_policy.data import (
        DuplexSequenceDataset,
        collate_duplex_sequences,
        grouped_dialogue_split,
    )

    path = tmp_path / "tiny.jsonl"
    rows = [_record(f"d{i}", n=3 + i % 3) for i in range(12)]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    ds = DuplexSequenceDataset.from_jsonl(path)
    batch = collate_duplex_sequences([ds[0], ds[1], ds[2]])
    assert batch.acoustic.shape == (3, 5, 2)
    assert batch.semantic.shape == (3, 5, 3)
    assert batch.valid_mask.sum().item() == 3 + 4 + 5
    assert batch.take_event_index.tolist() == [2, 3, 4]

    splits = grouped_dialogue_split([r["dialogue_id"] for r in rows], seed=42)
    assert set(splits) == {"train", "val", "calibration", "test"}
    assert not (set(splits["train"]) & set(splits["test"]))
    assert set().union(*map(set, splits.values())) == {r["dialogue_id"] for r in rows}

    from tfd.duplex_policy.data import grouped_entity_split

    pairs = [("a1", "speaker-a"), ("a2", "speaker-a"), ("b1", "speaker-b"),
             ("c1", "speaker-c"), ("d1", "speaker-d"), ("e1", "speaker-e")]
    entity_splits = grouped_entity_split(pairs, seed=42)
    locations = {
        dialogue_id: split
        for split, ids in entity_splits.items()
        for dialogue_id in ids
    }
    assert locations["a1"] == locations["a2"]


def test_dataset_rejects_dimension_drift(tmp_path: Path):
    from tfd.duplex_policy.data import DuplexSequenceDataset

    bad = _record("bad")
    bad["semantic"][2] = [1.0, 2.0]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic feature dimension"):
        DuplexSequenceDataset.from_jsonl(path)


def test_split_manifest_rejects_stale_data_and_cross_split_overlap(tmp_path: Path):
    from tfd.duplex_policy.feature_store import (
        load_sequence_dataset, sha256_file, validate_split_manifest,
    )

    path = tmp_path / "records.jsonl"
    rows = [_record(f"d{i}") for i in range(4)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data = load_sequence_dataset(path)
    payload = {"source_sha256": sha256_file(path), "splits": {
        "train": ["d0"], "val": ["d1"], "calibration": ["d2"], "test": ["d3"],
    }}
    validate_split_manifest(data, payload, path)
    with pytest.raises(ValueError, match="overlapping"):
        validate_split_manifest(data, {**payload, "splits": {
            **payload["splits"], "test": ["d0"]}}, path)
    grouped_rows = [dict(row) for row in rows]
    grouped_rows[1]["condition"] = {**grouped_rows[1]["condition"], "split_group": "same"}
    grouped_rows[2]["condition"] = {**grouped_rows[2]["condition"], "split_group": "same"}
    grouped_path = tmp_path / "grouped.jsonl"
    grouped_path.write_text("".join(json.dumps(row) + "\n" for row in grouped_rows), encoding="utf-8")
    grouped_data = load_sequence_dataset(grouped_path)
    with pytest.raises(ValueError, match="leaks across"):
        validate_split_manifest(grouped_data, {"splits": payload["splits"]}, grouped_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        validate_split_manifest(data, payload, path)


def test_risk_coverage_curve_orders_by_confidence():
    from tfd.duplex_policy.metrics import risk_coverage_curve

    confidence = np.asarray([0.95, 0.80, 0.55, 0.20])
    errors = np.asarray([0.0, 0.0, 1.0, 1.0])
    curve = risk_coverage_curve(confidence, errors)
    assert curve[0][0] == pytest.approx(0.25)
    assert curve[0][1] == pytest.approx(0.0)
    assert curve[-1][0] == pytest.approx(1.0)
    assert curve[-1][1] == pytest.approx(0.5)
    assert all(math.isfinite(x) for pair in curve for x in pair)


def test_synthetic_generator_is_deterministic_and_schema_valid(tmp_path: Path):
    from tfd.duplex_policy.data import DuplexSequenceDataset
    from tfd.duplex_policy.synthetic import generate_synthetic_records, write_jsonl

    first = generate_synthetic_records(8, seed=17, acoustic_dim=8, semantic_dim=12)
    second = generate_synthetic_records(8, seed=17, acoustic_dim=8, semantic_dim=12)
    assert first == second
    assert {row["scenario"] for row in first} >= {"complete", "hesitation", "risky"}
    path = tmp_path / "synthetic.jsonl"
    write_jsonl(first, path)
    ds = DuplexSequenceDataset.from_jsonl(path)
    assert len(ds) == 8
    assert ds.acoustic_dim == 8 and ds.semantic_dim == 12


def test_train_epoch_and_checkpoint_round_trip(tmp_path: Path):
    from torch.utils.data import DataLoader

    from tfd.duplex_policy.data import (
        DuplexSequenceDataset,
        collate_duplex_sequences,
    )
    from tfd.duplex_policy.model import CausalDuplexPolicy, DuplexPolicyConfig
    from tfd.duplex_policy.synthetic import generate_synthetic_records, write_jsonl
    from tfd.duplex_policy.trainer import DuplexPolicyTrainer, load_checkpoint, save_checkpoint

    torch.manual_seed(3)
    path = tmp_path / "train.jsonl"
    write_jsonl(
        generate_synthetic_records(12, seed=3, acoustic_dim=4, semantic_dim=6), path
    )
    ds = DuplexSequenceDataset.from_jsonl(path)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_duplex_sequences)
    cfg = DuplexPolicyConfig(
        acoustic_dim=4,
        semantic_dim=6,
        model_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = CausalDuplexPolicy(cfg)
    trainer = DuplexPolicyTrainer(model, lr=1e-3, device="cpu")
    metrics = trainer.train_epoch(loader)
    assert math.isfinite(metrics["loss"]) and metrics["loss"] > 0

    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(checkpoint, model=model, optimizer=trainer.optimizer, epoch=1, metrics=metrics)
    restored, payload = load_checkpoint(checkpoint, map_location="cpu")
    assert payload["epoch"] == 1
    with torch.no_grad():
        for left, right in zip(model.parameters(), restored.parameters()):
            assert torch.equal(left, right)


def test_event_timing_metrics_separate_false_missed_and_latency():
    from tfd.duplex_policy.metrics import event_timing_metrics

    # d0/d1 are matched, d2 is a false takeover, d3 is a missed takeover.
    predicted = np.asarray([5, 8, 3, -1, -1])
    target = np.asarray([6, 8, -1, 4, -1])
    report = event_timing_metrics(predicted, target, frame_ms=160, tolerance_ms=200)
    assert report["matched_events"] == 2
    assert report["false_event_rate"] == pytest.approx(0.5)
    assert report["missed_event_rate"] == pytest.approx(1 / 3)
    assert report["within_tolerance_rate"] == pytest.approx(1.0)
    assert report["latency_ms_p50"] == pytest.approx(-80.0)
    assert report["absolute_latency_ms_p95"] == pytest.approx(152.0)


def test_factorized_reward_and_constrained_group_advantages():
    from tfd.duplex_policy.rewards import (
        ConstrainedGRPOState,
        DuplexOutcome,
        factorized_reward,
        group_relative_advantages,
    )

    safe = factorized_reward(
        DuplexOutcome(
            expected_event=True,
            event_delay_ms=120.0,
            task_success=True,
            unsafe_execute=False,
            false_takeover=False,
            inappropriate_backchannel=False,
        )
    )
    unsafe = factorized_reward(
        DuplexOutcome(
            expected_event=True,
            event_delay_ms=120.0,
            task_success=True,
            unsafe_execute=True,
            false_takeover=False,
            inappropriate_backchannel=False,
        )
    )
    assert safe.total > unsafe.total
    assert unsafe.components["safety"] < 0

    rewards = torch.tensor([[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    costs = torch.tensor([[0.0, 0.5, 2.0], [0.0, 0.0, 0.0]])
    advantages = group_relative_advantages(rewards, costs, lagrange_multiplier=1.0)
    assert advantages.shape == rewards.shape
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-6)
    state = ConstrainedGRPOState(lagrange_multiplier=0.2, cost_budget=0.1, dual_lr=0.5)
    state.update(costs)
    assert state.lagrange_multiplier > 0.2


def test_stream_feature_tap_captures_audio_and_causal_hidden_states():
    from tfd.duplex_policy.feature_tap import StreamFeatureTap, align_token_features

    class FakeDecoder:
        def feed(self, embeds: torch.Tensor, return_logits: bool = False):
            hidden = embeds.unsqueeze(0) + 10.0
            logits = hidden[:, -1].sum(dim=-1, keepdim=True)
            return (logits, hidden) if return_logits else None

    decoder = FakeDecoder()
    tap = StreamFeatureTap(decoder)
    original_method = decoder.feed
    with tap:
        tap.arm()
        embeds = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        returned = decoder.feed(embeds, return_logits=True)
        capture = tap.pop()
        assert torch.equal(returned[1], embeds.unsqueeze(0) + 10.0)
    assert decoder.feed.__func__ is original_method.__func__
    assert capture.acoustic.shape == (5, 4)
    assert capture.semantic.shape == (5, 4)
    acoustic, semantic = align_token_features(
        capture.acoustic, capture.semantic, duration_ms=800, frame_ms=160
    )
    assert acoustic.shape == semantic.shape == (5, 4)


def test_streaming_alignment_does_not_expose_future_native_chunks():
    from tfd.duplex_policy.feature_tap import (
        StreamFeatureCapture,
        align_completed_chunks,
        causal_acoustic_frames,
    )

    captures = [
        StreamFeatureCapture(
            acoustic=torch.full((3, 4), value),
            semantic=torch.full((3, 5), value + 10),
        )
        for value in (1.0, 2.0, 3.0)
    ]
    fast, slow, available = align_completed_chunks(
        captures, chunk_ms=[1035, 1000, 1000], duration_ms=2500, frame_ms=160
    )
    assert fast.shape == (16, 4) and slow.shape == (16, 5)
    assert available[:6] == [False] * 6
    assert available[6:12] == [True] * 6
    assert torch.equal(fast[6], torch.ones(4))
    assert torch.equal(fast[11], torch.ones(4))
    assert torch.equal(fast[12], torch.full((4,), 2.0))
    assert torch.equal(fast[-1], torch.full((4,), 2.0))

    changed = list(captures)
    changed[1] = StreamFeatureCapture(
        acoustic=torch.full((3, 4), 999.0),
        semantic=torch.full((3, 5), 999.0),
    )
    fast_changed, slow_changed, _ = align_completed_chunks(
        changed, chunk_ms=[1035, 1000, 1000], duration_ms=2500, frame_ms=160
    )
    assert torch.equal(fast[:12], fast_changed[:12])
    assert torch.equal(slow[:12], slow_changed[:12])

    audio = np.linspace(-0.1, 0.1, 40000, dtype=np.float32)
    audio_changed = audio.copy()
    audio_changed[16000:] = 0.8
    raw = causal_acoustic_frames(audio, frame_ms=160)
    raw_changed = causal_acoustic_frames(audio_changed, frame_ms=160)
    assert torch.equal(raw[:6], raw_changed[:6])


def test_system_playback_state_is_frame_causal():
    from tfd.duplex_policy.feature_tap import system_activity_frames

    intervals = [{"start_ms": 300, "end_ms": 620}]
    activity = system_activity_frames(intervals, duration_ms=800, frame_ms=160)
    assert activity.flatten().tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    changed_future = intervals + [{"start_ms": 700, "end_ms": 800}]
    later = system_activity_frames(changed_future, duration_ms=800, frame_ms=160)
    assert torch.equal(activity[:4], later[:4])
    with pytest.raises(ValueError, match="outside recording"):
        system_activity_frames([{"start_ms": 0, "end_ms": 900}], duration_ms=800)


def test_production_dual_track_feature_dimensions_train_on_cpu():
    from tfd.duplex_policy.model import CausalDuplexPolicy, DuplexPolicyConfig

    model = CausalDuplexPolicy(DuplexPolicyConfig(
        acoustic_dim=4118, semantic_dim=4096, model_dim=16,
        num_heads=4, num_layers=1, dropout=0.0,
    ))
    acoustic = torch.randn(1, 4, 4118)
    semantic = torch.randn(1, 4, 4096)
    output = model(acoustic, semantic)
    assert output.action_logits.shape == (1, 4, 7)
    output.action_logits.mean().backward()
    assert model.acoustic_projection[1].weight.grad is not None
    assert model.semantic_projection[1].weight.grad is not None


def test_annotation_normalizer_rejects_out_of_recording_events():
    sys.path.insert(0, str(ROOT / "scripts"))
    script = ROOT / "scripts/74_normalize_duplex_annotations.py"
    spec = importlib.util.spec_from_file_location("tfd_normalize_v3", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    record = {
        "dialogue_id": "d0",
        "audio_path": "d0.wav",
        "duration_ms": 500,
        "source": "consented",
        "license": "research-consent",
        "action_intervals": [{"start_ms": 320, "end_ms": 500, "action": "take"}],
        "risk_intervals": [],
        "take_event_ms": 320,
    }
    assert module.normalize(record, 160)["take_event_index"] == 2
    assert module.normalize({**record, "audio_path": "..\\audio\\d0.wav"}, 160)["audio_path"] == "../audio/d0.wav"
    with pytest.raises(ValueError, match="event time"):
        module.normalize({**record, "take_event_ms": 700}, 160)
    with pytest.raises(ValueError, match="interval must satisfy"):
        module.normalize({**record, "action_intervals": [
            {"start_ms": 320, "end_ms": 700, "action": "take"}
        ]}, 160)


def test_annotation_audio_path_relocation_across_manifest_directories(tmp_path: Path):
    module = _load_numbered_script("74_normalize_duplex_annotations.py")
    audio_dir = tmp_path / "audio"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "interim"
    for directory in (audio_dir, raw_dir, output_dir):
        directory.mkdir()
    recording = audio_dir / "x.wav"
    recording.write_bytes(b"test")
    system = audio_dir / "system.wav"
    system.write_bytes(b"system")
    relocated = module.relocate_audio_path(
        {"audio_path": "../audio/x.wav", "system_audio_path": "../audio/system.wav"},
        input_dir=raw_dir, output_dir=output_dir
    )
    assert (output_dir / relocated["audio_path"]).resolve() == recording.resolve()
    assert (output_dir / relocated["system_audio_path"]).resolve() == system.resolve()
    with pytest.raises(FileNotFoundError):
        module.relocate_audio_path(
            {"audio_path": "../audio/missing.wav"}, input_dir=raw_dir, output_dir=output_dir
        )


def test_npz_feature_manifest_round_trip(tmp_path: Path):
    from tfd.duplex_policy.feature_store import FeatureManifestDataset, write_feature_record

    feature_dir = tmp_path / "features"
    manifest = tmp_path / "manifest.jsonl"
    actions = ["listen", "listen", "take"]
    write_feature_record(
        feature_dir=feature_dir,
        manifest_path=manifest,
        dialogue_id="dialogue-1",
        acoustic=np.ones((3, 4), dtype=np.float32),
        semantic=np.ones((3, 6), dtype=np.float32) * 2,
        actions=actions,
        risk_labels=[0, 0, 1],
        frame_ms=160,
        take_event_index=2,
        yield_event_index=None,
        source="unit-test",
        license_name="CC-BY-4.0",
        condition={"scenario": "complete"},
    )
    dataset = FeatureManifestDataset(manifest)
    item = dataset[0]
    assert item.dialogue_id == "dialogue-1"
    assert item.acoustic.shape == (3, 4)
    assert item.semantic.shape == (3, 6)
    assert item.actions.tolist() == [0, 0, 2]


def _load_numbered_script(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts" / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audio_inventory_and_independent_annotation_packets(tmp_path: Path):
    import wave

    inventory = _load_numbered_script("79_inventory_duplex_audio.py")
    packets = _load_numbered_script("80_prepare_duplex_annotation_packets.py")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sound = np.full(16000, 1000, dtype="<i2")
    for name in ("a.wav", "copy.WAV"):
        with wave.open(str(audio_dir / name), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(sound.tobytes())
    output = tmp_path / "inventory.csv"
    rows, errors = inventory.inventory(audio_dir, output)
    assert len(rows) == 2 and not errors
    assert len({row["sha256"] for row in rows}) == 1
    assert rows[0]["duration_ms"] == 1000
    assert "Duplicate audio copies: 1" in inventory.quality_report(rows, errors, audio_dir)
    packet = packets.prepare(rows, inventory_dir=output.parent, packet_dir=tmp_path, annotator="A")
    assert len(packet) == 2
    assert packet[0]["annotation_complete"] is False
    assert packet[0]["action_intervals"] is None


def test_double_annotation_comparison_flags_timing_and_missing_review():
    compare = _load_numbered_script("81_compare_duplex_annotations.py").compare
    base = {
        "dialogue_id": "d1", "audio_path": "d1.wav", "sha256": "hash",
        "duration_ms": 800, "source": "consented", "license": "research-consent",
        "annotation_complete": True, "action_intervals": [
            {"start_ms": 640, "end_ms": 800, "action": "take"}],
        "risk_intervals": [], "take_event_ms": 640, "yield_event_ms": None,
    }
    a = {**base, "annotator_id": "A"}
    b = {**base, "annotator_id": "B"}
    agreed = compare({"d1": a}, {"d1": b})
    assert agreed[0]["needs_adjudication"] is False
    late = {**b, "take_event_ms": 160}
    assert "take_event_ms_timing_disagreement" in compare({"d1": a}, {"d1": late})[0]["issues"]
    assert "incomplete_annotation" in compare({"d1": a}, {"d1": {**b, "annotation_complete": False}})[0]["issues"]


def test_annotation_audit_fails_closed_when_groups_cannot_be_split():
    audit = _load_numbered_script("78_audit_duplex_annotations.py").audit
    rows = [{"dialogue_id": f"d{i}", "split_group": "one-speaker", "source": "lab",
             "license": "consent", "condition": {"scientific_label": False}}
            for i in range(5)]
    report = audit(rows, min_dialogues=1)
    assert report["ready_for_gpu_extraction"] is False
    assert any("cannot make leakage-safe four-way split" in item for item in report["problems"])


def test_gpu_preflight_rejects_mislabeled_user_only_and_duplicate_tracks(tmp_path: Path):
    import hashlib
    import wave

    preflight = _load_numbered_script("82_preflight_duplex_gpu.py")
    sound = np.zeros(16000, dtype="<i2")
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(sound.tobytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "schema_version": "3.0-extraction", "dialogue_id": "one", "duration_ms": 1000,
        "frame_ms": 160, "audio_path": "voice.wav", "sha256": digest,
        "actions": ["listen"] * 7, "risk_labels": [0] * 7,
        "condition": {"scientific_label": False},
    }
    good = preflight.check([row], manifest_dir=tmp_path, mode="probe")
    assert good["ready_for_gpu_probe"] is True and good["expected_acoustic_dim"] == 4118
    claimed = preflight.check([{**row, "condition": {"scientific_label": True}}],
                              manifest_dir=tmp_path, mode="probe")
    assert claimed["ready_for_gpu_probe"] is False
    duplicate = preflight.check([{**row, "system_audio_path": "voice.wav",
                                  "system_active_intervals": []}],
                                manifest_dir=tmp_path, mode="probe")
    assert any("same file" in issue for issue in duplicate["problems"])


def test_native_tap_invariance_probe_detects_no_change_with_fake_decoder():
    compare = _load_numbered_script("83_probe_feature_tap_invariance.py").compare

    class Decoder:
        cache_length = 0

        def get_cache_length(self):
            return self.cache_length

        def feed(self, embeds, return_logits=False):
            self.cache_length += len(embeds)
            hidden = embeds.unsqueeze(0) + 10.0
            logits = hidden[:, -1].sum(-1, keepdim=True)
            return (logits, hidden) if return_logits else None

    class Duplex:
        decoder = Decoder()
        pending_logits = None

    class Model:
        duplex = Duplex()
        audio_past_key_values = object()

        def reset_session(self, **_kwargs):
            self.audio_past_key_values = None

        def duplex_prepare(self, **_kwargs):
            self.duplex.decoder.cache_length = 0
            self.duplex.pending_logits = None

        def duplex_prefill(self, audio_waveform):
            embeds = torch.tensor(audio_waveform, dtype=torch.float32).reshape(2, 4)
            self.duplex.pending_logits, _ = self.duplex.decoder.feed(embeds, return_logits=True)
            return {"success": True}

    report = compare(Model(), np.arange(8, dtype=np.float32))
    assert report["passed"] is True
    assert report["max_abs_logit_difference"] == 0.0
    assert report["plain_cache_length"] == report["tapped_cache_length"] == 2


def test_clipped_constrained_grpo_loss_is_finite_and_prefers_positive_candidate():
    from tfd.duplex_policy.grpo import clipped_grpo_loss

    new_log_probs = torch.log(torch.tensor([[0.55, 0.45], [0.40, 0.60]]))
    old_log_probs = torch.log(torch.full((2, 2), 0.5))
    advantages = torch.tensor([[1.0, -1.0], [-0.5, 0.5]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    loss, parts = clipped_grpo_loss(
        new_log_probs,
        old_log_probs,
        advantages,
        valid,
        reference_log_probs=old_log_probs,
        clip_epsilon=0.2,
        kl_coefficient=0.01,
    )
    assert torch.isfinite(loss)
    assert parts["policy_loss"].item() < 0
    assert parts["approx_kl"].item() >= 0
