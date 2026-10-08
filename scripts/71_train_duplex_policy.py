#!/usr/bin/env python3
"""Train the TFD-STAR 3.0 causal sidecar policy."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bootstrap import load_config, resolve
from tfd.duplex_policy.data import collate_duplex_sequences
from tfd.duplex_policy.feature_store import dialogue_ids, load_sequence_dataset, validate_split_manifest
from tfd.duplex_policy.losses import DuplexLossWeights
from tfd.duplex_policy.model import CausalDuplexPolicy, DuplexPolicyConfig
from tfd.duplex_policy.trainer import DuplexPolicyTrainer, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _subset(dataset, allowed: set[str]):
    indices = [index for index, value in enumerate(dialogue_ids(dataset)) if value in allowed]
    if not indices:
        raise ValueError("requested split is empty")
    return Subset(dataset, indices)


def main() -> int:
    args = parse_args()
    cfg = load_config("duplex_policy")
    train_cfg = cfg["training"]
    seed = int(args.seed if args.seed is not None else train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_path = args.data or Path(resolve(cfg["paths"]["smoke_data"]))
    split_path = args.splits or Path(resolve(cfg["paths"]["split_manifest"]))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    dataset = load_sequence_dataset(data_path)
    validate_split_manifest(dataset, split_payload, data_path)
    print(
        f"[data] resolved feature dimensions: acoustic={dataset.acoustic_dim}, "
        f"semantic={dataset.semantic_dim}"
    )
    train_ds = _subset(dataset, set(split_payload["splits"]["train"]))
    val_ds = _subset(dataset, set(split_payload["splits"]["val"]))
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_duplex_sequences,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_duplex_sequences,
    )

    model_cfg = DuplexPolicyConfig(
        acoustic_dim=dataset.acoustic_dim,
        semantic_dim=dataset.semantic_dim,
        max_frames=int(cfg["clock"]["max_frames"]),
        **cfg["model"],
    )
    model = CausalDuplexPolicy(model_cfg)
    weights = DuplexLossWeights(**train_cfg["loss_weights"])
    trainer = DuplexPolicyTrainer(
        model,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        grad_clip=float(train_cfg["grad_clip"]),
        device=args.device,
        loss_weights=weights,
    )

    output_dir = args.output_dir or Path(resolve(cfg["paths"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or int(train_cfg["epochs"])
    history = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} action_acc={val_metrics['action_accuracy']:.4f}"
        )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=trainer.optimizer,
            epoch=epoch,
            metrics=val_metrics,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                metrics=val_metrics,
            )
    report = {
        "format": "tfd-star-duplex-policy-train-report-v3",
        "data": str(data_path.resolve()),
        "splits": str(split_path.resolve()),
        "synthetic_smoke_only": bool(split_payload.get("synthetic_smoke_only")),
        "weak_label_only": bool(split_payload.get("weak_label_only")),
        "scientific_evidence_eligible": bool(
            split_payload.get("scientific_evidence_eligible", False)
        ),
        "seed": seed,
        "history": history,
        "best_val_loss": best_loss,
        "claim_boundary": (
            "Training mechanics only; results are not research evidence."
            if not split_payload.get("scientific_evidence_eligible", False)
            else "Eligible for the preregistered real-data evaluation protocol."
        ),
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] checkpoints and report -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
