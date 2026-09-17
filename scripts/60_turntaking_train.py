#!/usr/bin/env python3
"""60 · turn-taking 决策器：数据合成 + MLP 训练（CPU 可跑，无需 GPU）。

产出：
  data/turntaking/train_feat.npy/.npz 特征与标签缓存（可复现）
  outputs/turntaking/learned_mlp.pt  训练好的决策器权重
  outputs/turntaking/train_report.json 训练/验证精度

用法：
  python scripts/60_turntaking_train.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from bootstrap import load_config, ROOT
from tfd.turntaking import (
    synth_dataset, frame_labels, featurize_utterance,
    TrainableMLP, HOLD, TAKE, BACKCHANNEL,
)


def build_split(utts, frame_ms: int, label_delay_s: float):
    X, ys, metas = [], [], []
    for u in utts:
        feats, _dbs = featurize_utterance(u, frame_ms=frame_ms)
        labels = frame_labels(u, frame_ms=frame_ms, label_delay_s=label_delay_s)
        # 帧数由音频驱动（段级 round 累积的样本误差可致 ±1 帧），以特征为准对齐
        if len(labels) < len(feats):
            labels = labels + [HOLD] * (len(feats) - len(labels))
        labels = labels[:len(feats)]
        X.append(feats)
        ys.extend(labels)
        metas.append({
            "final_pause_start_ms": u.final_pause_start * 1000,
            "n_frames": len(labels),
        })
    return np.concatenate(X, 0), ys, metas


def main():
    cfg = load_config("turntaking")
    frame_ms = cfg["frame_ms"]
    d = cfg["data"]
    print("=" * 72)
    print("60 · turn-taking 决策器训练（合成流式决策点，CPU）")
    print("=" * 72)

    tr = synth_dataset(d["n_train"], seed=d["seed"])
    va = synth_dataset(d["n_val"], seed=d["seed"] + 1)
    print(f"话语数: train={len(tr)} val={len(va)}  (seed={d['seed']})")

    Xtr, ytr, _ = build_split(tr, frame_ms, cfg["label_delay_s"])
    Xva, yva, _ = build_split(va, frame_ms, cfg["label_delay_s"])
    from collections import Counter
    print(f"帧数: train={len(ytr)} val={len(yva)}")
    print(f"train 标签分布: {dict(Counter(ytr))}")

    s = cfg["strategies"]["learned"]
    mlp = TrainableMLP(hidden=s["hidden"], lr=s["lr"], seed=d["seed"])
    print(f"\n训练 MLP(8->{s['hidden']}->3): lr={s['lr']} epochs={s['epochs']} "
          f"class_weights={s['class_weights']}")
    losses = mlp.fit(Xtr, ytr, epochs=s["epochs"],
                     class_weights=s["class_weights"], verbose_every=100)

    acc_tr = mlp.accuracy(Xtr, ytr)
    acc_va = mlp.accuracy(Xva, yva)
    print(f"\n帧级精度: train={acc_tr:.3f}  val={acc_va:.3f}  "
          f"(loss {losses[0]:.3f} -> {losses[-1]:.3f})")

    # 每类召回（val，帧级无平滑）：take 召回直接决定漏接率
    strat = mlp.to_strategy(confirm_frames=1)
    per_class = {}
    for a in (HOLD, TAKE, BACKCHANNEL):
        idx = [i for i, y in enumerate(yva) if y == a]
        if not idx:
            continue
        pred = [strat.feed(list(Xva[i])) for i in idx]
        per_class[a] = round(sum(p == a for p in pred) / len(idx), 3)
    print(f"val 每类召回: {per_class}")

    out_dir = ROOT / "outputs" / "turntaking"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "learned_mlp.pt"
    mlp.save(str(ckpt))
    data_dir = ROOT / "data" / "turntaking"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / "frames.npz", Xtr=Xtr, ytr=np.array(ytr),
             Xva=Xva, yva=np.array(yva))

    report = {
        "n_utterances": {"train": len(tr), "val": len(va)},
        "n_frames": {"train": len(ytr), "val": len(yva)},
        "label_dist": dict(Counter(ytr)),
        "frame_acc": {"train": round(acc_tr, 4), "val": round(acc_va, 4)},
        "val_recall_per_class": per_class,
        "loss": {"first": round(losses[0], 4), "last": round(losses[-1], 4)},
        "ckpt": str(ckpt.relative_to(ROOT)),
        "seed": d["seed"],
    }
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] 权重 -> {ckpt}\n     报告 -> {out_dir / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
