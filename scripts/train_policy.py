#!/usr/bin/env python
"""训练出牌策略网（BC），导出纯 Python 产物 + 训练记录。

用法::

    PYTHONPATH=src uv run python scripts/train_policy.py \\
        --data runs/bc-data/train.npz --valid runs/bc-data/valid.npz \\
        --seed 20260928 --hidden 128,128 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from nnrl import bc
from nnrl.model import dataset_fingerprint
from nnrl import policy_net


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练出牌策略网（行为克隆）")
    ap.add_argument("--data", default="runs/bc-data/train.npz")
    ap.add_argument("--valid", default="runs/bc-data/valid.npz")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--vram-fraction", type=float, default=0.5)
    ap.add_argument("--name", default="")
    args = ap.parse_args(argv)

    import torch

    train = bc.load_bc(args.data)
    valid = bc.load_bc(args.valid)
    name = args.name or f"policy-bc-s{args.seed}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    fp = dataset_fingerprint(train.x, train.y)
    hidden = tuple(int(v) for v in args.hidden.split(",") if v.strip())
    cfg = policy_net.PolicyConfig(
        hidden=hidden, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        vram_fraction=args.vram_fraction,
    )

    ta = policy_net.teacher_agreement(valid)
    print(f"train n={len(train)} valid n={len(valid)} fingerprint={fp[:12]}…")
    print(f"教师自一致率（valid，argmax total，不含 tiebreak）= {ta:.4f}")
    t0 = time.perf_counter()
    result = policy_net.train_policy(train, valid, cfg)
    dt = time.perf_counter() - t0

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    record = policy_net.export_policy(
        result, out / "model.json", cfg=cfg, fingerprint=fp, gpu=gpu,
        torch_version=torch.__version__,
    )
    record["elapsed_seconds"] = round(dt, 1)
    record["train_samples"] = len(train)
    record["valid_samples"] = len(valid)
    record["teacher_self_agreement_valid"] = ta
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (out / "log.jsonl").open("w", encoding="utf-8") as fh:
        for row in result.history:
            fh.write(json.dumps(row) + "\n")

    print(f"best valid top-1 一致率 = {result.best_valid_agreement:.4f}（{dt:.1f}s, {gpu}）")
    print(f"教师自一致率参照        = {ta:.4f}")
    print(f"产物 {out/'model.json'}  记录 {out/'record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
