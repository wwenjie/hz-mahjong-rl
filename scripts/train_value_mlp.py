#!/usr/bin/env python
"""训练 MLP 价值模型并导出纯 Python 产物 + 训练记录。

用法::

    PYTHONPATH=src uv run python scripts/train_value_mlp.py --seed 20260928 --epochs 40

产出（本仓库内，不写主仓库）：

- ``runs/<name>/model.json``   纯 Python 可推理产物
- ``runs/<name>/record.json``  训练记录（数据指纹/超参/种子/依赖/GPU）
- ``runs/<name>/log.jsonl``    每 epoch 指标
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from nnrl import data
from nnrl.model import TrainConfig, dataset_fingerprint, export_pure_python, train_mlp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练 MLP 价值模型（纯 Python 导出）")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=str, default="64,64")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--vram-fraction", type=float, default=0.5)
    ap.add_argument("--name", default="")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    train = data.load_value_train()
    valid = data.load_value_valid()
    name = args.name or f"mlp-value-s{args.seed}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    fingerprint = dataset_fingerprint(train.x, train.y)
    hidden = tuple(int(v) for v in args.hidden.split(",") if v.strip())
    cfg = TrainConfig(
        hidden=hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        vram_fraction=args.vram_fraction,
    )
    print(f"train n={len(train)} valid n={len(valid)} fingerprint={fingerprint[:12]}…")
    started = time.perf_counter()
    result = train_mlp(train.x, train.y, (valid.x, valid.y), cfg)
    elapsed = time.perf_counter() - started

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    record = export_pure_python(
        result,
        out / "model.json",
        cfg=cfg,
        fingerprint=fingerprint,
        gpu=gpu,
        torch_version=torch.__version__,
    )
    record["elapsed_seconds"] = round(elapsed, 1)
    record["train_samples"] = len(train)
    record["valid_samples"] = len(valid)
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (out / "log.jsonl").open("w", encoding="utf-8") as fh:
        for row in result.history:
            fh.write(json.dumps(row) + "\n")

    # 指标对照：常数基线
    const_mse = float(np.mean((valid.y - train.y.mean()) ** 2))
    print(f"best valid MSE = {result.best_valid_mse:.4f}（用时 {elapsed:.1f}s, {gpu}）")
    print(f"常数基线 MSE  = {const_mse:.4f}")
    print(f"产物 {out/'model.json'}  记录 {out/'record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
