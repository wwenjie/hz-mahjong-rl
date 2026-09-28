#!/usr/bin/env python
"""策略网容量扫描：看 BC 一致率随网络容量如何变化。

门禁口径：
- ``teacher_agreement``：平凡基线 = argmax(total)，即"教师自一致率"（教师的 tiebreak 使其 <1）。
- 学生若**达不到**该平凡基线，说明连输入里现成的 ``total`` 都没学会，加容量前先查表征/优化。

用法::
    PYTHONPATH=src uv run python scripts/sweep_policy.py --hiddens 32,32;64,64;128,128;256,256;512,512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from nnrl import bc, policy_net
from nnrl.model import dataset_fingerprint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="策略网容量扫描")
    ap.add_argument("--data", default="runs/bc-data/train.npz")
    ap.add_argument("--valid", default="runs/bc-data/valid.npz")
    ap.add_argument("--hiddens", default="32,32;64,64;128,128;256,256")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--out", default="records/sweep-policy.json")
    args = ap.parse_args(argv)

    train = bc.load_bc(args.data)
    valid = bc.load_bc(args.valid)
    fp = dataset_fingerprint(train.x, train.y)
    ta = policy_net.teacher_agreement(valid)
    print(f"train n={len(train)} valid n={len(valid)}")
    print(f"平凡基线（argmax total，valid）= {ta:.4f}")

    rows = []
    for spec in args.hiddens.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        hidden = tuple(int(v) for v in spec.split(",") if v.strip())
        cfg = policy_net.PolicyConfig(
            hidden=hidden, epochs=args.epochs, seed=args.seed,
        )
        t0 = time.perf_counter()
        res = policy_net.train_policy(train, valid, cfg)
        dt = time.perf_counter() - t0
        params = sum(
            int(np.prod(p.shape)) for p in res.model.parameters()
        )
        rows.append({
            "hidden": list(hidden),
            "params": params,
            "valid_agreement": res.best_valid_agreement,
            "elapsed_s": round(dt, 1),
        })
        print(
            f"  hidden={spec:>10s}  params={params:7d}  "
            f"valid_top1={res.best_valid_agreement:.4f}  "
            f"({'≥基线' if res.best_valid_agreement >= ta else '<基线'})  {dt:.1f}s"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fingerprint": fp, "baseline_teacher_agreement": ta,
        "epochs": args.epochs, "seed": args.seed, "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"记录写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
