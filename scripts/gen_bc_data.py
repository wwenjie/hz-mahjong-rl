#!/usr/bin/env python
"""生成并缓存 BC 数据集（train / valid 用不同种子，避免泄漏）。

用法::

    PYTHONPATH=src uv run python scripts/gen_bc_data.py --train-matches 8 --valid-matches 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from nnrl import bc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 BC 数据集（自对弈记录启发式决策）")
    ap.add_argument("--train-matches", type=int, default=8)
    ap.add_argument("--valid-matches", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=20260928)
    ap.add_argument("--valid-seed", type=int, default=771014)
    ap.add_argument("--out-dir", default="runs/bc-data")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, matches, seed in (
        ("train", args.train_matches, args.train_seed),
        ("valid", args.valid_matches, args.valid_seed),
    ):
        t0 = time.perf_counter()
        data = bc.generate_bc_data(matches=matches, rounds=args.rounds, seed=seed)
        dt = time.perf_counter() - t0
        target = out / f"{name}.npz"
        bc.save_bc(data, target)
        per_round = matches * args.rounds
        print(
            f"[{name}] 样本 {len(data):6d}  局 {per_round:4d}  用时 {dt:6.1f}s "
            f"（{dt / max(per_round,1) * 1000:.0f} ms/局）  -> {target}"
        )
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
