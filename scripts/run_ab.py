#!/usr/bin/env python
"""多档位 / 多种子配对 A/B 运行器（四座位旋转），产出 JSON 记录。

纪律（继承主仓库，见 README）：

- 四座位旋转（评测器内建）。
- 默认场地 = baseline ⇒ **对任何偏离都给正分**，所以正号不能单独作为采纳依据；
  必须同时看机制量（名次分/胡次数）与跨种子一致性。
- 报跨种子分布，不单看一个种子。

用法::

    PYTHONPATH=src uv run python scripts/run_ab.py \\
        --arm mlp-value=runs/mlp-value-s20260928/model.json \\
        --baseline heuristic --matches 40 --seeds 20260928,771014
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from nnrl import eval as ev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="多档位多种子配对 A/B")
    ap.add_argument("--arm", action="append", default=[], help="name=model.json（纯 Python 产物）")
    ap.add_argument("--baseline", default="heuristic")
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seeds", default="20260928,771014")
    ap.add_argument("--field", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms: list[str] = []
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            args_ = None
            arms.append(name)  # 直接用主仓库档位名
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        from nnrl.decider import register_mlp_value_arm

        arms.append(register_mlp_value_arm(payload, label=name))

    results: dict = {"baseline": args.baseline, "matches": args.matches, "seeds": seeds, "arms": {}}
    for arm in arms:
        per_seed = {}
        for seed in seeds:
            started = time.perf_counter()
            r = ev.paired_ab(
                arm,
                args.baseline,
                matches=args.matches,
                rounds=args.rounds,
                seed=seed,
                field=args.field or None,
            )
            summary = {}
            for label, values in r.differences.items():
                n = len(values)
                mean = sum(values) / n
                var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
                se = (var / n) ** 0.5 if n > 1 else 0.0
                summary[label] = {
                    "mean": mean,
                    "se": se,
                    "t": (mean / se) if se else 0.0,
                    "n": n,
                }
            summary["_elapsed"] = round(time.perf_counter() - started, 1)
            per_seed[str(seed)] = summary
            print(f"[{arm} seed={seed}] 用时 {summary['_elapsed']:.0f}s")
            for label, s in summary.items():
                if label.startswith("_"):
                    continue
                print(f"    {label:8s} 均值 {s['mean']:+8.3f}  t {s['t']:+6.2f}")
        results["arms"][arm] = per_seed

    out = Path(args.out or f"records/ab-{'-'.join(arms)}-{args.matches}m.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"记录写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
