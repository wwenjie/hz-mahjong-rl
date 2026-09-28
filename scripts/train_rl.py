#!/usr/bin/env python
"""RL 自对弈训练（纯 numpy REINFORCE，增量式）。

起点 = 启发式核心排序（``delta≡0``），只学习出牌重排。用法::

    PYTHONPATH=src uv run python scripts/train_rl.py --episodes 40 --seed 20260928

纪律（继承主仓库）：
- 自对弈是**有偏判据**，训练中的胜率只作监控；采纳与否看**四座位旋转配对对拍**（另跑）。
- 种子必填；记录含数据指纹（归一化统计来源）、超参、依赖版本、GPU。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from nnrl import bc as bc_mod
from nnrl import rl_net
from nnrl.rl_play import RLPolicy


def _norm_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """从既有 BC 数据取归一化统计（与训练/推理共用同一套，避免分布漂移）。"""
    d = bc_mod.load_bc(path)
    from nnrl import policy_net as P

    flat = P.assemble(d.x, d.cand).reshape(-1, P.INPUT_PER_CAND)
    sel = d.mask.reshape(-1) > 0
    f = flat[sel]
    return f.mean(0), f.std(0) + 1e-8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RL 自对弈训练（增量式）")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", default="rl-run")
    ap.add_argument("--stats-from", default="runs/bc-data/train.npz")
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reward-scale", type=float, default=10.0)
    ap.add_argument("--save-every", type=int, default=10, help="每 N 个 episode 存一次检查点（抗环境回收）")
    ap.add_argument("--init-from", default="", help="可选：从已有 params.json 热启")
    args = ap.parse_args(argv)

    bc_mod._bootstrap()
    from majiang.sim.batch import run_match

    mean, std = _norm_stats(args.stats_from)
    from nnrl import policy_net as P

    input_dim = P.INPUT_PER_CAND
    total_idx = P._flat_index("total")
    hidden = tuple(int(v) for v in args.hidden.split(",") if v.strip())
    cfg = rl_net.RLCfg(hidden=hidden, temperature=args.temperature, entropy_coef=args.entropy,
                       lr=args.lr, batch_decisions=args.batch)
    rng = np.random.default_rng(args.seed)

    if args.init_from:
        params = rl_net.load_params(args.init_from)
        print(f"热启自 {args.init_from}")
    else:
        params = rl_net.init_params(cfg, rng, mean, std, total_idx=total_idx, input_dim=input_dim)

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w", encoding="utf-8")

    baseline = 0.0
    batch: list = []
    t0 = time.perf_counter()
    total_decisions = 0

    # 起点自检：delta≡0 时策略 = argmax(total)，即「增量修正」的零位
    print(f"起点自检：delta 分支全零（w3 范数={np.linalg.norm(params['w3']):.6f}，b3={float(params['b3'][0]):.6f}）")

    for ep in range(1, args.episodes + 1):
        rl = [RLPolicy(params, cfg=cfg, sample=True, seed=args.seed + 1000 * s, label=f"rl{s}")
              for s in range(4)]
        for r in range(args.rounds):
            from majiang.sim.round import run_round

            outcome = run_round(
                rl, dealer=r % 4, round_no=r + 1, base_score=1,
                rng=__import__("random").Random(args.seed * 7919 + ep * 101 + r),
            )
            for seat in range(4):
                for (rows, mask, pick) in rl[seat].take_pending():
                    batch.append((rows, mask, pick, outcome.scores[seat] / args.reward_scale))
                    total_decisions += 1

        if len(batch) >= cfg.batch_decisions:
            mean_r, used = rl_net.train_step(params, batch, cfg, baseline=baseline)
            baseline = rl_net.update_baseline(baseline, mean_r, cfg)
            row = {"episode": ep, "decisions": total_decisions, "mean_reward": round(mean_r, 4),
                   "baseline_before": round(used, 4), "w3_norm": round(float(np.linalg.norm(params["w3"])), 5),
                   "b3": round(float(params["b3"][0]), 5)}
            print(json.dumps(row))
            log.write(json.dumps(row) + "\n")
            log.flush()
            batch = []
            # 周期性检查点：环境会周期性回收长跑进程，不能只在结束时保存
            if cfg and args.save_every and ep % args.save_every == 0:
                rl_net.save_params(params, out / "params.json", cfg=cfg,
                                   extra={"seed": args.seed, "input_dim": input_dim,
                                          "total_idx": total_idx, "episodes_done": ep,
                                          "stats_from": args.stats_from, "checkpoint": True})
                rl_net.save_params(params, out / f"params.ep{ep}.json", cfg=cfg,
                                   extra={"episodes_done": ep, "checkpoint": True})

    rl_net.save_params(params, out / "params.json", cfg=cfg,
                       extra={"seed": args.seed, "episodes": args.episodes,
                              "stats_from": args.stats_from, "input_dim": input_dim,
                              "total_idx": total_idx, "total_decisions": total_decisions})
    record = {
        "version": rl_net.RL_VERSION, "seed": args.seed, "episodes": args.episodes,
        "rounds_per_episode": args.rounds, "cfg": cfg.as_dict(),
        "input_dim": input_dim, "total_idx": total_idx,
        "total_decisions": total_decisions, "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "stats_source": args.stats_from, "artifact": str(out / "params.json"),
    }
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.close()
    print(f"完成：{total_decisions} 决策，{time.perf_counter()-t0:.1f}s -> {out/'params.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
