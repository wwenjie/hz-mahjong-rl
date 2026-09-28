#!/usr/bin/env python
"""RL 自对弈训练（**并行版**）：多进程扇出对局，与单进程版数学完全一致。

为什么并行：剖析显示瓶颈在**模拟器的进张/向听计算**（单局 77% 时间，
`_blocks_value` 被调 460 万次），网络前向 <0.1%。整条链单线程，
而本机 16 核只用 1 核。对局之间**完全独立**，可直接扇出。

与 `train_rl.py` 的唯一差别是「谁在算 rollout」：
- 单进程版：主进程顺序跑 episode。
- 并行版：worker 进程并行跑**单局**，主进程收集到一批后做**同一次**梯度更新。

参数快照按任务广播（~7k float，开销可忽略）；worker 只读快照、不回传梯度，
因此**更新语义与单进程完全一致**（等价于并行 rollout 的同步 A2C 式采样）。

用法::

    PYTHONPATH=src python scripts/train_rl_par.py --episodes 40 --seed 20260928 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp

import numpy as np

from nnrl import bc as bc_mod
from nnrl import rl_net

_CTX: dict = {}


def _init_worker(model: dict) -> None:
    bc_mod._bootstrap()
    _CTX["cfg"] = rl_net.RLCfg(**model["cfg"])
    _CTX["base_seed"] = model["base_seed"]
    _CTX["input_dim"] = model["input_dim"]


def _worker_round(task: tuple[int, int, int, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """跑一局，返回该局全部出牌决策的 (rows, mask, pick, reward)。

    ``params`` **随任务传入**：worker 侧不持有可变快照，因此无需重建进程池。
    7k float 的广播开销相对整局模拟可忽略。
    """
    ep, rnd, dealer, params = task
    from nnrl.rl_play import RLPolicy
    from majiang.sim.round import run_round

    cfg, base_seed = _CTX["cfg"], _CTX["base_seed"]
    seed = base_seed * 7919 + ep * 101 + rnd
    rl = [RLPolicy(params, cfg=cfg, sample=True, seed=seed + 1000 * s, label=f"rl{s}") for s in range(4)]
    outcome = run_round(rl, dealer=dealer, round_no=rnd + 1, base_score=1,
                        rng=__import__("random").Random(seed))
    rows, masks, picks, rewards = [], [], [], []
    for seat in range(4):
        for (r, m, p) in rl[seat].take_pending():
            rows.append(r)
            masks.append(m)
            picks.append(p)
            rewards.append(outcome.scores[seat])
    if not rows:
        k = _CTX["input_dim"]
        return (np.zeros((0, 34, k), np.float32), np.zeros((0, 34), np.float32),
                np.zeros((0,), np.int64), np.zeros((0,), np.float32))
    return (np.stack(rows).astype(np.float32), np.stack(masks).astype(np.float32),
            np.asarray(picks, np.int64), np.asarray(rewards, np.float32))


def _norm_stats(path: str):
    d = bc_mod.load_bc(path)
    from nnrl import policy_net as P

    flat = P.assemble(d.x, d.cand).reshape(-1, P.INPUT_PER_CAND)
    sel = d.mask.reshape(-1) > 0
    f = flat[sel]
    return f.mean(0), f.std(0) + 1e-8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RL 自对弈训练（并行）")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", default="rl-par")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--stats-from", default="runs/bc-data/train.npz")
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reward-scale", type=float, default=10.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--init-from", default="")
    args = ap.parse_args(argv)

    bc_mod._bootstrap()
    from nnrl import policy_net as P

    mean, std = _norm_stats(args.stats_from)
    input_dim = P.INPUT_PER_CAND
    total_idx = P._flat_index("total")
    hidden = tuple(int(v) for v in args.hidden.split(",") if v.strip())
    cfg = rl_net.RLCfg(hidden=hidden, temperature=args.temperature, entropy_coef=args.entropy,
                       lr=args.lr, batch_decisions=args.batch)
    rng = np.random.default_rng(args.seed)
    params = (rl_net.load_params(args.init_from) if args.init_from
              else rl_net.init_params(cfg, rng, mean, std, total_idx=total_idx, input_dim=input_dim))

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w", encoding="utf-8")
    print(f"并行训练：workers={args.workers}，起点 w3 范数={np.linalg.norm(params['w3']):.6f}")

    cfg_dict = cfg.as_dict()
    plan: list[tuple[int, int, int]] = []
    for ep in range(1, args.episodes + 1):
        for rnd in range(args.rounds):
            plan.append((ep, rnd, rnd % 4))

    baseline = 0.0
    buf: list = []
    total_decisions = 0
    t0 = time.perf_counter()
    done_ep = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(({"cfg": cfg_dict, "base_seed": args.seed,
                              "input_dim": input_dim}),)) as pool:
        for chunk_start in range(0, len(plan), args.workers):
            wave = [(ep, rnd, dealer, params) for (ep, rnd, dealer) in
                    plan[chunk_start:chunk_start + args.workers]]
            for rows, masks, picks, rewards in pool.map(_worker_round, wave, chunksize=1):
                for i in range(len(picks)):
                    buf.append((rows[i], masks[i], int(picks[i]), float(rewards[i]) / args.reward_scale))
                total_decisions += len(picks)
            while len(buf) >= cfg.batch_decisions:
                mean_r, used = rl_net.train_step(params, buf, cfg, baseline=baseline)
                baseline = rl_net.update_baseline(baseline, mean_r, cfg)
                row = {"decisions": total_decisions, "mean_reward": round(mean_r, 4),
                       "baseline_before": round(used, 4),
                       "w3_norm": round(float(np.linalg.norm(params["w3"])), 5),
                       "b3": round(float(params["b3"][0]), 5),
                       "elapsed": round(time.perf_counter() - t0, 1)}
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
                buf = []
    rl_net.save_params(params, out / "params.json", cfg=cfg,
                       extra={"seed": args.seed, "input_dim": input_dim, "total_idx": total_idx,
                              "stats_from": args.stats_from, "total_decisions": total_decisions})
    record = {"version": rl_net.RL_VERSION + "-par", "seed": args.seed, "episodes": args.episodes,
              "workers": args.workers, "cfg": cfg.as_dict(), "input_dim": input_dim,
              "total_idx": total_idx, "total_decisions": total_decisions,
              "elapsed_seconds": round(time.perf_counter() - t0, 1),
              "stats_source": args.stats_from, "artifact": str(out / "params.json")}
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.close()
    print(f"完成：{total_decisions} 决策，{time.perf_counter()-t0:.1f}s -> {out/'params.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
