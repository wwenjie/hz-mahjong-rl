#!/usr/bin/env python3
"""Top 1 探针：同局面分叉 → 偏好对可分辨率 / 信号检测。

**不改动任何现有逻辑**，只读复用主仓库引擎（`nnrl.paths` 只读边界）。

方法（research-rl-optimization.md §3.4 / §6 Top 1）：

1. 用启发式跑一局，在探针座位（seat 0）的第 k 个**出牌决策**处记录候选集
   按启发式 `total` 的排序（这就是「同一局面」）。
2. 用**同一 seed**（⇒ 同一副牌、同一牌墙、分叉前决策完全一致）重放两次：
   - 分支 A：在 k 处**强制打出**启发式候选的第 1 名；
   - 分支 B：在 k 处**强制打出**启发式候选的第 2 名；
   分叉后两侧都交回启发式。两个分支在 k 之前**逐位相同**，故是严格的同局面分叉。
3. 比较分支 A/B 中探针座位的**名次分**（`place_points_of`，低方差判据）。

产出判据：
- **可分辨率** = P(place_A ≠ place_B)：同局面分叉能否产生不同结局。
- **启发式排序预测力** = P(place(top1) > place(top2))：若显著 > 0.5，说明
  「同一局面内的名次序数」携带**可学习信号**（Top 1 的 Go 条件）。
- **分叉方差 / 跨局面方差** 之比：配对能把噪声压掉多少（Top 3 的前提）。

纪律：`nice -n 15`；同一时刻只跑一个；结果写本仓库 `records/`。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

from nnrl import bc as bc_mod
from nnrl import paths

bc_mod._bootstrap()

from majiang.rules.action import Action  # noqa: E402
from majiang.sim.batch import place_points_of  # noqa: E402
from majiang.sim.round import deal, play_round  # noqa: E402

DISCARD = "discard"
PROBE_SEAT = 0
TOTAL_COL = 9  # CAND_FEATURE_NAMES 中 "total" 的下标

# 供 fork worker 使用的模块级全局（由 main 设定）
_SEED_BASE = 20260928
_KS: list[int] = [2, 4, 6]
_GAPS: list[int] = [1, 3, 7]  # gap=1 即 top1 vs top2；3=第4名；7=第8名


class _Probe:
    """探针座位的决策器：可记录候选排序 / 在指定决策强制出某张。"""

    def __init__(self, target_k: int | None, force_tile: int | None, record: bool) -> None:
        bc_mod._bootstrap()
        from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

        self.inner = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
        self.target_k = target_k
        self.force_tile = force_tile
        self.record = record
        self.k = 0
        self.records: list[tuple[int, list[int], int]] = []  # (k, 候选排序, 启发式选择)
        self.forced: int | None = None
        self.name = "probe"
        self.last_reason = ""
        self.last_detail: dict = {}

    def configure(self, tournament) -> None:
        self.inner.configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        if choice is not None and getattr(choice, "kind", None) == DISCARD:
            k = self.k
            self.k += 1
            # 候选 = **合法**出牌动作（抓打圈下仅限刚摸的牌），不是全部手牌
            legal_discards = [int(a.tile) for a in actions
                              if getattr(a, "kind", None) == DISCARD and a.tile is not None]
            if self.record or k == self.target_k:
                x, cand, mask = bc_mod.candidate_features(self.inner, situation)
                totals = np.asarray(cand, dtype=np.float64)[:, TOTAL_COL]
                order = sorted(legal_discards, key=lambda t: (-float(totals[t]), t))
                self.records.append((k, order, int(choice.tile)))
            if k == self.target_k and self.force_tile is not None and int(self.force_tile) in legal_discards:
                self.forced = int(self.force_tile)
                return Action(DISCARD, tile=int(self.force_tile))
        return choice


class _Heur:
    def __init__(self) -> None:
        bc_mod._bootstrap()
        from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

        self.inner = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
        self.name = "heur"
        self.last_reason = ""
        self.last_detail: dict = {}

    def configure(self, tournament) -> None:
        self.inner.configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        return self.inner.choose(situation, actions, budget_ms=budget_ms)


def _one_round(seed: int, target_k: int | None, force_tile: int | None, record: bool):
    """跑一局（探针坐 seat 0），返回 (scores, place, records, is_flow)。"""
    rng = random.Random(seed)
    probe = _Probe(target_k, force_tile, record)
    deciders = [probe, _Heur(), _Heur(), _Heur()]
    state = deal(rng, dealer=seed % 4, round_no=1)
    res = play_round(state, deciders, base_score=1)
    place = place_points_of(res.scores)
    return res.scores, place, probe.records, res.is_flow


def _fork_worker(task: tuple) -> dict:
    """worker：给定 (seed, target_k, force_tile)，返回探针座位名次分。

    模块级以便 fork 传递；每个任务完全确定 ⇒ 与 worker 数无关。
    """
    seed, target_k, force_tile = task
    scores, place, _rec, flow = _one_round(seed, target_k, force_tile, record=False)
    return {"seed": seed, "k": target_k, "tile": force_tile, "place": int(place[PROBE_SEAT]),
            "score": int(scores[PROBE_SEAT]), "flow": bool(flow)}


def _corpus_worker(index: int) -> list[dict]:
    """worker：跑一个 seed，返回该 seed 在各 k 上的分叉点候选（含候选全序）。"""
    seed = _SEED_BASE * 100003 + index
    _s, _p, records, _f = _one_round(seed, target_k=None, force_tile=None, record=True)
    by_k = {k: (order, chosen) for (k, order, chosen) in records}
    out = []
    for k in _KS:
        if k not in by_k:
            continue
        order, chosen = by_k[k]
        if len(order) < 2:
            continue
        out.append({"index": index, "seed": seed, "k": k,
                    "order": [int(t) for t in order], "chosen": int(chosen)})
    return out


def _run(tasks: list[tuple], workers: int):
    if workers and workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            return pool.map(_fork_worker, tasks, chunksize=1)
    return [_fork_worker(t) for t in tasks]


def _run_corpus(indices: list[int], workers: int) -> list[dict]:
    if workers and workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, len(indices))) as pool:
            nested = pool.map(_corpus_worker, indices, chunksize=1)
    else:
        nested = [_corpus_worker(i) for i in indices]
    return [c for group in nested for c in group]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200, help="分叉对数量")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--rounds", type=int, default=8, help="每个 seed 覆盖的局（取每局第 k 决策）")
    ap.add_argument("--ks", default="2,4,6", help="在这些出牌决策序号上分叉")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="records/probe-pref-pairs.json")
    args = ap.parse_args()

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 4)
    ks = [int(s) for s in args.ks.split(",") if s.strip()]

    global _SEED_BASE, _KS
    _SEED_BASE = args.seed
    _KS = ks

    # 阶段 1：记录每个 seed 在目标 k 上的启发式候选排序（并行）
    print(f"阶段 1：记录候选排序（seeds={args.pairs}，ks={ks}，workers={workers}）")
    t0 = time.perf_counter()
    corpus = _run_corpus(list(range(args.pairs)), workers)
    t1 = time.perf_counter()
    print(f"  可用分叉点 {len(corpus)}（{t1 - t0:.0f}s）")

    # 阶段 2：每个分叉点跑 branches 个分支（top1 + 各档间隔对手）
    # 关键：不只比 top1 vs top2（可能本就等价），还比 top1 vs 更低名次，
    # 若大幅扰动仍无信号，no-go 才站得住。
    tasks = []
    for c in corpus:
        order = c["order"]
        picks = [order[0]]
        for gap in _GAPS:
            if len(order) > gap:
                picks.append(order[gap])
        for t in dict.fromkeys(picks):  # 去重保序
            tasks.append((c["seed"], c["k"], t))
    print(f"阶段 2：跑 {len(tasks)} 个分支（workers={workers}，gaps={_GAPS}）...")
    t2 = time.perf_counter()
    results = _run(tasks, workers)
    t3 = time.perf_counter()
    print(f"  完成（{t3 - t2:.0f}s）")

    # 阶段 3：统计（按每档间隔分别汇总）
    by_seed_k = {}
    for r in results:
        by_seed_k.setdefault((r["seed"], r["k"]), {})[r["tile"]] = r

    def _gap_stats(gap: int) -> dict:
        diffs = []
        score_diffs = []
        wins = loss = ties = 0
        for c in corpus:
            order = c["order"]
            if len(order) <= gap:
                continue
            cell = by_seed_k.get((c["seed"], c["k"]))
            if not cell:
                continue
            a = cell.get(order[0])
            b = cell.get(order[gap])
            if a is None or b is None:
                continue
            d = a["place"] - b["place"]
            diffs.append(d)
            score_diffs.append(a["score"] - b["score"])
            if d > 0:
                wins += 1
            elif d < 0:
                loss += 1
            else:
                ties += 1
        nn = len(diffs)
        res = wins + loss
        m = sum(diffs) / nn if nn else 0.0
        v = sum((x - m) ** 2 for x in diffs) / (nn - 1) if nn > 1 else 0.0
        ms = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0
        vs = (sum((x - ms) ** 2 for x in score_diffs) / (len(score_diffs) - 1)
              if len(score_diffs) > 1 else 0.0)
        se_s = (math.sqrt(vs) / math.sqrt(nn)) if nn > 1 else 0.0
        return {
            "gap": gap, "n": nn, "resolvable": res,
            "frac_resolvable": round(res / nn, 4) if nn else 0.0,
            "top1_win_frac": round(wins / res, 4) if res else None,
            "place_diff_mean": round(m, 4), "place_diff_sd": round(math.sqrt(v), 4),
            "score_diff_mean": round(ms, 4),
            "score_diff_t": round(ms / se_s, 3) if se_s > 0 else 0.0,
        }

    gaps_report = {str(g): _gap_stats(g) for g in _GAPS}
    # 主口径：gap=1（top1 vs top2）
    g1 = gaps_report["1"]
    n = g1["n"]
    resolvable = g1["resolvable"]
    frac_res = g1["frac_resolvable"]
    mean_sd = g1["score_diff_mean"]
    t_score = g1["score_diff_t"]
    n_sd = n
    flows = sum(1 for r in results if r["flow"])

    # 跨局面方差：纯启发式跑同样 seed，取探针座位名次分（并行）
    base_places = [r["place"] for r in _run(
        [(args.seed * 100003 + index, None, None) for index in range(min(args.pairs, 200))],
        workers,
    )]
    bp_mean = sum(base_places) / len(base_places) if base_places else 0.0
    bp_var = (sum((x - bp_mean) ** 2 for x in base_places) / (len(base_places) - 1)
              if len(base_places) > 1 else 0.0)

    report = {
        "config": {"pairs": args.pairs, "ks": ks, "seed": args.seed, "workers": workers,
                   "gaps": _GAPS},
        "n_pairs": n,
        "resolvable": resolvable,
        "frac_resolvable": round(frac_res, 4),
        "place_diff_mean": g1["place_diff_mean"],
        "place_diff_sd": g1["place_diff_sd"],
        "score_diff_mean": mean_sd,
        "score_diff_t": t_score,
        "n_score_diffs": n_sd,
        "across_var": round(bp_var, 4),
        "flows": flows,
        "elapsed_s": round(t3 - t0, 1),
        "gaps_report": gaps_report,
        "verdict_hint": (
            "GO: gap=1 得分差 t 显著 ⇒ 同局面携带可学习信号"
            if (n_sd >= 300 and abs(t_score) > 1.96) else
            ("NO-GO: 所有档位（含大幅扰动 gap=3/7）均无预测力 ⇒ 局面不足以预测结局，终止「出牌重排」上投 RL"
             if (n_sd >= 300 and all(abs(gaps_report[str(g)]["score_diff_t"]) < 1.0 for g in _GAPS)) else
             "INCONCLUSIVE: 样本不足或结果居中，需加样本")
        ),
    }
    paths.forbid_write(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
