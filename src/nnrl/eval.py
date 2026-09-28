"""四座位旋转的配对 A/B 评测（只读复用主仓库的自对弈引擎）。

设计对齐主仓库 `tools/ab_test.py`（其纪律是踩过坑的结论，必须照抄）：

1. **同种子重放**：``seed * 100003 + match_index``，两侧拿到同一副牌。
2. **四座位旋转**：treatment 依次坐 0/1/2/3，抵消座次/庄位偏置。
3. **逐场配对差分**：报均值、标准误、t、95%CI；区间跨 0 即不显著。
4. **低方差判据**：名次分 / 胡次数优先，总得分作参考（重尾）。
5. **场地可配**：``field`` 控制另三座坐谁；默认 baseline（即"三个自己的复制品"）。
   注意：默认场地对任何偏离都给正分（主仓库实测），故**正号不能单独作为采纳依据**。

对主仓库的访问全部经 :func:`nnrl.paths.repo_root` 只读完成；本模块不写主仓库。
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field as dc_field

from . import paths

SEATS = 4


def _bootstrap_main_repo() -> None:
    """把主仓库 ``src/`` 加入 import 路径（只读）。"""
    src = paths.repo_root() / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"主仓库 src/ 不存在：{src}")
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)


@dataclass
class ArmResult:
    differences: dict[str, list[float]] = dc_field(default_factory=dict)
    pairs: int = 0
    elapsed: float = 0.0
    treatment_score: int = 0
    baseline_score: int = 0

    def report(self) -> str:
        lines = [
            f"配对样本 {self.pairs}（4 旋转），用时 {self.elapsed:.1f}s",
            f"总得分合计 treatment {self.treatment_score} vs baseline {self.baseline_score}"
            f"（{self.treatment_score - self.baseline_score:+d}）",
            "",
            "逐场配对差分（treatment − baseline，正值=treatment 更好）",
        ]
        for label, values in self.differences.items():
            lines.append("  " + _describe(values, label))
        return "\n".join(lines)


def _describe(values: list[float], label: str) -> str:
    n = len(values)
    if n < 2:
        return f"{label:10s} 样本不足"
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return f"{label:10s} 差分恒为 {mean:+.3f}（无方差）"
    t = mean / se
    margin = 1.96 * se
    verdict = "显著" if abs(t) > 1.96 else "**不显著**"
    return (
        f"{label:10s} 均值 {mean:+8.3f}  标准误 {se:6.3f}  t {t:+6.2f}  "
        f"95%CI [{mean - margin:+.3f}, {mean + margin:+.3f}]  {verdict}"
    )


# 由调用方注册的自研决策器工厂：name -> 零参工厂（每次返回新对象）
CUSTOM: dict[str, "object"] = {}


def _build(name: str):
    if name in CUSTOM:
        return CUSTOM[name]()
    from majiang.cli import DECIDERS, make_decider
    from majiang.strategy import versions
    from majiang.strategy.policy import Mode

    tally = {"heuristic": Mode.QUALIFIER, "final": Mode.FINAL, "qualifier": Mode.QUALIFIER}
    if name in tally:
        return make_decider("heuristic", tally[name])
    if name not in DECIDERS and not versions.is_version(name):
        raise SystemExit(
            f"未知策略 {name!r}，可选 {sorted(set(DECIDERS) | set(tally))} "
            f"或版本 {sorted(versions.BY_ID)}"
        )
    return make_decider(name, Mode.QUALIFIER)


def _play(names: list[str], index: int, *, rounds: int, base_score: int, seed: int):
    from majiang.sim.batch import run_match

    deciders = [_build(n) for n in names]
    result = run_match(
        deciders,
        rounds=rounds,
        base_score=base_score,
        seed=seed * 100003 + index,
        labels=names,
        start_dealer=index % SEATS,
    )
    return tuple(
        (s.total_score, s.place_points, s.god_count, s.wins, s.fan_total) for s in result.seats
    )


def paired_ab(
    treatment: str,
    baseline: str = "heuristic",
    *,
    matches: int = 120,
    rounds: int = 8,
    base_score: int = 1,
    seed: int = 20260928,
    field: str | None = None,
) -> ArmResult:
    """四座位旋转配对 A/B。``field`` 为另三座名字（默认=baseline）。"""
    _bootstrap_main_repo()
    field_list = [n.strip() for n in (field or baseline).split(",") if n.strip()]
    if len(field_list) == 1:
        field_list = field_list * 3
    if len(field_list) != 3:
        raise ValueError("field 只接受 1 个或 3 个名字")

    def names_for(rotation: int, seat_name: str) -> list[str]:
        row = [""] * SEATS
        row[rotation] = seat_name
        for seat, name in zip([s for s in range(SEATS) if s != rotation], field_list):
            row[seat] = name
        return row

    started = time.perf_counter()
    same_field = len(set(field_list)) == 1 and field_list[0] == baseline
    if same_field:
        shared = [
            _play([baseline] * SEATS, i, rounds=rounds, base_score=base_score, seed=seed)
            for i in range(matches)
        ]
        baseline_runs = [[row[s] for s in range(SEATS)] for row in shared]
    else:
        baseline_runs = [
            [
                _play(names_for(r, baseline), i, rounds=rounds, base_score=base_score, seed=seed)[r]
                for r in range(SEATS)
            ]
            for i in range(matches)
        ]

    labels = ("总得分", "名次分", "白板数", "胡次数", "番数总和")
    diffs: dict[str, list[float]] = {label: [] for label in labels}
    t_score = b_score = 0
    for rotation in range(SEATS):
        names = names_for(rotation, treatment)
        for i in range(matches):
            mine = _play(names, i, rounds=rounds, base_score=base_score, seed=seed)[rotation]
            theirs = baseline_runs[i][rotation]
            for label, a, b in zip(labels, mine, theirs):
                diffs[label].append(a - b)
            t_score += mine[0]
            b_score += theirs[0]
    return ArmResult(
        differences=diffs,
        pairs=matches * SEATS,
        elapsed=time.perf_counter() - started,
        treatment_score=t_score,
        baseline_score=b_score,
    )


__all__ = ["SEATS", "ArmResult", "paired_ab"]
