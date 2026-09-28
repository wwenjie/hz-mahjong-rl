"""行为克隆（BC）数据管线：从自对弈中记录启发式的每个出牌决策。

设计约束（来自 `notes/research-nn-design.md` 与 M 线负结果）：

- **不整段替换**：学生学的是「在启发式的候选上如何选」，所以每条样本都带着
  启发式对该局面的**候选评分明细**（`DiscardScore` 的全部数值字段）。
- **特征与推理同源**：全局特征直接复用主仓库 `strategy.features.extract`
  （29 维，与线上价值模型同一套），避免 train-serve skew。
- **只记出牌**：胡/杠/碰/吃不进本数据集——文献与 M 线都指向「别把动作类型混进
  出牌维度」，且 M 线的教训是整段替换会绕过胡杠判定。

对主仓库的访问一律经 `nnrl.paths` 只读完成；本模块不写主仓库。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DISCARD = "discard"

# 每张候选牌的特征（在 29 维全局特征之外）
CAND_FEATURE_NAMES: tuple[str, ...] = (
    "in_hand_count",
    "is_god",
    "shanten",
    "blocks",
    "pair_value",
    "meld_value",
    "route_value",
    "feed_cost",
    "god_penalty",
    "total",
    # 精确进张数（只对最小向听候选计算；其余置 0）。
    # 这是教师 tiebreak 唯一使用的量，也是 A 定位的「听口宽度」缺口所在——
    # 缺了它，学生既克隆不了教师（差 6.34% 的 tiebreak），也无法在 RL 中改善听口。
    "ukeire",
)
CAND_FEATURE_COUNT = len(CAND_FEATURE_NAMES)


def _bootstrap() -> None:
    from . import eval as ev

    ev._bootstrap_main_repo()


@dataclass
class BCData:
    """一批 BC 样本。

    - ``x``     : (n, 29) 全局局面特征
    - ``cand``  : (n, 34, 10) 每张牌的启发式评分明细
    - ``mask``  : (n, 34) 该牌是否在候选集内（1/0）
    - ``y``     : (n,) 教师选择的牌索引
    """

    x: np.ndarray
    cand: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    decisions: int = 0

    def __len__(self) -> int:
        return int(self.x.shape[0])


class _Accumulator:
    def __init__(self) -> None:
        self.x: list[list[float]] = []
        self.cand: list[np.ndarray] = []
        self.mask: list[np.ndarray] = []
        self.y: list[int] = []

    def build(self) -> BCData:
        if not self.x:
            z = np.zeros((0, 29), dtype=np.float32)
            return BCData(z, np.zeros((0, 34, CAND_FEATURE_COUNT), np.float32),
                          np.zeros((0, 34), np.float32), np.zeros((0,), np.int64), 0)
        return BCData(
            x=np.asarray(self.x, dtype=np.float32),
            cand=np.stack(self.cand).astype(np.float32),
            mask=np.stack(self.mask).astype(np.float32),
            y=np.asarray(self.y, dtype=np.int64),
            decisions=len(self.y),
        )


def candidate_features(heuristic_inner, situation):
    """构造某局面下的 (全局特征, 候选明细矩阵, 掩码)。

    **训练与推理共用本函数**（避免 train-serve skew）。``heuristic_inner`` 需提供
    ``_score_discard``（主仓库 `HeuristicDecider`）。
    """
    from majiang.rules import tiles
    from majiang.strategy.features import extract

    n_kinds = tiles.TILE_KINDS
    x = extract(situation)
    cand = np.zeros((n_kinds, CAND_FEATURE_COUNT), dtype=np.float32)
    mask = np.zeros((n_kinds,), dtype=np.float32)
    from majiang.rules.action import Action

    scored: dict[int, object] = {}
    for tile in range(n_kinds):
        if situation.hand.counts[tile] <= 0:
            continue
        try:
            scored[tile] = heuristic_inner._score_discard(situation, Action(DISCARD, tile=tile))  # noqa: SLF001
        except Exception:
            continue

    # 精确进张：只算向听最小的那组候选（教师 tiebreak 的适用范围），共用 memo 提速
    ukeire: dict[int, float] = {}
    if scored:
        min_shanten = min(score.shanten for score in scored.values())
        visible = None
        memo: dict = {}
        for tile, score in scored.items():
            if score.shanten != min_shanten:
                continue
            counts = list(situation.hand.counts)
            counts[tile] -= 1
            if visible is None:
                from majiang.rules import shanten as _shanten

                visible = _shanten.visible_counts(
                    situation.hand.counts,
                    [meld.tiles for meld in situation.all_melds],
                    situation.discards,
                )
            from majiang.rules import shanten as _shanten

            try:
                entries = _shanten.ukeire(
                    counts, situation.hand.meld_count, visible=visible, memo=memo
                )
                ukeire[tile] = float(sum(copy for _, copy in entries))
            except Exception:
                ukeire[tile] = 0.0

    for tile, score in scored.items():
        mask[tile] = 1.0
        cand[tile] = (
            float(situation.hand.counts[tile]),
            1.0 if tile == tiles.GOD else 0.0,
            float(score.shanten),
            float(score.blocks),
            float(score.pair_value),
            float(score.meld_value),
            float(score.route_value),
            float(score.feed_cost),
            float(score.god_penalty),
            float(score.total),
            ukeire.get(tile, 0.0),
        )
    return x, cand, mask


class RecordingHeuristic:
    """包住主仓库启发式，边对局边记录出牌决策样本。

    刻意**不实现** ``observe_state``：该钩子只为「完全信息上界」这类仅用于测量的
    对象准备，比赛决策器不得实现（读对手暗牌违规）。
    """

    def __init__(self, acc: _Accumulator, *, label: str = "heuristic[rec]") -> None:
        _bootstrap()
        from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

        self.inner = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
        self.acc = acc
        self.name = label
        self.last_reason = ""
        self.last_detail: dict[str, object] = {}

    def configure(self, tournament) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = getattr(self.inner, "last_reason", "")
        self.last_detail = dict(getattr(self.inner, "last_detail", {}) or {})
        if (
            choice is not None
            and getattr(choice, "kind", None) == DISCARD
            and choice.tile is not None
            and situation.drawn_tile is not None
        ):
            self._record(situation, choice)
        return choice

    def _record(self, situation, choice) -> None:
        x, cand, mask = candidate_features(self.inner, situation)
        if mask[choice.tile] <= 0:
            return
        self.acc.x.append(list(x))
        self.acc.cand.append(cand)
        self.acc.mask.append(mask)
        self.acc.y.append(int(choice.tile))


def save_bc(data: BCData, path: "str | Path") -> None:
    """缓存 BC 数据集（本仓库内）。"""
    from pathlib import Path as _Path

    from . import paths

    target = _Path(path)
    paths.forbid_write(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, x=data.x, cand=data.cand, mask=data.mask, y=data.y,
        decisions=np.asarray([data.decisions]),
    )


def load_bc(path: "str | Path") -> BCData:
    """读取缓存的 BC 数据集。"""
    with np.load(path) as d:
        return BCData(
            x=d["x"], cand=d["cand"], mask=d["mask"], y=d["y"],
            decisions=int(d["decisions"][0]) if "decisions" in d else int(d["x"].shape[0]),
        )


def generate_bc_data(*, matches: int, rounds: int = 8, seed: int = 20260928) -> BCData:
    """跑启发式自对弈，返回 BC 数据集（四座位旋转初始庄家）。"""
    _bootstrap()
    from majiang.sim.batch import run_match

    acc = _Accumulator()
    for index in range(matches):
        deciders = [RecordingHeuristic(acc) for _ in range(4)]
        run_match(
            deciders,
            rounds=rounds,
            seed=seed * 100003 + index,
            labels=["rec"] * 4,
            start_dealer=index % 4,
        )
    return acc.build()
