"""出牌策略网决策器：在启发式候选上做**增量**选择，其余动作完全委托启发式。

M 线负结果（整段替换 → 名次分 −3.425, t −6.67）是这里的设计前提：
网络只在启发式给出的候选集合内挑牌，且候选明细里带着启发式自己的 `total`，
网络学的是残差而非从零重建。
"""

from __future__ import annotations

import numpy as np

from . import bc
from .policy_net import policy_pick

DISCARD = "discard"


class PolicyNetDecider:
    """先问启发式；若它要出牌，则用策略网在**同一候选集**内重选。"""

    def __init__(self, payload: dict, *, label: str = "policy-net") -> None:
        bc._bootstrap()
        from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

        self.inner = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
        self.payload = payload
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
            choice is None
            or getattr(choice, "kind", None) != DISCARD
            or situation.drawn_tile is None
        ):
            return choice

        x, cand, mask = bc.candidate_features(self.inner, situation)
        if mask.sum() <= 1:
            return choice
        tile = policy_pick(self.payload, list(x), cand.tolist(), mask.tolist())
        if mask[tile] <= 0:
            return choice
        self.last_reason = f"策略网重选：{tile}（启发式原选 {choice.tile}）"
        self.last_detail["policy_pick"] = tile
        self.last_detail["heuristic_pick"] = choice.tile
        from majiang.rules.action import Action

        return Action(DISCARD, tile=tile)


def register_policy_net_arm(payload: dict, label: str = "policy-net") -> str:
    """注册进评测器，返回可用于 A/B 的档位名。"""
    from . import eval as ev

    ev._bootstrap_main_repo()

    def factory():
        return PolicyNetDecider(payload, label=label)

    ev.CUSTOM[label] = factory
    return label


__all__ = ["PolicyNetDecider", "register_policy_net_arm"]
