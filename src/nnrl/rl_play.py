"""RL 决策器：在**启发式候选集内**重排出牌，其余决策一律委托启发式。

这是 M 线负结果的直接对策（`notes/REPORT-6.6-nnrl.md` §2.1）：

- 整段替换 ⇒ 丢弃启发式结构 ⇒ 显著更差。故此处**只动出牌这一项**，
  且候选集仍由启发式给出，网络只做组内重排（``logit = total + delta``）。
- 胡 / 杠 / 碰 / 吃 / 响应 一律返回启发式的选择，网络不参与。
- 非出牌局面**不产生训练样本**（避免把响应决策的收益错误归因到出牌上）。
"""

from __future__ import annotations

import numpy as np

from . import bc as bc_mod
from . import rl_net


class RLPolicy:
    """包住主仓库启发式，出牌用网络重排。满足 ``Decider`` 窄接口。"""

    def __init__(self, params: dict, *, cfg: rl_net.RLCfg | None = None, sample: bool = True,
                 seed: int = 0, label: str = "rl") -> None:
        bc_mod._bootstrap()
        from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

        self.inner = HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
        self.params = params
        self.cfg = cfg or rl_net.RLCfg()
        self.sample = sample
        self.name = label
        self.gen = np.random.default_rng(seed)
        self.last_reason = ""
        self.last_detail: dict = {}
        self.pending: list[tuple[np.ndarray, np.ndarray, int]] = []

    def configure(self, tournament) -> None:  # 真机路径
        self.inner.configure(tournament)

    # ---- 特征组装（训练与推理共用） --------------------------------------
    def _state_matrix(self, situation):
        """返回 ``(full, mask, pickable)``：每张候选牌的输入向量 + 掩码。"""
        x, cand, mask = bc_mod.candidate_features(self.inner, situation)
        g = int(np.asarray(x).size)
        rows = np.concatenate(
            [
                np.tile(np.asarray(x, dtype=np.float64).reshape(1, g), (mask.size, 1)),
                np.asarray(cand, dtype=np.float64),
            ],
            axis=1,
        )
        rows = rl_net.normalize(self.params, rows)
        return rows, np.asarray(mask, dtype=np.float64), g

    def choose(self, situation, actions, *, budget_ms: int = 0):
        if not actions:
            return None
        # 先让启发式决定「做什么」（胡/杠/吃碰/出牌）
        choice = self.inner.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = getattr(self.inner, "last_reason", "")
        self.last_detail = dict(getattr(self.inner, "last_detail", {}) or {})
        if choice is None or choice.kind != "discard":
            return choice
        # 只重排「打哪张」
        rows, mask, _ = self._state_matrix(situation)
        pick = rl_net.act(
            self.params, rows, mask, sample=self.sample, gen=self.gen,
            temperature=self.cfg.temperature,
        )
        self.pending.append((rows, mask, int(pick)))
        from majiang.rules.action import Action

        return Action("discard", tile=int(pick))

    def take_pending(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
        out, self.pending = self.pending, []
        return out


def heuristic_only_factory():
    """纯启发式决策器工厂（对拍基线 / 混合场地用）。"""
    bc_mod._bootstrap()
    from majiang.strategy.policy import HeuristicDecider, Mode, PolicyConfig

    return HeuristicDecider(PolicyConfig.for_mode(Mode.QUALIFIER))
