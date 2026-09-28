"""把本线的纯 Python MLP 价值模型包成主仓库接口的决策器。

对齐主仓库 `strategy/value.py` 的做法：**只覆盖出牌那一步**，胡/杠/弃胡飘仍交给启发式。
这是踩过坑的结论——从候选排序整体入手会绕过胡/杠判定，导致策略永不自摸。

本模块定义的决策器实现主仓库 `sim.round._ask` 需要的接口：

- ``choose(situation, actions, *, budget_ms) -> Action | None``
- 可选 ``observe_state``（本线不用）
- ``configure(tournament)``（自对弈路径不调用；实现为空以兼容）
- ``name`` 属性（日志可分辨档位）

**不 import 主仓库任何模块**：主仓库对象由调用方（评测器）注入为 ``heuristic`` 参数，
因此本模块在无主仓库 `sys.path` 时也可导入。
"""

from __future__ import annotations

from .model import pure_forward

DISCARD = "discard"


class MLPValueDecider:
    """用纯 Python MLP 给**出牌**打分；其余动作委托启发式。"""

    def __init__(self, payload: dict, heuristic, *, feature_extract, label: str = "mlp-value") -> None:
        self.payload = payload
        self.heuristic = heuristic
        self.feature_extract = feature_extract
        self.name = label
        self.last_reason = ""
        self.last_detail: dict[str, object] = {}

    def configure(self, tournament) -> None:  # pragma: no cover - 自对弈不调用
        configure = getattr(self.heuristic, "configure", None)
        if configure is not None:
            configure(tournament)

    def choose(self, situation, actions, *, budget_ms: int = 0):
        self.last_reason = ""
        self.last_detail = {}
        first = self.heuristic.choose(situation, actions, budget_ms=budget_ms)
        self.last_reason = getattr(self.heuristic, "last_reason", "")
        self.last_detail = dict(getattr(self.heuristic, "last_detail", {}) or {})
        if first is None or getattr(first, "kind", None) != DISCARD:
            return first
        if situation.drawn_tile is None:
            return first

        best = None
        scored: list[tuple[float, str]] = []
        for action in actions:
            if getattr(action, "kind", None) != DISCARD or action.tile is None:
                continue
            if situation.hand.counts[action.tile] <= 0:
                continue
            after = _without_tile(situation, action.tile)
            value = pure_forward(self.payload, self.feature_extract(after))
            scored.append((value, action.describe()))
            if best is None or value > best[0]:
                best = (value, action)
        if best is None:
            return first
        scored.sort(reverse=True)
        self.last_detail["value"] = [f"{n}={v:.2f}" for v, n in scored[:4]]
        self.last_reason = f"MLP 价值：{scored[0][1]} 价值 {scored[0][0]:.2f}"
        return best[1]


def register_mlp_value_arm(payload: dict, label: str = "mlp-value") -> str:
    """把 MLP 价值决策器注册进评测器，返回可用于 A/B 的档位名。

    需要在能 import 主仓库的进程里调用（评测器会自动加 ``src/`` 到路径）。
    """
    from . import eval as ev
    from .model import pure_forward  # noqa: F401  (文档用途)

    ev._bootstrap_main_repo()
    from majiang.strategy.features import extract
    from majiang.strategy.policy import HeuristicDecider, PolicyConfig

    def factory():
        return MLPValueDecider(
            payload, HeuristicDecider(PolicyConfig.for_mode(_qualifier_mode())),
            feature_extract=extract, label=label,
        )

    ev.CUSTOM[label] = factory
    return label


def _qualifier_mode():
    from majiang.strategy.policy import Mode

    return Mode.QUALIFIER


def _without_tile(situation, tile: int):
    """返回打出 ``tile`` 后的局面副本。

    优先用主仓库 ``dataclasses.replace`` 语义（经启发式对象不可得时用 duck typing）。
    """
    from dataclasses import replace as _replace

    return _replace(situation, hand=situation.hand.without_tile(tile))


__all__ = ["MLPValueDecider"]
