"""组内 softmax 策略网络 —— 纯 numpy REINFORCE（**不需要 torch**）。

设计动机来自 BC 线的负结果（见 `notes/REPORT-6.6-nnrl.md`）：

- 训练目标必须是**真实对局结果**，不是模仿教师（模仿的最优解 = 教师本人，零增益）。
- 架构必须是**增量式**：M 线已证「整段替换」显著更差。

因此打分形式固定为

    归一化 logit_i = total_i + delta_i,   delta = MLP(候选扁平特征)

``total`` 是启发式自己的评分（候选特征第 9 列），``w3``/``b3`` **零初始化 ⇒ delta≡0**，
故**策略起点 = argmax(total) = 启发式的核心排序规则**。训练只推动 ``delta`` 分支，
不可能把启发式的结构整体丢掉。

网络规模：40 → 64 → 64 → 1，约 6.8k 参数。前向/反向手写，零第三方依赖
（与参赛运行路径的约束一致，便于日后直接移植）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

RL_VERSION = "nnrl-rl-1"


@dataclass
class RLCfg:
    hidden: tuple[int, int] = (64, 64)
    temperature: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 0.01
    l2: float = 0.0
    batch_decisions: int = 1024
    baseline_momentum: float = 0.99

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


def init_params(cfg: RLCfg, rng: np.random.Generator, mean, std, *, total_idx: int, input_dim: int) -> dict:
    """初始化参数。``w3``/``b3`` **必须为零** —— 这是「起点=启发式」的保证。"""
    h1, h2 = cfg.hidden

    def draw(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float64)

    return {
        "W1": draw((input_dim, h1), 1.0 / np.sqrt(input_dim)),
        "b1": np.zeros(h1),
        "W2": draw((h1, h2), 1.0 / np.sqrt(h1)),
        "b2": np.zeros(h2),
        "w3": np.zeros(h2),
        "b3": np.zeros(1),
        "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "std": np.asarray(std, dtype=np.float64).reshape(-1) + 1e-8,
        "total_idx": int(total_idx),
        "input_dim": int(input_dim),
    }


def normalize(p: dict, flat: np.ndarray) -> np.ndarray:
    return (np.asarray(flat, dtype=np.float64) - p["mean"]) / p["std"]


def _forward(p: dict, F: np.ndarray):
    pre1 = F @ p["W1"] + p["b1"]
    h1 = np.maximum(pre1, 0.0)
    pre2 = h1 @ p["W2"] + p["b2"]
    h2 = np.maximum(pre2, 0.0)
    delta = h2 @ p["w3"] + p["b3"]
    return pre1, h1, pre2, h2, delta


def _dist(p: dict, F: np.ndarray, mask: np.ndarray, temperature: float):
    _, _, _, _, delta = _forward(p, F)
    logits = (F[:, p["total_idx"]] + delta) / float(temperature)
    logits = np.where(mask > 0, logits, -1e18)
    z = logits - logits.max()
    e = np.exp(z) * (mask > 0)
    total = e.sum()
    if total <= 0:  # 极端兜底：掩码全空 ⇒ 均匀
        e = mask.astype(np.float64).copy()
        if e.sum() <= 0:
            e = np.ones_like(e)
        total = e.sum()
    return logits, e / total


def act(p: dict, F: np.ndarray, mask: np.ndarray, *, sample: bool, gen: np.random.Generator | None = None,
        temperature: float | None = None) -> int:
    """选一张牌。``sample=True`` 训练期按分布采样；``False`` 取 argmax（评测）。"""
    T = p.get("_temperature", 1.0) if temperature is None else temperature
    _, pr = _dist(p, F, mask, T)
    if sample:
        return int(gen.choice(len(pr), p=pr))
    return int(np.argmax(pr))


def train_step(p: dict, batch, cfg: RLCfg, *, baseline: float) -> tuple[float, float]:
    """一次 REINFORCE 更新。``batch`` 为 ``(F归一化, mask, chosen, reward)`` 列表。

    返回 ``(平均奖励, 更新前基线)``。
    """
    keys = ("W1", "b1", "W2", "b2", "w3", "b3")
    grads = {k: np.zeros_like(p[k], dtype=np.float64) for k in keys}
    rewards: list[float] = []
    for F, mask, chosen, reward in batch:
        rewards.append(float(reward))
        pre1, h1, pre2, h2, _ = _forward(p, F)
        logits, pr = _dist(p, F, mask, cfg.temperature)
        adv = float(reward) - float(baseline)

        # dLoss/dlogit
        dl = adv * pr
        dl[chosen] -= adv
        H = -float(np.sum(pr * np.log(pr + 1e-12)))
        dl = dl + cfg.entropy_coef * pr * (np.log(pr + 1e-12) + H)
        dlogit = dl / float(cfg.temperature)  # 链式穿过 /T

        # 反传到 MLP
        grads["w3"] += (h2 * dlogit[:, None]).sum(0)
        grads["b3"] += dlogit.sum()
        dh2 = dlogit[:, None] * p["w3"][None, :]
        dpre2 = dh2 * (pre2 > 0)
        grads["W2"] += np.einsum("mi,mj->ij", h1, dpre2)
        grads["b2"] += dpre2.sum(0)
        dh1 = dpre2 @ p["W2"].T
        dpre1 = dh1 * (pre1 > 0)
        grads["W1"] += np.einsum("mi,mj->ij", F, dpre1)
        grads["b1"] += dpre1.sum(0)

    n = max(1, len(batch))
    used = float(baseline)
    for k in keys:
        g = grads[k] / n
        if cfg.l2 and k in ("W1", "W2"):
            g = g + cfg.l2 * p[k]
        p[k] = p[k] - cfg.lr * g
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    return mean_reward, used


def update_baseline(baseline: float, mean_reward: float, cfg: RLCfg) -> float:
    m = cfg.baseline_momentum
    return m * baseline + (1.0 - m) * mean_reward


def save_params(p: dict, path: str | Path, *, cfg: RLCfg, extra: dict | None = None) -> dict:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    payload = {"version": RL_VERSION, "cfg": cfg.as_dict(), "params": out, "extra": extra or {}}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load_params(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    p = {}
    for k, v in doc["params"].items():
        p[k] = np.asarray(v, dtype=np.float64) if isinstance(v, list) else v
    return p


__all__ = [
    "RL_VERSION",
    "RLCfg",
    "init_params",
    "normalize",
    "act",
    "train_step",
    "update_baseline",
    "save_params",
    "load_params",
]
