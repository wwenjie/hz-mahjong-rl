"""出牌策略网：在启发式候选上逐张打分（学残差），训练 + 纯 Python 导出 + 纯 Python 前向。

设计依据（`notes/research-nn-design.md`）：

- **动作表示**：逐张打分（Suphx/Meowjong 同构），吃/碰/杠/胡不压进本网络。
- **增量而非替换**：每张候选的输入里已含启发式的全部评分明细（含 `total`），
  网络只需学"在哪些局面上该偏离启发式、往哪偏"。
- **纯 Python 可推理**：沿用 `model.py` 的导出模式，前向只用标准库。

M 线负结果（整段替换 → 名次分 −3.425, t −6.67）是本模块存在的原因：
网络不得重新发明结构性知识，只能在其上做修正。
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bc import CAND_FEATURE_COUNT

GLOBAL_FEATURE_COUNT = 29
POLICY_VERSION = "nnrl-policy-1"
KINDS = 34
INPUT_PER_CAND = GLOBAL_FEATURE_COUNT + CAND_FEATURE_COUNT  # 39


@dataclass
class PolicyConfig:
    hidden: tuple[int, ...] = (128, 128)
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    vram_fraction: float = 0.5

    def as_dict(self) -> dict:
        return {
            "hidden": list(self.hidden),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "vram_fraction": self.vram_fraction,
        }


@dataclass
class PolicyResult:
    model: "object"
    mean: "object"
    std: "object"
    best_valid_agreement: float
    history: list[dict] = field(default_factory=list)


def assemble(x, cand) -> np.ndarray:
    """(n,29) + (n,34,10) -> (n,34,39)：每张候选一份「全局 + 该候选明细」。"""
    n = x.shape[0]
    glob = np.repeat(x[:, None, :], KINDS, axis=1)
    return np.concatenate([glob, cand], axis=2).astype(np.float32)


def _masked_logits(model, flat, shape):
    import torch

    logits = model(flat).reshape(shape[0], shape[1])
    return logits


def train_policy(data, valid, cfg: PolicyConfig) -> PolicyResult:
    import torch

    if not cfg.seed:
        raise SystemExit("拒绝执行：必须显式给 --seed")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    from .model import set_vram_budget

    set_vram_budget(cfg.vram_fraction)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xtr, xtr_c = data.x, data.cand
    mtr, ytr = data.mask, data.y
    xva, xva_c = valid.x, valid.cand
    mva, yva = valid.mask, valid.y

    tr_all = assemble(xtr, xtr_c)  # (n,34,39)
    # 归一化统计量只在**被掩码的候选**上计算
    sel = mtr.reshape(-1) > 0
    flat_tr = tr_all.reshape(-1, INPUT_PER_CAND)[sel]
    mean = flat_tr.mean(axis=0, keepdims=True)
    std = flat_tr.std(axis=0, keepdims=True) + 1e-8
    del flat_tr

    def prep(x, cand, mask):
        flat = assemble(x, cand).reshape(-1, INPUT_PER_CAND)
        flat = ((flat - mean) / std).astype(np.float32)
        return (
            torch.from_numpy(flat).to(device),
            torch.from_numpy(mask.astype(np.float32)).to(device),
            torch.from_numpy(mask.shape).to(device),
        )

    ftr, mtr_t, shp_tr = prep(xtr, xtr_c, mtr)
    fva, mva_t, shp_va = prep(xva, xva_c, mva)
    ytr_t = torch.from_numpy(ytr.astype(np.int64)).to(device)
    yva_t = torch.from_numpy(yva.astype(np.int64)).to(device)

    layers: list = []
    prev = INPUT_PER_CAND
    for width in cfg.hidden:
        layers += [torch.nn.Linear(prev, width), torch.nn.ReLU()]
        prev = width
    layers.append(torch.nn.Linear(prev, 1))
    model = torch.nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    n = shp_tr[0].item()
    best = -1.0
    best_state = None
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            # 该 batch 的候选行索引
            rows = (idx[:, None] * KINDS + torch.arange(KINDS, device=device)[None, :]).reshape(-1)
            b_shape = torch.tensor([idx.numel(), KINDS], device=device)
            logits = _masked_logits(model, ftr[rows], b_shape)
            masked = logits.masked_fill(mtr_t[idx] <= 0, -1e9)
            loss = ce(masked, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            rows = torch.arange(mva_t.shape[0], device=device)[:, None] * KINDS + torch.arange(
                KINDS, device=device
            )[None, :]
            rows = rows.reshape(-1)
            logits = _masked_logits(model, fva[rows], shp_va)
            masked = logits.masked_fill(mva_t <= 0, -1e9)
            agree = float((masked.argmax(dim=1) == yva_t).float().mean())
        history.append({"epoch": epoch, "valid_agreement": agree})
        if agree > best:
            best = agree
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return PolicyResult(
        model=model.cpu(),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        best_valid_agreement=best,
        history=history,
    )


def teacher_agreement(data) -> float:
    """启发式自身（argmax total，忽略 tiebreak）在候选上的 top-1 一致率。

    这是"教师可否被学生逼近"的参照上界之一；<1 说明教师含 total 之外的 tiebreak 逻辑。
    """
    total = data.cand[:, :, CAND_FEATURE_COUNT - 1]
    guess = np.argmax(np.where(data.mask > 0, total, -1e9), axis=1)
    return float((guess == data.y).mean())


def export_policy(result: PolicyResult, path: str | Path, *, cfg: PolicyConfig,
                  fingerprint: str, gpu: str, torch_version: str) -> dict:
    linears = [m for m in result.model if hasattr(m, "weight")]
    layers = [
        {
            "w": lin.weight.detach().numpy().tolist(),
            "b": lin.bias.detach().numpy().tolist(),
            "act": "linear" if i == len(linears) - 1 else "relu",
        }
        for i, lin in enumerate(linears)
    ]
    payload = {
        "version": POLICY_VERSION,
        "input_per_cand": INPUT_PER_CAND,
        "global_feature_count": GLOBAL_FEATURE_COUNT,
        "cand_feature_count": CAND_FEATURE_COUNT,
        "kinds": KINDS,
        "mean": result.mean.reshape(-1).tolist(),
        "std": result.std.reshape(-1).tolist(),
        "layers": layers,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "version": POLICY_VERSION,
        "data_fingerprint": fingerprint,
        "config": cfg.as_dict(),
        "valid_agreement": result.best_valid_agreement,
        "epochs": len(result.history),
        "torch": torch_version,
        "gpu": gpu,
        "python": platform.python_version(),
        "artifact": str(target),
    }


# --------------------------------------------------------------------------- #
# 纯 Python 前向（运行路径：只用标准库）
# --------------------------------------------------------------------------- #


def policy_scores(payload: dict, global_features: list[float], cand_matrix: list[list[float]],
                  mask: list[float]) -> list[float]:
    """对 34 张牌各返回一个分数；被掩码的返回 -inf。纯标准库。"""
    out: list[float] = []
    for tile in range(payload["kinds"]):
        if mask[tile] <= 0:
            out.append(float("-inf"))
            continue
        vec = [float(v) for v in global_features] + [float(v) for v in cand_matrix[tile]]
        vec = [
            (v - float(m)) / float(s)
            for v, m, s in zip(vec, payload["mean"], payload["std"])
        ]
        for layer in payload["layers"]:
            w, b, act = layer["w"], layer["b"], layer["act"]
            nxt = []
            for row, bias in zip(w, b):
                acc = float(bias)
                for weight, value in zip(row, vec):
                    acc += float(weight) * value
                nxt.append(acc if act == "linear" else max(0.0, acc))
            vec = nxt
        out.append(vec[0])
    return out


def policy_pick(payload: dict, global_features: list[float], cand_matrix: list[list[float]],
                mask: list[float]) -> int:
    """纯 Python 选牌（argmax）。"""
    scores = policy_scores(payload, global_features, cand_matrix, mask)
    best = max(range(len(scores)), key=lambda t: scores[t])
    return best


__all__ = [
    "POLICY_VERSION",
    "INPUT_PER_CAND",
    "KINDS",
    "PolicyConfig",
    "PolicyResult",
    "assemble",
    "train_policy",
    "teacher_agreement",
    "export_policy",
    "policy_scores",
    "policy_pick",
]
