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
    # 残差架构：logit = skip × 归一化 total + MLP 修正。
    # 目的是让学生**从起点就不劣于教师**（M 线教训：整段替换会丢掉教师的强结构）。
    residual: bool = True

    def as_dict(self) -> dict:
        return {
            "hidden": list(self.hidden),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "vram_fraction": self.vram_fraction,
            "residual": self.residual,
        }


@dataclass
class PolicyResult:
    model: "object"
    mean: "object"
    std: "object"
    best_valid_agreement: float
    skip: float = 1.0
    history: list[dict] = field(default_factory=list)


def assemble(x, cand) -> np.ndarray:
    """(n,29) + (n,34,10) -> (n,34,39)：每张候选一份「全局 + 该候选明细」。"""
    n = x.shape[0]
    glob = np.repeat(x[:, None, :], KINDS, axis=1)
    return np.concatenate([glob, cand], axis=2).astype(np.float32)


def _masked_logits(model, flat, shape, *, skip=None, residual=False):
    import torch

    logits = model(flat).reshape(shape[0], shape[1])
    if residual:
        # **用扁平索引**：候选特征在向量里有 GLOBAL_FEATURE_COUNT 的偏移
        z = flat.reshape(shape[0], shape[1], INPUT_PER_CAND)[:, :, _flat_index("total")]
        logits = logits + skip * z
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
        )

    ftr, mtr_t = prep(xtr, xtr_c, mtr)
    fva, mva_t = prep(xva, xva_c, mva)
    n_tr = int(mtr.shape[0])
    n_va = int(mva.shape[0])
    ytr_t = torch.from_numpy(ytr.astype(np.int64)).to(device)
    yva_t = torch.from_numpy(yva.astype(np.int64)).to(device)

    layers: list = []
    prev = INPUT_PER_CAND
    for width in cfg.hidden:
        layers += [torch.nn.Linear(prev, width), torch.nn.ReLU()]
        prev = width
    layers.append(torch.nn.Linear(prev, 1))
    model = torch.nn.Sequential(*layers).to(device)
    # 末层零初始化 ⇒ 初始 logits = skip×total，起点严格等于教师（平凡基线）
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.zero_()
    # 残差跳连：初始 1.0 ⇒ 学生起点 = 教师的 total 排序（平凡基线）
    skip = torch.nn.Parameter(torch.ones(1, device=device))
    params = list(model.parameters()) + ([skip] if cfg.residual else [])
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    def fwd(flat, shape):
        return _masked_logits(model, flat, shape, skip=skip, residual=cfg.residual)

    def agreement(fx, mx, yy):
        with torch.no_grad():
            lg = fwd(fx, (int(yy.numel()), KINDS)).masked_fill(mx <= 0, -1e9)
            return float((lg.argmax(1) == yy).float().mean())

    n = n_tr
    best = -1.0
    best_state = None
    history: list[dict] = []
    # 起点一致率（零 MLP + skip=1 应恰好等于教师基线）——用于确认管线无偏
    init_agree = agreement(
        fva[
            (
                torch.arange(n_va, device=device)[:, None] * KINDS
                + torch.arange(KINDS, device=device)[None, :]
            ).reshape(-1)
        ],
        mva_t,
        yva_t,
    )
    history.append({"epoch": 0, "valid_agreement": init_agree})
    # **起点必须作为候选最优**：残差架构下它就是教师基线，天然不劣。
    # 若只在“有提升”时保存，一旦训练未能超过起点，best_state 保持 None，
    # 模型会留在最后一轮的差权重上——导出后一致率远低于报告值（已踩坑）。
    best = init_agree
    best_state = {
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "skip": skip.detach().clone(),
    }
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            # 该 batch 的候选行索引
            rows = (idx[:, None] * KINDS + torch.arange(KINDS, device=device)[None, :]).reshape(-1)
            logits = fwd(ftr[rows], (int(idx.numel()), KINDS))
            masked = logits.masked_fill(mtr_t[idx] <= 0, -1e9)
            loss = ce(masked, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            rows = (
                torch.arange(n_va, device=device)[:, None] * KINDS
                + torch.arange(KINDS, device=device)[None, :]
            ).reshape(-1)
            logits = fwd(fva[rows], (n_va, KINDS))
            masked = logits.masked_fill(mva_t <= 0, -1e9)
            agree = float((masked.argmax(dim=1) == yva_t).float().mean())
        history.append({"epoch": epoch, "valid_agreement": agree})
        if agree > best:
            best = agree
            best_state = {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "skip": skip.detach().clone(),
            }
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        with torch.no_grad():
            skip.copy_(best_state["skip"])
    return PolicyResult(
        model=model.cpu(),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        best_valid_agreement=best,
        skip=float(skip.detach().cpu()),
        history=history,
    )


def _feature_index(name: str) -> int:
    """``total`` 等在**候选矩阵内**的列号（用于 ``data.cand``）。"""
    from .bc import CAND_FEATURE_NAMES

    return CAND_FEATURE_NAMES.index(name)


def _flat_index(name: str) -> int:
    """同一特征在**扁平 per-candidate 向量** ``[全局 | 候选]`` 中的位置。

    两个索引**不能混用**：曾因在扁平向量上用候选内索引（9）导致残差跳连
    接到全局特征上，学生初始化远低于基线。
    """
    return GLOBAL_FEATURE_COUNT + _feature_index(name)


def teacher_agreement(data) -> float:
    """启发式自身（argmax total，忽略 tiebreak）在候选上的 top-1 一致率。

    这是"教师可否被学生逼近"的参照上界之一；<1 说明教师含 total 之外的 tiebreak 逻辑。
    **按名称取列**（勿用 -1），否则新增特征会静默改变本函数的含义。
    """
    total = data.cand[:, :, _feature_index("total")]
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
        "residual": bool(cfg.residual),
        "skip": float(getattr(result, "skip", 1.0)),
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
        z_total = vec[_flat_index("total")] if payload.get("residual") else 0.0
        for layer in payload["layers"]:
            w, b, act = layer["w"], layer["b"], layer["act"]
            nxt = []
            for row, bias in zip(w, b):
                acc = float(bias)
                for weight, value in zip(row, vec):
                    acc += float(weight) * value
                nxt.append(acc if act == "linear" else max(0.0, acc))
            vec = nxt
        score = vec[0] + float(payload.get("skip", 1.0)) * z_total
        out.append(score)
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
