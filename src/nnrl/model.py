"""MLP 价值模型：训练（torch）+ 纯 Python 导出 + 纯 Python 前向。

沿用主仓库约定（`research/train_nn.py`、`strategy/gbdt.py`）：

1. **纯 Python 可推理**：导出 JSON（权重 + 归一化参数），前向只用标准库，
   `src/majiang/**` 的零第三方依赖约束不被破坏。
2. **种子必填**：省略即拒绝，随机性会让结论失去可比性。
3. **显存预算硬上限**：`set_per_process_memory_fraction`，不靠"小心点"。
4. **训练记录**：数据指纹 / 超参 / 种子 / 依赖版本 / GPU 型号。
5. **特征只来自公开信息**：直接复用主仓库 29 维 `features.extract()` 的输出。
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

FEATURE_COUNT = 29
MODEL_VERSION = "nnrl-mlp-1"

_DEFAULT_HIDDEN = (64, 64)


def set_vram_budget(fraction: float) -> None:
    """硬上限：超过即抛 CUDA OOM，不靠自觉。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def dataset_fingerprint(x, y) -> str:
    import numpy as np

    h = hashlib.sha256()
    for arr in (x, y):
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = _DEFAULT_HIDDEN
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
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
class TrainResult:
    model: "object"  # torch.nn.Module
    mean: "object"
    std: "object"
    best_valid_mse: float
    history: list[dict] = field(default_factory=list)


def _mlp(n_in: int, hidden: tuple[int, ...]):
    import torch

    layers: list = []
    prev = n_in
    for width in hidden:
        layers += [torch.nn.Linear(prev, width), torch.nn.ReLU()]
        prev = width
    layers.append(torch.nn.Linear(prev, 1))
    return torch.nn.Sequential(*layers)


def train_mlp(x, y, valid, cfg: TrainConfig) -> TrainResult:
    import numpy as np
    import torch

    if not cfg.seed:
        raise SystemExit("拒绝执行：必须显式给 --seed（随机性会让结论不可比）")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_vram_budget(cfg.vram_fraction)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8

    def prep(a, b):
        xn = ((a - mean) / std).astype(np.float32)
        return (
            torch.from_numpy(xn).to(device),
            torch.from_numpy(b.astype(np.float32)).reshape(-1, 1).to(device),
        )

    xt, yt = prep(x, y)
    xv, yv = prep(*valid)
    model = _mlp(x.shape[1], cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()

    n = xt.shape[0]
    best = float("inf")
    best_state = None
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.numel()
        model.eval()
        with torch.no_grad():
            valid_mse = float(loss_fn(model(xv), yv))
        history.append({"epoch": epoch, "train_mse": total / n, "valid_mse": valid_mse})
        if valid_mse < best:
            best = valid_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(
        model=model.cpu(),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        best_valid_mse=best,
        history=history,
    )


# --------------------------------------------------------------------------- #
# 纯 Python 导出 / 前向（运行路径：只用标准库）
# --------------------------------------------------------------------------- #


def export_pure_python(result: TrainResult, path: str | Path, *, cfg: TrainConfig, fingerprint: str,
                       gpu: str, torch_version: str) -> dict:
    """导出为纯 Python 可推理 JSON，并返回训练记录。"""
    layers = []
    seq = result.model
    linears = [m for m in seq if hasattr(m, "weight")]
    for i, lin in enumerate(linears):
        layers.append(
            {
                "w": lin.weight.detach().numpy().tolist(),
                "b": lin.bias.detach().numpy().tolist(),
                "act": "linear" if i == len(linears) - 1 else "relu",
            }
        )
    payload = {
        "version": MODEL_VERSION,
        "feature_count": FEATURE_COUNT,
        "mean": result.mean.reshape(-1).tolist(),
        "std": result.std.reshape(-1).tolist(),
        "layers": layers,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "version": MODEL_VERSION,
        "data_fingerprint": fingerprint,
        "config": cfg.as_dict(),
        "valid_mse": result.best_valid_mse,
        "epochs": len(result.history),
        "torch": torch_version,
        "gpu": gpu,
        "python": platform.python_version(),
        "artifact": str(target),
    }


def load_payload(path: str | Path) -> dict:
    """加载纯 Python 产物（只用标准库）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pure_forward(payload: dict, features: "list[float]") -> float:
    """纯 Python 前向：只 import 标准库。"""
    if len(features) != payload["feature_count"]:
        raise ValueError(f"特征维度不符：{len(features)} != {payload['feature_count']}")
    vec = [
        (float(v) - float(m)) / float(s)
        for v, m, s in zip(features, payload["mean"], payload["std"])
    ]
    for layer in payload["layers"]:
        w, b, act = layer["w"], layer["b"], layer["act"]
        out = []
        for row, bias in zip(w, b):
            acc = float(bias)
            for weight, value in zip(row, vec):
                acc += float(weight) * value
            out.append(acc if act == "linear" else max(0.0, acc))
        vec = out
    return vec[0]


__all__ = [
    "FEATURE_COUNT",
    "MODEL_VERSION",
    "TrainConfig",
    "TrainResult",
    "train_mlp",
    "export_pure_python",
    "pure_forward",
    "load_payload",
    "dataset_fingerprint",
    "set_vram_budget",
]
