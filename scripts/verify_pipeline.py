#!/usr/bin/env python
"""全链路验证：纯 Python 前向 = torch 前向；自研决策器可接进对局。

用法::
    PYTHONPATH=src uv run python scripts/verify_pipeline.py runs/<name>/model.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def check_parity(artifact: Path, n: int = 200) -> bool:
    """纯 Python 前向与 torch 前向在同一批样本上逐位对比。"""
    import torch

    from nnrl import data
    from nnrl.model import pure_forward

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    valid = data.load_value_valid()
    x = valid.x[:n]
    mean = np.array(payload["mean"], dtype=np.float32)
    std = np.array(payload["std"], dtype=np.float32)
    xn = (x - mean) / std

    # torch 侧
    vec = torch.from_numpy(xn.astype(np.float32))
    for layer in payload["layers"]:
        w = torch.tensor(layer["w"], dtype=torch.float32)
        b = torch.tensor(layer["b"], dtype=torch.float32)
        vec = vec @ w.t() + b
        if layer["act"] == "relu":
            vec = torch.relu(vec)
    torch_out = vec.reshape(-1).numpy()

    py_out = np.array([pure_forward(payload, row.tolist()) for row in x], dtype=np.float32)
    max_abs = float(np.max(np.abs(torch_out - py_out)))
    print(f"[parity] n={n} 最大绝对差 = {max_abs:.3e}")
    return max_abs < 1e-3


def check_in_match(artifact: Path) -> bool:
    """把 MLP 价值决策器接进对局（小样本冒烟，不作为统计结论）。"""
    from nnrl import eval as ev

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    from nnrl.decider import register_mlp_value_arm

    label = register_mlp_value_arm(payload, label="mlp-value")
    r = ev.paired_ab(label, "heuristic", matches=2, rounds=4, seed=20260928)
    print(r.report())
    return True


def main() -> int:
    artifact = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/mlp-value-s20260928/model.json")
    if not artifact.exists():
        raise SystemExit(f"产物不存在：{artifact}")
    ok1 = check_parity(artifact)
    ok2 = check_in_match(artifact)
    print("全链路", "通过" if (ok1 and ok2) else "失败")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
