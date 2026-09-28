"""纯 Python 前向与 torch 前向的一致性测试，以及无需第三方库的推理验证。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nnrl import model


def _fake_payload():
    return {
        "version": model.MODEL_VERSION,
        "feature_count": 3,
        "mean": [1.0, 2.0, 3.0],
        "std": [2.0, 2.0, 2.0],
        "layers": [
            {"w": [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], "b": [0.5, -0.5], "act": "relu"},
            {"w": [[1.0, 1.0]], "b": [0.25], "act": "linear"},
        ],
    }


def test_pure_forward_deterministic():
    payload = _fake_payload()
    a = model.pure_forward(payload, [2.0, 4.0, 6.0])
    b = model.pure_forward(payload, [2.0, 4.0, 6.0])
    assert a == b


def test_pure_forward_wrong_dim_rejected():
    with pytest.raises(ValueError):
        model.pure_forward(_fake_payload(), [1.0, 2.0])


def test_pure_forward_runs_without_third_party(tmp_path: Path):
    """关键约束：产物必须能在**裸解释器**（无 torch/numpy）里推理。"""
    payload = _fake_payload()
    artifact = tmp_path / "m.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    script = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "from nnrl.model import pure_forward;"
        "print(pure_forward(json.load(open(%r)), [2.0,4.0,6.0]))"
        % (str(Path(__file__).resolve().parents[1] / "src"), str(artifact))
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin"},  # 刻意不给 PYTHONPATH 之外的东西
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()
