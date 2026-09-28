"""离线数据加载（只读主仓库 ``data/``）。

所有函数都不写主仓库、不联网。数据口径与主仓库 `research/FEATURES.md` 一致：
29 维特征（`features.py`），标签 = 本局本人最终得分。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import paths


@dataclass(frozen=True)
class ValueBatch:
    x: "object"  # np.ndarray (n, 29) float32
    y: "object"  # np.ndarray (n,) float32
    seq: "object | None" = None  # np.ndarray (n, 4, 24) int8，仅序列分片有
    files: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.x.shape[0])


def _to_batch(files: list[str], arrays: dict) -> ValueBatch:
    return ValueBatch(
        x=arrays["x"],
        y=arrays["y"],
        seq=arrays.get("seq"),
        files=tuple(files),
    )


def load_value_train() -> ValueBatch:
    """训练集：`data/value_seq_train.w000.*.npz`（含弃牌序列）。"""
    files, arrays = paths.load_npz("value_seq_train.w000.*.npz")
    return _to_batch(files, arrays)


def load_value_valid() -> ValueBatch:
    """验证集：`data/value_valid*.npz`（独立种子批次）。"""
    files, arrays = paths.load_npz("value_valid*.npz")
    return _to_batch(files, arrays)


def load_value_post() -> ValueBatch:
    """后期批次：`data/value_post*.npz`（可作额外验证/时序外样本）。"""
    files, arrays = paths.load_npz("value_post*.npz")
    return _to_batch(files, arrays)


__all__ = ["ValueBatch", "load_value_train", "load_value_valid", "load_value_post"]
