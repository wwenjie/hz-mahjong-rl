"""对主仓库的**只读**访问桥。

本仓库（NN/RL/MLP 研究线）与参赛主仓库 `hz-mahjong-ai` 是两套所有权。为了在不违反
主仓库「同一文件只有一个所有人」铁律的前提下复用其**公开离线数据**，这里把所有跨仓
访问收敛到一个地方，并把「只读」做成代码约束而不是口头约定：

- :func:`repo_root` 解析主仓库根（``MAJIANG_REPO``，默认 ``/home/wuwenjie01/majiang_ai``）。
- :func:`data_dir` / :func:`iter_event_files` / :func:`load_npz` 是仅有的读取入口。
- :func:`forbid_write` 拒绝任何指向主仓库的写入；:class:`ReadOnlyViolation` 是它的异常。

**禁止在本模块之外拼接主仓库路径。** 需要新的只读入口时在这里加函数，不要在别处
直接 ``open(...)``。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

DEFAULT_REPO = "/home/wuwenjie01/majiang_ai"
ENV_REPO = "MAJIANG_REPO"


class ReadOnlyViolation(RuntimeError):
    """试图向主仓库写入时抛出。"""


def repo_root() -> Path:
    """主仓库根目录（只读）。"""
    return Path(os.environ.get(ENV_REPO, DEFAULT_REPO)).resolve()


def data_dir() -> Path:
    """主仓库的 ``data/`` 目录（只读）。"""
    return repo_root() / "data"


def _resolve_readable(path: str | Path) -> Path:
    root = repo_root()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = data_dir() / candidate
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ReadOnlyViolation(f"路径不在主仓库内：{candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"主仓库中不存在：{candidate}")
    return candidate


def forbid_write(path: str | Path) -> None:
    """任何写入前调用：路径若落在主仓库内，直接拒绝。

    本仓库自身的文件不受影响（主仓库不会被当作本仓库的子路径，两者是并列目录）。
    """
    root = repo_root()
    candidate = Path(path)
    try:
        candidate = candidate.resolve()
    except OSError:  # 目标尚不存在时 resolve 仍可工作；保守起见按字符串判断
        candidate = Path(os.path.abspath(str(path)))
    if candidate == root or root in candidate.parents:
        raise ReadOnlyViolation(
            f"拒绝写入主仓库（只读边界）：{candidate}\n"
            f"主仓库根本应只读；NN/RL 产物请写入本仓库自身。"
        )


def iter_event_files(repo: str | Path | None = None) -> Iterator[Path]:
    """遍历 ``data/auto_sessions/*/events/*.json``（真实对局事件流，只读）。

    按**房**（``auto_sessions/<房号>``）分组返回的迭代顺序由调用方在需要时懒处理；
    这里只保证逐文件产出，抽样纪律（按房分层）由调用方负责。
    """
    base = data_dir() if repo is None else _resolve_readable(repo)
    events_root = base / "auto_sessions"
    if not events_root.is_dir():
        return
    for room in sorted(p for p in events_root.iterdir() if p.is_dir()):
        events = room / "events"
        if not events.is_dir():
            continue
        for path in sorted(events.glob("*.json")):
            yield path


def load_npz(pattern: str):
    """以只读方式加载主仓库 ``data/`` 下的 ``.npz`` 分片。

    返回 ``(files, arrays)``：文件列表与按顺序拼接后的字典。需要 numpy；
    仅供离线训练使用，不进入任何运行路径。
    """
    import glob

    import numpy as np

    base = data_dir()
    files = sorted(glob.glob(str(base / pattern)))
    if not files:
        raise FileNotFoundError(f"未匹配到任何分片：{base / pattern}")
    collected: dict[str, list] = {}
    for path in files:
        _resolve_readable(path)
        with np.load(path) as doc:
            for key in doc.files:
                collected.setdefault(key, []).append(doc[key])
    merged = {key: np.concatenate(values) for key, values in collected.items()}
    return files, merged


__all__ = [
    "DEFAULT_REPO",
    "ENV_REPO",
    "ReadOnlyViolation",
    "repo_root",
    "data_dir",
    "forbid_write",
    "iter_event_files",
    "load_npz",
]
