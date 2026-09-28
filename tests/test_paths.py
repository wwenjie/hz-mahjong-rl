"""只读边界测试：主仓库不可写、可读入口存在、路径逃逸被拒。"""

from __future__ import annotations

from pathlib import Path

import pytest

from nnrl import paths


def test_repo_root_exists():
    assert paths.repo_root().is_dir()


def test_data_dir_readable():
    assert paths.data_dir().is_dir()


def test_forbid_write_rejects_repo_paths():
    target = paths.repo_root() / "notes" / "SHOULD-NEVER-EXIST.md"
    with pytest.raises(paths.ReadOnlyViolation):
        paths.forbid_write(target)


def test_forbid_write_allows_own_repo(tmp_path: Path):
    # 本仓库自身的文件不受边界影响
    paths.forbid_write(tmp_path / "runs" / "model.pt")


def test_resolve_readable_blocks_escape():
    with pytest.raises(paths.ReadOnlyViolation):
        paths._resolve_readable("/etc/passwd")


def test_iter_event_files_or_empty():
    # 主仓库可能暂无事件流；有则必须是只读目录内的 json
    for path in paths.iter_event_files():
        assert path.suffix == ".json"
        assert paths.data_dir() in path.parents
        break
