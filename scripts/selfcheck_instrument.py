"""仪表自检：对称设置（heuristic vs heuristic）应当差分恒为 0。

这是评测器的**已知答案检验**：四个座位、四个旋转全部是同一个决策器，任何非零差分
都只可能来自座次/庄位/时序偏置。主仓库的历史教训是「固定座位的对比测出 +5.7pp 假优势」，
所以本检查在每次评测口径变更后都应重跑。

用法（约 80 秒）::

    PYTHONPATH=src uv run python scripts/selfcheck_instrument.py
"""
from __future__ import annotations

import sys

from nnrl import eval as ev


def main() -> int:
    r = ev.paired_ab("heuristic", "heuristic", matches=3, rounds=4, seed=20260928)
    print(r.report())
    bad = any(any(abs(v) > 1e-9 for v in values) for values in r.differences.values())
    print("自检", "失败：出现非零差分" if bad else "通过：全指标差分恒为 0")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
