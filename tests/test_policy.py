"""BC 数据管线与策略网的一致性测试（不依赖 torch 的部分）。"""

from __future__ import annotations

import numpy as np

from nnrl import bc, policy_net


def _fake_bc() -> bc.BCData:
    n = 6
    x = np.random.RandomState(0).rand(n, 29).astype(np.float32)
    cand = np.zeros((n, 34, bc.CAND_FEATURE_COUNT), dtype=np.float32)
    mask = np.zeros((n, 34), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    rs = np.random.RandomState(1)
    for i in range(n):
        picks = rs.choice(34, size=4, replace=False)
        for t in picks:
            mask[i, t] = 1.0
            cand[i, t, -1] = rs.rand()  # total
        y[i] = picks[int(rs.randint(0, 4))]
    return bc.BCData(x, cand, mask, y, n)


def test_assemble_shape():
    data = _fake_bc()
    flat = policy_net.assemble(data.x, data.cand)
    assert flat.shape == (6, 34, policy_net.INPUT_PER_CAND)
    # 全局特征在每张候选上重复
    assert np.allclose(flat[0, :, 0], data.x[0, 0])


def test_teacher_agreement_bounds():
    data = _fake_bc()
    a = policy_net.teacher_agreement(data)
    assert 0.0 <= a <= 1.0


def test_pure_python_policy_pick_masks():
    """纯 Python 前向必须只在被掩码的候选里选。"""
    payload = {
        "version": policy_net.POLICY_VERSION,
        "input_per_cand": policy_net.INPUT_PER_CAND,
        "global_feature_count": 29,
        "cand_feature_count": bc.CAND_FEATURE_COUNT,
        "kinds": 34,
        "mean": [0.0] * policy_net.INPUT_PER_CAND,
        "std": [1.0] * policy_net.INPUT_PER_CAND,
        "layers": [
            {"w": [[1.0] * policy_net.INPUT_PER_CAND], "b": [0.0], "act": "linear"},
        ],
    }
    x = [0.0] * 29
    cand = [[0.0] * bc.CAND_FEATURE_COUNT for _ in range(34)]
    cand[5][-1] = 3.0  # 只有 5 号牌有大 total
    cand[9][-1] = 1.0
    mask = [0.0] * 34
    mask[5] = 1.0
    mask[9] = 1.0
    pick = policy_net.policy_pick(payload, x, cand, mask)
    assert pick in (5, 9)
    # 取 total 大者
    assert pick == 5


def test_pure_python_policy_scores_neg_inf_when_masked():
    payload = {
        "version": policy_net.POLICY_VERSION,
        "input_per_cand": policy_net.INPUT_PER_CAND,
        "global_feature_count": 29,
        "cand_feature_count": bc.CAND_FEATURE_COUNT,
        "kinds": 34,
        "mean": [0.0] * policy_net.INPUT_PER_CAND,
        "std": [1.0] * policy_net.INPUT_PER_CAND,
        "layers": [{"w": [[1.0] * policy_net.INPUT_PER_CAND], "b": [0.0], "act": "linear"}],
    }
    mask = [0.0] * 34
    mask[3] = 1.0
    scores = policy_net.policy_scores(payload, [0.0] * 29, [[0.0] * 10] * 34, mask)
    assert scores[3] != float("-inf")
    assert scores[0] == float("-inf")
