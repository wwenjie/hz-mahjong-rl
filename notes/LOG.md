# NN/RL/MLP 研究线 — 计划与结论（追加式）

本文件为追加式记录。每个结论必须含：**样本量、口径、噪声底、跨种子分布、产物路径**。

---

## 2026-09-28 — 立项

### 背景（独立核实，非转述）

主仓库 `hz-mahjong-ai` 现况（读自 `notes/STATUS.md` / `PROTOCOL.md` / `OWNERSHIP.md` /
`research/**` / `openspec/changes/openclaw-deploy`）：

- 三名成员 A（策略主线）、B（独立验证/稳定性）、C（部署/值守/RL-NN 研究）。
- 当前默认档 `heuristic`（`tiebreak=exact-ukeire`，即 v2 冻结版本）；`tools/ab_test.py`
  正跑 6 个 job；价值模型 `models/value_model.json` 是 29 维 GBDT，**未进运行路径**。
- C 的研究线（原 `research/**`）结论：**价值侧已触顶**——T1（NN vs GBDT）在噪声底内打平；
  T2（序列表征）净增益 +0.69 MSE / 143.7 = **0.48%**，且排序一致率无改善 → 该线判死。
- A 的机制级结论：**瓶颈在决策侧的听口宽度**（爆头是宽听口的下游，不是独立目标）。
  `openspec` 的 6.6「新模型 vs 基线对拍」仍是**未勾选项**。

### 边界

- 主仓库**只读**（`src/nnrl/paths.py` 代码层强制）；本仓库全权所有。
- 零平台请求；不碰 A 的采集进程与令牌；离线重活 `nice -n 15` 且同时只跑一个。

### 待用户拍板

1. 远端仓库地址（`github.com/wwenjie/<name>`）。
2. 目标：进 10/8 比赛 vs 赛后研究线。
3. GPU 使用授权边界（只读数据、不挤 A 的 CPU 队列）。
