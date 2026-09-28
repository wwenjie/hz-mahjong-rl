# 调研：在「学习策略去击败/增强一个强启发式」设定下，如何提高信噪比 / 拿到正增益

- **日期**：2026-09-28
- **作者**：team-researcher（子代理），受 `team-coordinator` 指派
- **仓库**：`/home/wuwenjie01/majiang_rl`（**只读调研**；本文件是本任务唯一写入物，未改代码、未改其它文件、未 commit）
- **检索方式**：arXiv API（`https://export.arxiv.org/api/query`，`id_list` + 关键词，`-L` 跟随 301）；OpenReview API（`api2.openreview.net`）作为非 arXiv 源的旁证。
  **本环境的 `web_search` 工具不可用**（`web_search is disabled or no provider is available`），因此**没有做通用网页检索**，可能遗漏不在 arXiv/OpenReview 上的资料（期刊、技术博客、比赛报告）。
- **证据等级标注**（贯穿全文）：
  - **【摘要确证】**＝本次检索中通过 arXiv API 取回了该文的**标题/作者/日期/摘要原文**（摘要原文已过目，未取全文正文）。
  - **【仅标题】**＝只核对到条目存在，**未读摘要**，不据此下结论。
  - **【推测】**＝我的推断或映射，**不是**论文结论。

---

## 0. 先把我们自己的现状钉死（后面所有建议都以它为准）

来自本仓库 `notes/REPORT-RL-20m.md`、`notes/REPORT-6.6-nnrl.md`、`notes/PLAN-nn-rl.md`、`notes/LOG.md`（均为**已读确证**的本地记录）：

| 线 | 形态 | 结果 |
|---|---|---|
| M 线 | MLP 价值网**整段替换**出牌（29 维 → (64,64) → 1） | 名次分 −3.425（t −6.67）/ −4.250（t −8.24），**双种子显著为负** |
| BC 线 | 从启发式教师学**候选排序** | valid 一致率最优 = **0.9462 = 教师本身**，**零增益**；教师行为 ≈ `argmax(total)`，剩余 5.38% 中 2.6% 非 argmax 可表达 |
| R 线 | 纯 numpy REINFORCE 自对弈，**在启发式候选集内重排**（`logit = total + delta`，`w3/b3` 零初始化 ⇒ 起点 = 启发式） | 名次分 −0.988（t −2.17）/ −1.575（t −3.04）；胡次数 −0.275（t −2.36）/ −0.438（t −2.94），**双种子一致为负** |

**R 线的关键结构事实（读代码确证，`scripts/train_rl_par.py` + `src/nnrl/rl_net.py`）：**

1. 样本量：**40 episode × 8 局 = 320 局**，共 **17,863 个出牌决策**；训练耗时 709.8s（8 worker）。
2. **奖励 = `outcome.scores[seat]`**（即**单局最终得分差**，代码 `rewards.append(outcome.scores[seat])`），压 `÷ reward_scale(=10.0)` 后**原封不动地赋给该局内该座位的每一个出牌决策**。
3. **没有任何** advantage normalization / GAE / RLOO / 组内相对 / 折扣 / 回报分解。
4. baseline 只有**单一标量滑动平均**（`momentum=0.99`），且**按 batch 更新、初值 0.0**——40 局只有 ~17 次 batch 更新，前期基线远未收敛。
5. **梯度按决策求和**（`grads[k] / n`，n=该 batch 决策数），**没有按「本局决策数」归一** ⇒ 长局（决策多、且往往流局/被自摸）对梯度的贡献被放大。
6. 起点 = 启发式，是**由构造保证**的等价（冒烟记录 `records/ab-rl-smoke.json`：名次分/胡次数/白板数**逐位 +0.000**）⇒ **训练把它改坏**，不是接口问题。
7. 判据的**噪底极硬**：主仓库 `verify/noise_floor.py` 口径下，两臂差 SD ≈ **1.33pp**，检出 1.25pp 需**每臂 ≈243 房 ≈ 102 小时串行**（`REPORT-6.6-nnrl.md` §1）。而我们现在的 A/B 只有 **80 配对**（20 场 × 4 旋转 × 2 种子）。

**由此得到三条"靶心"（后文每条方向都要对齐它们）：**

- **靶心 A（信用分配）**：奖励是**一局一个标量**，却要监督**局内几十个决策**。文献里这叫 delayed / sparse credit assignment。
- **靶心 B（重尾 + 估计量）**：奖励是"单局得分差"，**番型连乘 + 庄家 ×8**（`src/majiang/rules/score.py`：`DEALER_MULTIPLIER = 8`）⇒ 单局动辄 ±32/±96（回归用例给的是 `fan=4` 庄家自摸四家净分 `[-32, 96, -32, -32]`），**方差极大且与"这一步打得好不好"几乎无关**。
- **靶心 C（相对基准）**：我们要的是**"超过启发式"**，不是"任意可玩策略"。任何"从零学"的路径都会先花掉全部预算去**重新发明启发式已经精确掌握的结构**（精确进张排序 / 喂牌代价 / 财神罚 / 吃碰闸门）。

---

## 1. 方向一：稀疏/序列化奖励下的方差缩减（advantage normalization / GAE / RLOO / GRPO / 基线）

### 1.1 GRPO（组内相对优势）

- **论文/来源**：*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*，Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song 等，2024（arXiv:2402.03300，v3）。**【摘要确证】**
  链接：https://arxiv.org/abs/2402.03300
- **方法核心**：**不用 critic（价值网）**，对同一状态采样一组（group）动作，用**组内奖励的均值/标准差**把每个样本的回报变成相对优势 `A_i = (r_i − mean(r)) / std(r)`。等价于"用同状态的其它样本当 baseline"。
- **为什么可能解决我们的问题**：直接命中**靶心 A**——我们现在把"整局标量"硬贴给每个决策，而 GRPO 的思路是**在同一状态下做组内比较**。麻将里天然可以造组：同一局面 + 同一牌墙前缀，**只改一个出牌决策**，其余用启发式走完，比较 2–4 条分支的最终得分 ⇒ 优势 = 分支间差。这把"跨局重尾噪声"换成"同局配对差分"，**方差从跨局量级降到同局配对量级**。
- **可行性：中（改造后可到高）**。我们**已经有**这个能力：`train_rl_par.py` 用 `seed = base_seed*7919 + ep*101 + rnd` 精确复现同一局；`run_ab.py`/`eval.py` 已经实现"同种子重放 + 配对差分"。要做的不是新算法，而是把**rollout 改成"同局分叉"**。
- **最小验证实验（1–2 天）**：
  - **不训练**，先只做**信号检测**。取 200–400 个真实出牌局面（从 BC 数据 `runs/bc-data/train.npz` 抽），对每个局面：
    1. 记录启发式选牌 `a*`；
    2. 采样 3 个**不同的替代出牌** `a1,a2,a3`（启发式候选集里 `total` 排名 2–5 的牌）；
    3. 每个分支各自**续跑完本局**（其余决策全交启发式），得最终名次分。
  - 统计：**同局面内分支间的名次分差**的方差 vs **不同局面间**的方差。
  - **判据（可跑出"信号"的硬指标）**：若"同局面分支方差 / 跨局面方差"**显著小于 1**（例如 < 0.3），说明**组内相对优势**这条路能把 SNR 提升一个量级 → 值得建 pipeline。若比 ≈ 1，说明**当前局面特征对最终名次几乎无预测力**，这条路直接否掉（这也是个**极有价值的信息**，一天就能拿到）。
  - 成本估算：每分支续跑按现有 320 局 / 710s ≈ **2.2 s/局**（8 worker）⇒ 400 局面 × 4 分支 × 2.2s ≈ **59 分钟**，纯 CPU，`nice -n 15` 即可。

### 1.2 RLOO / "Back to Basics"（留一法基线）

- **论文/来源**：*Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*，Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer 等，2024（arXiv:2402.14740，v2）。**【摘要确证】**
  链接：https://arxiv.org/abs/2402.14740
- **方法核心**：**REINFORCE Leave-One-Out**——对同一 prompt 采 K 个样本，用**其余 K−1 个的均值**当第 i 个的 baseline（而不是整体均值），是**无偏**的方差缩减；论文的结论是"PPO 的很多组件在 RLHF 里其实没必要，朴素 REINFORCE + 好 baseline 就够"。
- **为什么可能解决我们的问题**：命中**靶心 A + B**。它比 GRPO 更"轻"（**不需要 std 归一**，避免 GRPO 在小 K 下 std 不稳），对我们这种**K 只有 2–4** 的场景更稳。GRPO 的 `(r−mean)/std` 在 K=2 时 std 退化成 `|r1−r2|/2`，噪声大；**RLOO 直接减均值，不除 std**，是更保守的起点。
- **可行性：高**。实现量极小（就是把 `train_step` 里的 `adv = reward − baseline` 换成 `adv = reward − mean(同局面其它分支)`）。
- **最小验证实验**：**就在 1.1 的数据上算**——同一份"每局面 4 分支 + 最终名次分"的数据，分别用
  (a) 现在的标量滑窗 baseline、(b) RLOO 组内均值 baseline，
  比较**优势估计的标准误**。目标：RLOO 的优势 SE 至少**小 3 倍**。零额外模拟成本（复用 1.1 的 rollout）。

### 1.3 GAE（广义优势估计）

- **论文/来源**：*High-Dimensional Continuous Control Using Generalized Advantage Estimation*，John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel，2015（arXiv:1506.02438，v6）。**【摘要确证】**
  链接：https://arxiv.org/abs/1506.02438
- **方法核心**：`A_t^GAE = Σ (γλ)^l δ_{t+l}`，用**指数加权**把"单步 TD 残差"滚成优势，**λ 调节偏差–方差权衡**（λ=0 纯 critic 低方差高偏差，λ=1 纯 Monte Carlo 无偏高方差）。
- **为什么可能解决我们的问题**：命中**靶心 A**。麻将一局 8–12 个小局（`PLAN-nn-rl.md` 口径），**每一小局是有终局得分的天然子回合**。GAE 允许我们**在"小局"粒度上做 bootstrap**，而不是等到整局结束。**但前提是先有一个 critic**——我们现在**没有价值网**（R 线是纯 REINFORCE）。
- **可行性：中**。需要新增一个小 critic（输入同 40/39 维候选特征，输出标量），训练不稳定风险要花时间调。
- **最小验证实验**：先**不做 GAE**，而是做它最有价值的那一半——**分小局的奖励**。把奖励从"整局最终名次分"改成"**该小局的得分 + 后续小局得分的模型预测**"。最小版：用**同一局内其它小局的平均得分**当后续回报的粗估计（不训练 critic），看优势 SE 是否下降。1 天内可完成。

### 1.4 Advantage normalization（全局 vs 局部归一）

- **论文/来源**：*REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization*，Jian Hu, Jason Klein Liu, Haotian Xu, Wei Shen，2025（arXiv:2501.03262，v9）。**【摘要确证】**
  链接：https://arxiv.org/abs/2501.03262
- **方法核心**：GRPO/RLOO 用的是**prompt 级（局部）归一**，论文论证它**是有偏估计**且易过拟合；改为**在整个全局 batch 上归一优势**更稳。
- **为什么可能解决我们的问题**：命中**靶心 B**。我们的奖励数值尺度随番型剧烈变化（±1 到 ±96），**全局归一**能让不同局面的梯度尺度可比，直接缓解重尾。**注意方向**：论文主张"**全局**优于局部"，而我们现在的"滑窗标量 baseline"既不是局部组内也不是全局归一，是**两头不靠**。
- **可行性：高**。这是**改动最小、最像"我们已写错"的那一条**：只需在 `train_step` 前对 batch 的 advantage 做 `(a − mean)/ (std + eps)`。
- **最小验证实验**：**单变量 A/B，一天**。同种子、同超参，只改 advantage 归一（bench 上跑 2 个种子）。看 `log.jsonl` 里的 `mean_reward` 与 `w3_norm` 轨迹**是否从"单调走坏"变成"有界震荡"**。判据不是"赢"，而是 **`w3` 范数是否不再持续膨胀**（我们现在的 `w3_norm` 是诊断参数更新的最直接仪表）。**成本 ≈ 2 × 710s ≈ 24 分钟**，这是本清单里**最便宜的一次实验**。

### 1.5 基于排名/排序的奖励（rank-based advantage）

- **论文/来源**：*Ranking-Augmented On-Policy Optimization with Adaptive Advantage-Normalization for Constrained Control*（A-GRPO），Md Ragib Rownak, Sidra Ghayour Bhatti, Qadeer Ahmed，2026（arXiv:2608.15359，v1）。**【摘要确证】**
  链接：https://arxiv.org/abs/2608.15359
- **方法核心**：**只有终局稀疏反馈**时，critic 型优势估计会失稳；作者改用**轨迹级排序**来重加权优势，并给出"逐时刻自适应归一可界定优势方差"的结论。
- **为什么可能解决我们的问题**：命中**靶心 B**，且**和我们的判据哲学完全一致**——我们**已经在用"名次分"这个低方差序数指标**（`REPORT-RL-20m.md` 主判据）。**用名次而不是得分当奖励/优势**，是把论文的洞见直接搬到我们身上。这是我认为**性价比最高的一条**。
- **可行性：高**。名次分 `place_points` 我们已经从 `eval._play` 的 `s.place_points` 拿得到；训练侧换奖励源 = 改一行。
- **最小验证实验**：**奖励替换消融，一天内**。R 线跑 3 组同种子同预算（各 ~12 分钟）：
  (a) 奖励 = 名次分（±3/±1 有序）；
  (b) 奖励 = 单局得分差（现状，对照）；
  (c) 奖励 = **名次指示 + 胡次数**（锦标赛真正在算的东西）。
  用同一 A/B 口径（`run_ab.py --matches 40 --rounds 8 --seeds ...`）比**训练后模型 vs 启发式**的名次分差，并**必须跨 2 种子**。判据：a/c 臂的差是否**明显优于 b 臂**（哪怕仍不显著为正，只要**"负得更少"且 `w3_norm` 不爆炸**就是进展）。

---

## 2. 方向二：麻将 / 不完美信息牌类博弈的最新 SOTA

### 2.1 Suphx（日麻，Tenhou，**最贴合我们**）

- **论文/来源**：*Suphx: Mastering Mahjong with Deep Reinforcement Learning*，Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon，2020（arXiv:2003.13590，v2）。**【摘要确证】**
  链接：https://arxiv.org/abs/2003.13590 ｜ 全文：https://ar5iv.labs.arxiv.org/html/2003.13590v2
- **方法核心（摘要确证 + 前次调研已核原文，见 `notes/research-nn-design.md`）**：三段式 **SL（人类对局）→ self-play RL → 运行时策略适配**；三项自研技术：**global reward prediction**、**oracle guiding**、**run-time policy adaptation**。摘要明说"demonstrated stronger performance than most top human players… rated above 99.99% of all the officially ranked human players"。
- **对我们最关键的两点**：
  1. **global reward prediction**＝训练一个**预测"整场最终回报"**的 predictor 当学习信号，而**不是**拿单局/单小局的即时得分当奖励。**原文动机（前次已核）**：「The loss of one round does not always mean that a player plays poorly for that round（e.g., the player may tactically lose the last round to ensure rank 1 of the game…）」。**这正是我们 R 线的病：把"这一小局得分"当成了"这一步打得好不好"。**
  2. **RL 臂叫 `RL-basic` = 用 SL 出牌模型初始化**（前次已核）——即 Suphx 的 "RL" **本身就以 BC 为前置**，**没有"裸网络从零 RL"**。
- **可行性：中**。
  - global reward predictor 的**思想**可直接用（训练一个从"当前局面特征"预测"整局最终名次分"的小网络），**成本低**；
  - 但 Suphx 的规模是 **1.5M 局/agent、44 GPU、2 天**（前次已核，`research-nn-design.md` 表），**我们的 320 局比它小 4 个数量级**——**不能照搬规模，只能照搬思想**。
- **最小验证实验（1 天，见 §6 Top-1）**：**不训策略**，只训一个**整局名次预测器**（输入 = 当前局面的 29 维全局特征，输出 = 本场最终名次分的期望）。用 BC 数据 + 现有自对弈记录，几百样本即可。**判据**：held-out 上名次分的**预测相关系数**是否显著 > 0。若 > 0，则"用预测回报当优势估计"这条 Suphx 路线**在我们数据上成立**；若 ≈ 0，说明**我们局面的信息量根本不足以预测结局**，那就该**停止在这个动作空间上投 RL**（这本身就是一份高价值结论）。

### 2.2 Meowjong（三人麻将 Sanma，**与我们最深可比**）

- **论文/来源**：*Building a 3-Player Mahjong AI using Deep Reinforcement Learning*，Xiangyu Zhao, Sean B. Holden，2022（arXiv:2202.12847，v3）。**【摘要确证】**
  链接：https://arxiv.org/abs/2202.12847
- **方法核心**：为 5 个动作（discard / Pon / Kan / Kita / Riichi）各预训练一个 CNN（4 conv，64/64/64/32 + 1 FC，**十万级参数**，前次已核）；**只对主动作（discard）用 Monte Carlo policy gradient 做 self-play RL 增强**，其余 4 个动作**不做 RL**。
- **为什么可能解决我们的问题**：
  - **结构同构**：Meowjong 的"**只 RL 增强出牌、其余动作交规则/已训模型**"**正是我们 R 线的设计**（`rl_play.py` 只重排出牌，胡/杠/碰/吃/响应全委托启发式）。这说明**我们的架构选择是对的**，问题在**估计量与样本量**，不在架构。
  - **规模量级**：RL 只跑了 **400 episodes**（前次已核）——**和我们 320 局同量级**，但它是**从 SL 初始化出发、且在 RL 上报告了"significant further enhancement"**。**这是与我们最可比的正对照**，值得**逐条对照它的差异**（见下）。
- **⚠️ 关键差异（【推测】，但逻辑很直接）**：Meowjong 的 400 episodes 是**从 SL 模型出发往"人类打法"方向微调**，SL 已经把策略放在一个**"人类级别"的盆地**里；而我们的起点是**启发式**，且我们**没有人类/SL 信号**，只有"启发式自己"。Meowjong 有 **SL 提供的、独立于 RL 的正则/锚点**，我们没有。**这可能是它 400 局能涨、我们 320 局往下掉的重要原因之一。**
- **可行性：中高**。可行的动作：**给出等价的"SL 锚"**——把启发式的 `total` 分数当成一种**离线先验策略**，在 RL 里加 **KL/正则项把它锚在启发式附近**（相当于我们自己造一个"SL 阶段"）。
- **最小验证实验**：给现有 `train_step` 的熵正则旁**加一项"与启发式分布的 KL"**（启发式分布 = softmax(`total`/T)），权重记为 β。跑 β ∈ {0, 0.1, 0.5} 三档，2 种子，看**是否能把"负得显著"压成"不显著"**。这是**"我们缺 SL 锚"这个假说的直接可判实验**。成本 ~3 × 12 分钟。

### 2.3 Mortal（开源 SOTA 级麻将 agent；**非论文，代码级证据**）

- **论文/来源**：开源项目 https://github.com/Equim-chan/Mortal （前次调研已核源码；本次**未重新取**，标注为**前次【源码确证】，本次未复核**）。前次确证的关键代码事实记录在 `notes/research-nn-design.md` §问题5：
  - 观测里**显式编码** `shanten`（向听数）、`required_tiles`（所需牌/进张）、`tenpai_probs`、`win_prob`、`candidate.ev`；
  - 训练侧：`cql_loss`（保守 Q 学习）+ **`next_rank` 辅助头**（预测四家名次概率），`next_rank_weight = 0.2`、`min_q_weight = 5`；
  - 奖励侧：`reward_calculator.py` 用 **`pts = [3, 1, -1, -3]` 算期望分差**，即**把"名次分"这个低方差目标显式建模出来**。
- **为什么重要**：Mortal 是**现存最强的开源麻将 agent 之一**，而它明确**不做"单局得分回归"**，而是**把名次当建模目标**。这与 §1.5 的 A-GRPO 从两个完全不同的方向指向**同一结论**：**用序数/名次信号，别用重尾得分。**
- **可行性：高（思想层面）/ 中（实现层面）**。"名次辅助头"我们**现在就能加**——我们已经有 `place_points`（`eval._play` 里的 `s.place_points`）。
- **最小验证实验**：在现有 6.8k 参数网络旁，**加一个只预测"本场最终名次"的小辅助头**（多任务 CE），权重 0.2；主任务仍是出牌优势。跑 2 种子，看主判据是否改善。**与 §2.1 的 Suphx predictor 是同一个实验的两面，可以合并做。**

### 2.4 Mahjax（2026，麻将在 JAX 上的大规模并行模拟器）

- **论文/来源**：*Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX*，Soichiro Nishimori, Shinri Okano, Keigo Habara, Sotetsu Koyamada, Eason Yu, Masashi Sugiyama，2026-05（arXiv:2605.20577，v1）。**【摘要确证】**
  链接：https://arxiv.org/abs/2605.20577
- **方法核心**：全向量化、JAX 实现的日麻环境，**8×A100 上最高 200 万 / 100 万 steps/s**（无红宝牌 / 有红宝牌规则）；论文强调其目的是**支持 tabula rasa（从零）研究**，并**验证了 agent 可以对基线策略取得名次改善**。
- **为什么可能解决我们的问题**：命中**靶心 A 的"样本量"根源**。我们的瓶颈之一是**样本量少 4 个数量级**（320 局 vs Suphx 1.5M 局），而我们的模拟器是**单机 16 核、2.2 s/局**。**MAhjax 的思路（向量化并行 rollout）是把样本量提上去的唯一现实途径。**
- **可行性：低（移植）/ 中（借思想）**。我们**不可能**把杭州麻将引擎搬进 JAX（规则引擎是主仓库的、且不能改），**但"把 rollout 并行度榨干"我们已经在做**（8→12 worker）。真正的抓手是 **§7 里提到的"单局模拟 77% 时间在精确进张/向听计算"**——**这是我们的真瓶颈，不是 GPU 不够**。
- **最小验证实验**：**性能剖析先行**（不做算法改动）。用 `cProfile` 确认 `_blocks_value` 是否仍占 ~77%；若是，试**在 rollout 期缓存/降精度进张计算**（只在候选集上算，而不是全牌墙），看单局耗时能否从 2.2 s 降到 < 1 s。**样本量翻倍 = 信噪比按 √2 改善**，这是**确定性收益**，不需要算法赌注。

### 2.5 其它牌类/不完美信息 SOTA（作为横向参照，**不逐一展开**）

| 来源 | 年份 | 我们要点 | 证据等级 |
|---|---|---|---|
| *DouZero: Mastering DouDizhu with Self-Play Deep RL*（Zha 等，arXiv:2106.06135） | 2021 | **DMC（Deep Monte Carlo）不用 bootstrap**，明确规避 DQN 的过估计；**大且逐回合变化的动作空间**可用。链接：https://arxiv.org/abs/2106.06135 | **摘要确证** |
| *DouRN: Improving DouZero by Residual Neural Networks*（Chen, Lyu, Zhang，arXiv:2403.14102） | 2024 | **把 DouZero 换成残差网络**即"significantly improves the winning rate"——**"残差结构在小样本牌类博弈上直接有增益"的牌类域证据**。链接：https://arxiv.org/abs/2403.14102 | **摘要确证** |
| *DanZero: Mastering GuanDan Game with RL*（Lu 等，arXiv:2210.17087） | 2022 | 掼蛋（4 人、长局、动作空间大）；**DMC + 分布式**训练，对**启发式规则基线** "outstanding performance"。链接：https://arxiv.org/abs/2210.17087 | **摘要确证** |
| *DanZero+: Dominating the GuanDan Game through RL*（Zhao 等，arXiv:2312.02561） | 2023 | 明确说"**巨大动作空间显著影响 policy-based 算法性能**"，对策是**用预训练模型加速训练** ⇒ **与我们"BC→RL"同构**。链接：https://arxiv.org/abs/2312.02561 | **摘要确证** |
| *AlphaDou: …End-to-End Doudizhu AI Integrating Bidding*（Lei & Lei，arXiv:2407.10279） | 2024 | 用**期望**做**动作空间剪枝**、用**胜率**生成策略。链接：https://arxiv.org/abs/2407.10279 | **摘要确证** |
| *Self-Play RL under Imperfect Information in Big 2*（Patwa，arXiv:2605.28863） | 2026 | **受控对比**：在统一环境/表示/预算/评测下，**PPO 胜过 Monte Carlo Q、SARSA、Q-learning**（对 random/greedy/heuristic 对手）；**适度熵正则防止策略过早确定**；**current-policy self-play 优于 checkpoint self-play 与固定对手**。链接：https://arxiv.org/abs/2605.28863 | **摘要确证** |
| *SkyNet: Belief-Aware Planning for Partially-Observable Stochastic Games*（Haile，arXiv:2603.27751） | 2026 | MuZero + **ego-conditioned 辅助头（胜者预测 / 名次估计）**；**对启发式对手 0.720 vs 基线 0.466 胜率**；作者强调**"belief-aware 模型一开始不如基线，只有训练吞吐足够后才反超"**。链接：https://arxiv.org/abs/2603.27751 | **摘要确证** |
| *Evo-Sparrow*（O'Connor 等，arXiv:2508.07522） | 2025 | Sparrow 麻将用 **CMA-ES 优化 LSTM**，**≥ random / rule-based，≈ PPO 基线** ⇒ **不用策略梯度也能打平**（进化策略作为替代路线）。链接：https://arxiv.org/abs/2508.07522 | **摘要确证** |
| *A Novel Reward Shaping Function for Single-Player Mahjong*（Chen 等，arXiv:2305.04145） | 2023 | **向听数（ShangTing）作 reward shaping**；1v1 对战中新 shaping "outperformed the default ShangTing function"。链接：https://arxiv.org/abs/2305.04145 | **摘要确证** |
| *CFR-p: CFR with Hierarchical Policy Abstraction … Two-player Mahjong*（Wang，arXiv:2307.12087） | 2023 | **二人**麻将的 CFR + 分层抽象（**不是 4 人**；参考价值有限）。链接：https://arxiv.org/abs/2307.12087 | **摘要确证** |
| *Adapting Rules of Official International Mahjong for Online Players*（Wang 等，arXiv:2601.08211） | 2026 | 用**世界冠军 AI 自对弈**做规则公平性分析（**先手优势**、**名次/分数设置问题**）⇒ **在 4 人麻将里"名次"才是稳定信号**。链接：https://arxiv.org/abs/2601.08211 | **摘要确证** |
| *RLCard: A Toolkit for RL in Card Games*（Zha 等，arXiv:1910.04376） | 2019 | 含 Mahjong 环境；定位"**large state and action space, and sparse reward**"。链接：https://arxiv.org/abs/1910.04376 | **摘要确证** |

**⚠️ 一条必须承认的检索缺口**：本次**没有检索到**"**杭州麻将 / 国标麻将（MCR）的深度 RL 论文**"，也**没有检索到**"**在 4 人麻将上直接做 residual/boosting 于一个强启发式之上**"的论文。检索命中的中文相关条目是规则公平性分析（2601.08211）与杭州麻将无关。**因此 §3 的范式是跨域迁移（机器人控制 / 棋类），不是麻将域内的既有结论。** 这一点在评估可行性时必须计入。

---

## 3. 方向三：从强启发式出发的改进范式

### 3.1 Residual RL / Residual Policy Learning（**与我们 R 线同构，是理论靠山**）

- **论文/来源**：
  - *Residual Policy Learning*，Tom Silver, Kelsey Allen, Josh Tenenbaum, Leslie Kaelbling，2018（arXiv:1812.06298，v2）。**【摘要确证】**
    链接：https://arxiv.org/abs/1812.06298
  - *Residual Reinforcement Learning from Demonstrations*，Minttu Alakuijala, Gabriel Dulac-Arnold, Julien Mairal, Jean Ponce, Cordelia Schmid，2021（arXiv:2106.08050，v1）。**【摘要确证】** 链接：https://arxiv.org/abs/2106.08050
  - *Accelerating Residual Reinforcement Learning with Uncertainty Estimation*，Lakshita Dodeja 等，2025（arXiv:2506.17564，v2）。**【摘要确证】** 链接：https://arxiv.org/abs/2506.17564
  - *What Makes Value Learning Efficient in Residual Reinforcement Learning?*，Guozheng Ma 等，2026（arXiv:2602.10539，v1）。**【摘要确证】** 链接：https://arxiv.org/abs/2602.10539
- **方法核心**：
  - RPL（2018）：**冻结/保留一个"好但不够好"的控制器，只学残差**。摘要原文："RPL thrives in complex robotic manipulation tasks where **good but imperfect controllers are available**. In these tasks, **reinforcement learning from scratch remains data-inefficient or intractable, but learning a residual on top of the initial controller can yield substantial improvements**."，并且"**RPL can perform long-horizon, sparse-reward tasks for which reinforcement learning alone fails**"。
  - 2602.10539（2026）：识别出 residual RL **价值学习的两个瓶颈**——**"cold start pathology"（critic 不了解 base policy 周围的价值地形）** 与 **"structural scale mismatch"（残差相对 base 动作太小，被淹没）**；解法是**用 base-policy 的转移做 value anchor（隐式 warmup）**与 **critic normalization**。
- **为什么可能解决我们的问题**：**这是对我们 R 线架构最强的文献背书，同时精确解释了我们的失败模式。**
  - 我们**已经**在做 residual（`logit = total + delta`，`delta≡0` 起步）——**方向正确**。
  - 2602.10539 的 **"structural scale mismatch"** 与我们 `REPORT-RL-20m.md` 的诊断**独立吻合**：`REPORT-RL-20m.md` 指出"更新方向实为有害…训练却在把修正分支推向一个更差的重排"。**残差被噪声淹掉**正是这个现象的名字。
  - 2506.17564 给了两个具体补丁：**用 base policy 的不确定性来聚焦探索** + **让 residual 能观察到 base action**（我们的候选特征里**已经含 `total`**，符合后者）。
- **可行性：高**。**这是"我们已经在正确的路上，只是缺两个已知补丁"的结论。**
- **最小验证实验（1 天，与 §1.4 合并做）**：
  1. **critic/advantage warmup**：训练**最初 N 个 batch 只用启发式自己的局面**（不采样偏离），把 baseline 先校准到"启发式自己的期望回报"，再开始推动 delta。实现 = `train_rl_par.py` 加一个 `--warmup-batches`。
  2. **残差尺度约束**：给 `delta` 加**有界激活**（如 `tanh` 或直接 clip 到 ±τ）并扫 τ ∈ {0.5, 1, 2}。动机是 2602.10539 的 scale mismatch。
  3. 判据：`w3_norm` 不再单调膨胀，且 A/B 名次分差**从 −1.0 收窄到区间跨 0**。

### 3.2 DAgger（数据集聚合）——**修复我们 BC 失败的方式**

- **论文/来源**：*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*，Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell，2010（arXiv:1011.0686，v3）。**【摘要确证】**
  链接：https://arxiv.org/abs/1011.0686
- **方法核心**：模仿学习在**自回归**（当前动作改变未来状态分布）下会累积误差。DAgger 迭代：用**学生自己产生的状态分布**去查询专家标签，再重训，从而把训练分布拉回测试分布。
- **为什么可能解决我们的问题**：命中**靶心 C 的一个具体缺陷**。我们的 BC 线是**纯离线**（教师在同一分布上，学生一致率 = 教师本身 ⇒ 零增益）。**但注意我们 BC 的失败原因与 DAgger 要修的"distribution shift"不同**——我们的问题是"**教师是确定性 argmax，学生最有解就是复制它**"（`REPORT-6.6-nnrl.md` §2.2）。**DAgger 修不了"天花板 = 教师"这个根本问题**。
- **⚠️ 结论（重要，用于避坑）**：**DAgger 对我们大概率无效**，因为我们的瓶颈**不是分布漂移，而是目标函数本身没有超越教师的信号**。文献给的机制解释要诚实套用：DAgger 的收益来自"学生偏离到新状态后专家能给出正确修正"，而**我们的专家（启发式）在任何状态下给出的都是 argmax(total)，学生最优解恒为恒等映射**。
- **可行性：低**。**建议不做**，除非先解决"教师标签本身包含超越教师的信息"（例如用**名次更好的 rollout 当标签**，那就不是 DAgger 而是 §3.4 的离线偏好学习了）。
- **最小验证实验**：**不建议投入**。若要证伪，最便宜的是**分析已有 BC 数据**：统计"学生偏离后教师给出的修正是否有系统方向"——若修正的熵 ≈ argmax 的 tie-break 噪声，则确认无信息。

### 3.3 AlphaZero 式 MCTS + 策略网（以及 Expert Iteration）

- **论文/来源**：
  - *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*（AlphaZero），Silver 等，2017（arXiv:1712.01815，v1）。**【摘要确证】** 链接：https://arxiv.org/abs/1712.01815
  - *Thinking Fast and Slow with Deep Learning and Tree Search*（Expert Iteration），Anthony, Tian, Barber，2017（arXiv:1705.08439，v4）。**【摘要确证】** 摘要原文："**We show that ExIt outperforms REINFORCE for training a neural network to play the board game Hex**" —— **这是"ExIt/搜索 > 纯 REINFORCE"的直接对照证据**。链接：https://arxiv.org/abs/1705.08439
  - *Accelerating Self-Play Learning in Go*（KataGo），David J. Wu，2019（arXiv:1902.10565，v5）。**【摘要确证】** 链接：https://arxiv.org/abs/1902.10565
  - *Learning and Planning in Complex Action Spaces*（Sampled MuZero / Gumbel MuZero），Hubert 等，2021（arXiv:2104.06303，v1）。**【摘要确证】** 摘要原文："Many important real-world problems have action spaces that are high-dimensional, continuous or both… **only small subsets of actions can be sampled for the purpose of policy evaluation and improvement**." 链接：https://arxiv.org/abs/2104.06303
  - *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*（MuZero），Schrittwieser 等，2019（arXiv:1911.08265，v2）。**【摘要确证】仅标题/日期核对** 链接：https://arxiv.org/abs/1911.08265
- **方法核心**：用**搜索（MCTS）**产生比策略网更强的动作分布，再让策略网**回归搜索的结果**（policy improvement operator）。**这是"用一个比现有策略更强的算子去改进策略"的标准范式**——而我们的问题是：**我们没有一个比启发式更强的算子**。
- **为什么可能（也可能不可能）解决我们的问题**：
  - **可能**：如果我们能在**完全信息的模拟器内**做**逐张前向搜索**（对每张候选牌做 N 次随机 rollout 到局末，取名次分期望），那么**搜索本身就是一个比启发式强的算子** ⇒ 用它当"教师"，BC/策略网只需**回归搜索**，就绕开了 R 线"从重尾标量里学"的难题。**这与 Suphx 的 look-ahead features 同源**（`research-nn-design.md` 已核：Suphx 把"打这张后经替换成胡的概率与番数"逐张算出喂给网络）。
  - **风险（必须诚实）**：麻将的**信息集不是"同一状态 + 不同动作"**——我们**看不到牌墙和对手手牌**。`PLAN-nn-rl.md` 已明确："自对弈天然不需要对手手牌（绕开推理输入不得含暗牌的红线）"，但**搜索需要采样隐藏信息**（PIMC）。麻将 PIMC 的**策略融合（strategy fusion）**是已知难题（桥牌文献给的是 `αμ` 类方法，见下）。
  - **另一个硬风险**：我们的**单局模拟 2.2 s**。若每个局面要 N 张候选 × M 次 rollout，计算量爆炸。Gumbel MuZero 的答案正是"**只在采样的动作子集上做 policy evaluation/improvement**"——**这条能救计算量**。
- **可行**：参照 *The αμ Search Algorithm for the Game of Bridge*（arXiv:1911.07960）与 *Perfect Information Monte Carlo with Postponing Reasoning*（arXiv:2408.02380）。**两者本次仅【仅标题】核对到条目存在**，未读摘要 ⇒ **不作为结论依据，只作为"下一步该读什么"的路标**。
- **可行性：低（全 MCTS 架构）/ 中（只借"逐张 look-ahead 当特征/教师"）**。
- **最小验证实验——只做"搜索当教师"，不建 MCTS 树**：
  - 取 100 个出牌局面，对**启发式 top-3 候选**各做 **K=20 次随机 rollout**（其余决策交启发式），算每张牌的**平均名次分**。
  - **判据 1（教师是否比启发式强）**：`argmax_k(平均名次分)` 与启发式首选 `a*` 的**一致率**。若一致率 **> 0.9** ⇒ 搜索几乎总是同意启发式 ⇒ **搜索给不出新信息，路线否掉**（这一天省下几周）。若一致率在 **0.6–0.8** ⇒ 搜索确实在偏离，值得用它生成**偏好对**（直接接到 §3.4）。
  - **判据 2（噪声）**：同一张牌两次 K=20 rollout 的**均值差的标准误**。若 SE 大于牌间真实差 ⇒ **K 不够**，先算清 K 要多大再谈。
  - 成本：100 局面 × 3 牌 × 20 rollout × 2.2 s ≈ **3.7 小时**（8 worker ⇒ ~28 分钟）。可接受。

### 3.4 离线偏好 / 排序学习（**替代在线 REINFORCE**）

- **论文/来源**：
  - *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*，Rafailov, Sharma, Mitchell, Ermon, Manning, Finn，2023（arXiv:2305.18290，v3）。**【摘要确证】** 摘要原文："The resulting algorithm… is **stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning**"。链接：https://arxiv.org/abs/2305.18290
  - *Deep reinforcement learning from human preferences*，Christiano, Leike, Brown, Martic, Legg, Amodei，2017（arXiv:1706.03741，v4）。**【摘要确证】** 链接：https://arxiv.org/abs/1706.03741
  - *Policy-Gradient Training of Language Models for Ranking*（Neural PG-RANK），Gao, Chang, Cardie, Brantley, Joachims，2023（arXiv:2310.04407，v2）。**【摘要确证】** 把 LLM 当成 **Plackett-Luce 排序策略**，用 policy gradient 直接优化**下游决策质量**；摘要原文："The results demonstrate that **when the training objective aligns with the evaluation setup, Neural PG-RANK yields remarkable in-domain performance improvement**"。链接：https://arxiv.org/abs/2310.04407
- **方法核心**：
  - DPO：**把"从偏好中学"化成简单的分类损失**，**不需要采样、不需要在线 RL**（无需 reward model + PPO）。
  - 排序版（PG-RANK）：**动作 = 一个候选排序**（Plackett-Luce），**目标直接对齐评测指标**。
- **为什么可能解决我们的问题**：**这是四条方向里我评价最高的。**
  - 命中**靶心 A/B**：我们的核心痛点是把"重尾标量"当学习信号。**偏好学习的输入是"A 比 B 好"这个序数**——**扔掉数值、只要序数**，直接把 §1.5 的排名思想推向极致，**方差天然小得多**。
  - 命中**靶心 C**：我们可以只用**"最终名次更好的那个分支"**造偏好对（同局面、同牌墙前缀，分支 A 名次 2、分支 B 名次 4 ⇒ 偏好 A）。**这就是"超越启发式"的直接监督信号**，而且**它不需要我们预先知道"正确动作"**——只需要能**评判结果**（我们的模拟器天然能做）。
  - **和我们的基础设施完全咬合**：`eval.py` 已实现**同种子重放**（`seed*100003+index`）与**四座位旋转**；`INITIAL_WALL`/`run_round` 支持**从任意局面续跑**；`place_points` 是现成的序数。**我们缺的只是"成对采样 + 排序损失"这一层皮。**
- **可行性：高**。理由：① 无需 critic；② 无需在线更新（**天然无探索-利用不稳定**）；③ 数据可**离线批量生成**（并行 rollout 我们已经很熟）；④ 损失是可加权的 pairwise logistic / Plackett-Luce，**纯 numpy 可手写**（与参赛路径"零第三方依赖"的约束一致）。
- **最小验证实验（1 天，本清单 Top-2）**：
  1. **离线偏好对生成**：抽 **300–500 个局面**；每局面取启发式 top-2 候选 `a1,a2`，各自续跑本局（其余交启发式），得 `r1,r2`（用**名次分**，序数、低方差）。
  2. 保留 `r1 ≠ r2` 的对（弃平局，或按 §2.3 的 `[3,1,-1,-3]` 折算期望分差）。
  3. **训练**：在现有 6.8k 参数网络（`rl_net`）上加 **pairwise logistic 损失**：`−log σ( (logit_{a1} − logit_{a2}) · sign(r1 − r2) )`，**w3 仍零初始化**（保持"起点 = 启发式"）。
  4. **判据**：held-out 局面上，**偏好对准确率**（预测的排序与真实名次排序一致）是否 **显著 > 0.5**。这一步**不需要跑 A/B**，一天内出结果。
  5. 若 > 0.5，**再**上 A/B（`run_ab.py --matches 40 --rounds 8 --seeds ...`）验证是否转成名次分正增益。
  - 成本：500 局面 × 2 分支 × 2.2 s ÷ 8 worker ≈ **4.6 分钟**（rollout）+ 几分钟训练。**极便宜。**

### 3.5 RUDDER / 回报分解与 hindsight credit assignment

- **论文/来源**：
  - *RUDDER: Return Decomposition for Delayed Rewards*，Arjona-Medina, Gillhofer, Widrich, Unterthiner, Brandstetter, Hochreiter，2018（arXiv:1806.07857，v3）。**【摘要确证】** 摘要原文：目标是"**making the expected future rewards zero**"，通过 **reward redistribution 得到 return-equivalent 决策过程（最优策略不变）**，并"**transforms the reinforcement learning task into a regression task at which deep learning excels**"；"On artificial tasks with delayed rewards, **RUDDER is significantly faster than MC and exponentially faster than Monte Carlo Tree Search (MCTS), TD(λ), and reward shaping approaches**"。链接：https://arxiv.org/abs/1806.07857
  - *Hindsight Credit Assignment*，Harutyunyan 等，2019（arXiv:1912.02503，v1）。**【摘要确证】仅标题/日期核对**。链接：https://arxiv.org/abs/1912.02503
- **方法核心**：**用回归把延迟回报拆到每一步**，且保证**最优策略不变**（return-equivalence）——**这是"重尾局末奖励"的正面对策**，且**它对齐我们的靶心 A**。
- **为什么可能解决我们的问题**：**RUDDER 的"return-equivalent reward redistribution"恰好匹配我们"不改最优策略、只改信用分配"的需求**。而且它的社区实现是**开源（github.com/ml-jku/rudder）**。**⚠️ 但 RUDDER 通常是给"游戏通关/Atari 关键帧"这类单一稀疏成功信号用的**；我们的奖励是"每小局都有分"（不是全 0 到局末），**稀疏度比 Atari 低**，所以**收益可能不如它的论文里那么大**——这是我的**推测**，需要实验判定。
- **可行性：中**。RUDDER 的贡献分析需要训练一个 LSTM 类模型，实现量明显大于 §3.4。
- **最小验证实验**：**先做最廉价的替代**——把奖励从"整局名次分"改成"**小局增量 + 整局名次分的混合**"，并做**per-decision 归一**（用该决策所在小局的番数缩放）。若这一步就有效，就不必上 RUDDER。**这是 §1.5 实验的一个扩展，边际成本近零。**

---

## 4. 关键反面证据：哪些"从零 RL 打麻将 / 打牌"的做法被证明低效

**这一节用于"避免重走"。**

1. **"裸策略网络 + 从零自对弈"在麻将上没有成功案例（本仓库前次调研已核，本次复核一致）。**
   - 前次调研（`notes/research-nn-design.md`，检索方法为 arXiv API）明确结论：Suphx / Meowjong / Mortal / Mahjax **全部是"有监督或离线/预训练"路径或其等价**，**没有找到裸策略网络从零成功的麻将案例**。本次复核**未推翻**该结论：
     - **Suphx**：`RL-basic` 臂 = **用 SL 出牌模型初始化**（摘要确证三段式）。
     - **Meowjong**：**先 pretrain 5 个 CNN，再只对 discard 做 RL**（摘要确证）。
     - **Mahjax（2026）**：摘要明确说"prior research has **heavily relied on supervised learning from human play logs to pre-train the policy**"，其自身定位是**提供从零研究的基础设施**，**不是**已证明从零成功（摘要确证）。
   - ⇒ **对我们的直接含义：继续投"从零/从启发式出发的在线 REINFORCE"是与整个领域证据相反的方向。**

2. **纯 REINFORCE 在同类问题上被 Expert Iteration 直接比下去（有对照数据）。**
   - *Thinking Fast and Slow*（1705.08439）摘要原文："**We show that ExIt outperforms REINFORCE** for training a neural network to play the board game Hex"。**【摘要确证】**
   - ⇒ 我们 R 线用 REINFORCE 是**文献里明确说"不如搜索式"的那一支**。

3. **大动作空间下 policy-based 方法本身被点名"性能显著受影响"。**
   - *DanZero+*（2312.02561）摘要原文：为应对"**huge action space, which will significantly impact the performance of policy-based algorithms**"，对策是"**adopt the pre-trained model to facilitate the training process**"。**【摘要确证，为前次转述的复核】**
   - ⇒ 我们必须**先有预训练（BC）再做 policy-based**；我们已有 BC 产物（`REPORT-6.6-nnrl.md` §4：起点 = 教师本人的构造等价）——**这是我们对的地方**。

4. **小样本下的评估本身不可靠（这是"不要被自己骗"的纪律证据）。**
   - *Deep Reinforcement Learning at the Edge of the Statistical Precipice*，Agarwal 等，2021（arXiv:2108.13264，v4）。**【摘要确证】** 摘要原文："reliable evaluation in the **few run deep RL regime cannot ignore the uncertainty in results**"，并给出 **performance profiles / interquartile mean** 等稳健聚合。
     链接：https://arxiv.org/abs/2108.13264
   - ⇒ **直接适用**：我们的 A/B 只有 80 配对、2 种子。按本仓库自己的噪底口径（两臂差 SD 1.33pp），**任何"接近 0 的差值"都不能当作结论**。**这条支持"先把噪声压下去，再谈算法"。**

5. **非传递性/策略循环：自对弈的评估信号可能与真实强度解耦。**
   - *Open-ended Learning in Symmetric Zero-sum Games*，Balduzzi 等，2019（arXiv:1901.08106，v2）。**【摘要确证】** 链接：https://arxiv.org/abs/1901.08106
   - *Real World Games Look Like Spinning Tops*，Czarnecki 等，2020（arXiv:2004.09468，v2）。**【摘要确证】** 摘要原文："**populations of strategies are necessary for training of agents**"。
     链接：https://arxiv.org/abs/2004.09468
   - ⇒ 这与本仓库 `PLAN-nn-rl.md` 的既有判断一致（"默认场地对任何偏离都给正分，符号不可信"）。**本次调研再次确认：单一批对手场地的正号不可作采纳依据；需要异质对手池。**

6. **"纯离线模仿确定性教师"的天花板 = 教师（这不是文献结论，是我们的实测）。**
   - `REPORT-6.6-nnrl.md` §2.2 已确证：教师自一致率 0.9462，学生最优 = 0.9462 = **恒等映射**。
   - ⇒ **任何"回归教师动作"的路线（含 DAgger）都不能产生增益**，除非标签源本身携带超越教师的信息（即必须换成"结果/偏好"标签）。**这排除了 §3.2。**

7. **重尾奖励下"不做分解/归一"的朴素做法是已知的高方差来源。**
   - GAE（1506.02438）摘要原文："The **two main challenges** are the **large number of samples** typically required, and the difficulty of obtaining stable and steady improvement despite the nonstationarity"——**我们的 R 线两个坑都踩了**。**【摘要确证】**
   - RUDDER（1806.07857）摘要把"delayed rewards"下的 MC 高方差与 TD 偏差明确对立起来并给出分解解。**【摘要确证】**
   - ⇒ **我们现在的配置（单局标量 + 无归一 + 无分解 + 无折扣）恰好是文献里被点名的"最高方差"组合。**（这一点我按**推断**陈述：三篇论文分别指出各成分的方差问题，但**没有**任何论文专门研究"杭州麻将 + 单局得分 + 40 局"这个具体组合。）

---

## 5. 汇总表：可行性评估 + 最小验证实验

**评估口径**：可行性 = 在**当前代码结构**（`src/nnrl/*`、`scripts/train_rl_par.py`、`scripts/run_ab.py`）、**当前样本规模**（~18k 决策 / 320 局）、**当前算力**（单机 16 核 / 单卡 8GB 级、单局 2.2 s）下的**落地难度**。

| # | 方向 | 命中靶心 | 可行性 | 理由（一句话） | 最小验证实验 | 成本 |
|---|---|---|---|---|---|---|
| 1 | **离线偏好/排序学习（§3.4）** | A+B+C | **高** | 只需"同局面分叉 + 序数标签 + pairwise 损失"，无 critic、无在线不稳定；基础设施已具备 | 500 局面 × top-2 分叉续跑 → 偏好对准确率是否 > 0.5 | **~10 分钟** |
| 2 | **奖励换成名次/序数（§1.5 + §2.3）** | B+C | **高** | 改奖励源 = 改一行；与 Mortal（`pts=[3,1,-1,-3]`）和 A-GRPO 两个独立来源同向 | 奖励消融 3 臂 × 2 种子，比 `w3_norm` 轨迹与名次分差 | **~40 分钟** |
| 3 | **组内相对优势 RLOO/GRPO（§1.1–1.2）** | A | **中高** | 同局面分叉造"组"，把跨局方差换成同局配对方差；RLOO 在 K=2–4 时比 GRPO 更稳 | 同局面分支方差 / 跨局面方差 < 0.3？ | **~1 小时** |
| 4 | **Suphx global reward predictor / Mortal 名次辅助头（§2.1/§2.3）** | A | **中** | 思想可直接用、成本低；但 Suphx 规模大 4 个数量级，不能照搬 | 训一个整局名次预测器，看 held-out 相关系数是否 > 0 | **~半天** |
| 5 | **advantage 全局归一（§1.4）** | B | **高** | 纯实现改动，论文论证"局部归一有偏"；我们现在两头不靠 | 只改归一，2 种子，看 `w3_norm` 是否不再膨胀 | **~25 分钟** |
| 6 | **Residual RL 的两个已知补丁（§3.1）** | A+C | **中高** | 我们已在正确范式内，缺"value warmup"与"残差尺度约束" | 加 `--warmup-batches` + `tanh` 限幅，扫 τ | **~1 小时** |
| 7 | **Meowjong 式"KL 锚回启发式"（§2.2）** | C | **中** | 补上我们相对 Meowjong 缺的"SL 锚"；改 1 项正则 | KL 权重 β ∈ {0, 0.1, 0.5} × 2 种子 | **~40 分钟** |
| 8 | **搜索当教师 / look-ahead 特征（§3.3）** | A+C | **中低** | 需要 PIMC 采样隐藏信息；单局 2.2 s 是硬约束（Gumbel 式"采样子集"可缓解） | 100 局面 × top-3 × K=20 rollout：top-1 一致率是否 < 0.9 | **~30 分钟** |
| 9 | **GAE + 新 critic（§1.3）** | A | **中低** | 需新增价值网并调稳，风险/工作量显著高于 #1–#5 | 先用"分小局回报"替代，再决定是否上 GAE | **~1 天** |
| 10 | **RUDDER 回报分解（§3.5）** | A | **低** | 需 LSTM 式贡献分析；且我们不是全 0→局末，稀疏度低于其原设定 | 先用"小局增量 + per-decision 归一"替代 | **~1 天** |
| 11 | **DAgger（§3.2）** | — | **低（不推荐）** | 天花板 = 教师，DAgger 只修分布漂移，修不了"标签无超越信号" | 分析已有 BC 数据的"偏离后修正方向"是否有信息 | 半天（**建议省掉**） |
| 12 | **模拟器加速以扩样本（§2.4）** | A | **中高** | 样本量 ×2 ⇒ SNR ×√2，**确定性收益、无算法赌注** | cProfile 确认瓶颈；缓存/降精度进张计算，目标 < 1 s/局 | **半天** |
| 13 | **异质对手池（§4.5）** | 评估有效性 | **中** | 单一场地符号不可信（本仓库已实测）；但建池要整合历史档位 | 用现有各实验档位混场，重跑一次 R 线对拍 | 半天 |

---

## 6. Top 3 行动清单（按"性价比"排序）

### 🥇 Top 1 — 离线偏好/排序学习：用"同局面分叉 + 名次序数"造偏好对（**§3.4 + §1.5 + §2.3**）

**为什么第一**：**唯一一条同时命中三个靶心、且基础设施已具备、成本 < 15 分钟就能拿到第一个信号的方向。**

- **动作**：
  1. 从 `runs/bc-data/train.npz` 抽 300–500 个出牌局面（**不训练、不改代码逻辑**，写一个独立探针脚本）。
  2. 对每个局面，取启发式候选的 **top-2**，各自**续跑完本局**（其余决策交启发式），拿 **`place_points`（名次）** 当标签。
  3. 统计 **偏好对可分辨率**（`r1 ≠ r2` 的比例）与**分叉方差**。
  4. 若可分辨率可观（> 30%），**再加** pairwise logistic 损失训练现有 6.8k 网络（`w3` 仍零初始化），测 **held-out 偏好准确率**。
- **一天内可得的信号**：
  - **Go**：偏好准确率 **> 0.5 且显著** ⇒ 说明"名次序数 + 同局面配对"确实携带可学习信号 ⇒ 转入完整 A/B 验证。
  - **No-Go**：偏好准确率 ≈ 0.5 ⇒ 说明**我们局面的信息量不足以预测结局**（同 §2.1 的判据）⇒ **正式终止在"出牌重排"这个动作空间上投 RL**，把预算转去特征/规则侧。**这同样是一份高价值结论。**
- **成本**：rollout ~5 分钟 + 训练几分钟。**`nice -n 15`，同时只跑一个（遵循本仓库算力纪律）。**
- **依据**：DPO（2305.18290）、Christiano（1706.03741）、PG-RANK（2310.04407）均为**【摘要确证】**；Mortal 的 `pts=[3,1,-1,-3]`（**前次源码确证**）；A-GRPO（2608.15359）**【摘要确证】**。

### 🥈 Top 2 — 把奖励从"单局得分"换成"名次/序数"，并加 advantage 全局归一（**§1.5 + §1.4**）

**为什么第二**：**最便宜的一次单变量实验（~25–40 分钟），且直击 R 线诊断里"更新方向有害"的那一点。**

- **动作**：在 `train_rl_par.py` 上将 `outcome.scores[seat]`（单局得分）替换为 **`place_points` 等价序数**；同时加 **batch 级 advantage 归一**。跑 3 臂 × 2 种子（reward ∈ {得分, 名次, 名次+胡次数}），同超参同预算。
- **一天内可得的信号**：**看 `log.jsonl` 的 `w3_norm` 轨迹**——现状是持续膨胀（参数被噪声推远）；若名次臂的 `w3_norm` **有界震荡**、且 A/B 名次分差**从 −1.0 收窄到跨 0**，即为正进展。**不需要"一次跑赢"就下结论**；先把"显著为负"变成"不显著"，是通往正增益的必经一步。
- **成本**：3 臂 × 2 种子 × ~12 分钟 ≈ **70 分钟**（含对拍）。
- **依据**：A-GRPO（2608.15359）、REINFORCE++（2501.03262）、Mortal 名次建模、Official Intl Mahjong 规则分析（2601.08211）均为**【摘要确证】**。

### 🥉 Top 3 — 组内相对优势（RLOO）：把"跨局重尾"换成"同局面配对差分"（**§1.1 + §1.2**）

**为什么第三**：**它是 Top 1 的"在线版"，能持续利用新数据；但实现比 Top 1 重、且需要确认"同局面方差确实更小"这个前提。**

- **动作**：先做**纯信号检测**（不训练）：400 局面 × 4 分支续跑，算"同局面分支间名次分方差 / 跨局面方差"。
- **一天内可得的信号**：比值 **< 0.3** ⇒ 建 pipeline（RLOO 优势替换现有标量 baseline）⇒ 这一步的收益上限很高；比值 **≈ 1** ⇒ **立即止损**，回去做 Top 1/2。
- **成本**：~1 小时信号检测；pipeline 改动 1 天。
- **依据**：GRPO/DeepSeekMath（2402.03300）、RLOO/Back to Basics（2402.14740）、Tree-style GRPO 方差分析（2509.24494）均为**【摘要确证】**。

---

## 7. 不确定性与未决问题（**必读**）

1. **最大的检索缺口**：**没有找到杭州麻将 / 国标麻将的深度 RL 论文**，也**没有找到"在 4 人麻将上 residual 于强启发式"的论文**。§3 的 residual/boosting 结论**全部来自机器人控制与棋类**，属于**跨域迁移**，**不是麻将对局域的既有结论**。
2. **`web_search` 不可用** ⇒ 未做通用网页检索，可能遗漏**非 arXiv** 的麻将比赛报告、技术博客、期刊论文。
3. **Gumbel MuZero 的条目在 arXiv API 中检索不到**（多组关键词均空）。本次只在 OpenReview 上核到**二手条目**（*Multiagent Gumbel MuZero: Efficient Planning in Combinatorial Action Spaces*，DBLP 记录，id `fsDPGlgLiG`）与 *An Empirical Analysis of Gumbel MuZero…*。**其正式出处为 *Policy improvement by planning with Gumbel*（Danihelka 等，ICLR 2022，OpenReview id `rDqI8G0RUt`）——该 ID 来自我的记忆（paper 级），本次未取到可核对页面，标注为【推断/未核实】。** 本文正文中我因此**只引其更可靠的同族 arXiv 版本 2104.06303（Sampled MuZero，已摘要确证）**。
4. **达标的"麻将 + 偏好学习"论文：未找到。** §3.4 的落地方式是**我的构造**（用名次造偏好对），**不是**某篇麻将论文的做法。这是**本文件里最重要的"未验证设计"**。
5. **"同局面分叉"在麻将里的有效性未经验证**：我推断"同一局面 + 同一牌墙前缀，只改一个决策"能显著降方差，但**麻将的后续摸牌路径会因一个决策而完全分岔**（牌墙消耗顺序不同）⇒ 配对可能没有我预期的那么紧。**这正是 Top 3 的第一步要测的东西。**
6. **Suphx 的规模不可比**：1.5M 局 / 44 GPU / 2 天（前次已核）vs 我们 320 局 · 单机。**任何"照 Suphx 做就能涨"的推断都是错的**——只能借思想。
7. **Meowjong 的 400 episodes 能涨而我们不能**：我的解释是"它有 SL/人类锚、我们只有启发式自身"（**§2.2，【推测】**）。这个假说**逻辑上直接、但无直接证据**；可由 §5 #7 的 KL 锚实验**证伪或支持**。
8. **两个"跨域前提"未验证**：① residual RL 的补丁（2602.10539）是**连续控制**实验，**在离散排序动作空间上是否成立未知**；② RUDDER 的 return-equivalence 在**非全零稀疏奖励**下是否仍有其论文量级的收益，未知。
9. **本仓库的算力/纪律约束**：`PLAN-nn-rl.md` §3 要求"离线重活 `nice -n 15`、同一时刻只跑一个"；**上述所有实验都必须遵守**，因此真实日历时间会比表中的"机时"长。
10. **所有实验结论都必须过噪底**：本仓库口径为**两臂差 SD 1.33pp**；在 80 配对下，"跨 0"就是"不显著"。**不要用单种子或单场地的正号下任何结论**（`REPORT-6.6-nnrl.md` §1、§4.5）。

---

## 附录 A：本文引用来源清单（全部附可核对链接）

**A.1 麻将 / 牌类博弈（本次经 arXiv API 核对标题+作者+日期+摘要）**

| 短名 | 完整标题 | 作者 | 年份 | 链接 |
|---|---|---|---|---|
| Suphx | Suphx: Mastering Mahjong with Deep Reinforcement Learning | Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang 等 | 2020 | https://arxiv.org/abs/2003.13590 |
| Meowjong | Building a 3-Player Mahjong AI using Deep Reinforcement Learning | Xiangyu Zhao, Sean B. Holden | 2022 | https://arxiv.org/abs/2202.12847 |
|Mahjax | Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX | Soichiro Nishimori, Shinri Okano, Keigo Habara, Sotetsu Koyamada, Eason Yu, Masashi Sugiyama | 2026 | https://arxiv.org/abs/2605.20577 |
| Mahjong reward shaping | A Novel Reward Shaping Function for Single-Player Mahjong | Kai Jun Chen, Lok Him Lai, Zi Iun Lai | 2023 | https://arxiv.org/abs/2305.04145 |
| Deficiency number | A Fast Algorithm for Computing the Deficiency Number of a Mahjong Hand | （摘要确证） | 2021 | https://arxiv.org/abs/2108.06832 |
| CFR-p | CFR-p: Counterfactual Regret Minimization with Hierarchical Policy Abstraction, and its Application to Two-player Mahjong | Shiheng Wang | 2023 | https://arxiv.org/abs/2307.12087 |
| OIM rules | Adapting Rules of Official International Mahjong for Online Players | Chucai Wang, Lingfeng Li, Yunlong Lu, Wenxin Li | 2026 | https://arxiv.org/abs/2601.08211 |
| Evo-Sparrow | Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong | Jim O'Connor, Derin Gezgin, Gary B. Parker | 2025 | https://arxiv.org/abs/2508.07522 |
| DouZero | DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning | Daochen Zha, Jingru Xie, Wenye Ma, Sheng Zhang, Xiangru Lian 等 | 2021 | https://arxiv.org/abs/2106.06135 |
| DouRN | DouRN: Improving DouZero by Residual Neural Networks | Yiquan Chen, Yingchao Lyu, Di Zhang | 2024 | https://arxiv.org/abs/2403.14102 |
| DanZero | DanZero: Mastering GuanDan Game with Reinforcement Learning | Yudong Lu, Jian Zhao, Youpeng Zhao, Wengang Zhou, Houqiang Li | 2022 | https://arxiv.org/abs/2210.17087 |
| DanZero+ | DanZero+: Dominating the GuanDan Game through Reinforcement Learning | Youpeng Zhao, Yudong Lu, Jian Zhao, Wengang Zhou, Houqiang Li | 2023 | https://arxiv.org/abs/2312.02561 |
| AlphaDou | AlphaDou: High-Performance End-to-End Doudizhu AI Integrating Bidding | Chang Lei, Huan Lei | 2024 | https://arxiv.org/abs/2407.10279 |
| Big 2 self-play | Self-Play Reinforcement Learning under Imperfect Information in Big 2 | Aalok Patwa | 2026 | https://arxiv.org/abs/2605.28863 |
| SkyNet | SkyNet: Belief-Aware Planning for Partially-Observable Stochastic Games | Adam Haile | 2026 | https://arxiv.org/abs/2603.27751 |
| RLCard | RLCard: A Toolkit for Reinforcement Learning in Card Games | Daochen Zha 等 | 2019 | https://arxiv.org/abs/1910.04376 |

**A.2 方差缩减 / 优势估计 / 偏好学习（**摘要确证**）**

| 短名 | 完整标题 | 作者 | 年份 | 链接 |
|---|---|---|---|---|
| GAE | High-Dimensional Continuous Control Using Generalized Advantage Estimation | John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel | 2015 | https://arxiv.org/abs/1506.02438 |
| GRPO | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song 等 | 2024 | https://arxiv.org/abs/2402.03300 |
| RLOO | Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs | Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer 等 | 2024 | https://arxiv.org/abs/2402.14740 |
| REINFORCE++ | REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization | Jian Hu, Jason Klein Liu, Haotian Xu, Wei Shen | 2025 | https://arxiv.org/abs/2501.03262 |
| Dr. GRPO | Understanding R1-Zero-Like Training: A Critical Perspective | Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin | 2025 | https://arxiv.org/abs/2503.20783 |
| Tree-style GRPO variance | Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO | Hongcheng Wang, Yinuo Huang, Sukai Wang, Guanghui Ren, Hao Dong | 2025 | https://arxiv.org/abs/2509.24494 |
| A-GRPO | Ranking-Augmented On-Policy Optimization with Adaptive Advantage-Normalization for Constrained Control | Md Ragib Rownak, Sidra Ghayour Bhatti, Qadeer Ahmed | 2026 | https://arxiv.org/abs/2608.15359 |
| DPO | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn | 2023 | https://arxiv.org/abs/2305.18290 |
| Christiano pref. | Deep reinforcement learning from human preferences | Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei | 2017 | https://arxiv.org/abs/1706.03741 |
| PG-RANK | Policy-Gradient Training of Language Models for Ranking | Ge Gao, Jonathan D. Chang, Claire Cardie, Kianté Brantley, Thorsten Joachims | 2023 | https://arxiv.org/abs/2310.04407 |
| RUDDER | RUDDER: Return Decomposition for Delayed Rewards | Jose A. Arjona-Medina 等 | 2018 | https://arxiv.org/abs/1806.07857 |
| Hindsight CA | Hindsight Credit Assignment | （仅标题核对） | 2019 | https://arxiv.org/abs/1912.02503 |
| Action-dep. baseline | Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines | Cathy Wu, Aravind Rajeswaran, Yan Duan, Vikash Kumar, Alexandre M. Bayen, Sham Kakade, Igor Mordatch, Pieter Abbeel | 2018 | https://arxiv.org/abs/1803.07246 |
| Traj. control variates | Trajectory-wise Control Variates for Variance Reduction in Policy Gradient Methods | Ching-An Cheng, Xinyan Yan, Byron Boots | 2019 | https://arxiv.org/abs/1908.03263 |
| Marginal PG | Marginal Policy Gradients: A Unified Family of Estimators for Bounded Action Spaces | Carson Eisenach, Haichuan Yang, Ji Liu, Han Liu | 2018 | https://arxiv.org/abs/1806.05134 |
| Statistical precipice | Deep Reinforcement Learning at the Edge of the Statistical Precipice | Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare | 2021 | https://arxiv.org/abs/2108.13264 |

**A.3 从启发式/示范出发的改进范式（**摘要确证**，除注明外）**

| 短名 | 完整标题 | 作者 | 年份 | 链接 |
|---|---|---|---|---|
| RPL | Residual Policy Learning | Tom Silver, Kelsey Allen, Josh Tenenbaum, Leslie Kaelbling | 2018 | https://arxiv.org/abs/1812.06298 |
| Residual RL from Demos | Residual Reinforcement Learning from Demonstrations | Minttu Alakuijala, Gabriel Dulac-Arnold, Julien Mairal, Jean Ponce, Cordelia Schmid | 2021 | https://arxiv.org/abs/2106.08050 |
| Residual RL + uncertainty | Accelerating Residual Reinforcement Learning with Uncertainty Estimation | Lakshita Dodeja, Karl Schmeckpeper, Shivam Vats, Thomas Weng, Mingxi Jia 等 | 2025 | https://arxiv.org/abs/2506.17564 |
| Residual RL value learning | What Makes Value Learning Efficient in Residual Reinforcement Learning? | Guozheng Ma, Lu Li, Haoyu Wang, Zixuan Liu, Pierre-Luc Bacon 等 | 2026 | https://arxiv.org/abs/2602.10539 |
| DAgger | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell | 2010 | https://arxiv.org/abs/1011.0686 |
| ExIt | Thinking Fast and Slow with Deep Learning and Tree Search | Thomas Anthony, Zheng Tian, David Barber | 2017 | https://arxiv.org/abs/1705.08439 |
| AlphaZero | Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm | David Silver 等 | 2017 | https://arxiv.org/abs/1712.01815 |
| KataGo | Accelerating Self-Play Learning in Go | David J. Wu | 2019 | https://arxiv.org/abs/1902.10565 |
| Sampled/Gumbel MuZero | Learning and Planning in Complex Action Spaces | Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain, Simon Schmitt, David Silver | 2021 | https://arxiv.org/abs/2104.06303 |
| MuZero | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model | Julian Schrittwieser 等 | 2019 | https://arxiv.org/abs/1911.08265 |
| AWAC | AWAC: Accelerating Online Reinforcement Learning with Offline Datasets | Ashvin Nair, Abhishek Gupta, Murtaza Dalal, Sergey Levine | 2020 | https://arxiv.org/abs/2006.09359 |
| DDPGfD | Overcoming Exploration in Reinforcement Learning with Demonstrations | Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel | 2017 | https://arxiv.org/abs/1709.10089 |
| Balduzzi open-ended | Open-ended Learning in Symmetric Zero-sum Games | David Balduzzi, Marta Garnelo, Yoram Bachrach, Wojciech M. Czarnecki, Julien Perolat, Max Jaderberg, Thore Graepel | 2019 | https://arxiv.org/abs/1901.08106 |
| Spinning tops | Real World Games Look Like Spinning Tops | Wojciech Marian Czarnecki, Gauthier Gidel, Brendan Tracey, Karl Tuyls, Shayegan Omidshafiei, David Balduzzi, Max Jaderberg | 2020 | https://arxiv.org/abs/2004.09468 |

**A.4 路标（本次仅核对条目存在，**未读摘要**，不作为结论依据）**

| 标题 | 链接 | 用途 |
|---|---|---|
| The αμ Search Algorithm for the Game of Bridge | https://arxiv.org/abs/1911.07960 | 不完美信息搜索（§3.3 的后续读物） |
| Perfect Information Monte Carlo with Postponing Reasoning | https://arxiv.org/abs/2408.02380 | PIMC 的策略融合（§3.3） |
| Improving Search with Supervised Learning in Trick-Based Card Games | https://arxiv.org/abs/1903.09604 | SL + 搜索的牌类结合（§3.3） |
| Securing Equal Share: A Principled Approach for Learning Multiplayer Symmetric Games | https://arxiv.org/abs/2406.04201 | 多人对称博弈的学习（§4.5 对手池） |
| Multiagent Gumbel MuZero: Efficient Planning in Combinatorial Action Spaces | OpenReview id `fsDPGlgLiG` | Gumbel MuZero 的近期进展（§3.3） |

**A.5 本仓库内部证据（只读引用，未改动）**

| 文件 | 用途 |
|---|---|
| `notes/REPORT-RL-20m.md` | R 线 20 场 × 2 种子负结果、归因 |
| `notes/REPORT-6.6-nnrl.md` | M/BC 线负结果、噪声底口径、RL 起点产物 |
| `notes/PLAN-nn-rl.md` | 架构级约束、三段式计划、算力纪律 |
| `notes/LOG.md` | 起点=启发式的冒烟证据、bug 更正 |
| `notes/research-nn-design.md` | 前次 NN 设计调研（Suphx/Mortal/DouZero 原文摘录、动作表示、输入维度） |
| `src/nnrl/rl_net.py` / `rl_play.py` | REINFORCE 实现、残差打分形式、动作空间 |
| `scripts/train_rl_par.py` | 奖励源、baseline、batch 语义（**§0 结构事实的来源**） |
| `src/nnrl/eval.py` | 四座位旋转、配对差分、判据定义、`place_points` 来源 |
| `src/majiang/rules/score.py`（主仓库，只读） | `DEALER_MULTIPLIER = 8`、番型倍率 ⇒ 重尾来源 |
| `src/majiang/sim/round.py`（主仓库，只读） | `scores=seat_deltas(...)`、流局 `scores=(0,0,0,0)` |
| 主仓库 `verify/noise_floor.py`（只读引用） | 噪底 1.33pp、检出 1.25pp 需 243 房 |
