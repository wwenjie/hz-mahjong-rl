# NN/RL 训练线设计依据（文献与公开资料检索）

> 归属：`hz-mahjong-rl`（研究仓库，全权所有）。
> 主仓库 `/home/wuwenjie01/majiang_ai` 全程**只读**，本次检索未写入其中任何一个字节。
> 检索日期：2026-09-28（Asia/Shanghai）。
> 检索方式：arXiv API（`export.arxiv.org`）+ ar5iv 全文 + 官方文档 / GitHub 源码原文。
> 所有链接在本次检索中实际取回（HTTP 200）；引用 ID 已用 arXiv `id_list` 查询逐一核对标题与作者。

**口径声明（先读这段，避免误读）**：
- 本文件区分三种证据等级：**【原文】**＝我直接取回了论文/源码正文并引用原句；**【转述】**＝我取回的是**别的论文对第三方工作的描述**（不是被引论文本身）；**【未找到】**＝本次检索未发现可靠来源，不做推测填充。
- 学习标的差异很大：学界主流是**日本立直麻将（Riichi）**与**斗地主/GuanDan/大老二**，**没有检索到杭州麻将（或任何"自摸胡、无点炮、财神百搭、番型连乘"规则）的公开 AI 论文**。规则差异会影响结论可迁移性——见各节「不确定性」。
- 杭州麻将的关键结构差异（自摸胡、无点炮、财神百搭、番连乘至 512、流局归零）**没有任何文献覆盖**。

---

## 问题 1：出牌策略网络的动作表示

### 结论

业界**没有单一主流做法**，实际分成三派，且**与"动作空间是否可变"强相关**：

| 做法 | 代表 | 适用前提 | 主要代价 |
|---|---|---|---|
| **逐张打分**（34 类分类 / 34 输出神经元） | Suphx 出牌模型、Meowjong 出牌模型 | 出牌动作集合**固定 = 34 种牌** | 无法直接表达"吃/碰/杠/胡"；需要**并列的多个模型或 head** |
| **定长动作槽位 + 合法动作掩码** | Mortal（37 维 DQN）、Big2（1695 维） | 动作可枚举成定长槽 | 换规则/新增动作类型就要改输出层；**动作槽必须含全部动作类型** |
| **候选动作编码 + 逐候选打分（Q(s,a)）** | DouZero（牌矩阵）、PerfectDou 同类 | 动作空间**巨大且可变** | 每个候选要跑一次网络（推理成本 ∝ 候选数） |

**对本项目的直接含义**：杭州麻将出牌面 = 34 种牌，**逐张打分是最自然的起点**（与 Suphx/Meowjong 同构）；但吃/碰/杠/胡是**另一组动作类型**，文献里的标准解法是**并列多个 head 或模型**，而不是把出牌和吃碰压进同一个 34 维向量。Mortal 的做法（出牌+杠 = 37 维定长槽 + 掩码，其余动作用 `can_*` 布尔掩码分支）是"单网络 + 掩码"的工程折中。

### 证据与出处

**【原文】Suphx —— 逐张打分，5 个独立模型**
- 链接：https://arxiv.org/abs/2003.13590 ｜ 全文：https://ar5iv.labs.arxiv.org/html/2003.13590v2
- 原文（Table 1 / 2.1 节）：Suphx 学 **5 个模型**——`Discard / Riichi / Chow / Pong / Kong`，各处理不同情形。
- 原文（2.2 节）：「The discard model has **34 output neurons corresponding to 34 unique tiles**, the Riichi/Chow/Pong/Kong models have only **two output neurons** corresponding to whether or not to take a certain action.」
- 原文（Table 2）：输入/输出维度 = Discard `34×838` → 34；Riichi `34×838` → 2；Chow/Pong/Kong `34×958` → 2。
- 原文（2.2 节）：「there is **no pooling layer** in our models, because every column of a channel has its semantic meaning and pooling will lead to information loss.」——即 34 张牌是"语义列"，不是图像。输入端用 **4 个通道编码本人手牌**（第 n 通道第 m 列 = 手中有 n 张第 m 种牌）。
- 编排方式：出牌模型是唯一"动作空间大"的模型，其余 4 个是**二分类**。这正是"逐张打分 + 并列小模型"的组合。

**【原文】Meowjong（三人麻将）—— 逐张 softmax 分类，5 个 CNN**
- 链接：https://arxiv.org/abs/2202.12847
- 原文摘要：「We pre-train **5 convolutional neural networks** (CNNs) for Sanma's 5 actions—discard, Pon, Kan, Kita and Riichi, and enhance **the major action's model, namely the discard model**, via self-play reinforcement learning」。
- 原文（网络结构）：「a CNN structure with **4 convolutional layers** followed by a fully-connected layer. Each of the first 3 convolutional layers has **64 filters**, and the last convolutional layer has **32 filters**」。
- 网络规模量级：**4 conv 层、64/64/64/32 filters + 1 FC**——即**十万级参数**，远小于 Suphx。

**【原文】Mortal —— 定长 37 维 DQN + 合法动作掩码**
- 仓库：https://github.com/Equim-chan/Mortal ｜ 文档：https://mortal.ekyu.moe
- 源码 `libriichi/src/consts.rs`：`pub const ACTION_SPACE: usize = 37 // discard | kan (choice)`
- 源码 `mortal/model.py`：`DQN` 的 `a_head = nn.Linear(512, ACTION_SPACE)`（或 `Linear(1024, 1+ACTION_SPACE)`），前向里做
  ```
  q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
  ```
  即**定长动作槽 + 掩码把非法动作置 −∞**，再加 Dueling 的均值中心化。
- 源码 `libriichi/src/state/action.rs`：`ActionCandidate` 用一组布尔量（`can_discard / can_chi_low|mid|high / can_pon / can_daiminkan / can_kakan / can_ankan / can_riichi / can_tsumo_agari / can_ron_agari / can_ryukyoku`）表达**可行动作集合**——即"动作掩码由规则引擎给"。

**【原文】DouZero —— 候选动作编码 + Q(s,a) 打分（动作特征泛化）**
- 链接：https://arxiv.org/abs/2106.06135 ｜ 全文：https://arxiv.org/html/2106.06135v1
- 原文（4.1 节）：「We encode each card combination with a **one-hot matrix**… For the state, we extract several card matrices… **Similarly, we use one card matrix to encode the action.**」
- 原文（3.2 节，**核心权衡**）：「While policy gradients methods work well in large action space, they **cannot use the action features to reason about previously unseen actions** (Dulac-Arnold et al., 2015)… by encoding the actions into card matrices, it can naturally **generalize over the actions that are not frequently seen** throughout the training process.」
- 网络规模：**LSTM（历史动作）+ 6 层 MLP、hidden 512**，对 state-action 拼接打分。
- 这是"候选动作编码"派的**理由原文**：当动作空间巨大且长尾时，逐候选编码 → 泛化；**纯策略梯度**做不到这点。

**【原文】Big 2（四人非完全信息牌类）—— 定长 1695 维 masked softmax**
- 链接：https://arxiv.org/abs/1808.10442
- 原文：「a linear output layer of **1695 units** which represents a probability weighting of each potentially allowable move. This is then **combined with the actually allowable moves to produce an actual probability distribution**.」
- 网络规模：**初始共享隐层 512 ReLU → 两个 256 ReLU 隐层**，一支出 value、一支接 1695 维动作头（**全连接**）。
- 这证明"定长动作槽 + 掩码"在**四人、非完全信息、可变动作空间**的牌类里可用。

**【原文】AlphaZero —— 动作 = 空间平面或扁平向量**
- 链接：https://arxiv.org/abs/1712.01815 ｜ 全文：https://ar5iv.labs.arxiv.org/html/1712.01815
- 原文：「The actions are encoded by either **spatial planes or a flat vector**, again based only on the basic rules for each game (see Methods).」
- 即围棋走子天然用平面，象棋/将棋用扁平向量——**按"动作是否空间化"分别选择**。

### 网络规模量级汇总

| 系统 | 参数量级（可读到的部分） |
|---|---|
| Meowjong (Sanma) | 4 conv（64/64/64/32）+ 1 FC → **~10^5** |
| Big 2 | 512 → 256/256 + 1695 输出头 → **~10^5–10^6** |
| DouZero | LSTM + 6×MLP(hidden 512) → **~10^6** |
| Mortal | ResNet，**conv_channels=192、num_blocks=40** + 1024 维表征 → **~10^7**（镜像规格，非官方模型参数） |
| Suphx | 5 个独立 CNN，无 pooling；具体层数/通道数在 arXiv 正文以图给出，**未能读到文字化的数值** |

> Mortal 规格出处：https://github.com/Equim-chan/Mortal/blob/main/mortal/config.example.toml（`conv_channels = 192`、`num_blocks = 40`、`batch_size = 512`、`next_rank_weight = 0.2`）。
> 本项目的 64×64 MLP（约 4×10^3 参数）比上表**最小者还小两个数量级**——结合问题 2 的冷启动证据，这解释了为什么"整段替换"会崩。

### 不确定性

1. **杭州麻将规则无文献覆盖**。"财神百搭"会让"34 类逐张打分"的语义发生变化（白板是万能牌，其"打出去"的代价与普通牌完全不同），文献里的 34 类表示**没有处理百搭牌**——这是一个需要自行设计的缺口。
2. **番型连乘（可达 512）** 意味着动作的长期收益极其重尾，文献里 Riichi 的"番"量级（常见 1–4 番）小得多；"逐张打分"输出的是**策略 logits**，不是番数，因此这一点主要影响**回报设计**而非动作表示。
3. **Suphx 网络的具体层数/通道数**我没能从可读正文中取到数值（Figure 4/5 是图片）；只确认了输入维度 `34×838` / `34×958` 与"无 pooling"。
4. `ACTION_SPACE = 37` 的**确切组成**（34 出牌 + 3 种杠选择？）源码注释只写了 `discard | kan (choice)`，我未逐行验证其展开方式。

---

## 问题 2：行为克隆 warm start 是否必要？冷启动有多严重？

### 结论

**在麻将/复杂非完全信息牌类里，"先模仿（人类或启发式）再 RL 微调"是压倒性的标准范式**，而且是**近年的默认工程实践**——包括 2026 年的最新工作仍在用 BC 初始化，并把"从零"明确列为**未来工作**。

更关键的是：**DouZero 论文直接给出了"从零 RL 打不过简单启发式"的定量反例**（<20% 胜率、训练 20 天）。结合本项目已有的 MLP 实测（名次分 −3.425，t −6.67），**结论是：不做行为克隆直接上策略网络，是已知会失败的路**。

引用方自己的证据链也非常清晰：Suphx 的 RL 臂叫做 **`RL-basic` = 「用 SL 出牌模型初始化」**——即"RL"这个名字本身就以 BC 为前置。

### 证据与出处

**【原文】Suphx —— 三段式：SL → self-play RL → 在线适配**
- 原文（3 节）：「First, we train the five models of Suphx by **supervised learning, using (state, action) pairs of top human players** collected from the Tenhou platform. **Second, we improve the supervised models through self-play reinforcement learning**…」
- 原文（4.2 节）：「**RL-basic**: In RL-basic, the discard model was **initialized with the SL discard model** and then boosted through the policy gradient method with round scores as reward and entropy regularization.」
- 原文（4.1 节，BC 的可行性与上限）：SL 出牌模型 **15M 样本、测试准确率 76.7%**（对比此前工作 68.8%）；Riichi 5M/85.7%；Chow 10M/95.0%；Pong 10M/91.9%；Kong 4M/94.0%（**Table 3**）。
- 解读：**出牌是 34 类分类，准确率只有 76.7%**——即人类顶尖牌手在同一局面下的出牌有近 1/4 不一致。**这给"行为克隆启发式"设了一个天花板参考**：模仿启发式时，监督准确率不必（也不可能）追到 100%。

**【原文】Meowjong —— 同样先 SL 再 RL，且 RL 只动出牌模型**
- 原文摘要：「We **pre-train** 5 CNNs for Sanma's 5 actions…, and **enhance the major action's model, namely the discard model, via self-play RL**」。
- 原文（VII-B 节）：「The discard model **without data standardization** was improved through self-play RL, using the REINFORCE (Monte Carlo policy gradient) algorithm, **for 400 episodes**.」

**【原文】Mahjax（2026）—— BC 初始化是前提，从零是"未来工作"**
- 链接：https://arxiv.org/abs/2605.20577
- 摘要：「While prior research has heavily relied on **supervised learning from human play logs to pre-train the policy**, algorithms capable of learning **tabula rasa (from scratch)** offer greater potential…」（即：承认从零更有普适价值，但仍未做到）
- 结果节：「The agent consistently achieves an average rank better (lower) than the neutral 2.5 baseline, indicating successful **policy improvement over the BC initialization**.」
- 局限节：「while our current RL training, **leveraging BC pre-training**, successfully demonstrates the simulator's reliability…, **we aim to move toward learning from scratch.**」

**【原文/转述】DouZero —— 冷启动失败的定量反例（关键）**
- 同域引述原文（DouZero 1 节引 You et al., 2019 与 Zha et al., 2019a）：
  - 「DQN and A3C are shown to have **less than 20% winning percentage against simple rule-based agents even with twenty days of training**」
  - 「the DQN in (Zha et al., 2019a) is **only slightly better than random agents** that sample legal moves uniformly」
  - 「CQN… **can not even beat simple heuristic rules after twenty days of training**」
- **注意**：这是 DouZero 对第三方工作的**转述**（原始出处为 You et al. 2019 / Zha et al. 2019a，我未直接取回原文）。但它是同领域、同量级的公开记录，可作为冷启动风险的**旁证**，不宜作为唯一依据。

**【原文】DouZero 自身 = 从零，但用的是"动作编码 + DMC 值学习"而非策略网络**
- DouZero 摘要明说「without the abstraction of the state/action space or any human knowledge」。**但它不是策略梯度、而是对 state-action 打分的 Deep Monte-Carlo**，并且用了动作特征泛化（见问题 1）。**"从零成功"的功劳是否可归因于"动作编码"而非"从零"，原文没有做消融**——所以不能用它证明"麻将类可以裸策略网络从零训"。

**【原文】通用范式：Kickstarting 蒸馏**
- 链接：https://arxiv.org/abs/1803.03835（DeepMind）
- 摘要：「We present a method for using previously-trained **'teacher' agents to kickstart the training of a new 'student' agent**… it **regulates itself to allow the students to surpass their teachers** in performance… kickstarted training **improves the data efficiency** of new agents.」
- 对本项目的直接映射：**手写启发式就是现成的 teacher**；kickstarting = 用蒸馏把启发式能力灌进学生网络，且**允许学生最终超越老师**（不是把网络锁死在启发式水平）。

**【原文】Mortal —— 离线 RL（人类牌谱 + CQL），而非在线从零**
- 源码 `mortal/train.py`：`cql_loss = q_out.logsumexp(-1).mean() - q.mean()`，仅在 `not online` 时加入；`dqn_loss = 0.5 * mse(q, q_target_mc)`，其中 `q_target_mc = gamma ** steps_to_done * kyoku_rewards`（**直接用本局最终回报做 MC 目标**，无 bootstrap）。
- 即 Mortal = **离线数据 + DQN + CQL 正则**。`config.example.toml` 有独立的 `[offline]` / `[online]` 两段。

**【原文】Mxplainer —— 逆问题：从既有麻将 agent 反推可解释规则**
- 链接：https://arxiv.org/abs/2506.14246
- 摘要：提出参数化搜索算法，把黑箱麻将 agent 的行为**转成可读条件规则**，对大多数局面可"解释"黑箱。
- 与本项目的关系：提供了"**用行为克隆把已有 agent 的能力提取出来**"的同类思路证据（模仿是可行且被研究的）。

### 不确定性

1. **"行为克隆 warm start 是否必要"这个问法在文献里没有以否定形式出现过**——我没有找到任何一篇论证"在麻将类可以不用 BC 而裸 RL 成功"的论文。因此更准确的说法是：**现有公开证据全部一致地支持 BC 前置，无相反证据**（这是"没有反例"而非"已证明不可能"）。
2. **BC 的目标是"人类职业牌手"还是"自己的启发式"**，文献里都是前者。用**启发式当 teacher** 的可行性来自 Kickstarting（通用、非麻将）——**在麻将域没有直接实验**。启发式可能含有"人类不下但机器偏好"的确定性怪招，蒸馏时是否会被放大，**未找到来源**。
3. DouZero 引述的 <20% / 20 天数据是**斗地主**，不是麻将；跨游戏外推有风险。
4. Suphx 的 `76.7%` 出牌准确率是**人类一致性**上限，**不等于**"模仿启发式的难度"。若启发式是确定性的，BC 准确率原则上可以接近 100%（单标签），此时 BC 的瓶颈会从"标签噪声"变成"表征容量"——这一点文献未讨论。

---

## 问题 3：从零自对弈需要多少局才能超过一个体面的手写启发式？

### 结论

**直接回答"多少局能超过手写启发式"的可靠来源：未找到。**

文献里能给的是**量级参照**，而且几乎都是"**达到人类高手水平**"而非"超过手写启发式"：

| 系统 | 训练规模 | 训练资源 | 起点 | 对手/评价 |
|---|---|---|---|---|
| **Big 2**（四人非完全信息，最接近的参照） | **约 300 万局** | 单机 4 核 + 1 GPU，**约 2 天** | 从零（纯自对弈） | 超过**业余**人类 |
| Suphx | 每个 RL agent **150 万局** | 44 GPU（4 Titan XP + 40 Tesla K80），**2 天** | **SL 初始化** | 对 3 个 SL-weak；评价用 100 万局 |
| Suphx SL 阶段 | 单模型 **4M–15M** 样本（5 模型合计约 4400 万） | 未在正文给 | — | 出牌 34 类 acc 76.7% |
| AlphaZero（象棋） | **700,000 steps × batch 4,096** | **5,000 一代 TPU 生成 + 64 二代 TPU 训练** | 从零 | **4 小时（300k steps）就超过 Stockfish** |
| DouZero | "几天" | 单机 **4 GPU** | 从零 + 动作编码 | 超过全部已有 DouDizhu 程序与规则程序 |
| DanZero (GuanDan) | **30 天** | **160 CPU + 1 GPU** | 从零（DMC） | 超过 8 个启发式基线，达人类水平 |
| Mahjax | **1 亿环境步**（≈5.8 小时 / 单张 GH200） | 1×GH200 | **BC 初始化** | 平均名次 < 2.5（对 3 个固定 BC 策略） |
| Meowjong | RL **400 episodes** | 未在正文给 | SL 初始化 | 对 2 个基线 agent |

**对本项目的量化含义（我的推断，非文献结论）**：
- 本项目真机数据 ≈ **2,400 局**，比上表**最小的"成功案例"（Big 2 的 300 万局）还小三个数量级**。
- 但**本项目的场景与上表不同**：我们有**规则引擎 + 可无限造数据的自对弈模拟器**，所以"数据量"不是瓶颈，**算力与自对弈质量**才是。Big 2 的 300 万局 / 2 天 / 单 GPU 说明：**四人非完全信息牌类在"家用级算力 + 百万局级"是有机会的**——这是对本项目最乐观、也最相关的一条外部证据。

### 证据与出处

**【原文】Big 2 —— 300 万局、单机、2 天、从零**
- 链接：https://arxiv.org/abs/1808.10442
- 原文：「We then run this for **[math] total steps ([math] training updates) which corresponds to approximately 3 million games**. This was carried out on **a single PC with four cores and a GPU and took about 2 days** to complete.」
- 原文（自对弈对手）：「We did **not** find that it was necessary to use any kind of opponent sampling… and so the neural networks were **always playing the most recent copies of themselves** throughout the entire duration.」
- 原文（水平）：「it is able to reach a level which **outperforms amateur human players** after only a relatively short amount of training time.」
- 用途：**这是"四人非完全信息 + 从零 + 可复现资源"的最强参照**。

**【原文】Suphx —— 1.5M 局/agent、44 GPU、2 天（但 SL 初始化）**
- 原文（4.2 节）：「for fair comparison, **each RL agent was trained using 1.5 million games**. The training of each agent costs **44 GPUs (4 Titan XP for the parameter server and 40 Tesla K80 for self-play workers) and two days**.」
- 原文（评估协议）：「we randomly generated **one million games**… the evaluation of one agent took **20 Tesla K80 GPUs for two days**」
- **注意**：这 1.5M 是"**从 SL 顶点再往上**"所需的局数，**不是**从零到超过启发式所需局数。

**【原文】AlphaZero —— 700k steps，300k steps / 4 小时超过 Stockfish**
- 原文：「Training proceeded for **700,000 steps (mini-batches of size 4,096) starting from randomly initialised parameters**, using **5,000 first-generation TPUs** to generate self-play games and **64 second-generation TPUs** to train the neural networks.」
- 原文：「In chess, AlphaZero **outperformed Stockfish after just 4 hours (300k steps)**」
- 用途：给出"从零 → 超过强基线"的**资源量级**参照（TPU 集群级）。**注意这不是牌类、也不是非完全信息**。

**【原文】DouZero —— 单机 4 GPU、数天、从零**
- 摘要：「Starting from scratch in a **single server with four GPUs**, DouZero outperformed all the existing DouDizhu AI programs **in days of training**」
- 用途：从零成功的**算力下限**参照；但**算法不是策略梯度**（见问题 2 不确定项）。

**【原文】DanZero（GuanDan）—— 30 天 / 160 CPU + 1 GPU**
- 链接：https://arxiv.org/abs/2210.17087
- 原文：「After training for **30 days using 160 CPUs and 1 GPU**, we get our DanZero bot. We compare it with **8 baseline AI programs which are based on heuristic rules** and the results reveal the outstanding performance of DanZero.」
- 用途：**"从零 RL 超过启发式规则基线"的少见直接案例**，但代价是 30 天 + 160 CPU。这是问题 3 最贴近"超过手写启发式"的一条证据。

**【原文】Mahjax —— 1 亿环境步**
- 原文：「The training run spanned **100 million environmental steps**, taking approximately **5.8 hours on a single NVIDIA GH200** Grace Hopper GPU.」（**BC 初始化**）

**【原文】Mahjax 的重要旁证：算力瓶颈是公开共识**
- 原文：「**AlphaHoldem required 6.5 billion training steps** to master heads-up no-limit poker. Given that Mahjong involves four players and longer horizons than poker, existing **CPU based simulators create a computational bottleneck** for practical training.」
- 用途：**"想从零训麻将，先要解决模拟器吞吐"**——这与本项目的"自对弈模拟器"路线直接对应。

### 不确定性

1. **"多少局超过一个体面的手写启发式"没有直接来源，标注：未找到可靠来源。** 所有可比数字都是"超过人类高手"或"超过特定基线程序"，且基线的强度未知。
2. **DanZero 的 8 个"heuristic baselines"具体强度未读到**，无法判断是否等价于本项目 `heuristic`（v2, `tiebreak=exact-ukeire`）的水平。
3. **跨游戏不可直接换算**：Big 2 手牌 13 张、无副露系统、无财神、计分简单；杭州麻将**有副露、有百搭、番型连乘、流局归零**，每局的有效决策深度与回报方差都更大。
4. **"局"的定义不统一**：Suphx 的 "game" 含 8–12 个 round（原文：「each game contains multiple rounds, e.g., 8-12 rounds」）；本项目的"局"与"房"要按自己的口径对齐，跨表比较时需换算。
5. AlphaZero 的 `700,000 steps` 是**训练步**不是局数，**不能**当"局数"用（每 step 一个 batch of 4,096 个局面）。

---

## 问题 4：已知的自对弈偏差/陷阱，与对付手段

### 结论

有明确的、理论 + 工程两层文献，**而且与本项目已观测到的现象高度吻合**：

**(a) 理论层：自对弈在"非传递"策略空间里不保证变强。**
- 真实游戏的几何结构 = **传递轴（强度）× 非传递径向（循环）**，"像陀螺（spinning top）"。在**存在循环**的策略区域，自对弈可能**不产生单调变强的策略序列**（Balduzzi 等）。这直接对应本项目 LOG.md 里"A 的机制级结论"与"feed-high 仪器检定"想区分的东西。

**(b) 工程层：对付手段有三种，且都已被大系统采用。**
| 手段 | 代表 | 机制 |
|---|---|---|
| **联盟/剥削者（league + exploiters）** | AlphaStar（**转述**）/ TStarBot-X | 主智能体之外专设 `main exploiter` / `league exploiter` **专门训练去打败当前策略的弱点**；用 **PFSP** 选对手，并维持策略多样性 |
| **只用可观测信息 + 训练期特权信息退火** | **Suphx `oracle guiding`** | 训练时给一个"能看全部暗牌"的 oracle agent，**逐步 dropout 特权信息**，最后转成只看可观测信息的正常 agent |
| **离线 RL + 保守正则 + 辅助预测头** | **Mortal** | 离线数据 + **CQL** 抑制 Q 过估计；`aux_net` 预测**最终名次**（`next_rank_weight=0.2`）；`GRP` 预测**名次概率分布** |

**(c) 本项目的本地实证（这是最有力的部分，且我核对过原文）**：
- `/home/wuwenjie01/majiang_rl/notes/LOG.md` 与主仓库 `notes/STATUS.md` 都记录了 `feed-high` 的**仪器检定假设**：「16 个单旋钮档位里 **14 个总得分小幅为正**（合并均值约 +0.40，单种子标准误 0.4–0.7）。若 feed-high 也为正 ⇒ 我们的**自对弈场地（三个自己的复制品）对任何偏离都给正分**，那么所有 ±0.5 量级的自对弈正号都不能作为采纳依据，判据必须改用机制指标。」
- 这正是文献所说"**对手是自己复制品 → 评分失真**"的教科书级实例：**用自己当对手时，评估信号无法区分"真的更好"和"只是更不一样"**。
- 主仓库 `notes/STATUS.md` 里 A 的机制结论「我们第 4 摸均向听就落后对手 0.16」也是在**同一场地**测的——同样受这条偏差约束。

### 证据与出处

**【原文】真实游戏像陀螺（非传递性）**
- 链接：https://arxiv.org/abs/2004.09468 ｜ 全文：https://ar5iv.labs.arxiv.org/html/2004.09468v2
- 摘要原文：「We hypothesise that their geometrical structure resembles a **spinning top**, with the **upright axis representing transitive strength**, and the **radial axis representing the non-transitive dimension, which corresponds to the number of cycles that exist at a particular transitive strength**.」

**【原文】非传递零和博弈中的开放式学习**
- 链接：https://arxiv.org/abs/1901.08106（Balduzzi 等，DeepMind）
- 摘要原文：「If the game is approximately transitive, then **self-play generates sequences of agents of increasing strength**. However, **nontransitive**…」（即：非传递时**不保证**递增）

**【原文】策略空间多样性（指出"多样性指标本身不够"）**
- 链接：https://arxiv.org/abs/2306.16884 ｜ 全文：https://ar5iv.labs.arxiv.org/html/2306.16884v2
- 摘要原文：「A major weakness in existing diversity metrics is that **a more diverse (according to their diversity metrics) population does not necessarily mean (as we proved in the paper) a better approximation to a NE**.」→ 提出 `PSPD`（Policy Space Diversity 变体）作为 best response 的正则项。

**【转述】AlphaStar 联盟（未取到 Nature 原文）**
- AlphaStar 原文（Nature 575, 2019）**未能取回**。以下来自**二手转述**：
  - TStarBot-X（https://arxiv.org/abs/2011.13729）原文：「In **AlphaStar's league, there are three main agents** each of which is for one StarCraft race. Each main agent is **equipped with one main exploiter and two league exploiters**.」
  - TStarBot-X 自身配置原文：「we construct the league with **one main agent (MA), two main exploiters (ME) and two league exploiters (LE)**… the league training can consistently improve the agent's performance, [but] the **strategic diversity and strength in the entire league is significantly limited**.」
  - TStarBot-X 关于 PFSP 的原文：「using **prioritized fictitious self-play (PFSP)**… can both let the training agent consistently surpass its historical models. However, due to the complexities in SC2, especially for its large space of **cyclic and non-transitive strategies**, **using PFSP is not sufficient** to let the agent discover robust or novel policies. **Diversity in the policy population thus plays an important role.**」
  - OpenAI Five（https://arxiv.org/abs/1912.06680）原文（对 AlphaStar 的转述）：「**AlphaStar used a league consisting of multiple agents, where agents were trained to beat certain subsets of other agents.**」
- **证据等级：转述。** 方向可信（两篇独立论文描述一致），但细节（如 PFSP 公式、exploiter 的损失设计）**未从 AlphaStar 原文核实**。

**【原文】Suphx —— oracle guiding（对付"信息缺失导致的信用分配难"）**
- 原文（1 节）：「**Oracle guiding introduces an oracle agent that can see the perfect information including the private tiles of other players and the wall tiles.**… In our RL training process, we **gradually drop the perfect information from the oracle agent**, and finally convert it to a normal agent which only takes observable information as input. **With the help of the oracle agent, our normal agent improves much faster than standard RL training which only leverages observable information.**」
- 原文（3.3 节，附录里给出了退火公式）：`0. When [math] , all the prefect features are dropped out and the model transits from the [oracle] to [normal]`（数学符号在 ar5iv 中渲染为 `[math]`，**具体退火函数我未能读到**）。

**【原文】Suphx —— global reward prediction（对付"跨局信用分配"）**
- 原文：「**Global reward prediction trains a predictor to predict the final reward (after several future rounds) of a game** based on the information of the current and previous rounds. This predictor provides **effective learning signals**…」
- 原文（1 节，动机）：「The loss of one round does not always mean that a player plays poorly for that round (e.g., the player may tactically lose the last round to ensure rank 1 of the game…)」
- **对本项目的直接映射**：杭州麻将"**流局归零**"与"**番型连乘**"让单局回报极度重尾；Suphx 的答案是**别用单局分当即时奖励，改预测整场最终回报**。

**【原文】Mortal —— 离线 + CQL + 名次辅助头**
- 源码 `mortal/train.py`：
  - `cql_loss = q_out.logsumexp(-1).mean() - q.mean()`（离线时启用）——**保守 Q 学习，抑制对未见动作的过估计**。
  - `next_rank_logits, = aux_net(phi)`；`next_rank_loss = ce(next_rank_logits, player_ranks)`；`loss = dqn_loss + cql_loss*min_q_weight + next_rank_loss*next_rank_weight`。
  - `min_q_weight = 5`、`next_rank_weight = 0.2`（`config.example.toml`）。
- 源码 `mortal/reward_calculator.py`：`GRP` 预测**四个玩家的名次概率矩阵**，用 `pts = [3, 1, -1, -3]` 算期望分差 `reward = exp_pts[1:] - exp_pts[:-1]`——即**把"名次分"这个低方差目标显式建模出来**，与本项目"用名次分/胡次数而非每手分"的纪律完全一致。

**【原文】DouZero —— 为什么不能简单用纯自对弈 + 大动作空间**
- 原文（3.2 节）：「DQN… is susceptible to **overestimation bias**」；DouZero 的解法是 **DMC（不 bootstrap）**，即「the approximation of the target Q-value in DMC is conducted without bias, diverging from Q-learning which relies on bootstrapping methodologies, making DMC avoid the overestimating issue suffered by deep Q-learning.」（转述见 DanZero+ https://arxiv.org/abs/2312.02561）

### 不确定性

1. **AlphaStar 的联盟机制是转述**（Nature 付费，未取原文）。PFSP 的准确公式、exploiter 的奖励塑形**未核实**。
2. **"循环/非传递"在杭州麻将里到底有多严重**，没有文献可查——文献测的是围棋/象棋/星际。**本项目 LOG.md 里 14/16 正号的现象是本地证据，说明"至少存在评估偏差"**，但是否构成真正的**策略循环（exploit cycle）**，**未找到来源**，也没有本地实验区分这两者。
3. **Suphx 的 oracle guiding 与本项目的合规红线存在潜在冲突**：它**在训练期读对手暗牌**（用后逐步丢弃）。本项目 `research/FEATURES.md` 的规则是「对手手牌可以出现在**生成标签**的代码里，**不得出现**在**推理输入**的代码路径里」——**oracle guiding 属于"训练期输入"**，是否落在红线的允许侧，**需要人来判定，本文件不代为决定**。这是本文件中最需要人工裁量的一条。
4. **CQL / GRP / aux head 的收益在本项目未量化**；Mortal 的配置是工程参数，不是消融结论。

---

## 问题 5：听口宽度 / 进张数能否作为辅助目标或特征？

### 结论

- **"把听口宽度/进张数作为显式辅助目标（auxiliary loss）"的综述性来源：未找到可靠来源。** 没有检索到以 `ukeire` / `tenpai` 为辅助任务的麻将论文。
- **但"把 shanten（向听数）/ 进张（所需牌）/ 听牌概率作为网络输入特征"有直接的、代码级的证据**——开源 SOTA 级 agent **Mortal 就把它们编码进观测**。这是问题 5 最硬的一条。
- **把"向听数"作为 reward shaping 也有直接证据**（单机麻将，ShangTing 函数）。
- **Suphx 的 `look-ahead features`** 是"听口/进张类结构量的前向搜索结果"，且原文明确说它是决策的关键支撑——**这是与"听口宽度是瓶颈"最接近的公开设计**。
- **通用辅助任务的收益有理论/实验支持**（Jaderberg 等 2016），但**不是麻将域**。

**对本项目的直接含义**：文献支持「**把进张/听口作为特征喂进去**」，也支持「**向听数作为 reward shaping**」，但**不支持**"把听口宽度做成一个被论文验证过的独立辅助目标"——那条路**在本项目属于未验证设计**，需要自行消融。

### 证据与出处

**【原文+代码】Mortal —— shanten / 进张 / 听牌概率**直接进观测**
- 源码：https://github.com/Equim-chan/Mortal/blob/main/libriichi/src/state/obs_repr.rs
- 取到的关键片段（原文行号来自我取回的源码）：
  - `let n = state.shanten as usize;`（**向听数进入编码路径**）
  - `IntegerEncoder::new(doras_unseen as usize, 5 * 4 + 3)`（整数特征分桶编码）
  - `discard_candidates_with_unconditional_tenpai()`（**无条件听牌的候选**——即"打这张能不能听"，且"无条件"意味着**听口张数足够**）
  - `keep_shanten_discards` / `next_shanten_discards`（按向听变化分类候选）
  - `// Encode required tiles.` → `for r in &candidate.required_tiles { ... self.arr.assign(self.idx + discard_tid, required_tid, 1.); }`（**显式把"所需牌/进张"编成通道**）
  - `candidate.tenpai_probs`、`candidate.win_prob`、`candidate.ev`（**听牌概率 / 和牌概率 / 期望值**都被编码）
  - `self.arr.assign(idx, tid, tenpai_prob);`
- 解读：Mortal 的网络输入里**同时存在**「向听数」「打某张后的所需牌集合」「听牌概率」「和牌概率」「EV」——即**进张/听口信息是被显式特征化的，不是让网络从原始牌面自己推断**。

**【原文】Suphx —— `look-ahead features`（前向搜索出的成胡概率与得分）**
- 原文（2.2 节）：「In addition to the directly observable information, we also design some **look-ahead features, which indicate the probability and round score of winning a hand if we discard a specific tile from the current hand tiles and then draw tiles from the wall to replace some other hand tiles.**」
- 原文（1 节，作用）：「we design look-ahead features to **encode the rich possibilities of different winning hands and their winning scores** of the round, **as a support to the decision making of our RL agent**.」
- 原文（2.2 节末）：「a feature represents whether discarding a specific tile can lead to a winning hand of 12,000 round score with replacing 3 hand tiles」（即**"打这张 → 替换若干张 → 能否成胡 + 得多少分"**逐张编码）
- 输入维度：出牌模型 `34×838`（其中 838 远大于纯状态通道数，差额就在 look-ahead 类特征）。
- **对本项目的直接映射**：本项目 A 的机制结论是"瓶颈在决策侧的听口宽度"。Suphx 的对应设计就是**把"打这张之后的成胡概率与番数"逐张算出来喂给网络**——**这正是"听口/进张"信息的工业级实现方式**。

**【原文】缺陷数/向听数的快速算法 —— 明确说它服务于决策**
- 链接：https://arxiv.org/abs/2108.06832
- 摘要原文：「An important notion in Mahjong is the **deficiency number (a.k.a. shanten number in Japanese Mahjong)** of a hand, which estimates how many tile changes are necessary to complete the hand into a winning hand. The deficiency number plays an **essential role in major decision-making tasks such as selecting a tile to discard**.」
- 摘要原文（可迁移性）：「The algorithm can be used as a **basic procedure in all Mahjong variants by both rule-based and machine learning-based Mahjong AI**.」（**明确点名 ML-based AI 也可以把它当前置过程**）

**【原文】ShangTing 作为 reward shaping（把向听数当辅助目标）**
- 链接：https://arxiv.org/abs/2305.04145
- 摘要原文：「Mahjong is a complex game with an intractably large state space with **extremely sparse rewards**… To overcome this, the **ShangTing function was adopted as a reward shaping function**.」
- 结果原文：「In a simulated 1-v-1 battle, usage of the new reward function **outperformed the default ShangTing function**, winning an average of $1.37 over 1000 games.」
- 用途：**"向听数类结构量可以做 reward shaping，并且不同 shaping 函数之间可测出差异"** 的直接证据。**注意：这是单机麻将 / 1v1，不是 4 人；也不是 self-play。**

**【原文】通用辅助任务（非麻将域，理论支撑）**
- 链接：https://arxiv.org/abs/1611.05397（Jaderberg 等，DeepMind）
- 标题即结论：「**Reinforcement Learning with Unsupervised Auxiliary Tasks**」——用辅助任务改善表征与数据效率。

**【原文】Mortal 的辅助头（域内实例）**
- `aux_net = AuxNet((4,))` → 预测 4 个玩家的最终名次；`next_rank_weight = 0.2`。
- 这证明"**在主任务之外加预测头**"在该域是被采用的（虽然内容是名次，不是听口）。

### 不确定性

1. **"听口宽度/进张数作为独立辅助目标（auxiliary loss）"——标注：未找到可靠来源。** 我检索了 arXiv 的 `shanten / ukeire / tenpai / waiting tiles / effective tiles` 等关键词，**只命中 2 篇**（`2108.06832` 缺陷数算法、`2305.04145` reward shaping），**均为单机或 1v1、非辅助损失范式**。任何"听口宽度做 aux loss"的设计**在本项目属于原创假设**，必须自行消融。
2. **Mortal 的 `obs_repr.rs` 证据是"代码级"而非"论文级"**：我读的是源码，**没有**读到 Mortal 对这些特征做消融（"去掉进张特征会掉多少"）。所以"Mortal 用了"≠"Mortal 证明它有用"。
3. **Suphx 的 look-ahead features 是"前向模拟搜索结果"**，与"直接用规则引擎算精确进张"**不等价**：前者是**在不知道牌墙时的概率估计**，后者是**在完全信息的模拟器里的精确计数**。本项目 A 已经发现"我们**精确进张**反而用得不够广"——文献没有覆盖"精确进张次排序在真机的边际收益"这个问题。
4. **合规约束**：本项目 `research/FEATURES.md` 明确「**推理路径不 import 任何读对手手牌的接口**」。Mortal 的 `required_tiles` / `tenpai_probs` 依赖"可见牌"（弃牌+副露），**原则上合规**；但**我在源码层面无法 100% 确认它没有间接读到暗牌信息**（例如对手模型）。若本项目要照搬，**必须逐项回溯来源**（这本来就是 `FEATURES.md` 规定的审计方法）。
5. **`tenpai_probs` 在 Mortal 里是"逐巡概率预测"**（代码里按 `turn` 取 `candidate.tenpai_probs.first()`，且有 `shanten >= 4` 的截断分支），**其精度依赖其内部估计器**，我**未评估**该估计器的可靠性。

---

## 附：检索方法与覆盖度（便于他人复核）

- **检索源**：arXiv API（`https://export.arxiv.org/api/query`，`sortBy=relevance`）；全文优先用 `https://ar5iv.labs.arxiv.org/html/<id>`，其次 `https://arxiv.org/html/<id>`；开源 agent 用 GitHub `raw.githubusercontent.com` 与官方文档站原文。
- **实际使用的检索式（节选）**：`all:mahjong AND all:"reinforcement learning"`；`abs:Mahjong AND abs:"self-play"`；`abs:"Mahjong" AND (abs:"policy" OR abs:"action")`；`all:"shanten" OR all:"ukeire"`；`all:"non-transitive" AND (all:"self-play" OR all:"games")`；`all:"kickstarting" AND all:"reinforcement learning"`；`abs:"DouZero" OR abs:"Doudizhu"`；`all:"action space" AND all:"card game" AND all:"neural network"`；`all:"waiting tiles" OR all:"effective tiles"` 等。
- **已用 `id_list` 核对标题/作者的引用**：`2003.13590`、`2202.12847`、`2106.06135`、`1808.10442`、`1712.01815`、`2605.20577`、`2011.13729`、`1901.08106`、`2004.09468`、`2306.16884`、`2108.06832`、`2305.04145`、`2210.17087`、`1803.03835`、`1611.05397`、`1912.06680`。
- **`web_search` 工具在本次环境不可用**（`provider_error: web_search is disabled or no provider is available`）；因此**没有做通用网页搜索**，只用了 arXiv API + 直接 URL 抓取。这可能遗漏**不在 arXiv 上的**资料（例如期刊论文、技术博客、比赛报告）。**已标注为"未找到"的问题 3 与问题 5，其结论受此限制。**

## 附：本文件对《LOG.md》两条结论的独立支持

本文件不是转述，而是独立检索。两条与 `notes/LOG.md` 的呼应：

1. LOG.md：「任何学习模型**不能**单点替换整段决策，只能（a）在启发式之上做增量改进，或（b）先行为克隆把启发式的能力完整吸收，再微调。」
   → **文献一致**：Suphx / Meowjong / Mortal / Mahjax **全部是 (b) 或其等价的离线/预训练路径**；没有找到任何 (a) 之外的、裸策略网络从零成功的麻将案例。**（问题 2）**
2. LOG.md：`feed-high` 仪器检定担忧"我们的自对弈场地（三个自己的复制品）对任何偏离都给正分"。
   → **文献给出机制解释**：非传递策略空间 + 纯自对弈 → 评估信号与真实强度解耦（Balduzzi 1901.08106；Czarnecki 2004.09468）；**工业级对策是联盟/exploiters（AlphaStar，转述）或 特权信息退火（Suphx oracle guiding）**。**（问题 4）**

## 不确定性的总清单（人工裁量项）

| # | 事项 | 为什么需要人 |
|---|---|---|
| 1 | **oracle guiding（训练期读暗牌，逐步丢弃）** | 与本项目 `FEATURES.md` 的"只许用于离线标签"红线可能冲突，**需人判定**（问题 4 不确定项 3） |
| 2 | **财神（白板）百搭如何进入动作表示** | 无任何文献覆盖；34 类表示不处理百搭（问题 1 不确定项 1） |
| 3 | **"多少局超过启发式"** | 未找到可靠来源；给的是量级参照，不能当判据（问题 3） |
| 4 | **听口宽度做 aux loss** | 未找到可靠来源，属原创假设（问题 5 不确定项 1） |
| 5 | **没有做通用网页检索** | `web_search` 不可用，可能遗漏非 arXiv 资料（见"检索方法"） |
| 6 | **AlphaStar 联盟机制为转述** | Nature 付费，未取原文（问题 4 不确定项 1） |
| 7 | **Suphx 网络层数/通道数、oracle 退火函数** | ar5iv 正文中为图片/`[math]`，未取到数值（问题 1、4） |

---

## 补检（第二轮）：问题 3/5/1 的缺口

**本轮环境更正（重要）**：上一轮报告 `web_search` 不可用。**本轮核心 `web_search`（`openclaw:core`）仍返回 `disabled`，但 MCP 工具 `websearch__web_search`（AIGW 网关）可用，`web_fetch`（core）可用**——本轮所有直取 URL 均 **HTTP 200**。因此**本轮补上了上一轮明确缺失的"通用网页检索"**，只补三个缺口，前文不改。

---

### 补检 1（对应问题 3）：从零自对弈需要多少局才能超过一个体面的手写启发式——量级参照

**结论**：**"多少局"仍无直接来源（维持"未找到可靠来源"）**；但本轮取到**最贴近"超过手写启发式"的定量时间锚**——DouZero「**单机 48 核 + 4×1080Ti，从零训练半天即超过启发式规则**」，以及一条**反向锚**（朴素 DQN/A3C 训 20 天仍打不过简单规则）。

**【原文】DouZero —— "半天超过启发式规则"（最贴近问题 3 的定量锚）**
- 链接（HTTP 200）：https://arxiv.org/html/2106.06135v1
- 原文：「Trained from scratch in a single server with only 48 cores and four 1080Ti GPUs, DouZero **outperforms CQN and the heuristic rules in half a day**, beats our internal supervised agents in two days, and surpasses DeltaDou in ten days.」
- 原文（反向锚，说明"训不动"是常态）：「In practice, **CQN can not even beat simple heuristic rules after twenty days of training**.」；「DQN and A3C are shown to have **less than 20% winning percentage against simple rule-based agents even with twenty days of training**」
- 用途：**"从零 + 家用级算力 + 超过手写启发式"最近的定量证据**。**注意：给的是训练时间不是局数**，正文未给局数，故**"局数"缺口仍未闭合**。

**【一手】LuckyJ（腾讯）—— 唯一"从零 <1500 场到 10 段"**
- 链接（HTTP 200）：https://haobofu.github.io/（作者本人主页）
- 原文：「On 30th May 2023, our Mahjong AI LuckyJ reached 10 dan at Tenhou.net. It is by far **the only Mahjong AI that reached 10 dan from scratch using under 1500 matches**.」
- 证据等级：**【一手（作者主页声明）】**。**非论文**；"场（match）"≠"局"；对手是**人类天梯**不是手写启发式。用途：**"从零 + 极少样本"在麻将域被宣称存在**，但**不能作为"超过启发式所需局数"的判据**。

**【原文】麻将域其他量级锚（本轮直取正文）**
- Suphx 稳定段位 **8.74 dan**——Meowjong 论文 Related Works 段原文：「Suphx eventually reached a stable rank of **8.74 dan** on Tenhou, which is about 2 dan higher than Bakuuchi, and is higher than 99.99% of all the officially ranked human players on Tenhou, **though at a cost of needing extremely heavy computational resources for training**.」链接（HTTP 200）：https://arxiv.org/html/2202.12847v3 ——【原文转述 Suphx】。
- Mahjax —— 「applying it to Mahjong requires a computational infrastructure capable of generating **billions of game steps**」；训练 **1 亿环境步 / 单卡 GH200 约 5.8 小时**。链接（HTTP 200）：https://www.alphaxiv.org/abs/2605.20577 ——【原文（alphaxiv 页面）】。
- 其余（DouZero"天数"级、DanZero 30 天/160 CPU+1 GPU、Big 2 ~300 万局/2 天、AlphaZero 300k steps/4 小时超 Stockfish）见前文问题 3 表，本轮不重复。

**小结（问题 3）**：量级参照宜用**"算力 × 时间"**表达——家用级 4 GPU / 数天；麻将域从零到 10 段可 <1500 场（LuckyJ，非论文）；**"超过手写启发式所需的局数"仍无可靠来源**。

---

### 补检 2（对应问题 1/5）：听口宽度 / 进张数（ukeire）作为**输入特征** vs 作为**训练目标**

**结论（严格区分两类证据）**：
- **作为输入特征**：**有直接证据**（前文 Mortal 代码级 / Suphx look-ahead 特征级；本轮新增"听牌/听张被显式建模"的域内先例）。
- **作为 RL 辅助目标（auxiliary loss）**：**仍无直接来源**（维持"未找到可靠来源"）。本轮新增一条**域内先例**：把"对手是否听牌 / 听什么牌"做成独立预测任务并与主模型合并——**但这是监督式多任务（独立头），不是 RL aux loss**。

**【原文】Zheng 等 2019 —— 四网络合并成一个模型，含"预测对手听牌 / 听张 / 点数变化"**
- 链接（HTTP 200）：https://www.jstage.jst.go.jp/article/jsaisigtwo/2019/SAI-034/2019_05/_article/-char/en
- 摘要原文：「**Four deep neural network for discarding and predicting opponents' waiting, waiting tiles and point changes are combined into one model** and performs good during games.」；「**Predicting opponents moves and hidden states is important in imperfect information games.**」
- 证据等级：**【原文（摘要）】**。
- 用途：**麻将域内"把听牌状态 / 听张（waiting tiles）做成显式预测任务、并与出牌模型合并"的直接先例**。**关键区别**：这是**监督式多任务/独立头**（各网络分别训练后合并），**不是**把"自己手牌进张宽度"当 RL 主策略的 **auxiliary loss**；且预测对象是**对手**（偏防守）。

**【原文】辅助任务通用理论 + 明确的"可能有害"警告**
- Liebel & Körner 2018「Auxiliary Tasks in Multi-task Learning」，链接（HTTP 200）：https://arxiv.org/abs/1805.06334 ——【原文（摘要）】：辅助任务可提升最终结果与训练时间。
- Jaderberg 等 2016「Reinforcement Learning with Unsupervised Auxiliary Tasks」（https://arxiv.org/abs/1611.05397，**上一轮已录，本轮未重取**）——【原文】通用辅助任务可提升表征/数据效率（非麻将域）。
- **反例警告（仅取到检索摘要，未取正文，标【转述】）**：https://arxiv.org/html/2412.19547v1 摘要片段「**Inadequately trained auxiliary tasks negatively impact the primary task's performance**」——辅助任务**并非无条件有益**。

**小结（问题 1/5）**：「听口/进张作为**特征**」有域内证据；「**对手**听牌/听张作为**预测目标**」本轮新增域内先例（Zheng 2019，监督多任务）；「**自己手牌进张宽度**作为 **RL auxiliary loss**」**仍属原创假设，无可靠来源**，且通用文献提示辅助任务**可能有害，必须消融**。

---

### 补检 3（对应问题 4）：非传递性 / 策略循环的公开证据，与 league / exploiter 标准做法

**结论**：
- **麻将域专门的"非传递/策略循环实证"：仍未找到可靠来源。**
- **非完全信息博弈（一般）层面：证据强且直接**（本轮新增两篇关于"循环式最优反应/非传递"的原文）。
- **AlphaStar 的循环说明与 league/exploiter：本轮首次取到官方一手博文，等级由上一轮"转述"升级为【原文（一手博文）】**。
- **league / exploiter 标准做法**：一手（DeepMind 博文）+ 复现（mini-AlphaStar）+ 改进论文（NeurIPS 2023 OAL，转述）。

**【原文】非完全信息博弈中自对弈"循环/灾难性"的直接陈述**
- 链接（HTTP 200）：https://arxiv.org/html/2502.08938v1（Reevaluating Policy Gradient Methods for IIGs）
- 原文：「because **imperfect information induces cyclical best response dynamics**, such an approach can **fail catastrophically, yielding policies that are maximally exploitable**.」
- 原文：「While PG methods can at least express non-deterministic policies, **their learning dynamics generally cycle, diverge or exhibit chaotic behavior, rather than converge to Nash equilibria**.」
- 用途：**"纯自对弈在非完全信息博弈里会转圈/发散"的直接文献背书**——对应 LOG.md 的"自对弈场地对任何偏离都给正分"。

**【原文】非传递是"真实游戏的普遍结构"（PSD-PSRO，NeurIPS 2023，腾讯 AI Lab + PKU）**
- 链接（HTTP 200）：https://arxiv.org/html/2306.16884v1
- 原文：「**Most real-world games demonstrate strong non-transitivity**, where the winning rule follows a cyclic pattern (e.g., the strategy cycle in Rock-Paper-Scissors).」
- 原文：「**Traditional algorithms, like simple self-play, fail to converge to a NE in games with strong non-transitivity**.」
- 原文（指出常见多样性指标不够）：「a more diverse (according to their diversity metrics) population **⇏** closer to a full game NE」——**对应 LOG.md"多样性指标本身不够"的担忧**。
- 用途：标准对策是 **PSRO 家族 + Population Exploitability / Policy Hull 度量**，而非单靠自对弈均分。

**【原文（一手）】AlphaStar 官方博文：RPS 式循环 + main/exploiter 联盟**
- 链接（HTTP 200）：https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- 原文（循环）：「in the game rock-paper-scissors, an agent may currently prefer to play rock... As self-play progresses, a new agent will then choose to switch to **paper**... Later, the agent will switch to **scissors**, and eventually back to **rock**, creating a cycle. **Fictitious self-play**... is one solution to cope with this challenge.」
- 原文（league/exploiter）：「**we need both main agents whose goal is to win versus everyone, and also exploiter agents that focus on helping the main agent grow stronger by exposing its flaws**, rather than maximising their own win rate against all players.」
- 证据等级：**【原文（DeepMind 一手博文）】**——**上一轮"AlphaStar 联盟为转述"在本轮升级为已取到一手来源（博文层面）**。
- **保留不确定性**：**Nature 正文（doi:10.1038/s41586-019-1724-z）只取到 Data/Code availability 与参考文献段**，正文被 cookie 墙挡住（HTTP 200 但内容不含方法正文），故 **PFSP/exploiter 的精确超参仍未取到 Nature 原文**。

**【原文】mini-AlphaStar 复现：main exploiter 的对手选择逻辑**
- 链接（HTTP 200）：https://arxiv.org/html/2104.06890v2
- 原文摘要：「For main exploiters, they first arbitrarily choose a main player. **If the win rate against it is above 0.1, they will return it as the opponent**...」；「AS uses **128,000 CPU cores and 384 TPUs** for training while lasting **44 training days**」
- 证据等级：**【原文（第三方复现描述）】**。用途：exploiter"选谁当对手"是**显式规则**，可移植。

**【转述（仅检索摘要，未取正文）】AlphaStar 联盟结构（NeurIPS 2023 改进论文）**
- 摘要：「The AlphaStar league consists of **four (yet three types) constantly-learning agents: one main agent, one main exploiter, and two league exploiters**.」来源 https://neurips.cc/virtual/2023/poster/70220 —— **标【转述】**。

**【原文】EGTA —— 检测/分析非传递策略的标准方法论**
- 链接（HTTP 200）：https://arxiv.org/html/2403.04018v1（Wellman, Tuyls 等，EGTA 综述）
- 原文：「the model of the game... is derived by **interrogation of a procedural description of the game environment**.」；「the empirical game's payoffs are **induced from noisy or sparse simulation data**, and so are subject to **approximation error**」
- 用途：**"把自对弈产出诱导成经验收益矩阵、再求元层均衡"是检测非传递/循环的标准做法**；也直接解释本项目"自己的复制品当对手→评分失真"。

**小结（问题 4）**：麻将域非传递**专门实证仍缺**；但**非完全信息博弈层面证据充足**（2502.08938 循环/灾难性；2306.16884 非传递普遍 + PSRO；Balduzzi 2004.09468 陀螺）；**AlphaStar 的 RPS 循环与 exploiter 联盟已取到一手博文**；**标准做法 = main + exploiter 联盟 / PSRO 家族，并用 exploitability 类指标（而非自对弈得分）评估**。

---

### 本轮仍存的不确定性

1. **"多少局超过手写启发式"仍无直接来源**：最近锚是 DouZero"半天超启发式"（时间为刻度，非局数）。**未闭合。**
2. **听口宽度做 RL auxiliary loss 仍无可靠来源**：仅有"对手听牌/听张做监督预测头"的域内先例（Zheng 2019）；通用文献提示辅助任务**可能有害**。
3. **麻将域"策略循环"的直接实证仍缺**：证据来自一般非完全信息博弈 + RPS。
4. **AlphaStar Nature 正文未取到**（cookie/付费墙），PFSP 精确机制为"博文 + 第三方复现 + 检索摘要"混合等级。
5. **LuckyJ 的"<1500 场"是一手主页声明、非论文**，口径是"场/天梯段位"，不可直接换算为"超过启发式的局数"。
6. **DouZero 的"半天"对应局数未知**（正文未给），故仍不能换算成"局数"。
