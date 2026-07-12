# RoboBenchMart × π0.5：机器人 VLA 监督微调与后训练系统

> 本文用于简历项目描述、技术面试讲解和项目复盘。内容依据本机 RoboBenchMart、
> OpenPI、RLinf 代码、实验文档、TensorBoard 与评测日志整理。文中将“最终效果”、
> “工程链路验证”和“未获得收益的研究尝试”明确区分，避免夸大实验结论。

## 0. 阅读导引：从任务流程理解核心概念

如果你不知道 KL-PPO、RECAP-lite、advantage、critic 是什么，可以先只记住下面这条
主线。这个项目做的事情，本质上是教一个“看得见、听得懂、能控制机器人”的模型把
指定商品放进篮子：

```text
第一步：使用专家示范学习基础策略（SFT）
第二步：让模型自己在仿真环境里尝试，根据结果微调（KL-PPO）
第三步：把成功和失败经历整理成教材，再继续模仿学习（RECAP-lite）
第四步：用从未参与训练的商品考试，检查它是真会了还是只背会了训练题
```

| 项目术语 | 核心作用 | 在本项目中做什么 |
|---|---|---|
| SFT（监督微调） | 从专家输入—动作对中学习行为映射 | 模仿 744 条专家抓取轨迹 |
| PPO（强化学习） | 根据在线交互结果优化动作策略 | 在线采集轨迹并更新模型 |
| KL 约束 | 限制新策略偏离初始策略的幅度 | 防止 PPO 破坏可用的 SFT 能力 |
| RECAP-lite | 按成败条件组织经验并继续监督训练 | 用成败标签引导 flow matching |
| ID | 与练习题同类型的考试 | Fanta、Nivea、Stars 三个训练商品 |
| OOD | 题型或外观发生变化的考试 | Nestle、Slam 两个未见商品 |

### 0.1 模型到底输入和输出什么

一次决策时，模型收到三张相机图像、机器人关节状态，以及一句类似“把 Fanta 放进
篮子”的文字指令。它输出的不是一句话，而是一串连续数值，用来控制机械臂移动、旋转
和夹爪开合。一次输出未来 50 个动作，叫作 **action chunk**。环境逐个执行这些动作，
然后模型再根据新画面做下一次决策。

因此这不是普通图片分类。模型需要连续完成“看见目标—靠近—夹住—抬起—移动—放下”
的长链路；任意一步出错，整条轨迹都可能失败。

### 0.2 为什么已经做了 SFT，还要做后训练

SFT 教模型拟合专家动作，但训练时模型看到的几乎都是专家能够正确处理的状态。真正
部署时，模型会产生小误差，随后进入专家示范里很少出现的状态，例如夹偏、把商品撞倒
或走到篮子旁边却没放下。后训练希望模型从自己的闭环经历里学会恢复。

难点是在线失败很多、最终成功奖励又很稀少。错误的奖励或不可靠的价值判断会让模型
把“靠近篮子”“抬起商品”误认为完成任务，甚至忘掉 SFT 学到的基本动作。这也是引入
KL-PPO、安全门控和 RECAP-lite 的原因。

### 0.3 核心术语速查

| 名词 | 本文中的含义 |
|---|---|
| policy / actor | 根据当前观测生成动作的策略模型 |
| rollout | 让当前模型在环境里完整尝试若干次，并记录过程 |
| episode / trajectory | 从环境重置到成功或超时的一整次尝试 |
| reward | 环境给动作的分数；成功分最高，中间进度只给少量分 |
| return | 从当前位置开始，未来累计能拿到多少分 |
| value / critic | 对当前状态未来累计回报的估计 |
| advantage | 实际结果相对预期好多少；正数鼓励，负数抑制 |
| log probability | 模型认为已执行动作有多合理的数值表示 |
| reference model | 冻结的原始 SFT 模型，是“不许偏太远”的锚点 |
| checkpoint | 某个训练时刻保存的模型、优化器和训练进度 |
| fixed seed | 固定仿真随机数，让不同模型面对尽量相同的考题 |

### 0.4 建议阅读顺序

初学者建议按第 3、4、5、6、8、14 章阅读。每章先看“为什么做”，再看结果；公式和
变量只是把直觉写得精确，不需要第一次就记住。准备面试时再看第 1、2、9、10、11
章；需要复现实验时看第 12 章及运行手册。

## 1. 简历版项目描述

### 项目名称

**RoboBenchMart × π0.5：面向商品抓取入篮任务的 VLA 微调与在线/离线后训练系统**

### 一段话版本

基于 Physical Intelligence 开源的 π0.5 视觉-语言-动作模型，在 RoboBenchMart
PickToBasket 仿真任务上完成从数据转换、JAX SFT、PyTorch/FSDP 推理适配，到
KL-PPO、Proxy-mix PPO 和 RECAP-lite 的完整后训练闭环。将 744 条 H5 专家轨迹
转换为包含三视角、15 维状态、13 维动作和语言指令的 LeRobot 数据集，并在官方
checkpoint 上训练 20k steps；JAX 评测中，三个训练物体平均成功率由 47.8% 提升到
74.4%。进一步在 RLinf 中实现冻结 SFT reference 的 KL-PPO、事件/进度型奖励、
critic warmup、SFT rehearsal、训练安全门控和 advantage-conditioned CFG-SFT；
定位并修复训练/推理去噪策略不一致、episode horizon、bootstrap、跨进程数据采集、
checkpoint 恢复及显存峰值等系统问题。后训练最终未显著提升 OOD，但通过严格对照
确认了失败源于稀疏成功信号、价值估计不足和 ID 数据无法提供 OOD 视觉监督，而非
简单的训练脚本故障。

### 简历要点版本

- 将 RoboBenchMart 的 744 条、307,319 帧 PickToBasket H5 专家示范转换为
  LeRobot v2.1，完成三视角 RGB、15 维 qpos、13 维连续动作、语言 prompt、
  normalization statistics 与 π0.5 输入/输出 adapter 的全链路适配。
- 基于官方 π0.5 checkpoint 进行 20,000-step JAX SFT，batch size 32、AdamW、
  gradient clip 1.0、学习率 1e-5；三个 ID 任务总体成功率从 43/90（47.8%）提升至
  67/90（74.4%），其中 Fanta、Nivea、Stars 分别提升 30.0、13.3、36.7 个百分点。
- 在 RLinf 的 Ray + FSDP + HuggingFace/OpenPI actor-env-rollout 架构中适配
  π0.5 flow policy PPO，实现 PPO clipped objective、GAE、value head、冻结 SFT
  reference KL、自适应 KL 系数、SFT rehearsal、critic-only warmup 和成功样本门控。
- 重构 PickToBasket 奖励为 success/lift/place 事件奖励与 basket-distance potential
  增量，修复重复成功奖励、首步虚假 progress、done 后重复 reward、600-step
  truncation 和错误 bootstrap，建立 90 条 fixed-seed ID、60 条 OOD、30 条 Proxy
  的可复现评测协议。
- 基于 π*0.6 RECAP 思路实现 RECAP-lite：采集 90 条 ID autonomous rollouts，生成
  episode-level success advantage sidecar，完成正/负 advantage prompt、CFG dropout、
  三视角数据 transform、episode-balanced/quota-balanced 混合采样与 CFG-SFT。
- 通过 action parity、TensorBoard、轨迹分阶段指标和 checkpoint 参数审计定位多个
  隐蔽问题；最终 PPO 首轮 actor 更新在固定评测上由 59/90 提升至 60/90，但 Stars
  下降 3 条，RECAP mixed 为 52/90，因此按预设门控停止扩训，形成可证伪、可复现的
  VLA 后训练实验方法。

## 2. 面试时如何介绍这个项目

### 30 秒版本

我的项目是把 π0.5 适配到 RoboBenchMart 的商品抓取入篮任务。首先把三种商品的
744 条专家轨迹转换成 LeRobot，增加三相机、状态、连续动作和语言指令适配，在官方
checkpoint 上做了 20k-step SFT，使 ID 成功率从 47.8% 提升到 74.4%。之后我基于
RLinf 搭建 KL-PPO 和 RECAP-lite 后训练闭环，重点解决 flow policy 的 logprob、冻结
reference KL、奖励、rollout 采集、FSDP 训练和 checkpoint 恢复。虽然后训练没有带来
可靠 OOD 提升，但我通过对照实验定位出核心限制是训练信号和分布覆盖，而不是继续调
学习率就能解决的问题。

### 3 分钟版本

1. **任务**：机器人接收“把某商品放进篮子”的语言指令，根据三个相机和机器人
   qpos 输出 13 维连续 action chunk。训练商品是 Fanta、Nivea、Stars；Nestle、Slam
   只用于 OOD 测试，不能加入训练。
2. **SFT**：我把 RoboBenchMart 原始 H5 专家轨迹转换成 LeRobot v2.1，并为 OpenPI
   新增 RBM data config 和 policy adapter。以官方 π0.5 为初始化训练 20k steps，ID
   总成功由 43/90 提升到 67/90，但 OOD 从 5/60 降到 2/60，暴露出专门化与泛化冲突。
3. **KL-PPO**：我在 RLinf 中增加冻结 SFT reference，比较 current logprob 与同一
   observation、action、flow-noise 下的 reference logprob；PPO clip 管单轮更新，
   reference KL 管累计漂移。同时设计 success-dominant shaped reward、critic warmup
   和 SFT replay。
4. **工程难点**：flow policy 不是离散 token policy，训练时必须保存动作生成所需的
   noise/timestep 等 forward inputs。项目中还出现了训练与评测去噪过程不同、40-step
   horizon 导致几乎没有有效 chunk、Ray pipeline 数据串线、SFT/PPO 双图 OOM、恢复
   checkpoint 后 reference 错位等问题，我逐项用 fixed-seed 对照和指标断言修复。
5. **RECAP-lite**：考虑到 PPO 的 value/advantage 噪声，我又实现了 advantage-
   conditioned CFG-SFT。SFT demo 标 positive，ID autonomous rollout 按整条 episode
   success 标 positive/negative，并做 episode/quota-balanced sampling，防止长失败
   轨迹按帧淹没成功轨迹。
6. **结论**：最终 PPO 只从 59/90 到 60/90，且任务间有能力迁移；RECAP mixed 降到
   52/90。项目价值不只是一个最好数字，而是完整建立了 VLA 的数据、SFT、在线 RL、
   离线 experience replay、评测、诊断和停止准则，并能解释为什么某条路线无效。

## 3. 任务与评测协议

这一章回答三个基本问题：机器人要完成什么、哪些商品参与训练、怎样公平判断模型有没有
进步。必须先固定考试规则再比较算法，否则成功率变化可能只是题目变了。

### 3.1 PickToBasket 任务

策略输入包括：

| 输入 | 内容 |
|---|---|
| 主相机 | 货架、目标物体、篮子和机器人全局画面 |
| 手腕相机 | 末端执行器附近细节 |
| 额外相机 | 第二个外部视角 |
| state | 15 维机器人 qpos |
| prompt | `move to shelf and pick <item> to basket` |

策略输出为 13 维连续动作。π0.5 内部使用 32 维统一 action space，因此训练时将
RBM 动作 pad 到 32 维，部署时只截取前 13 维。模型一次预测长度为 50 的 action
chunk，推理使用 10 个 flow denoising steps。

### 3.2 数据划分

- **ID/train**：Fanta Sabor Naranja 2L、Nivea Body Milk、Nestle Honey Stars
  （文中简称 Stars）。
- **OOD/unseen item**：Nestle、Slam；只用于最终测试，不进入 SFT、PPO 或 RECAP
  训练数据。
- **Proxy**：从 PickToBasket 场景中选择非上述商品的随机物体，尝试用更广的 ID
  替代分布提供弱泛化信号，但不直接使用目标 OOD 商品。

### 3.3 成功定义

每条 episode 最多执行 600 个环境 action steps。目标物体被正确抓取并最终放入篮子
记为成功。成功率为：

```text
success_rate = successful_episodes / total_episodes
```

例如固定测试 90 次、成功 59 次，成功率就是 `59 / 90 = 65.6%`。不能只挑一次随机
测试中最好看的数字，也不能把不同任务数量、随机种子或最长执行步数的结果直接比较。

### 3.4 为什么重新建立评测协议

早期 RLinf PyTorch BASE 只有 38/90，而历史 JAX SFT 为 67/90。最初怀疑 JAX 到
PyTorch 转换损坏了策略，但进一步比较发现两者使用了不同的 episode seed、机器人
初始 pose 和 simulator backend。于是我没有继续调模型，而是先统一实验协议：

- 每个任务 30 个与历史 JAX 一致的 episode seeds；
- ID 不额外指定 robot pose seed，OOD 使用历史 pose seeds；
- `sim_backend=cpu`；
- 每个任务独立 pipeline stage；
- 600-step horizon，关闭 auto-reset 和 termination 忽略；
- action chunk 50、denoise steps 10；
- 固定 PyTorch rollout RNG，并输出逐任务指标。

随后又设计 fixed-observation、fixed-noise action parity：同一 observation、相同 flow
noise 下比较 JAX/PyTorch 动作统计，而不是要求两个 RNG 系统的一次随机样本逐元素
相等。该过程把“框架转换误差”和“随机采样/评测协议差异”分开。

## 4. 第一阶段：官方 π0.5 与 SFT

### 4.0 先用一句话理解 SFT

SFT（Supervised Fine-Tuning，监督微调）使用观测与专家动作组成的配对数据，使模型
输出尽量接近专家。它不需要设计 reward，也不让模型在线试错，因此通常是机器人
策略最稳定的起点。代价是模型只学到数据里出现过的内容，可能记住三个商品的外观，却
没有学到可以迁移到所有商品的抓取规则。

### 4.1 官方 JAX checkpoint 基线

| 任务 | 官方 π0.5 | 类型 |
|---|---:|---|
| Fanta | 13/30 = 43.3% | ID |
| Nivea | 17/30 = 56.7% | ID |
| Stars | 13/30 = 43.3% | ID |
| **ID 总计** | **43/90 = 47.8%** |  |
| Nestle | 4/30 = 13.3% | OOD |
| Slam | 1/30 = 3.3% | OOD |
| **OOD 总计** | **5/60 = 8.3%** |  |

### 4.2 SFT 数据从哪里来

RoboBenchMart 提供三个训练物体的 H5 expert demonstrations：

```text
pick_to_basket_fanta_248traj_4workers
pick_to_basket_nivea_248traj_4workers
pick_to_basket_stars_248traj_4workers
```

每个物体约 248 条轨迹，总计 744 episodes、307,319 frames、10 FPS。每一帧包含
三个相机、qpos、专家 action 和对应语言任务。数据完全来自 ID 专家示范，不包含
Nestle/Slam。

### 4.3 H5 到 LeRobot 的转换

OpenPI 的训练入口不能直接读取 RBM H5，因此实现：

```text
RoboBenchMart H5
  -> trajectory/episode 校验
  -> 图像、qpos、action、task 提取
  -> LeRobot v2.1 Parquet + metadata
  -> norm_stats
  -> OpenPI data loader
```

转换后数据关键字段：

| 字段 | 规格 |
|---|---|
| `image` | 256×256×3 RGB |
| `wrist_image` | 128×128×3 RGB |
| `extra_image` | 256×256×3 RGB |
| `state` | float32[15] |
| `actions` | float32[13] |
| `episode_index/frame_index/task_index` | episode 与任务索引 |

转换脚本为 `RoboBenchMart/scripts/convert_pick_to_basket_h5_to_lerobot.py`。相比直接
把数组拼起来，LeRobot 格式使后续 SFT、rollout replay、advantage sidecar 和多数据集
混合能够复用统一的数据接口。

### 4.4 对 OpenPI/π0.5 的适配

1. 新增 `robobenchmart_policy.py`，将 RBM 三视角映射为 π0.5 的
   `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`。
2. 将 15 维 state 和 13 维 action pad 到模型内部 32 维空间，输出时裁剪回 13 维。
3. 新增 `LeRobotRBMDataConfig`，负责字段 repack、prompt、normalization、图像 resize、
   tokenization 和 action transform。
4. 新增 `pi0_eval_rbm`、`pi05_eval_rbm` 和 `pi05_sft_rbm_pick_to_basket` 配置。
5. 对齐 RoboBenchMart 官方示范的 Fetch 初始 qpos；曾出现全 0 成功率，最终定位为
   环境初始姿态偏离训练分布，而不是 checkpoint 或 policy server 故障。

### 4.5 SFT 算法与参数

π0.5 是 flow-based VLA。训练不是分类 action token，而是在随机时间 `t` 和噪声动作
`x_t` 上回归把噪声输运到专家 action 的向量场，即 flow matching loss。模型同时用
视觉、语言和 state 表征作为条件。

可以把 flow matching 想成“把一团随机噪声逐步雕刻成机械臂动作”。训练时先把专家
动作和随机噪声混合，模型学习在当前噪声程度下应该往哪个方向修正；推理时从纯噪声
出发，重复修正 10 次，最终得到 50 步动作。这里的 `t` 是“当前噪声程度”的时间刻度，
`x_t` 是该时刻的带噪动作，不是机器人环境的第 t 帧。

表中的 learning rate 控制每次参数更新跨多大一步；batch size 表示一次用多少训练
样本估计更新方向；gradient clip 是防止某次异常梯度把参数推得过远的保险丝。

| 参数 | 值 |
|---|---:|
| 初始化 | 官方 RoboBenchMart π0.5 checkpoint |
| 训练框架 | OpenPI JAX |
| steps | 20,000，最终编号 19,999 |
| batch size | 32 |
| optimizer | AdamW |
| peak/decay LR | 1e-5 / 1e-5 |
| warmup | 1,000 steps |
| gradient clip | 1.0 |
| 保存间隔 | 5,000 steps |
| 最终 samples seen | 约 640k |
| 近似 epoch | 2.08 |

训练 loss 从 step 5,050 的 0.002101 降到 step 19,950 的 0.000988，后 20 个日志点
均值约 0.000970；grad norm 约 0.02～0.03，没有数值发散。

### 4.6 SFT 结果

| 任务 | 官方 π0.5 | SFT 19,999 | 变化 |
|---|---:|---:|---:|
| Fanta ID | 13/30 | 22/30 | +9 |
| Nivea ID | 17/30 | 21/30 | +4 |
| Stars ID | 13/30 | 24/30 | +11 |
| **ID 总计** | **43/90** | **67/90** | **+24** |
| Nestle OOD | 4/30 | 2/30 | -2 |
| Slam OOD | 1/30 | 0/30 | -1 |
| **OOD 总计** | **5/60** | **2/60** | **-3** |

这一阶段证明 SFT 有效学会了三个训练物体，但也给出了后训练问题：如何在不把
Nestle/Slam 放入训练数据的约束下，保持 ID 能力并恢复泛化。

## 5. 第二阶段：将 JAX SFT 接入 RLinf/PyTorch

这一阶段不研究新算法，而是“换发动机但保持车的行为不变”。原始 SFT 在 JAX/OpenPI
中训练，RLinf 的 PPO 主路径使用 PyTorch/FSDP。权重格式、图像预处理、随机噪声、
动作裁剪或机器人初始姿态中任何一项没对齐，都可能让同一模型表现不同。因此必须先
证明转换后的 PyTorch 模型仍能完成任务，才能把后续变化归因于 PPO。

RLinf 的 embodied PPO 主要采用 PyTorch actor，因此将 JAX SFT checkpoint 转换为
PyTorch `full_weights.pt`，并建立 actor、rollout、env 三组 Ray workers：

```text
EnvGroup(CPU simulator)
  -> observation/prompt
RolloutGroup(OpenPI policy)
  -> action chunk + old_logprob + value + flow forward inputs
EnvGroup
  -> reward/done/next observation
ActorGroup(FSDP)
  -> GAE + PPO/value/SFT/KL backward
  -> checkpoint + weight sync
```

三个 worker 组构成训练流水线：`EnvGroup` 运行仿真环境，`RolloutGroup` 根据观测生成
动作，`ActorGroup` 使用轨迹数据更新参数。Ray 负责调度进程，FSDP 负责
管理大模型训练的参数和显存。它们是工程手段，不是新的学习算法。

matched PyTorch SFT BASE 为：

| Fanta | Nivea | Stars | ID 总计 | Nestle | Slam | Proxy |
|---:|---:|---:|---:|---:|---:|---:|
| 15/30 | 23/30 | 21/30 | 59/90 | 1/30 | 0/30 | 2/30 |

它低于 JAX SFT 的 67/90，但 fixed-noise/action parity 与协议排查后，没有证据表明
权重转换发生大规模损坏。π0.5 flow policy 的高斯初始噪声、JAX/PyTorch 数值路径与
rollout 实现会影响有限样本成功率，因此后训练统一以 59/90 PyTorch BASE 为公平
对照，不混用 67/90 作为 PyTorch checkpoint 的直接基线。

## 6. 第三阶段：True SFT-reference KL-PPO

### 6.0 KL-PPO 到底是什么

PPO 的完整名字是 Proximal Policy Optimization。它根据模型自己的尝试更新策略：比
预期好的动作以后更常做，比预期差的动作以后少做。PPO clipping 只限制“当前模型”
和“刚刚采数据的旧模型”之间一次不要变化太大；连续训练很多轮后，模型仍可能一点点
远离最初可靠的 SFT 模型。

本项目的 **KL-PPO** 因此增加了长期正则约束：冻结一份 SFT 模型作为 reference，
每次更新都惩罚当前模型与它差得太远。KL 可以先理解为“两位驾驶员对同一动作的认可
程度差异”。`True SFT-reference` 强调比较对象真的是最初 SFT 权重，不是上一轮
rollout 的旧策略，也不是在错误时机复制的快照。

一条 PPO 数据变成更新的过程是：

```text
当前策略在仿真中做动作
-> 记录画面、动作、旧策略评分、critic 预测和 reward
-> 从后往前计算 return 与 advantage
-> 当前策略重新评价相同动作
-> PPO 根据 advantage 提高或降低动作概率
-> KL 把新策略拉回 SFT 附近
-> value loss 训练 critic 预测未来得分
```

| 变量名 | 它回答的问题 |
|---|---|
| `old_logprob` | 采数据时，旧策略有多认可这个动作？ |
| `current_logprob` | 更新时，当前策略现在有多认可同一动作？ |
| `ref_logprob` | 冻结的 SFT reference 对同一动作给出多高密度？ |
| `value` | critic 事前预计从这里还能拿多少分？ |
| `return` | 这次尝试后来实际累计拿了多少分？ |
| `advantage` | 实际结果比 critic 的预期好还是差？ |
| `ratio` | 当前策略相对采样时把动作概率改了多少？ |
| `kl_beta` | KL 正则强度；越大越不允许偏离 SFT |

假设 critic 预计某状态只能得 1 分，实际得到 4 分，advantage 就是明显的正信号，actor
会更愿意重复相关动作。如果 critic 的预测本身乱跳，它会把普通动作误判为惊喜，actor
便沿着噪声学习。本项目的 value head 是新加入的，开始时几乎不会预测，所以先做
critic-only warmup；成功轨迹少于门槛时也暂不更新 actor。

### 6.1 为什么不直接跑普通 PPO

PPO clip 只限制 `current policy` 相对本轮采样的 `old policy`，多轮更新仍可能累计
偏离 SFT。机器人 SFT 已经包含识别、接近、抓取和放置能力，RL 的目标应是小幅纠错，
而不是重新学习。因此我增加了一份冻结的 SFT reference：

```text
ratio = exp(logπ_current(a|s) - logπ_old(a|s))
L_clip = -min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)

L_total = L_clip
        + c_v * L_value
        + β * KL(π_current || π_SFT_reference)
        + λ_sft * L_flow_matching(SFT replay)
```

这里 PPO ratio 控制单轮变化，reference KL 控制相对初始 SFT 的长期漂移，SFT
rehearsal 则持续提供专家行为锚点。

公式中的 `s` 是状态与多模态观测，`a` 是 rollout 中实际执行的动作，`π(a|s)` 表示
策略在状态 `s` 下对动作 `a` 的密度；`A` 是 advantage，`ε=0.1` 是本实验的裁剪范围。
当 `A>0` 时，目标倾向于提高该动作概率；当 `A<0` 时，目标倾向于降低概率。`clip`
把 ratio 限制在 `[0.9, 1.1]` 附近，使单次更新不会因少量样本产生过大变化。

本项目使用 GAE 估计 advantage。其递推形式为：

```text
δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
R_t = A_t + V(s_t)
```

`δ_t` 表示一步之后的实际结果与 critic 预测之间的误差；`γ=0.99` 决定未来奖励的
权重，`λ=0.95` 在方差与偏差之间折中；`done_t` 防止一个 episode 的回报错误传播到
下一个 episode。`R_t` 是 value head 的训练目标。因此 done、truncation 和 bootstrap
实现错误会同时污染 advantage 与 critic，而不只是影响日志统计。

### 6.2 flow policy 的 logprob 难点

普通分类模型直接给每个选项一个概率；π0.5 输出的是经多步去噪得到的连续动作，没有
现成的“这个动作概率是多少”。实现需要在相同 observation、动作、flow 时间和噪声
条件下重放计算，得到可比较的近似 logprob。若 current、old、reference 使用的噪声或
去噪步骤不同，差异可能来自随机过程而非策略参数，KL 和 PPO ratio 就失去意义。

π0.5 输出连续 action flow，不是语言模型 token。为了对 rollout action 重算 current
与 reference logprob，必须复用同一组：

- observation、prompt 与 state；
- 已执行的 action chunk；
- flow noise；
- denoising timestep/index；
- action mask 和归一化参数。

否则比较的不是同一个条件概率，KL 和 PPO ratio 都没有意义。我扩展 rollout batch
保存 forward inputs，并在 actor 中用相同输入分别执行 current/reference forward。

### 6.3 Reference KL 实现

实现上先保存初始 SFT 权重的冻结快照。训练每个 batch 时，当前模型和 reference 都对
同一批输入及同一批动作打分，只让当前模型参与反向传播。损失的直观结构为：

```text
总损失 = PPO 成败学习 + critic 预测误差 + 偏离 SFT 的罚款 + 少量专家题复习
```

`SFT rehearsal` 相当于 PPO 训练时穿插专家示范。KL 约束“评分分布不要偏太远”，
rehearsal 直接提醒模型“正确专家动作怎样做”，两者作用互补。

- actor 初始化时复制冻结 SFT reference，不加入 optimizer；
- 首次 actor update 前检查 reference/current 的零漂移，`ref_kl_abs > 0.02` 直接停止；
- 采用非负低方差 KL 近似，避免有符号 Monte Carlo KL 抵消；
- KL target 为 0.01，初始 `β=0.1`；超过目标两倍则乘 1.5，低于一半则除 1.5，
  并限制在 `[1e-4, 10]`；
- checkpoint 保存 `actor_update_count`。恢复 critic-only checkpoint 后重新冻结 reference；
  若已做过 actor 更新，则保持原始 SFT reference，避免 reference 随策略漂移。

### 6.4 Reward 的设计与迭代

只在最终成功时给分最诚实，但机器人可能连续走 600 步才得到一次反馈，学习非常困难。
因此加入少量中间分：首次抬起、首次放置、朝篮子靠近。不过中间分必须是一次性事件或
距离的真实增量，否则模型可能一直举着商品或在篮子旁来回移动刷分。最终成功奖励必须
占主导，这就是 success-dominant shaped reward。

最初 per-step shaped reward 会重复奖励“已抬起”“已靠近篮子”等状态，策略可能通过
停留在中间状态刷分。最终改为 event + potential reward：

| 事件/信号 | 奖励 |
|---|---:|
| 首次任务成功 | +5.00 |
| 首次放入篮子 | +1.00 |
| 首次抬起目标 | +0.30 |
| 向篮子靠近的 potential 增量 | `0.50 × progress` |
| 放入且机械臂稳定 | +0.30 |
| 首次移动非目标物体 | -0.20 |
| 单步 clamp | [-0.2, 6.0] |

同时修复：success 只奖励一次；reset 时记录真实初始 potential，避免第一步虚假进度；
episode done 后屏蔽 chunk 中剩余 action 的 reward；terminal/truncated 不做错误 bootstrap。

### 6.5 PPO 训练参数

| 参数 | 最终保守设置 |
|---|---:|
| rollout | 每任务 4 条，共 12 条/runner epoch |
| episode horizon | 600 action steps |
| action chunk / denoise | 50 / 10 |
| actor micro/global batch | 1 / 12 |
| update epochs | 1 |
| actor LR | 5e-8 |
| value LR | 1e-5 |
| critic warmup | 12 optimizer steps，即完整第一轮 |
| PPO clip | ±0.1 |
| γ / GAE λ | 0.99 / 0.95 |
| SFT rehearsal weight | 1.0 |
| minimum successes | 2/12，否则 critic-only |
| optimizer/FSDP | AdamW，clip 1.0，FSDP no-shard |

### 6.6 关键大改一：episode horizon、done 与 bootstrap

**现象**：40-step runner epoch 配合 50-step action chunk，几乎没有完整有效决策；done
后的 chunk action 仍可能重复写 reward，value bootstrap 也与配置不一致。

**判断**：critic 学不到并不一定是模型容量问题，首先是时序数据语义错误。

**修改**：训练 horizon 改为 600；第一次 done 后屏蔽 chunk 内后续 reward/done；明确
truncation；`bootstrap_type:none` 不再向终止 reward 写 value。

### 6.7 关键大改二：critic 与 actor 的更新门控

**现象**：早期 rollout 0/12 success，critic explained variance 从 -0.413 降至
-1.568。如果直接更新 actor，advantage 主要来自不可靠 shaped reward。

**判断**：随机初始化 value head 需要先适配 π0.5 特征；没有成功样本时，PPO 没有
最终任务信号。

**修改**：第一整轮仅训练 value head；少于 2 条成功轨迹时强制 actor update 为 0；
explained variance 改为用完整 rollout 的更新前 values/returns 计算，而不是在 micro
batch size 1 上得到 NaN。

### 6.8 关键大改三：训练/推理去噪策略不一致

**现象**：修复评测后，训练 rollout 仍只有 0/12、1/12，而同模型评测是 59/90。

**定位**：`joint_logprob=True` 时，train policy 在 10 个 denoising steps 全部注入
`flow_noise`；eval policy 10 步全部走 `flow_ode`。RL 实际优化的不是部署策略分布。

**修改**：设为 `joint_logprob=False`，每个 action chunk 只随机选择一个 denoising
step 注入探索噪声，其余九步走 ODE。修改后训练路径基线达到 4/12，首次真实 actor
轮达到 5/12，证明此前的主要问题确实是 exploration policy mismatch。

### 6.9 关键大改四：OOM 与 checkpoint 恢复

**OOM 现象**：step 3 不是保存失败，而是在首次 PPO+SFT backward 尝试额外分配
7.56 GiB 时 OOM。原实现同时保留 PPO 和 SFT 两张计算图，且 SFT loader 错误继承
OpenPI preset batch 256。

**修改**：强制 SFT batch 等于 actor micro batch；先 backward PPO 并释放 activation，
再建立 SFT graph；rollout 模型训练期 offload；增加可扩展 CUDA allocator 配置。

**恢复现象**：从 critic-only checkpoint 恢复时 reference KL 突然为 1.615，而理论上
应接近 0。

**定位与修改**：RLinf 先复制 reference，后加载 resume checkpoint，导致 reference
与 current 来自两个时刻。为 checkpoint 增加 optimizer/warmup/actor-update 元数据，
按更新历史决定是否刷新 reference，并重建正确的 optimizer/scheduler。

### 6.10 KL-PPO 最终结果

首次真实 actor update 的在线指标：

- 5/12 successful trajectories；
- `policy_loss=-0.194`；
- `ref_kl_abs=0.0012`，`approx_kl=7.68e-4`；
- `clip_fraction=0.0069`；
- `actor_update_count=12`，checkpoint 正常保存；
- 完整 rollout critic EV 为 -0.0054，仍接近常数预测。

固定 90-episode 评测：

| checkpoint | Fanta | Nivea | Stars | 总计 |
|---|---:|---:|---:|---:|
| PyTorch SFT BASE | 15/30 | 23/30 | 21/30 | 59/90 |
| KL-PPO step 2 | 18/30 | 24/30 | 18/30 | 60/90 |
| 变化 | +3 | +1 | -3 | +1 |

总量提高 1 条，但 Stars 下降 3 条，无法证明稳定提升或泛化改善。因此没有把它包装成
成功结果，也没有直接扩到 6 epochs。这个结果表明工程链路已经有效，但 12 条 rollout
的稀疏成功信号会在任务间重新分配能力。

## 7. Proxy-mix PPO

Proxy-mix 的想法是：不能拿最终考试用的 Nestle/Slam 训练，但可以加入货架中其他商品，
让模型见到更多外观。`mix` 指三个原训练任务与 proxy 任务一起产生 PPO 数据。它不是
新的 PPO 公式，只是改变训练任务的混合方式。

### 7.1 设计动机

因为禁止使用 Nestle/Slam 训练，希望用货架内其他随机商品作为 proxy，给策略更多
物体外观与抓取状态，同时保留 Fanta/Nivea/Stars。训练环境采用四个 pipeline stages：
三个 ID 任务加一个 ProxyRandom 环境，并共享 PPO、reference KL 和 SFT rehearsal。

### 7.2 为什么没有继续

修复路径后 Proxy BASE 只有 2/30 成功，在线数据几乎全部为失败。这样的 proxy-mix：

- 很难形成正负 advantage 对照；
- shaped reward 可能奖励接近任意物体，而不是语言指定物体；
- proxy 商品分布与 Nestle/Slam OOD 的关系没有明确定义；
- 增加任务并发还会稀释三个 ID 任务的有效成功样本。

因此 Proxy-mix 没有得到可靠收益。它的重要结论是“加入更多任务”不等于“加入有效的
泛化监督”；代理分布必须和目标 OOD 因素建立可解释关系。

## 8. 第四阶段：RECAP-lite

### 8.0 RECAP-lite 的概念边界

完整 RECAP 从专家示范、自主交互和专家纠错中学习 advantage-conditioned policy。
本项目没有同等规模的 value model 与人工纠错数据，因此实现的是简化版本，称为
**RECAP-lite**。其中 CFG（classifier-free guidance）用于在推理阶段增强与正
advantage 条件相关的动作方向。

具体做法是给训练经验增加离散条件：成功轨迹使用 `Advantage: positive`，失败轨迹
使用 `Advantage: negative`。模型既学习原始任务条件下的动作，也学习附加成败条件后
的动作。部署时并不依赖未来成功标签，而是固定选择 positive 条件，并利用 conditional
与 unconditional 预测之差修正动作生成过程。

它和 PPO 的主要区别在于：PPO 通过 policy gradient 直接改变动作概率；RECAP-lite
仍使用 flow-matching 监督目标，只是将训练数据按 advantage 条件重新组织。这个目标
通常比在线 PPO 稳定，但失败轨迹仅提供“该行为未成功”的信息，并未给出失败状态下的
正确动作；缺少 expert correction 时，其反事实纠错能力有限。

### 8.1 为什么从 PPO 转向 RECAP

PPO 的瓶颈是：在线 success 少、critic 不可靠、advantage 方差大。π*0.6 的 RECAP
提供了另一种思路：不直接用 policy gradient，而是把 demonstrations、autonomous
rollouts 和 corrections 统一为 advantage-conditioned behavior learning。

本项目没有完整 RECAP 的大规模 value model 和人工 correction 条件，因此实现
RECAP-lite：用 episode success 作为离散 advantage 标签，再做 CFG flow-matching SFT。

### 8.2 核心算法

原任务 prompt 后增加条件：

```text
Advantage: positive
Advantage: negative
```

训练时以一定概率 dropout advantage 条件，联合学习 conditional/unconditional flow。
推理时使用 positive guidance：

```text
v_guided = v_uncond + scale * (v_positive - v_uncond)
```

式中的 `v` 不是 critic 输出的状态价值，而是 flow 模型预测的动作向量场：

- `v_uncond`：仅使用原始任务 prompt 时的动作修正方向；
- `v_positive`：加入 positive advantage 条件后的修正方向；
- `scale`：guidance 强度，决定条件差分被放大多少。

当 `scale = 1` 时，结果等于 positive 条件预测；大于 1 时会进一步放大正条件方向，
但过大也可能导致动作分布偏离训练数据。训练时以概率 `p_drop` 丢弃 advantage 条件：

```text
condition = original_prompt,                    probability = p_drop
condition = original_prompt + advantage_label, probability = 1 - p_drop
```

这个 dropout 使同一个模型同时学到 conditional 与 unconditional 分支。若从不丢弃
条件，就无法稳定估计 `v_uncond`；若总是丢弃，训练则退化为普通 SFT。

目标是让模型在部署时偏向训练数据中与成功相关的动作模式。SFT expert demo 直接标为
positive；autonomous rollout 根据整条 episode 最终 success 标为 positive/negative。

### 8.3 数据闭环

优势标签采用 sidecar 保存。sidecar 是一张按 `(episode_index, frame_index)` 与原始
LeRobot 数据对齐的轻量表；图像和动作仍保存在原 Parquet 中，训练读取时再拼接标签。
这种设计避免为了增加一个布尔标签而复制数十万帧图像数据，也便于重新生成不同标签。

实现独立采集脚本，从 PyTorch SFT BASE 在三个 ID 任务上各采 30 条，共 90 episodes：

| 任务 | 成功 | 失败 |
|---|---:|---:|
| Fanta | 15 | 15 |
| Nivea | 23 | 7 |
| Stars | 21 | 9 |

每条 episode 保存三视角、state、action、task 和 success。实现 advantage sidecar，只读
metadata 列并生成 `(episode_index, frame_index, advantage, reason, task_index)`；
success 标签按整条 episode 广播，兼容 episode 跨 parquet 文件的情况。

### 8.4 工程适配

- CFG OpenPI wrapper 增加 RBM 三视角输入；
- value-model dataset/checkpoint/advantage process 增加 `rbm/robobenchmart` transform；
- 修复多 pipeline stage 同时写 `rank_0/id_0` 导致覆盖的问题；
- writer 使用绝对 LeRobot root，自动发现三个 task shards；
- 将第三视角列从采集期别名统一迁移为 `extra_image`，使用临时文件原子替换；
- preflight 检查 schema、episode 碎片、每任务数量以及正负标签覆盖；
- 修复 Ray actor 被 kill 后 collective 把随机 int 当 object type 的次生异常；
- checkpoint 包含 full weights、optimizer、scheduler 和 trainer state，可用于恢复和评测。

### 8.5 采样策略为什么经历两次大改

**第一版：all-positive**

先把 744 条专家 demo 全标 positive，验证 CFG wrapper、forward/backward、保存和推理。
它只验证通路和遗忘风险，没有失败信息，理论上不能显著提升泛化。

**第二版：按帧混合 SFT 与 rollout**

最初按数据集/帧权重采样，例如 70% SFT + 30% rollout。但失败 episode 通常跑满
600 帧，成功 episode可能提前结束，按帧采样会让长失败轨迹占据更多概率；随机短训
还可能完全抽不到某个任务的 negative。

**第三版：episode/quota-balanced**

改为先确定数据源、任务和标签配额，再选 episode，最后在 episode 内选帧。正式短训
使用 5 steps × global batch 12 = 60 个样本：54 个 SFT positive；Fanta、Nivea、
Stars 各 2 个 rollout slots，并强制每个任务各 1 个 positive、1 个 negative。这样
保证一次短训真实覆盖全部任务和两类标签，而不是依赖随机碰到。

问题的本质是轨迹长度偏差：失败 episode 往往运行到 600-step 超时，包含的帧数多于
提前成功的 episode。若直接按帧均匀采样，负样本会因轨迹更长而获得更高权重。先按
episode 抽样，再规定 task/label 配额，采样分布才由实验设计而不是轨迹长度决定。

### 8.6 RECAP-lite 训练参数与结果

工程验证阶段使用 micro batch 1、global batch 4、LR 2e-7；正式保守 quota run 使用
global batch 12、低学习率与 5-step 周期，并逐 checkpoint 保存。旧 mixed step 5 的
90-episode 结果为：

| Fanta | Nivea | Stars | 总计 |
|---:|---:|---:|---:|
| 10/30 | 23/30 | 19/30 | 52/90 |

低于 PyTorch SFT BASE 59/90。修复为 quota-balanced 后，5-step checkpoint 在每任务
前 10 个 fixed seeds 上为 19/30，与对应 BASE 小样本总数相同，但参数审计显示 step 1
到 step 5 的 relative L2 change 仅 `5.36e-9`，因此 10-episode 波动不能解释为可靠
能力变化。

### 8.7 为什么 RECAP-lite 没有达到目标

1. **没有 OOD 视觉信号**：所有训练图像仍来自三个 ID 商品，无法直接教模型识别
   Nestle/Slam。
2. **negative condition 的含义有限**：模型学习的是“在 negative 条件下复现失败
   action”，并不是显式降低失败 action 的概率；只有 positive guidance 是否足以产生
   反事实纠错，在该小数据设置中没有保证。
3. **缺少 correction**：完整 RECAP 的价值不仅是成败标签，还包括人工 intervention
   或成功纠正轨迹。本项目只有自主失败，没有告诉模型失败状态下正确动作是什么。
4. **更新过小与评测噪声**：保守学习率保护了 SFT，但几步训练的参数变化远小于
   flow sampling 带来的成功率波动；盲目增加步数又会提高遗忘风险。

所以结论不是“RECAP 算法无效”，而是当前 RECAP-lite 信号不足以实现目标。

## 9. 贯穿项目的调试与决策方法

机器人训练链路较长，成功率变化不一定来自算法本身。环境 seed、机器人初始姿态、
图像字段映射、动作归一化、flow 噪声、episode 截断、checkpoint 加载与显存峰值都可能
改变结果。因此本项目使用 BASE、zero-step、critic-only、first-actor-update 等分层
对照，并尽量固定评测 seed，使每次实验只回答一个明确问题。

### 9.1 技术决策时间线

| 阶段 | 当时看到的现象 | 初始假设 | 验证与结论 | 因此采取的改动 |
|---|---|---|---|---|
| 官方 BASE | 有基础抓取能力但 ID 仅 43/90 | π0.5 尚未适配具体机器人/商品 | 专家数据充足且任务定义明确 | 转 LeRobot，在官方 checkpoint 上做 SFT |
| SFT 20k | ID 达到 67/90，OOD 降到 2/60 | SFT 专门化破坏泛化 | loss 降低只对应 ID action imitation | 尝试受 SFT reference 约束的后训练 |
| PyTorch BASE | 一度只有 38/90 | JAX→PyTorch 权重转换损坏 | seeds、pose、backend 均不一致 | 建立 matched fixed-seed 评测与 action parity |
| 初版 PPO | rollout 0/12，critic EV 很负 | critic 或 reward 不好 | horizon/done/bootstrap 先存在语义错误 | 修复 600-step 时序语义与 success-dominant reward |
| actor 首更 | step 3 OOM、checkpoint 未生成 | 保存逻辑故障 | 实际在保存前的 PPO+SFT backward OOM | 拆分两次 backward、限制 SFT batch、offload rollout |
| 恢复训练 | reference KL 起点为 1.615 | KL 公式或 beta 错 | reference 在加载 checkpoint 前被冻结 | 保存更新元数据并修复 reference 生命周期 |
| rollout/eval | 训练 0/12、1/12，评测 59/90 | 在线 seed 特别困难 | train 10 步全加噪，eval 10 步全 ODE | 单噪声 denoising，首次真实 actor 轮达到 5/12 |
| KL-PPO 结果 | 总计 +1，但 Stars -3 | 可能长训后继续提高 | critic EV≈0，12 条数据不足且任务间迁移 | 不用“先跌后涨”解释，停止直接扩到 6 epochs |
| Proxy-mix | Proxy BASE 2/30 | 更多商品能提供泛化 | 几乎全失败，没有有效正样本对照 | 停止把无定义的 proxy 当 OOD 替代信号 |
| RECAP all-positive | 通路可训练，但无明确收益 | positive CFG 可保护 SFT | 全 positive 没有失败/纠错信息 | 采集三个 ID 的 success/failure rollouts |
| RECAP mixed | 旧 step 5 为 52/90 | 负样本过多或任务采样失衡 | 长失败 episode 按帧占据更高概率 | episode-balanced + deterministic quota sampling |
| quota RECAP | 小样本结果接近 BASE | 修改可能开始有效 | 权重 relative L2 仅 5.36e-9 | 判定信号/更新不足，不再把采样波动当提升 |

### 9.2 不把现象直接归因于模型

- 成功率低：先核对 seeds、pose、backend、action horizon 和 checkpoint 加载；
- checkpoint 没出现：先看是否训练在保存前 OOM，而不是先改保存代码；
- critic EV 为 NaN：检查 micro-batch 统计定义，再判断 critic；
- KL 突然很大：检查 reference 生命周期，而不是直接降低 KL beta；
- rollout 低于 eval：比较 train/eval policy 的 denoising 过程。

### 9.3 使用对照和断言控制算力成本

- BASE、zero-step、critic-only、first-actor-update 分层验证；
- fixed seeds 和逐任务指标，不只看总成功率；
- actor update minimum successes；
- first-update reference zero-drift assertion；
- RECAP schema/label/quota preflight；
- 每次大训练前进行 transform、单步 backward、保存和恢复 smoke；
- 预先定义停止条件，不用“可能先下降再上升”解释持续退化。

### 9.4 失败实验的价值

该项目最终没有证明 PPO/RECAP 提高 OOD，但排除了多种错误解释：

- SFT 明确提升 ID，但损伤 OOD，说明训练目标与泛化目标不一致；
- PPO 工程链路通过后只净增 1/90，并在任务间交换能力；
- Proxy 几乎全失败，不能作为有效 advantage 数据源；
- RECAP-lite 数据闭环完整，但 success label 不能代替 correction 和 OOD coverage；
- 继续堆 epoch 没有证据，及时停止比得到一个偶然最好 checkpoint 更可信。

## 10. 项目中体现的能力

### 机器人学习与算法

- 理解 VLA、flow matching、action chunking、continuous-action logprob；
- 完成 behavior cloning/SFT、PPO、GAE、value learning、KL regularization；
- 理解 advantage-conditioned policy、classifier-free guidance 和 offline replay；
- 能区分 reward 上升、训练 loss 下降、ID success 和 OOD generalization。

### 数据工程

- H5 trajectory 到 LeRobot Parquet/metadata 的转换与校验；
- 多相机、state/action、task prompt、normalization schema 设计；
- episode/shard 级成功标签广播、原子 schema migration、混合数据配额采样；
- simulator autonomous rollout 的成功/失败数据闭环。

### 分布式训练与系统工程

- Ray actor/env/rollout worker 编排和 pipeline-stage 路由；
- PyTorch FSDP、模型 offload、梯度累积、optimizer/scheduler checkpoint；
- JAX checkpoint 训练与 PyTorch 权重转换/评测；
- CUDA OOM 定位、双计算图生命周期、micro/global batch 对齐；
- TensorBoard、日志、参数审计、恢复训练语义与自动 preflight。

### 研究方法

- 建立公平、可复现的 fixed-seed 评测协议；
- 用 parity test 和消融定位框架差异；
- 将假设、证据、改动和结果串成可证伪的决策链；
- 如实报告负结果，区分“代码可运行”和“算法有效”。

## 11. 如果被追问：下一步会怎么做

我不会直接继续 6-epoch PPO，也不会把 OOD 数据偷偷加入训练。下一步先对 Nestle/Slam
失败轨迹做阶段化标注：目标定位、接近、夹取、抬升、运输、放置、抓错物体。若主要是
控制失败，则用 ID 商品生成位置、朝向、篮子、相机和干扰物随机化的**动作一致**专家
数据；若主要是目标识别失败，则重点保护 π0.5 的预训练视觉语言表示，限制 action
expert 更新并增加通用 replay，而不是只做图像增强。这个决策继续遵守“先定位根因，
再投入训练算力”的原则。

更具体地说：若失败分析表明模型没有识别正确商品，应补充视觉—语言覆盖；若模型已
识别目标但夹取或放置失败，则应补充控制阶段的纠错轨迹。两类问题需要不同的数据，
不能仅通过增加 PPO steps 或提高学习率解决。

## 12. 关键代码与文档入口

### OpenPI/SFT

- `RoboBenchMart/scripts/convert_pick_to_basket_h5_to_lerobot.py`
- `openpi/src/openpi/policies/robobenchmart_policy.py`
- `openpi/src/openpi/training/config.py` 中的 `LeRobotRBMDataConfig`
- `RoboBenchMart/RBM_OpenPI_SFT_20k_Training_Report.md`

### RLinf/PPO

- `RLinf/examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml`
- `RLinf/rlinf/workers/actor/fsdp_actor_worker.py`
- `RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py`
- `RLinf/rlinf/envs/robobenchmart/robobenchmart_env.py`
- `RLinf/rlinf/utils/embodied_training_safety.py`
- `RLinf/scripts_local/06_rbm_ppo_smoke.sh`
- `RLinf/scripts_local/08_rbm_eval_matched.sh`

### RECAP-lite

- `RLinf/examples/recap/process/make_recap_lite_advantages.py`
- `RLinf/rlinf/data/datasets/recap/cfg_model.py`
- `RLinf/rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py`
- `RLinf/scripts_local/09_rbm_collect_indomain_rollouts.sh`
- `RLinf/scripts_local/10_rbm_recap_prepare_advantages.sh`
- `RLinf/scripts_local/11_rbm_recap_cfg_mixed.sh`

### 测试与复盘

- `RLinf/tests/unit_tests/test_rbm_post_training_safety.py`
- `RLinf/tests/unit_tests/test_recap_lite_advantages.py`
- `RLinf/docs/rbm_post_training_fixes_and_runbook.md`
- `RLinf/docs/rbm_true_kl_ppo_and_reward_changes.md`
- `RLinf/docs/rbm_recap_lite_changes.md`

## 13. 参考论文与开源项目

1. Physical Intelligence, **π0.5: a Vision-Language-Action Model with Open-World
   Generalization**, 2025. <https://arxiv.org/abs/2504.16054>
2. Physical Intelligence, **openpi**：π0/π0-FAST/π0.5 官方开源模型、checkpoint、
   JAX/PyTorch 训练与推理实现。<https://github.com/Physical-Intelligence/openpi>
3. Physical Intelligence, **π*0.6: a VLA That Learns From Experience**，RECAP 将
   demonstrations、on-policy experience 和 expert corrections 统一为 advantage-
   conditioned policy learning。<https://arxiv.org/abs/2511.14759>
4. Schulman et al., **Proximal Policy Optimization Algorithms**，PPO clipped surrogate
   objective。<https://arxiv.org/abs/1707.06347>
5. Lipman et al., **Flow Matching for Generative Modeling**，π0 系列连续动作 flow
   matching 的基础方法。<https://arxiv.org/abs/2210.02747>
6. Ho and Salimans, **Classifier-Free Diffusion Guidance**，RECAP-lite 正/负条件与
   unconditional dropout/guidance 的方法基础。<https://arxiv.org/abs/2207.12598>
7. **RLinf: Reinforcement Learning Infrastructure for Embodied and Agentic AI**，本项目
   actor/env/rollout、Ray 调度、FSDP 与 embodied PPO 的底层框架。
   <https://github.com/RLinf/RLinf>
8. **RLinf: RL on π0 and π0.5 Models**，OpenPI flow-noise/flow-SDE 与 embodied RL
   官方适配说明。<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html>
9. Hugging Face, **LeRobot**，机器人数据集、采集、训练与部署的开源基础设施，本项目
   使用其 episode/Parquet metadata 格式连接 SFT 与 rollout replay。
   <https://github.com/huggingface/lerobot>

## 14. 对项目结果的准确表述

推荐表述：

> 我完成了 π0.5 在 RoboBenchMart 的 SFT 与两类后训练闭环。SFT 在 JAX 官方协议下
> 将 ID 成功率从 47.8% 提升到 74.4%；KL-PPO 与 RECAP-lite 的系统链路均完成真实
> rollout、反向、保存、恢复和 fixed-seed 评测，但没有获得可靠 OOD 提升。通过这些
> 实验，我定位了训练/推理去噪不一致、价值估计与数据覆盖等根因，并按预设门控停止
> 无效扩训。

不推荐表述：

> PPO 将模型成功率提升到了 60/90，所以 PPO 有效。

原因是 59/90 到 60/90 只有一条净变化，并且 Stars 下降三条；它只能作为弱正向信号
和工程验证，不能作为统计上可靠的算法收益。
