# RoboBenchMart PickToBasket OpenPI SFT 20k 训练完整说明报告

生成时间：2026-07-01
项目：RoboBenchMart / OpenPI / RLinf
任务：RoboBenchMart `pick_to_basket`
模型：OpenPI `pi0.5` official checkpoint 上继续 SFT
最终 checkpoint：

```text
/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt/19999
```

本文档面向初学者，目标是把这次训练从“数据是什么、模型吃什么、输出什么、为什么要适配、怎么训练、指标怎么算、结果说明什么”完整讲清楚。

---

## 1. 一句话总结

本次实验做的是：

> 用 RoboBenchMart 的 PickToBasket 示范数据，对官方 OpenPI `pi0.5` 机器人策略模型继续做监督微调，也就是 SFT，让模型在 Fanta / Nivea / Stars 三个训练物体上更会把目标物体抓起并放进篮子。

最终结果：

| 任务 | 官方 pi0.5 | SFT 19999 | 变化 |
|---|---:|---:|---:|
| Fanta train | 13/30 = 43.3% | 22/30 = 73.3% | +30.0% |
| Nivea train | 17/30 = 56.7% | 21/30 = 70.0% | +13.3% |
| Stars train | 13/30 = 43.3% | 24/30 = 80.0% | +36.7% |
| Nestle OOD | 4/30 = 13.3% | 2/30 = 6.7% | -6.7% |
| Slam OOD | 1/30 = 3.3% | 0/30 = 0.0% | -3.3% |

核心结论：

- SFT 明显提升了训练分布内的 3 个 train item。
- SFT 没有提升 OOD item，反而略退化。
- 这说明监督微调学到了训练数据里的物体和动作模式，但没有自动获得更强的 unseen item 泛化能力。
- 后续如果要提升 Nestle / Slam OOD，需要 PPO、奖励设计、数据增强或加入 OOD 数据，而不是只靠当前三物体 SFT。

---

## 2. 本项目里几个名字是什么意思

### 2.1 RoboBenchMart 是什么

RoboBenchMart 是一个机器人任务 benchmark。它基于 ManiSkill / SAPIEN 仿真环境，提供类似便利店货架、商品、机器人抓取、移动、放置等任务。

本次用的任务是：

```text
pick_to_basket
```

意思是：

1. 机器人看到货架和商品。
2. 根据语言指令找到目标商品。
3. 抓起目标商品。
4. 把目标商品放进篮子。
5. 环境判断是否成功。

### 2.2 OpenPI 是什么

OpenPI 是一个视觉-语言-动作模型框架。它的输入通常包括：

- 多个相机图像
- 机器人状态
- 语言指令

输出是：

- 接下来一段时间要执行的机器人动作序列，也叫 action chunk。

本次使用的是 OpenPI 的 `pi0.5` 模型。它已经在官方数据上训练过，我们不是从零开始训练，而是在官方 checkpoint 上继续微调。

### 2.3 RLinf 是什么

RLinf 在这里主要提供运行环境和部分评测依赖。你使用的评测 Python 环境是：

```text
/root/autodl-tmp/projects/RLinf/.venv_openpi
```

OpenPI 自己也有 Python/uv 环境：

```text
/root/autodl-tmp/projects/openpi/.venv
```

训练一般在 OpenPI 项目里用 `uv run scripts/train.py ...`。
评测 RoboBenchMart 环境一般用 RLinf 的 `.venv_openpi/bin/python`。

---

## 3. 什么是 SFT

SFT 全称是 Supervised Fine-Tuning，中文通常叫监督微调。

在机器人策略模型里，可以理解为：

> 给模型看很多“专家示范”：某个图像、机器人状态、语言指令下，专家应该怎么动作。模型通过模仿这些动作来学习任务。

### 3.1 SFT 和 RL/PPO 的区别

SFT：

- 输入：专家数据。
- 学习目标：模仿专家动作。
- 优点：稳定、容易训练、能快速适应已有数据分布。
- 缺点：只会模仿数据里出现过的行为，对 OOD 泛化和长程纠错能力有限。

PPO / RL：

- 输入：环境交互。
- 学习目标：最大化 reward。
- 优点：可以针对成功率直接优化，也可以学会纠错。
- 缺点：训练更复杂、更慢、更容易不稳定。

本次训练是 SFT，不是 PPO。

---

## 4. 本次为什么选择在官方 pi0.5 checkpoint 上继续训练

有两种可能：

1. 从 `pi0.5 base` 训练。
2. 从 RoboBenchMart 官方 pi0.5 checkpoint 继续训练。

本次采用第二种：

```text
/root/autodl-tmp/projects/RoboBenchMart/models/pi05/params
```

原因：

- 官方 checkpoint 已经具备 RoboBenchMart 的基本能力。
- 从它继续 SFT 更容易保留已有视觉、语言、动作对齐能力。
- 和官方 baseline 对比也更直接，因为初始能力来自同一类模型。

风险：

- 如果 SFT 数据只覆盖 Fanta / Nivea / Stars，模型会更偏向 train item。
- 可能牺牲 Nestle / Slam 这种 OOD item。
- 本次结果确实出现了这种情况。

---

## 5. 数据是什么

### 5.1 原始数据

RoboBenchMart 原始数据是 H5 格式，里面包含很多 trajectory。

每条 trajectory 可以理解为一次专家示范：

```text
obs_0, action_0
obs_1, action_1
obs_2, action_2
...
obs_T, action_T
```

其中：

- `obs` 是当时环境观测。
- `action` 是专家下一步执行的动作。

本次 PickToBasket train item 包括：

```text
pick_to_basket_fanta_248traj_4workers
pick_to_basket_nivea_248traj_4workers
pick_to_basket_stars_248traj_4workers
```

每个 item 大约 248 条 trajectory，总计：

```text
744 episodes
307319 frames
```

### 5.2 训练数据里的输入

模型每一步看到：

| 数据 | 含义 |
|---|---|
| `observation/image` | 主相机图像 |
| `observation/extra_image` | 额外相机图像 |
| `observation/wrist_image` | 手腕相机图像 |
| `observation/state` | 机器人关节状态 qpos |
| `prompt` | 语言任务指令 |

### 5.3 训练数据里的输出

专家输出是 action：

```text
actions: (T, 13)
```

这里 13 维动作对应 RoboBenchMart 机器人控制所需的 action 维度。

OpenPI 模型内部 action dim 是 32，所以训练时会 pad 到 32，但真正送给环境执行时只取前 13 维。

---

## 6. 为什么需要数据转换

OpenPI 官方训练代码不直接吃 RoboBenchMart 原始 H5。它更习惯使用 LeRobot 格式数据。

所以需要转换：

```text
RoboBenchMart H5 -> LeRobot dataset
```

转换后的数据目录：

```text
/root/autodl-tmp/lerobot_data/rbm_dataset
```

转换后模型训练时通过：

```bash
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data
```

告诉 OpenPI：LeRobot 数据在哪里。

### 6.1 转换脚本位置

```text
/root/autodl-tmp/projects/RoboBenchMart/scripts/convert_pick_to_basket_h5_to_lerobot.py
```

### 6.2 转换做了什么

转换脚本主要做几件事：

1. 读取 RoboBenchMart H5 中的每条 trajectory。
2. 把图像、qpos、actions、prompt 抽出来。
3. 把数据写成 LeRobot v2.1 数据结构。
4. 保存 episode metadata。
5. 生成 OpenPI 可以加载的数据目录。

### 6.3 转换后的关键格式

| 字段 | 期望格式 |
|---|---|
| actions | `(T, 13)` |
| qpos/state | `(T, 15)` |
| image | RGB uint8 |
| prompt | 与目标物体对应的自然语言 |

训练时 OpenPI 再做以下处理：

- 图像 resize 到 224x224。
- state/action pad 到 32 维。
- prompt tokenization。
- action normalization。

---

## 7. 什么是适配 adapter

不同项目的数据字段名、相机名、动作维度不一样。OpenPI 原生不一定知道 RoboBenchMart 的输入输出长什么样。

所以需要适配层。

适配文件：

```text
/root/autodl-tmp/projects/openpi/src/openpi/policies/robobenchmart_policy.py
```

### 7.1 输入适配

RoboBenchMart eval script 准备 observation：

```python
observation = {
    "observation/image": left_base_camera_link,
    "observation/extra_image": right_base_camera_link,
    "observation/wrist_image": fetch_hand,
    "observation/state": qpos,
    "prompt": language_instruction,
}
```

OpenPI adapter 把它映射到模型需要的 image keys。

主要对应关系：

| RoboBenchMart key | OpenPI model key |
|---|---|
| `observation/image` | `base_0_rgb` |
| `observation/wrist_image` | `left_wrist_0_rgb` |
| `observation/extra_image` | `right_wrist_0_rgb` |
| `observation/state` | state |
| `prompt` | prompt |

### 7.2 输出适配

OpenPI 模型输出 action chunk，内部 action dim 是 32。

RoboBenchMart 环境实际只需要 13 维动作，所以输出时裁剪：

```python
actions[:, :13]
```

如果不裁剪，环境可能收到错误维度 action，导致评测失败或动作异常。

---

## 8. state 和 action 是什么

### 8.1 state / qpos

`qpos` 是机器人关节位置状态。本任务里原始 qpos 是 15 维。

可以粗略理解为：

- 底盘位置/旋转相关量
- torso 高度
- 机械臂关节角
- 夹爪开合状态

训练数据里：

```text
obs/agent/qpos: 15 dim
```

OpenPI 模型需要统一 shape，所以训练 transform 会 pad 到 32 维。

### 8.2 action

action 是机器人下一步要执行的控制量。本任务实际 action 是 13 维。

可以粗略理解为：

- base 移动/旋转控制
- torso 控制
- arm joint position 控制
- gripper 控制

训练数据里：

```text
actions: 13 dim
```

OpenPI 内部统一 pad 到 32 维，但最终执行只取前 13 维。

---

## 9. 语言 prompt 的作用

同一个环境中可能有多个商品，机器人需要知道目标是哪一个。

prompt 例如：

```text
move to shelf and pick Fanta Sabor Naranja 2L to basket
```

prompt 告诉模型：

- 任务类型：pick to basket
- 目标物体：Fanta / Nivea / Stars
- 目标位置：basket

OpenPI 会对 prompt 做 tokenization，把自然语言变成模型可处理的 token。

如果 prompt 错了，模型可能抓错物体。

---

## 10. 环境初始位姿为什么重要

之前官方 pi0.5 评测曾出现成功率全 0。根因不是 checkpoint 错，也不是端口错，而是 PickToBasketContEnv 的 Fetch 初始 qpos 和官方 demo/checkpoint 训练分布不一致。

正确初始 qpos 需要接近：

```text
[0, 0, 0, 0.34, 0, 0, 0, 1.4, 0, 0.76, 0, -2*pi/3, 0, 0.015, 0.015]
```

错误低位姿态类似：

```text
torso = 0
arm roughly [-pi/4, 0, pi/4, 0, pi/3]
```

如果初始姿态错，模型一开始就处在从未训练过的状态，动作会明显异常，成功率可能全 0。

本项目已检查并确认：

- `PickToBasketContEnv` reset 的 qpos 与 demo 初始 qpos 一致。
- 这是官方 pi0.5 baseline 和 SFT 公平对比的前提。

---

## 11. 本次训练配置

训练命令：

```bash
cd /root/autodl-tmp/projects/openpi

export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OMP_NUM_THREADS=8

EXP=pick_to_basket_sft_20k_official_ckpt

uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name=${EXP} \
  --num-train-steps=20000 \
  --save-interval=5000 \
  --log-interval=50 \
  --no-wandb-enabled \
  --resume
```

### 11.1 参数解释

| 参数 | 含义 |
|---|---|
| `HF_LEROBOT_HOME` | LeRobot 数据集所在目录 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` | JAX 最多预分配约 85% GPU 显存 |
| `OMP_NUM_THREADS=8` | CPU 并行线程数 |
| `pi05_sft_rbm_pick_to_basket` | OpenPI 训练配置名 |
| `--exp-name` | 实验名称，决定 checkpoint/log/tensorboard 子目录 |
| `--num-train-steps=20000` | 总训练 step 数，不是每个数据集样本跑 20000 次 |
| `--save-interval=5000` | 每 5000 step 保存一次 checkpoint |
| `--log-interval=50` | 每 50 step 写一次指标 |
| `--no-wandb-enabled` | 不使用 wandb |
| `--resume` | 从已有 checkpoint 继续训练 |

### 11.2 batch size 和 step 的关系

配置里 batch size 是：

```text
batch_size = 32
```

一次 train step 处理 32 个训练样本。

所以：

```text
samples_seen = step * batch_size
```

比如 step 19950：

```text
samples_seen = 638432
```

这不是 19950 个 episode，而是模型训练中看过的 batch 样本数量。

### 11.3 epoch 是什么

epoch 表示大约完整看过多少遍训练数据。

本次转换数据总 frame 数：

```text
total_frames = 307319
```

近似：

```text
approx_epoch = samples_seen / total_frames
```

最终约：

```text
approx_epoch = 2.0774
```

也就是大约看过 2.08 遍训练数据。

---

## 12. checkpoint 保存

实际保存目录：

```text
/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt
```

保存了：

```text
5000/
10000/
15000/
19999/
```

为什么最终是 `19999` 而不是 `20000`：

- 训练 loop 通常从当前 step 迭代到 `num_train_steps - 1`。
- 所以最后一个 step 是 `19999`。

每个 checkpoint 约：

```text
42G
```

总共约：

```text
168G
```

### 12.1 checkpoint 里有什么

一个完整 checkpoint 包含：

| 目录/文件 | 含义 |
|---|---|
| `_CHECKPOINT_METADATA` | Orbax checkpoint 元数据 |
| `params/` | 推理所需模型参数 |
| `train_state/` | 训练状态，包括 optimizer state 等 |
| `assets/rbm_dataset/norm_stats.json` | normalization 统计量 |

最终 checkpoint 完整性检查通过：

```text
19999/_CHECKPOINT_METADATA
19999/assets/rbm_dataset/norm_stats.json
19999/params/_METADATA
19999/train_state/_METADATA
```

### 12.2 为什么之前会有保存失败

之前出现过：

```text
10000.orbax-checkpoint-tmp-*
```

这是 checkpoint 没 finalize 的临时目录，不能用于 resume 或推理。

后续修复为同步 checkpoint：

```python
enable_async_checkpointing=False
checkpoint_manager.wait_until_finished()
```

现在每次保存成功会打印：

```text
Finished checkpoint save at step ... saved=True
```

本次 20k 训练最终没有 tmp 残留。

---

## 13. 训练日志和 TensorBoard

CSV 指标文件：

```text
/root/autodl-tmp/projects/openpi/logs/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt/metrics.csv
```

TensorBoard 目录：

```text
/root/autodl-tmp/projects/openpi/tensorboard/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt
```

启动 TensorBoard：

```bash
cd /root/autodl-tmp/projects/openpi

uv run tensorboard \
  --logdir /root/autodl-tmp/projects/openpi/tensorboard \
  --host 0.0.0.0 \
  --port 6006
```

注意：机器上可能有另一个旧 TensorBoard：

```text
port 6007 -> /root/tf-logs
```

这个不是 OpenPI 本次训练日志，不能看它。

---

## 14. 各训练指标是什么，怎么算，有什么意义

### 14.1 loss

loss 是 SFT 最核心的训练指标。

它表示：

> 模型预测的动作和专家动作之间差多少。

训练代码里大致是：

```python
chunked_loss = model.compute_loss(rng, observation, actions, train=True)
loss = mean(chunked_loss)
```

含义：

- 越低表示模型越能拟合训练数据里的专家动作。
- 但 loss 低不等于真实环境成功率一定高。
- 因为机器人任务有长程误差累积：一步动作稍微错，后面状态就可能偏离训练分布。

本次 loss：

```text
5050:  0.002101
10000: 0.001280
15000: 0.001136
19950: 0.000988
```

说明模型对训练数据拟合逐渐增强。

### 14.2 grad_norm

grad_norm 是梯度全局范数。

代码：

```python
optax.global_norm(grads)
```

含义：

- 梯度代表参数更新方向和大小。
- grad_norm 太大可能表示训练不稳定或梯度爆炸。
- grad_norm 接近 0 可能表示学习停滞。

本次 grad_norm 大致在：

```text
0.02 - 0.03
```

比较稳定，没有明显爆炸。

### 14.3 param_norm

param_norm 是参数范数。

代码大致对 kernel 参数求：

```python
optax.global_norm(kernel_params)
```

含义：

- 观察模型参数规模是否异常变化。
- 如果突然暴涨，可能训练发散。
- 如果平稳变化，一般说明训练数值稳定。

本次 param_norm 从约：

```text
2048.19 -> 2048.96
```

变化很小，说明训练稳定。

### 14.4 learning_rate

learning_rate 是学习率，表示每次参数更新幅度。

本次使用约：

```text
1e-5
```

日志中显示：

```text
9.999999747378752e-06
```

学习率较小，适合在已有 checkpoint 上微调，避免破坏原模型能力。

### 14.5 step_time_sec

每个训练 step 平均耗时。

计算：

```text
elapsed_time_between_logs / logged_steps
```

本次大约：

```text
3.0 秒 / step
```

### 14.6 samples_per_sec

训练吞吐量。

计算：

```text
logged_steps * batch_size / elapsed_time
```

本次大约：

```text
10.5 samples/sec
```

### 14.7 samples_seen

累计训练样本数。

计算：

```text
samples_seen = (step + 1) * batch_size
```

batch size 是 32。

### 14.8 approx_epoch

近似 epoch。

计算：

```text
approx_epoch = samples_seen / total_frames
```

本次：

```text
total_frames = 307319
final approx_epoch = 2.0774
```

说明模型大约看过 2.08 遍训练数据。

---

## 15. 训练过程数据分析

训练记录：

```text
rows = 299
first_step = 5050
last_step = 19950
```

关键指标：

| step | loss | grad_norm | samples_seen | approx_epoch |
|---:|---:|---:|---:|---:|
| 5050 | 0.002101 | 0.028705 | 161632 | 0.5259 |
| 10000 | 0.001280 | 0.022519 | 320032 | 1.0414 |
| 15000 | 0.001136 | 0.029008 | 480032 | 1.5620 |
| 19950 | 0.000988 | 0.026302 | 638432 | 2.0774 |

统计：

```text
loss_first = 0.002101
loss_last = 0.000988
loss_min = 0.000785
loss_mean_last20 = 0.000970

grad_first = 0.028705
grad_last = 0.026302
grad_mean_last20 = 0.024995
```

解释：

- loss 降幅明显，说明 SFT 数据拟合成功。
- loss 后期仍有波动，但整体较低，没有发散。
- grad_norm 平稳，说明训练数值稳定。
- 训练到 2.08 epoch，不是只看了很少数据。

---

## 16. 推理评测是怎么做的

训练时模型只是在数据上模仿专家动作。

评测时要让模型真的控制机器人：

1. env reset，创建仿真场景。
2. 取当前 observation：图像、state、prompt。
3. 发给 OpenPI policy server。
4. 模型输出 action chunk。
5. 环境执行 action。
6. 重复直到成功、失败或超过 `max_horizon=600`。
7. 统计 success rate。

### 16.1 policy server

启动最终模型：

```bash
cd /root/autodl-tmp/projects/openpi

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export OMP_NUM_THREADS=8

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_eval_rbm \
  --policy.dir=/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt/19999
```

### 16.2 eval client

评测脚本：

```text
/root/autodl-tmp/projects/RoboBenchMart/scripts/eval_policy_client.py
```

本次评测：

- `num_traj=30`
- `max_horizon=600`
- `sim_backend=cpu`
- 保存视频：`--save-video`
- 不保存轨迹：不加 `--save-traj`

注意：

- `--save-video` 会保存视频，比较占空间。
- 不加 `--save-traj` 就不会保存大 H5 轨迹。

---

## 17. 成功率怎么计算

每个 episode 结束后，环境返回：

```python
info["success"]
```

成功率：

```text
success_rate = num_success / num_traj
```

例如 Fanta：

```text
22 / 30 = 0.7333 = 73.3%
```

平均 episode length：

```text
avg_episode_length = mean(elapsed_steps)
```

如果模型更快完成任务，平均步长通常会下降。

但如果失败 episode 都跑满 600，平均步长会变大。

---

## 18. 评测结果总表

| 任务 | 官方 pi0.5 | SFT 19999 | 变化 |
|---|---:|---:|---:|
| Fanta train | 13/30 = 43.3% | 22/30 = 73.3% | +30.0% |
| Nivea train | 17/30 = 56.7% | 21/30 = 70.0% | +13.3% |
| Stars train | 13/30 = 43.3% | 24/30 = 80.0% | +36.7% |
| Nestle OOD | 4/30 = 13.3% | 2/30 = 6.7% | -6.7% |
| Slam OOD | 1/30 = 3.3% | 0/30 = 0.0% | -3.3% |

平均 episode length：

| 任务 | 官方 pi0.5 | SFT 19999 | 说明 |
|---|---:|---:|---|
| Fanta train | 511.7 | 470.0 | 更快 |
| Nivea train | 483.3 | 461.7 | 更快 |
| Stars train | 520.0 | 445.0 | 更快 |
| Nestle OOD | 573.3 | 590.0 | 更慢 |
| Slam OOD | 595.0 | 600.0 | 更慢 |

---

## 19. 逐任务结果解释

### 19.1 Fanta train

```text
官方: 13/30 = 43.3%
SFT:  22/30 = 73.3%
提升: +30.0%
```

新增成功 seed：

```text
[8, 12, 28, 32, 80, 88, 104, 108, 120, 128, 136, 140, 152]
```

丢失成功 seed：

```text
[16, 24, 56, 116]
```

解释：

- SFT 学到了 Fanta 相关示范，成功率明显提高。
- 仍有少数官方成功 seed 被 SFT 破坏，说明微调不是单调改进。
- 平均步长下降，说明整体执行效率也提升。

### 19.2 Nivea train

```text
官方: 17/30 = 56.7%
SFT:  21/30 = 70.0%
提升: +13.3%
```

新增成功 seed：

```text
[12, 40, 52, 64, 80, 88, 92, 108]
```

丢失成功 seed：

```text
[32, 56, 84, 124]
```

解释：

- Nivea 也提升，但幅度小于 Fanta 和 Stars。
- 可能官方 baseline 对 Nivea 已经相对较强，所以提升空间较小。

### 19.3 Stars train

```text
官方: 13/30 = 43.3%
SFT:  24/30 = 80.0%
提升: +36.7%
```

新增成功 seed：

```text
[0, 8, 28, 36, 40, 76, 88, 96, 104, 116, 120, 136, 144, 168]
```

丢失成功 seed：

```text
[20, 60, 156]
```

解释：

- Stars 是提升最大的 train task。
- 之前 5000 checkpoint 上 Stars 曾经 `0/5`，说明早期 checkpoint 不稳定。
- 训练到 19999 后 Stars 大幅恢复并超过官方 baseline。

### 19.4 Nestle OOD

```text
官方: 4/30 = 13.3%
SFT:  2/30 = 6.7%
下降: -6.7%
```

新增成功 seed：

```text
[42017, 42028]
```

丢失成功 seed：

```text
[42010, 42011, 42014, 42020]
```

解释：

- SFT 没有改善 Nestle OOD。
- 虽然有两个新 seed 成功，但丢掉了官方原本成功的四个 seed。
- 净效果是下降。

### 19.5 Slam OOD

```text
官方: 1/30 = 3.3%
SFT:  0/30 = 0.0%
下降: -3.3%
```

解释：

- Slam OOD 本来就很难。
- SFT 后没有成功 episode。
- 说明当前 SFT 对 unseen item 没有帮助。

---

## 20. 为什么 train 提升但 OOD 下降

这是机器人 SFT 里很常见的问题。

### 20.1 SFT 优化的是训练数据拟合

SFT 只要求模型模仿训练数据动作。

训练数据包含：

```text
Fanta / Nivea / Stars
```

不包含：

```text
Nestle / Slam
```

所以模型会更贴近 train item 的视觉外观和动作分布。

### 20.2 OOD 需要泛化

OOD item 和 train item 可能在以下方面不同：

- 外观颜色
- 包装形状
- 尺寸
- 重心
- 抓取姿态
- 与篮子的接触方式
- 目标识别难度

如果训练数据没有覆盖这些变化，SFT 可能不会提升 OOD。

### 20.3 微调可能破坏原有泛化

官方模型可能有一些通用能力。SFT 后模型更偏向特定训练物体，可能损失部分通用能力。

这就是为什么：

```text
train item 明显提升
OOD item 略微下降
```

---

## 21. loss 下降为什么不等于 OOD 成功率提升

loss 是在训练数据上算的。

如果训练数据只有 Fanta/Nivea/Stars，loss 下降表示：

> 模型更会模仿 Fanta/Nivea/Stars 的专家动作。

但 OOD 是 Nestle/Slam，分布不同。

所以可能出现：

```text
loss 下降
train success 上升
OOD success 下降
```

本次实验正是这个现象。

---

## 22. 当前模型该怎么使用

最终模型：

```text
/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt/19999
```

适合：

- Fanta train task
- Nivea train task
- Stars train task
- 作为 PPO 后训练初始化模型
- 作为 train distribution SFT baseline

不适合直接声称：

- Nestle OOD 提升
- Slam OOD 提升
- unseen item 泛化增强

---

## 23. 后续 PPO 路线建议

如果目标是提高 OOD：

```text
RoboBenchMart ManiSkill Env
  -> reset(scene, target, robot_init_seed)
  -> observation(image / wrist_image / extra_image / state / prompt)
  -> pi0.5/SFT policy 输出 action chunk
  -> env.step(action)
  -> reward / done / success / info
  -> PPO update
```

建议 PPO 重点：

1. 使用 `19999` SFT checkpoint 作为初始化。
2. 在 Nestle/Slam OOD 上 rollout。
3. reward 不只用 sparse success，也可以加入：
   - 目标物体接近篮子距离变化
   - 是否抓起目标物体
   - 是否放入篮子
   - 是否扰动非目标物体
   - 是否机器人静止
4. 保持 train task 回归评测，避免 PPO 提升 OOD 但破坏 train item。

---

## 24. 存储空间注意事项

当前 checkpoint：

```text
5000   42G
10000  42G
15000  42G
19999  42G
```

总计约：

```text
168G
```

视频评测也会占空间。

当前磁盘剩余约：

```text
47G
```

建议：

- 如果只保留最终模型，可以保留 `19999/`。
- 如果还担心 resume，可以保留 `5000/` 和 `19999/`。
- `10000/15000/` 可在确认不需要中间模型对比后删除。
- 视频文件可以单独归档或删除，只保留 `eval_summary.json`。

删除前一定确认路径，避免删错最终 checkpoint。

---

## 25. 关键路径汇总

RoboBenchMart：

```text
/root/autodl-tmp/projects/RoboBenchMart
```

OpenPI：

```text
/root/autodl-tmp/projects/openpi
```

RLinf 评测环境：

```text
/root/autodl-tmp/projects/RLinf/.venv_openpi
```

LeRobot 数据：

```text
/root/autodl-tmp/lerobot_data/rbm_dataset
```

转换脚本：

```text
/root/autodl-tmp/projects/RoboBenchMart/scripts/convert_pick_to_basket_h5_to_lerobot.py
```

OpenPI RBM adapter：

```text
/root/autodl-tmp/projects/openpi/src/openpi/policies/robobenchmart_policy.py
```

OpenPI config：

```text
/root/autodl-tmp/projects/openpi/src/openpi/training/config.py
```

checkpoint 保存代码：

```text
/root/autodl-tmp/projects/openpi/src/openpi/training/checkpoints.py
```

最终 SFT checkpoint：

```text
/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt/19999
```

训练指标：

```text
/root/autodl-tmp/projects/openpi/logs/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt/metrics.csv
```

TensorBoard：

```text
/root/autodl-tmp/projects/openpi/tensorboard/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_20k_official_ckpt
```

最终评测 summary：

```text
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket/evaluations/pi05_sft_19999_basket_fanta_train_n30_video/eval_summary.json
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket/evaluations/pi05_sft_19999_basket_nivea_train_n30_video/eval_summary.json
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket/evaluations/pi05_sft_19999_basket_stars_train_n30_video/eval_summary.json
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/test_unseen_items_pick_to_basket/evaluations/pi05_sft_19999_basket_nestle_ood_n30_video/eval_summary.json
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/test_unseen_items_pick_to_basket/evaluations/pi05_sft_19999_basket_slam_ood_n30_video/eval_summary.json
```

---

## 26. 最终结论

本次 SFT 20k 训练成功，checkpoint 保存完整，训练指标稳定下降，train task 成功率显著提升。

最终模型在训练物体上表现：

```text
Fanta: 73.3%
Nivea: 70.0%
Stars: 80.0%
```

相比官方 pi0.5 baseline 有显著提升。

但 OOD 表现：

```text
Nestle: 6.7%
Slam: 0.0%
```

没有提升，且略低于官方 baseline。

因此，本模型可以作为：

- PickToBasket train item SFT baseline
- PPO 后训练初始化模型
- 后续 OOD 强化学习实验起点

但不能作为 OOD 泛化提升的证据。
