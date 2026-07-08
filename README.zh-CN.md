# RoboBenchMart PickToBasket SFT + KL-PPO 后训练全流程项目

[English README](README.md)

这个仓库是一个个人研究工程项目，目标是在 **RoboBenchMart PickToBasket** 任务上，为 **OpenPI pi0.5** 建立一条完整的后训练链路：

```text
PickToBasket SFT
-> JAX checkpoint 评测
-> JAX checkpoint 转 PyTorch checkpoint
-> RLinf rollout / eval
-> SFT-reference KL-PPO 小规模闭环强化
-> 失败 episode 分析
-> correction data / RECAP-lite weighted SFT
-> train / robo / unseen / OOD 四类场景评测
```

这个项目的起点是一个已经完成 20000 步 SFT 的 OpenPI pi0.5 PickToBasket 模型。SFT 对 in-domain 任务有提升，但问题是：能不能继续用闭环强化学习让模型在 in-domain 和 OOD 任务上都更好，同时不破坏 SFT 已经学到的抓取、移动、放置能力。

这个仓库记录的就是为了回答这个问题，在 RLinf 中打通 OpenPI + RoboBenchMart + PPO 后训练所做的代码、配置、脚本和技术分析。

它不是模型权重发布仓库。checkpoint、训练日志、视频、数据集和仿真资产都不会放进仓库。

## 这个仓库是什么

这个仓库连接了三个系统：

```text
OpenPI         pi0.5 模型、SFT 训练、JAX checkpoint
RoboBenchMart  PickToBasket 仿真任务和评测环境
RLinf          PyTorch rollout、PPO actor/value 训练、分布式 worker
```

主要工作不是重新写一个 PPO 算法，而是把完整链路打通并排查清楚：

1. 用 OpenPI pi0.5 在 RoboBenchMart PickToBasket demonstration 上做 SFT。
2. 在官方 JAX/OpenPI 路径中评测 SFT checkpoint。
3. 把 JAX checkpoint 转成 RLinf 可加载的 PyTorch checkpoint。
4. 在 RLinf 中做 matched eval，确认转换后的模型行为仍然合理。
5. 在 RLinf 中跑保守 PPO，包含 action、logprob、value、old_logprob、advantage 和 SFT-reference KL。
6. 每个 PPO checkpoint 都和 SFT baseline 对比，而不是只看 reward 是否变大。
7. 用失败 rollout 生成 correction 数据，为后续 RECAP-lite / weighted SFT 做准备。

## 为什么要做这个项目

单纯 SFT 通常能提高训练分布内任务，但对 unseen / OOD 的泛化不一定稳定。闭环 RL 看起来很适合机器人任务，因为模型可以和环境交互，根据真实任务反馈继续优化。

但直接在一个已经比较强的 SFT 机器人策略上做 PPO 很容易出问题：

- reward 可能稀疏，或者和最终 success rate 不完全一致。
- shaped reward 可能被模型“钻空子”，例如只学会夹起物体但不放进篮子。
- PPO clip 只约束当前策略和 rollout old policy 的距离，不能保证多轮训练后仍然接近最初 SFT 模型。
- JAX eval 和 PyTorch rollout 的动作、图像、seed、action chunk 只要有一点不一致，PPO 就可能在错误链路上优化。

因此这个项目中，PPO 的定位是：

```text
小规模、保守、用于验证链路和暴露失败模式的闭环强化 probe。
```

更稳的下一步不是直接长训 PPO，而是：

```text
用 eval / PPO rollout 找失败模式 -> 构造 correction 数据 -> 做 RECAP-lite / weighted SFT。
```

## 整体技术路线

```text
阶段 0：OpenPI / RoboBenchMart SFT
  用 OpenPI JAX trainer 在 PickToBasket demonstration 上训练 pi0.5。

阶段 1：SFT 评测与 checkpoint 转换
  先在 JAX 路径评测 SFT checkpoint，再转成 RLinf PyTorch checkpoint。

阶段 2：RLinf matched eval
  用转换后的 PyTorch checkpoint 做确定性评测，并和 JAX 行为对齐。

阶段 3：SFT-reference KL-PPO
  小规模 PPO probe，训练 value，计算 old_logprob / current logprob / advantage，并加入 SFT reference KL。

阶段 4：失败 episode 收集
  保存失败视频、动作轨迹、任务 id、seed 和 reward 信息，用于分析失败原因。

阶段 5：RECAP-lite / weighted SFT
  把成功和修正后的失败 episode 组织成带权重的 SFT 数据。

阶段 6：统一评测
  在 train / robo / unseen / OOD 四类 split 上评测，并始终和 matched SFT baseline 对比。
```

## 仓库关键内容

```text
rlinf/envs/robobenchmart/
  RLinf EnvWorker 使用的 RoboBenchMart 环境 wrapper。

rlinf/envs/robobenchmart/robobenchmart_env.py
  PickToBasket 观测映射、任务选择、reset、seed 对齐、视频/debug hook 和 shaped reward。

rlinf/envs/robobenchmart/proxy_tasks.py
  proxy task 定义。用于训练时增加物体和布局变化，同时避免把 Nestle/Slam/Duff 这类 OOD 任务直接放进 PPO 训练。

rlinf/models/embodiment/openpi/dataconfig/robobenchmart_dataconfig.py
  RLinf 侧的 RoboBenchMart OpenPI data config。

rlinf/models/embodiment/openpi/policies/robobenchmart_policy.py
  把 RoboBenchMart observation/action 转成 OpenPI 模型输入输出的 policy adapter。

rlinf/workers/actor/fsdp_actor_worker.py
  在 embodied PPO actor 中加入真正的 SFT-reference KL 约束。

examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
  PickToBasket 主 PPO / eval 配置。

examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
  proxy-mix PPO 配置。

examples/embodiment/config/env/robobenchmart_pick_to_basket_*.yaml
  train、proxy、unseen、OOD 风格环境 split。

scripts_local/06_rbm_ppo_smoke.sh
scripts_local/07_rbm_ppo_proxy_mix.sh
  本地 smoke 和 proxy-mix PPO 启动脚本。

scripts_local/08_rbm_collect_jax_action_trace.py
scripts_local/09_compare_rbm_action_traces.py
  JAX policy 和 RLinf PyTorch policy 动作轨迹对比工具。

docs/rbm_stage1_rlinf_ppo_smoke_adaptation.md
  RLinf PPO 适配和 smoke 阶段记录。

docs/rbm_true_kl_ppo_and_reward_changes.md
  SFT-reference KL-PPO 和 PickToBasket reward 修改的详细解释。
```

## SFT 阶段

OpenPI 侧 RoboBenchMart SFT config 是：

```text
pi05_sft_rbm_pick_to_basket
```

本地实验设置：

```text
模型：OpenPI pi0.5
任务：RoboBenchMart PickToBasket
action_dim：13
训练步数：20000 steps
checkpoint 格式：OpenPI / JAX checkpoint
```

SFT checkpoint 必须先在 JAX/OpenPI 原始路径中评测。原因是：如果 SFT 自己都没有正确跑通，就不能进入 PPO；如果转换前后行为不一致，后续 PPO 的结果也没有意义。

## Checkpoint 转换

RLinf 的 PPO 路径使用 PyTorch OpenPI 模型，因此需要把 SFT JAX checkpoint 转成 PyTorch checkpoint：

```bash
python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint_dir /path/to/openpi_jax_sft_checkpoint/19999 \
  --output_path /path/to/rbm_pi05_sft_pytorch \
  --config_name pi05_eval_rbm
```

注意：

```text
SFT 训练 config：pi05_sft_rbm_pick_to_basket
评测/转换 config：pi05_eval_rbm
```

之前如果用 `pi05_rbm` 会报错，因为 OpenPI 里没有这个 config。

## PPO 阶段需要哪些量

PPO 不是只需要 action 和 reward。完整链路需要：

```text
action
logprob
value
old_logprob
ref_logprob
advantage
KL penalty
```

在这个项目里：

- `action`：rollout policy 在环境交互时采样出来的动作。
- `old_logprob`：采样这个 action 时，rollout policy 给出的 log probability。
- `value`：actor/value 模型对当前状态的价值估计。
- `advantage`：由 reward 和 value 计算出来，用于告诉 PPO 哪些动作比预期更好。
- `logprob`：PPO update 时，当前 actor 对同一批 action 重新计算的 log probability。
- `ref_logprob`：冻结的 SFT 初始模型对同一批 action 重新计算的 log probability。
- `KL(current, SFT)`：限制 PPO 后的当前模型不要偏离 SFT 模型太远。

最终 loss 变成：

```text
loss = PPO actor/value loss + kl_beta * reference_KL(current_logprob, ref_logprob)
```

这和普通 PPO 的 `approx_kl` 不一样：

```text
approx_kl：current policy vs rollout old policy
ref_kl：current policy vs 初始 SFT reference policy
```

你的技术路线里真正需要的是第二个，因为目标是“在 SFT 附近做保守闭环强化”。

## PickToBasket reward 设计

当前 reward 是事件型 + 进度型：

```text
+5.00 * success
+1.00 * first_placed
+0.30 * first_lifted
+0.50 * positive_progress_to_basket
+0.30 * first_placed_static
-0.20 * first_non_target_displacement
```

设计思路：

- `success` 权重最大，保证最终目标还是完成任务。
- `first_placed` 比 `first_lifted` 更大，因为放进篮子比夹起来更接近成功。
- `first_lifted` 只给一次，避免模型靠反复夹起刷 reward。
- `positive_progress_to_basket` 只奖励比历史更靠近篮子的正向进展，避免停在篮子附近刷分。
- `first_placed_static` 鼓励放入后稳定，避免刚碰到篮子就掉出。
- `first_non_target_displacement` 轻量惩罚扰动非目标物体。

## 当前结论

当前最重要的结论是：

```text
SFT 是可靠 baseline。
直接 PPO 很脆弱。
reward 变大不等于 success rate 变高。
没有 SFT-reference KL 的 PPO 结果不可信。
更值得继续的是 failure correction / RECAP-lite weighted SFT。
```

已有实验中，PPO probe 没有稳定超过 SFT baseline。因此这个仓库不会把 PPO 包装成已经完成的性能提升，而是把它作为后训练链路、失败模式分析和 correction 数据生成的一部分。

## 不包含哪些内容

仓库不会包含：

```text
模型 checkpoint
训练 logs
TensorBoard event files
评测视频
RoboBenchMart assets
ManiSkill assets
OpenPI cached assets
本地虚拟环境
私有数据集
```

这些文件体积大、路径依赖强，或者不适合放到公开 GitHub 仓库。

## 最小复现目录结构

建议本地目录：

```text
/path/to/RLinf
/path/to/openpi
/path/to/RoboBenchMart
/path/to/ManiSkill
/path/to/rbm_pi05_sft_pytorch
```

环境变量：

```bash
export PYTHONPATH=/path/to/RLinf:/path/to/RoboBenchMart:/path/to/openpi:${PYTHONPATH:-}
export MODEL_PATH=/path/to/rbm_pi05_sft_pytorch
export OMP_NUM_THREADS=1
```

PPO smoke：

```bash
bash scripts_local/06_rbm_ppo_smoke.sh
```

proxy-mix PPO：

```bash
bash scripts_local/07_rbm_ppo_proxy_mix.sh
```

动作轨迹对比：

```bash
python scripts_local/09_compare_rbm_action_traces.py \
  /path/to/jax_trace.npz \
  /path/to/rlinf_trace_dir \
  --stage-id 0 \
  --atol 1e-4
```

## 项目状态

这是一个研究工程仓库，适合展示从 SFT 到 RL 后训练的完整链路、排错过程和保守 KL-PPO 的设计理由。它不是最终 benchmark checkpoint 发布，也不声称 PPO 已经稳定超过 SFT。
