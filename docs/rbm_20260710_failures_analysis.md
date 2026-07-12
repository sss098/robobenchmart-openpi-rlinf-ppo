# 2026-07-10 RBM BASE、Proxy 和 PPO 故障复盘

## 结论摘要

本次三个问题都有日志证据，不能归结为随机波动：

1. 旧的 RLinf BASE 与历史 JAX SFT 结果使用了不同 episode seed、robot pose seed 和 simulator backend，因此 `38/90` 不能与 `67/90` 直接比较。
2. Proxy 失败是评测脚本引用了不存在的目录 `demo_envs/original_envs_configs_pick_to_basket`。
3. 3 epoch 训练不是 checkpoint 保存失败，而是首次 actor+SFT backward 在保存前 CUDA OOM。
4. 2 epoch checkpoint 为 critic-only，`37/90` 与同协议 BASE `38/90` 等价，没有证据表明 PPO actor 已经破坏 SFT。
5. critic explained variance 从 `-0.413` 变为 `-1.568`，critic 没有学好，当前不应启用 actor PPO。

## 1. BASE 为什么看起来比 JAX 差

### 旧 RLinf 结果

日志：

```text
logs/20260710-17:40:36-rbm_pick_to_basket_ppo_openpi_pi05/run_embodiment.log
```

结果：

```text
success_once = 0.42222223
num_trajectories = 90
即 38/90
```

### 历史 JAX 结果

```text
Fanta 22/30
Nivea 21/30
Stars 24/30
合计 67/90
```

### 比较协议实际不一致

历史 JAX train3 使用三个 demo JSON 前 30 条 episode seed。它们不是 0 到 29：

- Fanta：`4, 8, 12, 16, ..., 152`
- Nivea：`0, 4, 8, 12, ..., 140`
- Stars：`0, 4, 8, 16, ..., 168`

历史 JAX train3 没有传 `robot_init_pose_seed`，并使用 `sim_backend=cpu`。

旧 RLinf 脚本却使用：

```text
episode_seed_start=0       -> 连续 0..29
robot_init_pose_start_seed=10000
sim_backend=auto
```

OOD 也不一致：历史 JAX 使用 episode seed `42000..42029` 和 pose seed `10000..10029`，旧 RLinf 使用 episode seed `0..29`。

因此现有结果只能说明 PyTorch policy 在另一组场景/初始姿态上为 38/90，不能证明 JAX 到 PyTorch 转换损失了 29 次成功。

### 已修复

`scripts_local/08_rbm_eval_matched.sh` 和 RECAP 评测脚本现在复现上述 JAX seed、pose 和 CPU simulator，并输出逐任务 metric：

```text
eval/PickToBasketContFantaEnv/success_once
eval/PickToBasketContNiveaEnv/success_once
eval/PickToBasketContStarsEnv/success_once
```

重跑命令：

```bash
cd /root/autodl-tmp/projects/RLinf
ray stop --force || true
bash scripts_local/08_rbm_eval_matched.sh BASE
EVAL_SPLIT=ood bash scripts_local/08_rbm_eval_matched.sh BASE
```

只有这次结果仍显著低于 67/90，才进入 JAX/PyTorch action parity 检查。需要启动原 JAX server，并按 `docs/rbm_stage1_rlinf_ppo_smoke_adaptation.md` 的 action trace 流程比较相同 observation 下的输出分布。

JAX 与 PyTorch 默认从各自 RNG 采样 flow noise，即便 scene seed 相同，单条 action 不应预期 bitwise 相等。parity 应比较固定 noise 或多次采样统计，不能只比较一次随机 action。

## 2. Proxy 报错

日志中的真实异常：

```text
hydra.errors.MissingConfigException:
Primary config directory not found:
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/original_envs_configs_pick_to_basket
```

本机不存在该目录。Proxy 自定义环境本来就是从 train PickToBasket scenes 中采样非 Fanta/Nivea/Stars/OOD 商品，所以应使用：

```text
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket
```

脚本已修复。重跑：

```bash
cd /root/autodl-tmp/projects/RLinf
ray stop --force || true
EVAL_SPLIT=proxy bash scripts_local/08_rbm_eval_matched.sh BASE
```

当前会话尝试单独创建 CPU Proxy env 时被主机直接 SIGKILL，没有新的 Python traceback。因此这里只确认原目录错误已修复，不能声称完整 Proxy runtime 已验证成功。

## 3. 为什么第 3 个模型没有保存

训练日志：

```text
logs/20260710-18:10:00-rbm_pick_to_basket_ppo_openpi_pi05/run_embodiment.log
```

step 1、2 已正常保存：

```text
Saving checkpoint at step 1
Saving checkpoint at step 2
```

step 3 在 `run_training()` backward 中失败：

```text
torch.OutOfMemoryError: CUDA out of memory
Tried to allocate 7.56 GiB
GPU total 94.97 GiB, free 1.20 GiB
PyTorch allocated 70.16 GiB, reserved unused 7.30 GiB
```

所以不是 save checkpoint 代码出错，而是 step 3 没有完成训练，自然不会进入保存。

step 3 恰好越过原来的 20 optimizer-step critic warmup，第一次同时保留 PPO graph 和 SFT graph，再对合并 loss backward，造成显存峰值。

### 已修复

- PPO graph 先 backward 并释放 activation，再建立并 backward SFT graph。
- actor micro batch 从 4 降到 2，global batch 仍为 16。
- 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 设置 `OMP_NUM_THREADS=1`，消除日志中的无效 OMP 配置。
- critic warmup 从 20 延长到 40 optimizer step。3 epoch 约 27 step，仍为 critic-only，不会在 critic 明显失效时更新 actor。

显存修复已经通过静态检查，但当前会话无可用 GPU，尚未完成真实 post-warmup backward 验证。

## 4. 2 epoch checkpoint 分析

评测日志：

```text
logs/20260710-18:26:39-rbm_pick_to_basket_ppo_openpi_pi05/run_embodiment.log
```

结果：

```text
2 epoch checkpoint: 37/90 = 41.1%
同旧协议 BASE:      38/90 = 42.2%
```

只差 1 条，远小于 90 次 Bernoulli 评测的统计误差。旧 rollout worker 没有显式初始化 PyTorch RNG，现已增加固定 seed。更重要的是前 2 epoch 日志显示：

```text
actor/lr=0
actor/policy_loss=0
actor/approx_kl=0
actor/ratio=1
```

说明 actor 没有被 PPO 更新，只有 value head 改变。因此 37/90 不是“2 epoch PPO 导致退化”，而是相同 actor 在 stochastic flow sampling 下的一次重复测量。

critic 指标：

```text
epoch 1 explained_variance = -0.413
epoch 2 explained_variance = -1.568
epoch 1 success_once = 0/12
epoch 2 success_once = 0/12
```

这表明当前 online rollout 的 critic 信号很差。虽然 shaped return 非零，但 value prediction 比常数均值基线还差，不能据此更新 actor。

## 5. 当前正确执行顺序

1. 先用修正协议重跑 BASE train3 和 OOD。
2. 重跑 Proxy BASE，确认目录问题消失。
3. 暂停 PPO actor 更新，不使用旧 2 epoch checkpoint继续训练。
4. 若修正协议下 PyTorch BASE 仍远低于 JAX，先做 fixed-observation/fixed-noise parity，不训练 PPO。
5. BASE 对齐后，重新从 SFT 启动 3 epoch critic-only run；检查 explained variance。
6. 只有 explained variance 接近 0 或转正，才允许超过 40 optimizer step进入 actor 更新。

三 epoch critic-only 命令：

```bash
cd /root/autodl-tmp/projects/RLinf
ray stop --force || true
MAX_EPOCHS=3 SAVE_INTERVAL=1 bash scripts_local/06_rbm_ppo_smoke.sh
```

若第 3 epoch explained variance 仍明显为负，停止 PPO，优先转向带 in-domain 正负 rollout 的 mixed RECAP-lite。
