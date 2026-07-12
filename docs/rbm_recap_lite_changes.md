# RBM PickToBasket RECAP-lite 适配记录

> 注意：本文保留历史适配过程。最新参数、修复和执行命令以 `docs/rbm_post_training_fixes_and_runbook.md` 为准。

> 2026-07-11 GPU 复验：advantages 生成、CFG-SFT 单步反向与保存、BASE 推理、`full_weights.pt` 加载推理均已通过。当前安全默认值是 `MICRO_BATCH_SIZE=1`、`GLOBAL_BATCH_SIZE=4`、`NUM_WORKERS=0`、`LR=2e-7`。历史示例中的 `LR=5e-7`、200/1000 step 不再作为首跑建议。mixed 命令只有在真实 in-domain rollout 数据存在且同时含正负标签时才可运行。

> 2026-07-12 mixed 前置链路修复：新增 `09_rbm_collect_indomain_rollouts.sh`；修复多 pipeline stage writer 路径冲突；LeRobot writer 显式使用绝对 root；advantage 脚本支持自动遍历 rollout shards；mixed 脚本聚合所有 shard 的正负标签并均分 rollout 总权重。无 GPU 时可用 `CHECK_ONLY=1` 完成 mixed 输入预检。

> 同日正式训练修复：采集原始第三视角列统一为 `extra_image`；新增 `09_rbm_repair_rollout_schema.py` 原子迁移旧 shard；mixed preflight 检查 schema、碎片、最小 episode 数及每 task 正负覆盖。正式 90 episode 数据已完成 2-step GPU transform/backward/save 验证。

本文记录本次为了把 RoboBenchMart PickToBasket 从 KL-PPO/proxy-mix 转到 RECAP-lite 所做的代码和配置补齐。

## 1. 技术路线

pi*0.6 论文里的 RECAP 核心不是 PPO，而是把数据里的动作按 advantage 分成正/负样本，然后继续做监督式 action/flow matching 训练。策略输入会在原任务 prompt 后增加：

- `Advantage: positive`
- `Advantage: negative`

训练时根据每个样本的 advantage label 选择正/负 guidance prompt，并随机 dropout 成原始 prompt；推理时固定使用 positive guidance。论文里 demo 和 expert correction 可以直接强制标成 positive，autonomous rollout 再通过 value model 或成功/失败信号区分正负。

本项目当前的 `/root/autodl-tmp/lerobot_data/rbm_dataset` 是 SFT demo 数据，parquet 里有：

- `image`
- `wrist_image`
- `extra_image`
- `state`
- `actions`
- `episode_index`
- `frame_index`
- `task_index`

没有 `is_success` 或 reward/return 列。因此本次先实现 RECAP-lite：把 SFT demo 全部标为 positive，再用 CFG 方式从 SFT checkpoint 继续训。这个路线优先保护 in-domain SFT 能力，避免 PPO 一样用在线失败 reward 把已学到的行为打坏。

## 2. 修改文件

### `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py`

补齐 CFG policy 的 RBM 第三视角输入：

- 在 `obs_processor()` 中把 `env_obs["extra_view_images"]` 映射到 `observation/extra_view_image`。
- 这和普通 OpenPI policy 的 RBM 逻辑一致。
- 没有这个修改时，CFG 模型在 RBM 环境推理时只会收到 base/wrist 两路图像，而 `pi05_rbm` 的数据 transform 期望三路图像：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。

### `rlinf/data/datasets/recap/value_model.py`

给 full RECAP 的 value-model 数据集补 RBM 支持：

- 新增 `robobenchmart` 和 `rbm` 两个 `robot_type`。
- repack 映射为：
  - `image -> observation/image`
  - `wrist_image -> observation/wrist_image`
  - `extra_image -> observation/extra_view_image`
  - `state -> observation/state`
  - `actions -> actions`
  - `prompt -> prompt`
- value SFT transform 使用 `RoboBenchMartInputs`。

RECAP-lite 当前不训练 value model，但这个修改让后续 full RECAP 路线可以继续接 RBM。

### `rlinf/models/embodiment/value_model/checkpoint_utils.py`

给 `ValueCriticModel.from_checkpoint(..., env_type="robobenchmart"|"rbm")` 增加输入 transform：

- 使用 `RoboBenchMartInputs`。
- 支持 RBM 三视角 value inference。

### `examples/recap/process/compute_advantages.py`

给 full RECAP 的 value advantage 计算补 RBM observation key mapping：

- `image -> observation/image`
- `wrist_image -> observation/wrist_image`
- `extra_image -> observation/extra_view_image`
- `state -> observation/state`
- `task/task_index -> prompt`

这个文件仍然是 full RECAP 用的，需要 value checkpoint 和 returns sidecar。RECAP-lite 初始实验不依赖它。

### `examples/recap/process/make_recap_lite_advantages.py`

新增 RECAP-lite advantage sidecar 生成脚本。

作用：

- 只读 LeRobot parquet 的轻量 metadata 列，不读图像。
- 写入 `meta/advantages_<tag>.parquet`。
- 输出列包括：
  - `episode_index`
  - `frame_index`
  - `advantage`
  - `advantage_continuous`
  - `label_reason`
  - `source_file`
  - `task_index`，如果原数据有该列
- 同时更新 `mixture_config.yaml`，记录 tag、正负样本数和正样本比例。

支持的标注模式：

- `all-positive`：所有帧标为 positive。适合 SFT demo 和 correction。
- `success-column`：按 episode 最后一帧的 `is_success`/`success`/`episode_success` 标整条 episode。适合后续 in-domain rollout 数据。
- `advantage-column`：直接读取已有 per-frame boolean label。
- `auto`：SFT/demo/correction 自动 all-positive；rollout 自动 success-column。

### `examples/recap/cfg/config/rbm_pick_to_basket_recap_lite_cfg_openpi_pi05.yaml`

新增 RBM RECAP-lite CFG 训练配置。

关键设置：

- `actor.model.model_type: cfg_model`
- `actor.model.openpi.config_name: pi05_rbm`
- `actor.model.openpi.num_images_in_input: 3`
- `actor.model.openpi.train_expert_only: true`
- `actor.model.openpi.positive_only_conditional: true`
- `actor.model.openpi.guidance_type: positive`
- `actor.model.openpi.unconditional_prob: 0.3`
- `actor.optim.lr: 5e-7`
- 默认从 `/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch` 继续训。
- 默认读取 `/root/autodl-tmp/lerobot_data/rbm_dataset/meta/advantages_rbm_recap_lite_sft_pos.parquet`。

这里的 LR 比原始 RECAP 示例保守，目的是先验证不会破坏 in-domain SFT。

### `scripts_local/10_rbm_recap_prepare_advantages.sh`

本地一键生成 SFT demo positive sidecar。

默认等价于：

```bash
cd /root/autodl-tmp/projects/RLinf
bash scripts_local/10_rbm_recap_prepare_advantages.sh
```

可覆盖变量：

```bash
DATASET_PATH=/path/to/dataset \
TAG=rbm_recap_lite_sft_pos \
DATASET_TYPE=sft \
LABEL_MODE=auto \
bash scripts_local/10_rbm_recap_prepare_advantages.sh
```

### `scripts_local/11_rbm_recap_cfg_sft.sh`

本地一键启动 RBM RECAP-lite CFG 训练。

默认命令：

```bash
cd /root/autodl-tmp/projects/RLinf
bash scripts_local/11_rbm_recap_cfg_sft.sh
```

常用保守 smoke run：

```bash
MAX_STEPS=20 SAVE_INTERVAL=5 LR=2.0e-7 \
bash scripts_local/11_rbm_recap_cfg_sft.sh
```

常用正式小规模 run：

```bash
MAX_STEPS=100 SAVE_INTERVAL=20 LR=2.0e-7 \
bash scripts_local/11_rbm_recap_cfg_sft.sh
```

如果要 sweep CFG 强度：

```bash
GUIDANCE_SCALE=0.7 MAX_STEPS=1000 bash scripts_local/11_rbm_recap_cfg_sft.sh
GUIDANCE_SCALE=1.0 MAX_STEPS=1000 bash scripts_local/11_rbm_recap_cfg_sft.sh
GUIDANCE_SCALE=1.3 MAX_STEPS=1000 bash scripts_local/11_rbm_recap_cfg_sft.sh
```

### `scripts_local/12_rbm_recap_eval_checkpoint.sh`

评估 CFG checkpoint。

FSDP 保存的 CFG checkpoint 通常只有 `full_weights.pt`，不包含 OpenPI norm stats。这个脚本会：

- 用原始 SFT checkpoint `/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch` 提供 config 和 norm stats。
- 用 `runner.ckpt_path` 加载 CFG 训练出的 `full_weights.pt`。

in-domain 三任务评估：

```bash
bash scripts_local/12_rbm_recap_eval_checkpoint.sh \
  logs/cfg_sft/rbm_pick_to_basket_recap_lite_cfg_openpi_pi05-YYYYmmdd-HH:MM:SS/rbm_recap_lite_cfg_pi05/checkpoints/global_step_200
```

OOD 两任务评估：

```bash
EVAL_SPLIT=ood \
bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_200
```

保存视频：

```bash
SAVE_VIDEO=True VIDEO_DIR=/root/autodl-tmp/projects/RLinf/logs/rbm_recap_eval_videos \
bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_200
```

## 3. 推荐实验顺序

1. 生成 advantage sidecar：

```bash
cd /root/autodl-tmp/projects/RLinf
bash scripts_local/10_rbm_recap_prepare_advantages.sh
```

2. 先跑 200 step smoke：

```bash
MAX_STEPS=200 SAVE_INTERVAL=50 LR=5.0e-7 \
bash scripts_local/11_rbm_recap_cfg_sft.sh
```

3. 评估 `global_step_50/100/150/200`，先看 in-domain 是否不掉：

```bash
bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_200
```

4. 只有当 in-domain 接近或超过 SFT baseline，再评估 OOD：

```bash
EVAL_SPLIT=ood \
bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_200
```

5. 如果 in-domain 掉点明显，优先降强度：

```bash
LR=2.0e-7 GUIDANCE_SCALE=0.7 MAX_STEPS=500 \
bash scripts_local/11_rbm_recap_cfg_sft.sh
```

## 4. 结果解读

当前 RECAP-lite 只使用 SFT demo positive labels，没有负样本，也没有 value advantage 分位数。因此它主要验证两件事：

- CFG prompt/guidance 路径在 RBM 三视角上是否正确工作。
- positive guidance 继续 SFT 是否比 PPO 更少破坏 in-domain。

如果它没有提升 OOD，这不代表 RECAP 方向无效，只说明“只有 SFT positive demo”的信号太弱。下一步更合理的是收集 in-domain autonomous rollouts 或 in-domain corrections：

- 成功 episode 标 positive。
- 失败 episode 标 negative。
- 人工 correction 标 positive。
- 不加入 Nestle/Slam 这类 OOD 物体，仍满足“不把 OOD 加进训练数据”的约束。

有了成功/失败混合数据后，可以继续用 `make_recap_lite_advantages.py --dataset-type rollout --label-mode success-column` 做轻量版，也可以切 full RECAP：`compute_returns.py -> value SFT -> compute_advantages.py -> CFG training`。

## 5. 与 PPO/proxy-mix 的差异

PPO/proxy-mix 是在线 RL 更新，reward/advantage 噪声、KL 约束和 logprob 近似都可能直接推坏 SFT 已经学到的 pick/place 行为。

RECAP-lite 当前是离线监督训练：

- 不重采样 OOD。
- 不用 PPO ratio。
- 不用在线 reward 梯度。
- 训练目标仍是 flow/action matching。
- 通过 `Advantage: positive` 和 unconditional dropout 引入可控的 CFG 分支。

因此它更适合作为 proxy-mix 变差后的下一条保守路线。
