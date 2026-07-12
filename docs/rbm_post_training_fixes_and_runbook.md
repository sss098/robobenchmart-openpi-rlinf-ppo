# RoboBenchMart PickToBasket 后训练修复与执行手册

本文是本次代码审查、修复和后续实验的唯一执行基准。旧文档用于保留历史实现过程；参数和命令以本文为准。

## 1. 结论

此前 KL-PPO、proxy-mix PPO 和 RECAP-lite 结果变差，不能简单解释成“大模型 RL 先变差再变好”。机器人策略的在线 PPO 一旦偏离 SFT 分布，后续采样会继续来自变差后的策略，退化可能自我强化，并不保证恢复。

本次发现的主要工程问题足以污染已有结论：

1. 多 pipeline stage 共用了相同通信 key。3/4 个 stage 的 observation、action 和 rollout result 可能错配。
2. YAML 中的 `max_episode_steps: 600` 没有真正施加到环境，日志因此可能长期为 `env/num_trajectories=0`。
3. 改成 50 步 action chunk 后，训练仍设置每 epoch 40 步，整数除法得到 0 个有效 chunk。
4. chunk 内某个环境结束后仍累计后续 reward/done，成功奖励还可能重复发放。
5. shaped reward 的 basket potential 从 0 开始，reset 后第一步可能凭空得到正奖励。
6. PPO 的 KL 原先是 log-prob MSE、系数仅 `0.001`，约束很弱且不易解释。
7. SFT co-train 虽计算了 loss，但调用方没有接住返回值，实际上没有加入反向传播。
8. PPO actor 没有先让随机初始化的 value head 学习；负 explained variance 会直接污染 advantage。
9. RECAP 训练和评测曾使用 10/5，SFT 基线使用 50/10，比较协议不一致。
10. 仅使用全 positive SFT 数据的 RECAP-lite 没有偏好对比信号，本质是带新 prompt 的继续 SFT，而不是完整 RECAP。
11. PPO SFT co-train 的 OpenPI loader 漏传 batch override，实际使用 preset 的 256，而不是 actor micro batch 1；这是 post-warmup OOM 的直接原因。
12. rollout 模型在 actor 更新期间没有卸载，额外常驻约 8 GiB GPU 显存。
13. PPO checkpoint 没有保存 warmup/optimizer step 元数据，warmup 后恢复会按错误的 optimizer 结构加载或再次 warmup。

因此，旧的 PPO/proxy 结果不能用于判断路线本身是否无效。修复后仍不应直接长训，必须按本文的逐级 gate 执行。

路线优先级：

- 第一优先：固定协议复测 SFT BASE，确认通信与评测基线。
- 第二优先：先跑 1 epoch training-path BASE diagnostic。只有三任务至少出现有效成功，才允许保守 KL-PPO。
- 第三优先：有 in-domain 成败 rollout 后做 mixed RECAP-lite。这是最符合“保护 in-domain、提高泛化且不加入 OOD 数据”的路线。
- 暂停 proxy-mix：只有 proxy BASE 自身成功率非零且 reward 与真实任务相关时才恢复。

## 2. 已修改内容

### 2.1 Pipeline stage 通信

修改：

- `rlinf/workers/env/env_worker.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/utils/embodied_training_safety.py`

所有通信 key 现在包含 stage，例如：

```text
train_stage_0_obs
train_stage_0_actions
train_stage_0_rollout_results
train_stage_1_obs
...
```

env 和 rollout 两端共同调用 `stage_channel_extra()`，防止字符串再次漂移。作用是保证 Fanta/Nivea/Stars 或 proxy 的 observation、action、log-prob 和 reward 属于同一 stage。

### 2.2 Episode、chunk 和 bootstrap

修改：

- `rlinf/envs/maniskill/maniskill_env.py`
- `rlinf/workers/env/env_worker.py`

效果：

- 明确按配置的 600 步产生 truncation。
- chunk 内环境第一次 done 后，屏蔽该环境后续 action step 的 reward 和重复 done。
- `bootstrap_type: none` 现在真正不向 terminal/truncated reward 写入 value bootstrap。
- PPO 训练每 epoch 从 40 步改为 600 步；50 步 action chunk 下为 12 个有效 chunk，不再是 0。

### 2.3 PickToBasket reward

修改：

- `rlinf/envs/robobenchmart/robobenchmart_env.py`
- `rlinf/utils/embodied_training_safety.py`

当前 shaped reward：

```text
首次成功                 +5.00
首次放入                 +1.00
首次抬起                 +0.30
basket potential 增量    +0.50 * progress
首次放入且机械臂稳定      +0.30
首次移动非目标物体        -0.20
单步总 reward clamp       [-0.2, 6.0]
```

修复点：success 只奖励一次；reset 时使用真实初始 basket potential，因此第一步没有虚假 progress；所有事件状态按 episode reset。

### 2.4 KL-PPO 和 SFT rehearsal

修改：

- `rlinf/workers/actor/fsdp_actor_worker.py`
- `examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml`
- `examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml`

关键参数：

```yaml
algorithm:
  kl_penalty: low_var_kl
  bootstrap_type: none
  kl_beta: 0.1
  ref_kl_target: 0.01
  kl_beta_min: 1.0e-4
  kl_beta_max: 10.0
  kl_beta_adaptation_rate: 1.5

actor:
  enable_sft_co_train: true
  sft_loss_weight: 0.05
  optim:
    lr: 5.0e-8
    value_lr: 1.0e-5
    critic_warmup_steps: 3
```

作用：

- 用非负、低方差的 k3 KL 近似约束当前策略到冻结 SFT reference。
- KL 超过目标两倍时增大 beta，低于目标一半时减小 beta，并限制上下界。
- 前 3 个 optimizer step 只训练 value head，不计算 KL、不做 actor PPO、不做 SFT rehearsal。
- warmup 后 SFT loss 现在确实加入总 loss，降低遗忘。
- SFT co-train batch 被强制绑定到 `actor.micro_batch_size`，并有运行时断言，不再继承 OpenPI preset 的 256。
- rollout 每轮生成后卸载到 CPU，为 PPO + SFT 反向释放 GPU 显存。
- checkpoint 写入 `actor/trainer_state.json`；恢复时据此重建正确的 optimizer/scheduler 并恢复 step 计数。
- 训练和基线统一为 action chunk 50、denoise step 10。

### 2.5 RECAP-lite

修改或新增：

- `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py`
- `examples/recap/process/make_recap_lite_advantages.py`
- `examples/recap/cfg/config/rbm_pick_to_basket_recap_lite_cfg_openpi_pi05.yaml`
- `scripts_local/10_rbm_recap_prepare_advantages.sh`
- `scripts_local/11_rbm_recap_cfg_sft.sh`
- `scripts_local/11_rbm_recap_cfg_mixed.sh`
- `scripts_local/12_rbm_recap_eval_checkpoint.sh`

效果：

- RBM CFG 支持三视角输入。
- positive 和 negative 样本分别进入对应 guidance prompt；两类样本都支持 unconditional dropout。
- success-column 标签按完整 episode 广播；支持 episode 跨 parquet shard，空 success 不会误判为 true。
- 全 positive 入口只用于通路和遗忘测试。
- mixed 入口默认 70% SFT positive + 30% in-domain rollout，且强制 rollout 同时含正负标签。
- RECAP 训练/评测统一使用 50/10。

### 2.6 统一评测和测试

新增：

- `scripts_local/08_rbm_eval_matched.sh`
- `scripts_local/13_rbm_post_training_preflight.sh`
- `tests/unit_tests/test_rbm_post_training_safety.py`

固定评测协议：每个任务 1 个并行环境，每任务 30 个历史 JAX seed，最大 600 步，关闭 auto-reset，固定 PyTorch rollout RNG seed，50/10。OpenPI flow policy 仍使用高斯噪声采样，但相同代码和 seed 下可重复。train3 共 90 回合，OOD 共 60 回合，proxy 共 30 回合。

## 3. 执行前准备

所有命令从仓库根目录运行：

```bash
cd /root/autodl-tmp/projects/RLinf
```

默认路径为：

```text
RLinf:         /root/autodl-tmp/projects/RLinf
RoboBenchMart: /root/autodl-tmp/projects/RoboBenchMart
OpenPI:        /root/autodl-tmp/projects/openpi
SFT model:     /root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch
SFT dataset:   /root/autodl-tmp/lerobot_data/rbm_dataset
```

路径不同时，在命令前设置 `PROJECT`、`RBM_PROJECT`、`OPENPI_PROJECT`、`MODEL_PATH` 或 `BASE_MODEL_PATH`。

先清理残留 Ray 并运行预检：

```bash
ray stop --force || true
bash scripts_local/13_rbm_post_training_preflight.sh
```

只有看到以下内容才继续：

```text
10 passed
RBM post-training preflight passed.
```

## 4. 第一步：重新建立 SFT BASE

不要把旧评测数字和新代码直接比较。先用修复后的通信、horizon 和相同协议重新评测 BASE。

### 4.1 In-domain 三任务

```bash
bash scripts_local/08_rbm_eval_matched.sh BASE
```

### 4.2 OOD 两任务

```bash
EVAL_SPLIT=ood bash scripts_local/08_rbm_eval_matched.sh BASE
```

### 4.3 Proxy 基线

```bash
EVAL_SPLIT=proxy bash scripts_local/08_rbm_eval_matched.sh BASE
```

每条命令最后会打印 log 路径。日志中关注 `eval/success`、`eval/success_at_end` 和 `eval/num_trajectories`，具体 success key 取决于 RBM env 返回字段。每个真实任务必须正好有 30 条 trajectory。

查看 TensorBoard：

```bash
source .venv_openpi/bin/activate
tensorboard --logdir logs --bind_all --port 6006
```

浏览器访问 `http://服务器IP:6006`。如端口被占用，把 6006 改为 6007。

把结果记录为：

| checkpoint | Fanta/30 | Nivea/30 | Stars/30 | Nestle/30 | Slam/30 | Proxy/30 |
|---|---:|---:|---:|---:|---:|---:|
| fixed SFT BASE | | | | | | |

如果修复后的 BASE 与原 22/21/24 差异很大，先查看保存视频和逐 stage 日志，不进入训练。

## 5. 第二步：KL-PPO gate 与 smoke

### 5.1 先跑 training-path BASE diagnostic

```bash
ray stop --force || true
BASELINE_DIAGNOSTIC=1 MAX_EPOCHS=1 SAVE_INTERVAL=1 \
EXPERIMENT_NAME=rbm_train_path_base_diagnostic \
bash scripts_local/06_rbm_ppo_smoke.sh
```

当前是 Fanta/Nivea/Stars 各 1 个 CPU env，共 3 条固定 seed、每条最多 600 action step。必须在日志中看到三条独立的 `success_once/<task>` 和 `num_trajectories=3`。

当前实测 diagnostic 为三任务 `0/3`。因此脚本默认阻止 actor PPO；这不是运行错误，而是效果 gate。不要直接跑 6 epoch。只有扩大 BASE diagnostic 的固定 seed 样本后确认训练路径确实有成功，才显式设置 `ALLOW_ACTOR_PPO=1`。

另开终端看日志：

```bash
cd /root/autodl-tmp/projects/RLinf
PPO_RUN=$(ls -td logs/*-rbm_pick_to_basket_ppo_openpi_pi05 | head -1)
tail -f "$PPO_RUN/run_embodiment.log"
```

找 checkpoint：

```bash
PPO_RUN=$(ls -td logs/*-rbm_pick_to_basket_ppo_openpi_pi05 | head -1)
find "$PPO_RUN" -type f -name full_weights.pt -print
```

### 5.2 训练健康条件

必须同时满足：

1. `env/num_trajectories` 必须是 3，且三个 task-specific metric 都存在。
2. reward、advantage、value loss、grad norm 都是有限数，不出现 NaN/Inf。
3. 前 3 optimizer step 的 actor policy loss 为 0；这段只训练 critic。
4. `train/critic/explained_variance` 从负值向 0 或正值移动。若持续明显为负，停止。
5. warmup 后才出现 `train/actor/ref_kl_loss`、`ref_kl_beta` 和 `sft_loss`。
6. reference KL 目标约 0.01；若持续高于 0.02 且 beta 已连续增长，停止。
7. clip fraction 不应长期接近 1，actor grad norm 不应爆炸。

### 5.3 分 checkpoint 评测

warmup 按 optimizer step 而不是 epoch 计数。当前每 epoch 有 3 个 optimizer step，所以第 1 epoch 是 critic-only，`global_step_1` 是重要控制组：

```bash
CKPT=/把上一步找到的/global_step_1
bash scripts_local/08_rbm_eval_matched.sh "$CKPT"
```

工程级 actor-update smoke（不是效果实验）需要显式确认 gate：

```bash
CKPT=/path/to/global_step_1
ALLOW_ACTOR_PPO=1 ALLOW_RESUME=1 MAX_EPOCHS=2 \
EXPERIMENT_NAME=rbm_actor_update_runtime_smoke \
bash scripts_local/06_rbm_ppo_smoke.sh runner.resume_dir="$CKPT"
```

通过条件：

- `global_step_1` 与 fixed SFT BASE 基本一致；否则说明 checkpoint 加载、value head 或评测仍有问题。
- resume checkpoint 必须含 `actor/trainer_state.json`，否则脚本拒绝恢复。
- 30 次/任务的 95% 不确定度通常约为正负 10 个百分点，优先看相同 seed 的逐回合配对变化，不把 1 到 2 次成功差异解释成提升。

只有 BASE gate 通过、critic explained variance 改善且 actor smoke 的 KL/梯度有限，才考虑正式小跑。首次最多 2 个 actor epoch，并逐 checkpoint 评测：

```bash
ALLOW_ACTOR_PPO=1 MAX_EPOCHS=3 SAVE_INTERVAL=1 \
EXPERIMENT_NAME=rbm_pick_to_basket_ppo_stage2 \
bash scripts_local/06_rbm_ppo_smoke.sh
```

若第 3 epoch 已明显下降，不要期待长训自动恢复，停止并转 mixed RECAP-lite。

## 6. Proxy-mix PPO

脚本默认拒绝启动，避免误跑。先看第 4.3 节的 proxy BASE。

以下任一情况都不要继续 proxy-mix：

- BASE 在 proxy 上只有 0 到 2/30 成功，采样几乎全是失败。
- proxy shaped reward 提升但真实 Fanta/Nivea/Stars 成功率下降。
- proxy 的视觉或动力学扰动与目标 OOD 泛化没有明确关系。

满足条件后仅跑 3 epoch：

```bash
ALLOW_PROXY_MIX=1 MAX_EPOCHS=3 \
EXPERIMENT_NAME=rbm_proxy_mix_ppo_safety_smoke \
bash scripts_local/07_rbm_ppo_proxy_mix.sh
```

评测仍使用同一个脚本：

```bash
bash scripts_local/08_rbm_eval_matched.sh /path/to/proxy/global_step_3
EVAL_SPLIT=proxy bash scripts_local/08_rbm_eval_matched.sh /path/to/proxy/global_step_3
```

proxy 成功率上升但 train3 下降，说明 proxy reward 与主任务目标不对齐，该路线应停止。

## 7. RECAP-lite：先做零步控制

CFG wrapper 本身可能改变输出，所以先测零训练 checkpoint。

原 prompt/no-guide 控制：

```bash
GUIDANCE_TYPE=no_guide \
bash scripts_local/12_rbm_recap_eval_checkpoint.sh BASE
```

未训练 positive prompt 控制：

```bash
GUIDANCE_TYPE=positive GUIDANCE_SCALE=1.0 \
bash scripts_local/12_rbm_recap_eval_checkpoint.sh BASE
```

`no_guide BASE` 应接近第 4 节 fixed SFT BASE。若不接近，不训练 RECAP，先检查 CFG checkpoint/model wrapper 加载。

## 8. RECAP-lite：全 positive 通路实验

生成 sidecar：

```bash
DATASET_PATH=/root/autodl-tmp/lerobot_data/rbm_dataset \
TAG=rbm_recap_lite_sft_pos \
DATASET_TYPE=sft LABEL_MODE=all-positive \
bash scripts_local/10_rbm_recap_prepare_advantages.sh
```

先训练 20 step，每 5 step 保存；默认已固定 micro batch 1、global batch 4、loader worker 0：

```bash
MAX_STEPS=20 SAVE_INTERVAL=5 LR=2.0e-7 \
TAG=rbm_recap_lite_sft_pos \
bash scripts_local/11_rbm_recap_cfg_sft.sh
```

找 checkpoint：

```bash
RECAP_RUN=$(ls -td logs/cfg_sft/rbm_pick_to_basket_recap_lite_cfg_openpi_pi05-* | head -1)
find "$RECAP_RUN" -type f -name full_weights.pt -print
```

依次评测 step 5/10/15/20，先只看 train3：

```bash
bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_5
bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_10
```

只有 train3 不掉点的 checkpoint 才评测 OOD：

```bash
EVAL_SPLIT=ood \
bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_10
```

这一实验的目标是验证 CFG 通路和遗忘程度，不应期待稳定 OOD 提升。全 positive 数据没有告诉模型“哪些自主行为是坏的”。

## 9. RECAP-lite：推荐的 in-domain mixed 路线

### 9.1 数据要求

收集的自主 rollout 只能来自 Fanta/Nivea/Stars 及其允许的 in-domain 随机化，不能包含 Nestle/Slam。

rollout LeRobot 数据至少需要：

```text
episode_index
frame_index
image
wrist_image
extra_image
state
actions
task_index/prompt
is_success 或 success 或 episode_success
```

success 可只在成功帧为 true，脚本会按 episode 的任意成功帧广播到整条轨迹。人工 correction 可增加 `is_correction=true`，对应帧会强制标 positive。

### 9.2 采集 in-domain rollout

采集使用普通 PyTorch SFT BASE，只运行 Fanta/Nivea/Stars，保存成功和失败 episode。三个 pipeline stage 会分别写为 `rank_0/id_0`、`rank_1/id_0`、`rank_2/id_0`，后续脚本会自动发现这些 shard，无需手工合并。

LeRobot episode 使用同步 commit：每条 episode 的 parquet 和 metadata 完整落盘后评测才继续。不能改回后台写入，否则 Ray 在评测返回时可能终止仍在写盘的 actor，只留下部分 task shard。

先用 1 episode/task 做格式 smoke。必须使用一个新的空输出目录：

```bash
cd /root/autodl-tmp/projects/RLinf
ray stop --force || true

OUTPUT_ROOT=/root/autodl-tmp/lerobot_data/rbm_indomain_rollouts_smoke \
EVAL_EPOCHS=1 \
bash scripts_local/09_rbm_collect_indomain_rollouts.sh
```

确认输出包含 3 个 `meta/info.json`：

```bash
find /root/autodl-tmp/lerobot_data/rbm_indomain_rollouts_smoke \
  -type f -path '*/meta/info.json' -print
```

smoke 通过后正式收集 30 episode/task。不要复用 smoke 目录：

```bash
OUTPUT_ROOT=/root/autodl-tmp/lerobot_data/rbm_indomain_rollouts \
EVAL_EPOCHS=30 \
bash scripts_local/09_rbm_collect_indomain_rollouts.sh
```

脚本拒绝覆盖非空目录。若采集中断，保留现场排错，并换一个新目录重跑；不要把不完整 shard 加入训练。

### 9.3 为 SFT 和所有 rollout shard 生成相同 tag

下面的 `ROLLOUT` 路径是已经完成采集和 LeRobot 导出的输入，不是由 advantage 脚本自动创建。`10_rbm_recap_prepare_advantages.sh` 只生成标签 sidecar；若 `data/`、`meta/` 或 parquet 不存在，它会立即停止。不要手工创建空目录。

```bash
TAG=rbm_recap_indomain_v1
SFT=/root/autodl-tmp/lerobot_data/rbm_dataset
ROLLOUT=/root/autodl-tmp/lerobot_data/rbm_indomain_rollouts

DATASET_PATH="$SFT" TAG="$TAG" \
DATASET_TYPE=sft LABEL_MODE=all-positive \
bash scripts_local/10_rbm_recap_prepare_advantages.sh

DATASET_PATH="$ROLLOUT" TAG="$TAG" \
DATASET_TYPE=rollout LABEL_MODE=success-column \
bash scripts_local/10_rbm_recap_prepare_advantages.sh
```

当 `ROLLOUT` 是采集根目录时，脚本会自动遍历其下所有 `rank_*/id_*` LeRobot shard。

检查 rollout 标签分布：

```bash
source .venv_openpi/bin/activate
ROLLOUT="$ROLLOUT" TAG="$TAG" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["ROLLOUT"])
paths = sorted(root.glob(f"**/meta/advantages_{os.environ['TAG']}.parquet"))
if not paths:
    raise SystemExit("no rollout advantage sidecars found")
x = pd.concat(
    [pd.read_parquet(p, columns=["advantage"])["advantage"] for p in paths],
    ignore_index=True,
).astype(bool)
print({"positive": int(x.sum()), "negative": int((~x).sum()), "rate": float(x.mean())})
PY
```

正或负任一类为 0 都不能训练 mixed RECAP。

### 9.4 Mixed 配置预检与保守训练

先做不加载模型、不使用 GPU 的输入预检：

```bash
CHECK_ONLY=1 \
SFT_DATASET="$SFT" ROLLOUT_DATASET="$ROLLOUT" TAG="$TAG" \
bash scripts_local/11_rbm_recap_cfg_mixed.sh
```

必须看到 `Mixed RECAP preflight passed.`，并确认列出的数据源为 1 个 SFT 加全部 rollout shard。

正式 mixed 默认要求每个 task shard 至少 10 条 episode，并拒绝任何短于 50 帧的碎片。`ALLOW_SMALL_ROLLOUT_DATASET=1` 只允许用于工程 smoke，不能用于效果训练。

预检还会强制检查：每个 task shard 同时包含成功/失败 episode；RBM 第三视角原始 LeRobot 列必须为 `extra_image`。早期采集数据若使用了错误的 `extra_view_image`，无需重采，执行：

```bash
python scripts_local/09_rbm_repair_rollout_schema.py \
  /root/autodl-tmp/lerobot_data/rbm_indomain_rollouts
```

```bash
SFT_DATASET="$SFT" \
ROLLOUT_DATASET="$ROLLOUT" \
TAG="$TAG" \
SFT_WEIGHT=0.7 ROLLOUT_WEIGHT=0.3 \
MAX_STEPS=20 SAVE_INTERVAL=5 LR=2.0e-7 \
bash scripts_local/11_rbm_recap_cfg_mixed.sh
```

这里显式设置 `data.balance_dataset_weights=False`。SFT 总权重为 0.7，rollout 总权重为 0.3；若存在三个 rollout shard，脚本会自动为每个分配 0.1，不会把总 rollout 权重错误放大为 0.9。

评测顺序仍为：zero-step control -> train3 -> OOD。先选 train3 不下降的最佳 checkpoint，再看 OOD，不按 OOD 测试集反复调参。

## 10. Stop/Go 决策表

| 路线 | 继续条件 | 立即停止条件 |
|---|---|---|
| fixed BASE | 每任务 30 trajectories，结果可复现 | stage 数量/trajectory 数不对、no-guide 不匹配 |
| KL-PPO critic warmup | explained variance 改善，step 2 actor 不变 | value 持续负 EV、step 2 已掉点 |
| KL-PPO actor | KL 受控，step 3 train3 不坍塌 | KL 持续 >0.02、单任务明显掉点 |
| proxy-mix | proxy BASE 有有效成功且 train3 同步改善 | proxy 上升但 train3 下降 |
| all-positive RECAP | no-guide 对齐，短训 train3 不掉 | zero-step 已不一致或短训快速遗忘 |
| mixed RECAP | in-domain rollout 同时有正负，train3 保持 | 只有单一标签、负样本压倒性且无 correction |

## 11. 本次验证结果

已在当前环境完成：

```text
Ruff:                 passed
Python py_compile:    passed
Bash syntax:          passed
Hydra compose:        PPO / proxy / RECAP passed
Unit tests:           10 passed
SFT sidecar:          307319 frames, 100% positive, written successfully
Preflight:            passed
RECAP CFG-SFT:        real GPU step 1 + checkpoint save passed
RECAP BASE eval:      real GPU rollout passed (1 seed/task, 2/3)
RECAP step-1 eval:    checkpoint load + rollout passed (1 seed/task, 2/3)
PPO critic warmup:    rollout + 3 optimizer steps + checkpoint passed
PPO resume:           trainer metadata + optimizer restore passed
PPO actor update:     ref KL + PPO + SFT + optimizer + step-2 save passed
```

单元测试覆盖：pipeline key 隔离、600 步 truncation、done 后 reward 屏蔽、初始 potential 无奖励、成功 reward 一次且有界、CFG 正负路由、KL beta 自适应、跨 parquet episode 成功标签广播。

PPO actor smoke 的实测关键指标：`actor/lr=5e-8`、`ref_kl_loss=0.156`、`ref_kl_beta_next=0.15`、`sft_loss=6.06e-4`，无 OOM 并成功保存。注意该轮三任务仍为 `0/3`；它只证明代码链路可运行，不证明继续 PPO 会改善效果。当前不建议直接跑 6 epoch。

本次产生的 v3/v5/step-2 PPO 诊断 checkpoint 和 RECAP step-1 工程 checkpoint 已在验证后删除，回收约 68 GiB；SFT BASE、日志和 TensorBoard 记录保留。

2026-07-12 真实采集 smoke 已验证：三个 task shard 各 1 条完整 episode且无碎片。正式 90 episode 数据为 Fanta 15/15、Nivea 23/7、Stars 21/9 成败；第三视角 schema 迁移后，全 90 条 parquet 统一为 `extra_image`。

正式 mixed RECAP 已用上述数据完整跑通 20 step，覆盖 negative rollout transform、正负 CFG 路由、反向、第 5/10/15/20 步四次保存和正常退出。已验证产物在：

```text
logs/cfg_sft/rbm_pick_to_basket_recap_lite_cfg_openpi_pi05-20260712-02:49:32/rbm_recap_mixed_formal_verified/checkpoints
```

四个 checkpoint 都包含约 8.5 GB 模型权重、3.2 GB optimizer state、scheduler 和 trainer state。完整训练段约 1 分 51 秒，加上模型/数据初始化后总墙钟时间约 3 分钟。这只证明链路可靠，不等于第 20 步效果最好；必须使用同一批 fixed seeds 分别评测 step 5/10/15/20，先按 train3 不下降选点，再做 OOD 一次确认。

## 12. 2026-07-12 效果回归后的训练安全修复

旧 mixed RECAP step 5 的正式结果为 Fanta 10/30、Nivea 23/30、Stars
19/30，总计 52/90，低于 SFT BASE 的 59/90。旧 PPO 与旧 mixed RECAP
checkpoint 均不得续训。

### 12.1 Mixed RECAP

采样已改为确定性配额：先按固定周期选择数据集和标签，再等概率选 episode，
最后在 episode 内选帧，避免 600 帧失败轨迹压倒提前成功的短轨迹。默认
5 step × global batch 12 = 60 个样本：54 个 SFT，三个 rollout task 各两个，
并强制每个 task 各一个成功 episode 帧和一个失败 episode 帧。保守默认值为
SFT/rollout=0.9/0.1、LR=5e-8、5 step、每 step 保存。

```bash
cd /root/autodl-tmp/projects/RLinf
CHECK_ONLY=1 \
SFT_DATASET=/root/autodl-tmp/lerobot_data/rbm_dataset \
ROLLOUT_DATASET=/root/autodl-tmp/lerobot_data/rbm_indomain_rollouts \
TAG=rbm_recap_indomain_v1 \
bash scripts_local/11_rbm_recap_cfg_mixed.sh
```

GPU smoke/短训必须从 SFT BASE 重新开始：

```bash
SFT_DATASET=/root/autodl-tmp/lerobot_data/rbm_dataset \
ROLLOUT_DATASET=/root/autodl-tmp/lerobot_data/rbm_indomain_rollouts \
TAG=rbm_recap_indomain_v1 \
bash scripts_local/11_rbm_recap_cfg_mixed.sh \
  runner.logger.experiment_name=rbm_recap_quota_balanced_v3
```

不得单独修改 `MAX_STEPS`、`GLOBAL_BATCH_SIZE`、`QUOTA_CYCLE_SIZE` 或数据权重。
脚本会要求 `MAX_STEPS * GLOBAL_BATCH_SIZE` 是 quota cycle 的整数倍，并要求每个
rollout task 至少有两个槽位，否则在加载模型前退出。

### 12.2 KL-PPO

每个 runner epoch 现在采集每任务 4 条、合计 12 条轨迹。第一轮的 12 次
optimizer update 只训练 value head；之后每轮成功轨迹少于 2 条时强制 critic-only，
不会更新 actor。过程奖励已降为弱信号，单次成功事件仍为 5.0。

reference/current logprob 使用相同 train/eval 模式。首次 actor update 前强制检查
reference KL 零漂移，`ref_kl_abs > 0.02` 会立即停止，而不是带着有偏 KL 继续烧卡。
SFT co-train 权重从 0.05 提高到 1.0。

```bash
cd /root/autodl-tmp/projects/RLinf
ALLOW_ACTOR_PPO=1 MAX_EPOCHS=2 SAVE_INTERVAL=1 \
EXPERIMENT_NAME=rbm_kl_ppo_guarded_smoke_v2 \
bash scripts_local/06_rbm_ppo_smoke.sh
```

第 1 epoch 应看到 `actor/update_blocked_no_success` 或纯 critic warmup；第 2 epoch
只有满足成功门槛才允许 actor。首次 actor 更新必须通过 KL 零漂移断言。若
`critic/explained_variance` 仍为 NaN/负值，或连续两轮没有至少 2 条成功轨迹，停止
PPO，不得通过关闭门控强行训练。

2026-07-12 根因复核：`joint_logprob=True` 会让 train policy 在全部 10 个去噪步
注入 `flow_noise`，而 eval policy 全部走 `flow_ode`。实测两轮训练策略仅 0/12、
1/12 成功，因此 critic 接收到的并不是评测时 59/90 的策略分布。配置已固定为
`joint_logprob=False`：每个 action chunk 仅随机一个去噪步探索，其余九步走 ODE。

### 12.3 单噪声 PPO 实测与恢复语义

单噪声基线轮实测为 4/12 成功，随后从 critic-only checkpoint 恢复的首个 actor
轮为 5/12 成功。恢复流程已修正：checkpoint 记录 `actor_update_count`；若恢复点尚未
发生 actor 更新，则在加载 critic-only 权重后重新冻结 reference；若恢复点已经发生
actor 更新，则继续保留原始 SFT reference。这样不会再把“恢复前模型”误当 reference
产生伪 KL，也不会在真正 PPO 续训时漂移 reference。

首个真实 actor update 已完整验证并保存到：

```text
logs/20260712-13:13:03-rbm_pick_to_basket_ppo_openpi_pi05/
rbm_ppo_single_noise_first_actor_v3/checkpoints/global_step_2
```

关键指标：5/12 success，`update_blocked_no_success=0`，`policy_loss=-0.194`，
`ref_kl_abs=0.0012`，`approx_kl=7.68e-4`，`clip_fraction=0.0069`，checkpoint
中的 `actor_update_count=12`。因此单噪声 rollout、critic warmup、reference 恢复、
actor 更新与保存链路均已实际跑通。

`critic/explained_variance` 现在统一按完整 rollout、更新前的 values/returns 计算，
不再把 micro batch size 1 的 NaN 当成 critic 结论。SFT/PPO loss ratio 也在整轮
聚合后计算；单个 advantage 为零的微批次不会再产生误导性的 `inf` 告警。本次完整
rollout EV 为 -0.0054，说明 value head 目前接近常数预测，并未证明已学到可靠排序。

工程链路通过不等于已经证明 PPO 能提高成功率。下一步只比较同一套 30×3 fixed
seeds 下的 critic-only step 1 与首个 actor step 2；没有显著提升就停止 PPO，不从
step 2 继续扩到 6 epoch。Mixed RECAP 已完成配额修复，但其失败条件仍是对失败动作
做条件模仿，且没有任何 OOD 视觉覆盖，因此不再作为当前主路线继续烧卡。
