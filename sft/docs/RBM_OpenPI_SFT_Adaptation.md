# RoboBenchMart PickToBasket OpenPI SFT 适配说明

本文档记录当前机器上为 RoboBenchMart `pick_to_basket` 任务做 OpenPI / pi0.5 SFT 所新增和已有的适配，包括数据转换脚本、OpenPI 配置、字段映射、训练命令和注意事项。

## 1. 当前结论

当前 82G PickToBasket demo 数据已经下载在：

```text
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket/demos
```

其中三份主要训练 H5 是：

```text
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5
/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_stars_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5
```

这些 H5 是 RoboBenchMart 原始 demo 格式，已经包含视觉观测、qpos、action 和语言指令。它们可以作为 SFT 数据源，但不能被 OpenPI `scripts/train.py` 直接读取。OpenPI 官方训练入口读取的是 LeRobot dataset，因此需要先转换成 LeRobot 格式。

## 2. 新增转换脚本

新增脚本：

```text
/root/autodl-tmp/projects/RoboBenchMart/scripts/convert_pick_to_basket_h5_to_lerobot.py
```

作用：把 RoboBenchMart PickToBasket H5 demo 转成 OpenPI 可读的 LeRobot dataset，默认输出 repo id 为：

```text
rbm_dataset
```

### 2.1 字段映射

转换脚本写出的 LeRobot 字段与 OpenPI `LeRobotRBMDataConfig` 对齐：

```text
RoboBenchMart H5                                      LeRobot 字段
--------------------------------------------------------------------------------
obs/sensor_data/left_base_camera_link/rgb             image
obs/sensor_data/fetch_hand/rgb                        wrist_image
obs/sensor_data/right_base_camera_link/rgb            extra_image
obs/agent/qpos                                        state
actions                                               actions
obs/extra/language_instruction_bytes + mask           task
```

OpenPI 之后会通过 `PromptFromLeRobotTask` 把 `task` 转成 `prompt`。

### 2.2 为什么这样映射

RoboBenchMart eval client 发给 OpenPI server 的 observation key 是：

```text
observation/image
observation/extra_image
observation/wrist_image
observation/state
prompt
```

OpenPI RBM data config 会把 LeRobot 字段 repack 成同样的 key：

```text
image        -> observation/image
wrist_image  -> observation/wrist_image
extra_image  -> observation/extra_image
state        -> observation/state
actions      -> actions
prompt       -> prompt
```

这样训练和评测使用同一套 camera / state / action / prompt transform，避免训练和评测不一致。

## 3. 转换命令

建议先做小样本 smoke test，不要直接转换 82G 全量数据。

```bash
cd /root/autodl-tmp/projects/RoboBenchMart

/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python \
  scripts/convert_pick_to_basket_h5_to_lerobot.py \
  --root /root/autodl-tmp/lerobot_data \
  --repo-id rbm_dataset \
  --max-episodes 3 \
  --overwrite
```

如果小样本转换成功，再跑全量转换：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart

/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python \
  scripts/convert_pick_to_basket_h5_to_lerobot.py \
  --root /root/autodl-tmp/lerobot_data \
  --repo-id rbm_dataset \
  --overwrite
```

转换完成后，OpenPI 训练时需要设置：

```bash
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data
```

OpenPI 会在这个目录下查找：

```text
/root/autodl-tmp/lerobot_data/rbm_dataset
```

## 4. OpenPI 已有 RoboBenchMart 适配

### 4.1 RBM policy 输入输出适配

文件：

```text
/root/autodl-tmp/projects/openpi/src/openpi/policies/robobenchmart_policy.py
```

关键类：

```text
RBMInputs
RBMOutputs
```

用途：

1. 把 RoboBenchMart observation 转成 OpenPI 模型输入。
2. 把三路 RGB 图像映射到 OpenPI/pi0.5 期望的 image dict。
3. 保留 15 维机器人 qpos state。
4. 传入语言 prompt。
5. 把模型输出 action chunk 裁成 RoboBenchMart 环境需要的 13 维 action。

核心映射：

```python
inputs = {
    "state": data["observation/state"],
    "image": {
        "base_0_rgb": base_image,
        "left_wrist_0_rgb": wrist_image,
        "right_wrist_0_rgb": extra_image,
    },
    "image_mask": {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    },
}
```

模型输出裁剪：

```python
return {"actions": np.asarray(data["actions"][:, :13])}
```

这一步非常关键，因为 pi0.5 内部动作维度是 pad 后的 32 维，而 RoboBenchMart `pd_joint_pos` 控制只需要前 13 维。

### 4.2 RBM LeRobot data config

文件：

```text
/root/autodl-tmp/projects/openpi/src/openpi/training/config.py
```

关键类：

```text
LeRobotRBMDataConfig
```

用途：把 LeRobot dataset 中的字段转成 OpenPI RBM policy 使用的字段。

repack 映射：

```python
{
    "observation/image": "image",
    "observation/wrist_image": "wrist_image",
    "observation/extra_image": "extra_image",
    "observation/state": "state",
    "actions": "actions",
    "prompt": "prompt",
}
```

随后接上：

```text
RBMInputs
Normalize
ResizeImages
TokenizePrompt
PadStatesAndActions
```

也就是说，LeRobot 数据经过这套 config 后，训练时看到的输入格式会和评测时 OpenPI server 收到的格式一致。

## 5. OpenPI 已有 eval config

文件：

```text
/root/autodl-tmp/projects/openpi/src/openpi/training/config.py
```

已有配置：

```text
pi0_eval_rbm
pi05_eval_rbm
```

用途：启动 OpenPI policy server 做 RoboBenchMart 官方 checkpoint 推理评测。

启动 pi0.5 官方 checkpoint server 的命令：

```bash
cd /root/autodl-tmp/projects/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_eval_rbm \
  --policy.dir=/root/autodl-tmp/projects/RoboBenchMart/models/pi05 \
  --port 8000
```

注意：server 端口是 `8000`，不要误用 `8080`。

## 6. 新增 OpenPI SFT config

新增配置位于：

```text
/root/autodl-tmp/projects/openpi/src/openpi/training/config.py
```

配置名：

```text
pi05_sft_rbm_pick_to_basket
```

用途：从官方 RoboBenchMart pi0.5 checkpoint 继续 SFT 训练 PickToBasket。

核心内容：

```python
TrainConfig(
    name="pi05_sft_rbm_pick_to_basket",
    model=pi0_config.Pi0Config(pi05=True, discrete_state_input=False),
    data=LeRobotRBMDataConfig(
        repo_id="rbm_dataset",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="/root/autodl-tmp/projects/RoboBenchMart/models/pi05/assets",
            asset_id="rbm_dataset",
        ),
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/root/autodl-tmp/projects/RoboBenchMart/models/pi05/params"
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-5,
        decay_steps=1_000_000,
        decay_lr=1e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    batch_size=32,
    num_train_steps=20_000,
    save_interval=2_000,
    keep_period=5_000,
)
```

### 6.1 为什么从官方 checkpoint 继续训练

官方 pi0.5 RBM checkpoint 已经适配过 RoboBenchMart 的视觉、动作、相机、prompt 和环境分布。从它继续做 SFT，比从 `pi0.5 base` 开始更稳，也更适合做公平对比。

推荐对比方式：

```text
baseline: 官方 pi0.5 checkpoint，直接评测
SFT:      官方 pi0.5 checkpoint -> PickToBasket SFT -> 同一批 seed 评测
```

### 6.2 为什么复用官方 norm stats

新增 SFT config 指向：

```text
/root/autodl-tmp/projects/RoboBenchMart/models/pi05/assets/rbm_dataset/norm_stats.json
```

原因：

1. 官方 checkpoint 就是用这套 RBM norm stats 训练/归一化的。
2. PickToBasket demo 的 action/state 维度和官方 RBM 数据分布一致。
3. 先复用官方 norm stats 可以避免因为重新统计 norm 导致输入/动作尺度变化，从而影响和官方 baseline 的公平对比。

如果后续你混入新任务、新机器人、新 action space，才需要重新考虑 norm stats。

## 7. OpenPI SFT 训练命令

全量转换完成后，运行：

```bash
cd /root/autodl-tmp/projects/openpi
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data

mkdir -p logs

uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name=pick_to_basket_sft_from_official \
  2>&1 | tee logs/pick_to_basket_sft_from_official.log
```

默认输出 checkpoint 目录类似：

```text
/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_from_official/<step>/
```

如果要覆盖默认步数或 batch size，可以在命令行加参数，例如：

```bash
uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name=pick_to_basket_sft_debug \
  --num-train-steps=500 \
  --batch-size=16 \
  --save-interval=250
```

## 8. SFT 后评测方式

训练出 checkpoint 后，用同一个 `pi05_eval_rbm` config 启动 server，只是 `--policy.dir` 换成 SFT checkpoint 目录。

示例：

```bash
cd /root/autodl-tmp/projects/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_eval_rbm \
  --policy.dir=/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_from_official/<step> \
  --port 8000
```

然后在 RoboBenchMart 侧用同一个 eval script 和同一批 seed 评测。

训练分布内任务：

```text
fanta_train
nivea_train
stars_train
```

OOD 任务：

```text
nestle_ood
slam_ood
```

固定公平对比设置：

```text
同一批 seed
同一 eval script
同一 max_horizon=600
同一 env 修复版本
同一 camera/prompt/action transform
同一 num_traj=30
```

已有官方 baseline：

```text
Fanta:      13/30 = 43.3%
Nivea:      17/30 = 56.7%
Stars:      13/30 = 43.3%
Nestle OOD:  4/30 = 13.3%
Slam OOD:    1/30 = 3.3%
```

## 9. 当前已做的检查

已经完成：

```text
python -m py_compile scripts/convert_pick_to_basket_h5_to_lerobot.py
python -m py_compile openpi/src/openpi/training/config.py
python -m py_compile openpi/src/openpi/policies/robobenchmart_policy.py
```

结果：语法检查通过。

未完成：

```text
全量 LeRobot 转换
OpenPI data loader smoke test
实际 SFT 训练
SFT checkpoint 评测
```

原因：当前机器未开 GPU，且部分 OpenPI/LeRobot 完整 import 在当前资源状态下会被系统 kill。建议等 GPU 打开后，先做小样本转换和训练 data loader smoke test，再跑全量转换和训练。

## 10. 后续执行 checklist

1. 小样本转换：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart
/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python \
  scripts/convert_pick_to_basket_h5_to_lerobot.py \
  --root /root/autodl-tmp/lerobot_data \
  --repo-id rbm_dataset \
  --max-episodes 3 \
  --overwrite
```

2. 确认 LeRobot dataset 能被 OpenPI 读取。

3. 全量转换。

4. 运行 OpenPI SFT：

```bash
cd /root/autodl-tmp/projects/openpi
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data
uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name=pick_to_basket_sft_from_official
```

5. 启动 SFT checkpoint server。

6. 用 RoboBenchMart `scripts/eval_policy_client.py` 跑同一批 seed 的 5 项对比。

## 11. 注意事项

- 不需要重新跑 `run_mp_all.sh` 或 `replay.sh`，因为 PickToBasket 三份带视觉观测的 H5 已经存在。
- 不建议频繁评测时加 `--save-traj`，H5 很大；正式 baseline / final model 可以保存。
- 当前 SFT 预期主要提升 `fanta/nivea/stars` seen train items；`nestle/slam` OOD 不保证提升，可能下降，需要单独监控。
- 如果后续更换数据分布、加入新任务或新机器人，再考虑重新生成 norm stats；当前从官方 RBM checkpoint 继续 SFT，优先复用官方 norm stats。

## 12. 运行指令速查

本节把实际执行时最常用的命令集中列出来。

### 12.1 环境变量

RoboBenchMart 评测端使用 RLinf 的 `.venv_openpi`：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart
source /root/autodl-tmp/projects/RLinf/.venv_openpi/bin/activate
```

OpenPI 训练和推理 server 在 OpenPI 仓库执行：

```bash
cd /root/autodl-tmp/projects/openpi
```

LeRobot 数据根目录建议固定为：

```bash
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data
```

### 12.2 小样本转换 smoke test

先只转 3 条 episode，确认转换脚本和 LeRobot 写盘流程没问题：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart

/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python \
  scripts/convert_pick_to_basket_h5_to_lerobot.py \
  --root /root/autodl-tmp/lerobot_data \
  --repo-id rbm_dataset \
  --max-episodes 3 \
  --overwrite
```

### 12.3 全量转换

小样本没问题后，转换全部 fanta/nivea/stars 训练 demos：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart

/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python \
  scripts/convert_pick_to_basket_h5_to_lerobot.py \
  --root /root/autodl-tmp/lerobot_data \
  --repo-id rbm_dataset \
  --overwrite
```

输出目录：

```text
/root/autodl-tmp/lerobot_data/rbm_dataset
```

### 12.4 OpenPI SFT 训练

从官方 RoboBenchMart pi0.5 checkpoint 继续 SFT：

```bash
cd /root/autodl-tmp/projects/openpi
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data

uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name=pick_to_basket_sft_from_official
```

调试短训版本：

```bash
cd /root/autodl-tmp/projects/openpi
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data

uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name=pick_to_basket_sft_debug \
  --num-train-steps=500 \
  --batch-size=16 \
  --save-interval=250 \
  --wandb-enabled=false
```

正式训练如果显存不足，可以先降 batch size：

```bash
uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name=pick_to_basket_sft_bs16 \
  --batch-size=16
```

### 12.5 启动官方 baseline server

用于复现官方 pi0.5 baseline：

```bash
cd /root/autodl-tmp/projects/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_eval_rbm \
  --policy.dir=/root/autodl-tmp/projects/RoboBenchMart/models/pi05 \
  --port 8000
```

### 12.6 启动 SFT checkpoint server

训练完成后，将 `<STEP>` 替换为实际 checkpoint step。注意 `--policy.dir` 应指向包含 `params/` 和 assets 的 checkpoint 目录。

示例：

```bash
cd /root/autodl-tmp/projects/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_eval_rbm \
  --policy.dir=/root/autodl-tmp/projects/openpi/checkpoints/pi05_sft_rbm_pick_to_basket/pick_to_basket_sft_from_official/<STEP> \
  --port 8000
```

如果 checkpoint 目录结构是：

```text
.../pick_to_basket_sft_from_official/19999/params
```

则 `--policy.dir` 应写：

```text
.../pick_to_basket_sft_from_official/19999
```

### 12.7 评测 smoke test

另开终端，先跑 1 条 Nivea smoke test：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart
source /root/autodl-tmp/projects/RLinf/.venv_openpi/bin/activate

OMP_NUM_THREADS=1 python scripts/eval_policy_client.py \
  --host localhost --port 8000 \
  --scene-dir demo_envs/pick_to_basket \
  --json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_248traj_4workers.json \
  --eval-subdir smoke_pi05_sft_nivea_n1 \
  --max-horizon 600 \
  --num-traj 1 \
  --sim-backend cpu
```

### 12.8 正式 seen train item 评测

Fanta：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart
source /root/autodl-tmp/projects/RLinf/.venv_openpi/bin/activate

OMP_NUM_THREADS=1 python scripts/eval_policy_client.py \
  --host localhost --port 8000 \
  --scene-dir demo_envs/pick_to_basket \
  --json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_248traj_4workers.json \
  --eval-subdir sft_basket_fanta_train_n30 \
  --max-horizon 600 \
  --num-traj 30 \
  --sim-backend cpu \
  --save-video
```

Nivea：

```bash
OMP_NUM_THREADS=1 python scripts/eval_policy_client.py \
  --host localhost --port 8000 \
  --scene-dir demo_envs/pick_to_basket \
  --json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_248traj_4workers.json \
  --eval-subdir sft_basket_nivea_train_n30 \
  --max-horizon 600 \
  --num-traj 30 \
  --sim-backend cpu \
  --save-video
```

Stars：

```bash
OMP_NUM_THREADS=1 python scripts/eval_policy_client.py \
  --host localhost --port 8000 \
  --scene-dir demo_envs/pick_to_basket \
  --json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_stars_248traj_4workers.json \
  --eval-subdir sft_basket_stars_train_n30 \
  --max-horizon 600 \
  --num-traj 30 \
  --sim-backend cpu \
  --save-video
```

### 12.9 正式 OOD item 评测

Nestle OOD：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart
source /root/autodl-tmp/projects/RLinf/.venv_openpi/bin/activate

OMP_NUM_THREADS=1 python scripts/eval_policy_client.py \
  --host localhost --port 8000 \
  -e PickToBasketContNestleEnv \
  --scene-dir demo_envs/test_unseen_items_pick_to_basket \
  --robot-init-pose-start-seed 10000 \
  --eval-subdir sft_basket_nestle_ood_n30 \
  --max-horizon 600 \
  --num-traj 30 \
  --sim-backend cpu \
  --start-seed 42000 \
  --save-video
```

Slam OOD：

```bash
OMP_NUM_THREADS=1 python scripts/eval_policy_client.py \
  --host localhost --port 8000 \
  -e PickToBasketContSlamEnv \
  --scene-dir demo_envs/test_unseen_items_pick_to_basket \
  --robot-init-pose-start-seed 10000 \
  --eval-subdir sft_basket_slam_ood_n30 \
  --max-horizon 600 \
  --num-traj 30 \
  --sim-backend cpu \
  --start-seed 42000 \
  --save-video
```

### 12.10 查看评测结果

每个评测目录会生成：

```text
eval_summary.json
```

查看所有 PickToBasket summary：

```bash
find /root/autodl-tmp/projects/RoboBenchMart/demo_envs \
  -path '*/evaluations/*/eval_summary.json' \
  -print
```

快速打印成功率：

```bash
/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python - <<'PY'
import glob, json
for p in glob.glob('/root/autodl-tmp/projects/RoboBenchMart/demo_envs/*pick_to_basket*/evaluations/*/eval_summary.json'):
    with open(p) as f:
        s = json.load(f)
    print(p)
    print(' ', s.get('env_id'), s.get('num_success'), '/', s.get('num_traj'), '=', s.get('success_rate'))
PY
```

### 12.11 重要运行条件

正式训练和评测前必须开 GPU/Vulkan。未开 GPU 时之前已经观察到：

```text
torch.cuda.is_available = False
jax.devices = [CpuDevice(id=0)]
SAPIEN Vulkan ErrorIncompatibleDriver
```

未开 GPU 时不要跑正式 OpenPI server、RGB 环境评测或 SFT 训练。

## 13. 转换进度条与小样本检查更新

转换脚本已更新，新增：

```text
总 episode 进度条
单条 trajectory frame 进度条
saved episode 输出
Conversion complete 汇总
手写 LeRobot v2.1 parquet/metadata，避免 LeRobot save_episode 在本机被 kill
--max-frames-per-episode debug 参数
```

当前建议先用更快的小样本 smoke test：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart

/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python \
  scripts/convert_pick_to_basket_h5_to_lerobot.py \
  --root /root/autodl-tmp/lerobot_data \
  --repo-id rbm_dataset \
  --max-episodes 1 \
  --max-frames-per-episode 32 \
  --overwrite
```

注意：`--max-frames-per-episode` 只用于快速检查转换链路，不要用于最终 SFT 数据。正式转换必须不加这个参数。

判断转换结果是否可用：

```bash
find /root/autodl-tmp/lerobot_data/rbm_dataset -type f \
  \( -name '*.parquet' -o -name '*.jsonl' -o -name 'info.json' -o -name '*.mp4' \) \
  -printf '%s %p\n' | sort
```

可用的数据集至少应该包含：

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
data/chunk-000/episode_000000.parquet
```

并且 `meta/info.json` 中应显示：

```text
total_episodes > 0
total_frames > 0
total_tasks > 0
```

如果只看到：

```text
images/.../frame_*.png
meta/info.json 里 total_episodes = 0
```

说明转换进程在 `save_episode()` 前中断，数据还不能用于 OpenPI SFT，需要 `--overwrite` 后重跑。


## 14. 最新转换验证与正式运行命令

### 14.1 2026-06-28 转换脚本状态

当前转换脚本已修复为按 episode 写 LeRobot v2.1 parquet，并使用 PyArrow 直接写出与 HuggingFace `datasets.Image()` 相同的 image schema，避免 `Dataset.from_dict()` 造成额外内存复制。

已完成 smoke test：

```text
输出目录: /root/autodl-tmp/lerobot_data_one_ep/rbm_dataset
episodes: 1
frames: 471
parquet: data/chunk-000/episode_000000.parquet
大小: 44 MiB
耗时: 约 3 分 28 秒
```

LeRobot 读取验证通过：

```text
image       -> torch.float32 [3, 256, 256]
wrist_image -> torch.float32 [3, 128, 128]
extra_image -> torch.float32 [3, 256, 256]
state       -> torch.float32 [15]
actions     -> torch.float32 [13]
task        -> move to shelf and pick Fanta Sabor Naranja 2L to basket
```

三份训练 H5 的全量统计：

```text
fanta: 248 episodes, 102720 frames
nivea: 248 episodes, 100891 frames
stars: 248 episodes, 103708 frames
total: 744 episodes, 307319 frames
```

按 JPEG quality 95 的 1 条轨迹实测估算：

```text
全量 LeRobot 数据集: 约 28-30 GiB
CPU 转换耗时: 约 35-40 小时
```

注意：如果使用 `--image-format png`，会更接近无损原始图像，但转换更慢、占用空间更大；当前建议先用默认 `jpeg --jpeg-quality 95` 做 SFT。它是有损压缩，理论上有轻微训练/评测视觉分布差异风险，但在空间和时间上更现实。

### 14.2 正式全量转换命令

建议在 `tmux` 中运行，避免 SSH/终端中断导致转换进程被杀。旧的失败输出可直接用 `--overwrite` 覆盖；之前失败目录没有完整 `info.json` 和 parquet，不可用于训练。

```bash
cd /root/autodl-tmp/projects/RoboBenchMart
mkdir -p logs

tmux new -s rbm_convert
```

进入 tmux 后运行：

```bash
/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python   scripts/convert_pick_to_basket_h5_to_lerobot.py   --root /root/autodl-tmp/lerobot_data   --repo-id rbm_dataset   --image-format jpeg   --jpeg-quality 95   --overwrite 2>&1 | tee logs/convert_rbm_dataset.log
```

查看进度：

```bash
tail -f /root/autodl-tmp/projects/RoboBenchMart/logs/convert_rbm_dataset.log
```

如果不用 tmux，也可以用 nohup：

```bash
cd /root/autodl-tmp/projects/RoboBenchMart
mkdir -p logs

nohup /root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python   scripts/convert_pick_to_basket_h5_to_lerobot.py   --root /root/autodl-tmp/lerobot_data   --repo-id rbm_dataset   --image-format jpeg   --jpeg-quality 95   --overwrite   > logs/convert_rbm_dataset.log 2>&1 &
```

### 14.3 转换完成后验证

```bash
cd /root/autodl-tmp/projects/RoboBenchMart

/root/autodl-tmp/projects/RLinf/.venv_openpi/bin/python - <<'PY'
import os
os.environ['HF_LEROBOT_HOME'] = '/root/autodl-tmp/lerobot_data'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('rbm_dataset')
print('episodes', meta.total_episodes)
print('frames', meta.total_frames)
print('tasks', meta.tasks)
ds = LeRobotDataset('rbm_dataset')
print('len', len(ds))
s = ds[0]
for k in ['image', 'wrist_image', 'extra_image', 'state', 'actions', 'task']:
    v = s[k]
    print(k, type(v), getattr(v, 'shape', None), getattr(v, 'dtype', None), v if k == 'task' else '')
PY
```

期望结果：

```text
episodes = 744
frames = 307319
state shape = [15]
actions shape = [13]
三路图像都能读成 torch Tensor
```
