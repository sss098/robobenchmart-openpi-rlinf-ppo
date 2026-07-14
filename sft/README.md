# RoboBenchMart π0.5 SFT 源码包

本目录收录监督微调阶段的关键自定义代码，解决公开仓库此前只有 SFT 结果、无法审查
数据转换和 OpenPI 配置的问题。

## 文件说明

| 文件 | 用途 |
|---|---|
| `convert_pick_to_basket_h5_to_lerobot.py` | 将 PickToBasket H5 专家数据转换为 LeRobot v2.1 |
| `openpi/robobenchmart_policy.py` | OpenPI 三视角、15 维 state 与 13 维 action adapter |
| `openpi/openpi_rbm_sft_config.patch` | OpenPI data config、eval/SFT config 和 checkpoint 修改 |
| `docs/RBM_OpenPI_SFT_Adaptation.md` | 数据、policy、训练和评测适配过程 |
| `docs/RBM_OpenPI_SFT_20k_Training_Report.md` | 20k-step 参数、日志与 ID/OOD 结果 |

## 数据转换

```bash
python sft/convert_pick_to_basket_h5_to_lerobot.py \
  --root /path/to/lerobot_data \
  --repo-id rbm_dataset \
  --image-format jpeg \
  --jpeg-quality 95 \
  --overwrite
```

输出包含三视角图像、15 维 state、13 维 action、任务文本以及 episode/frame/task
metadata。转换器会检查轨迹长度、相机分辨率、维度和 dtype，并生成 LeRobot v2.1 的
Parquet、`info.json`、`episodes.jsonl`、`episodes_stats.jsonl` 和 `tasks.jsonl`。

## 应用 OpenPI 适配

补丁基于本项目使用的 OpenPI 工作树生成。请先在兼容版本上检查：

```bash
cd /path/to/openpi
git apply --check /path/to/RLinf/sft/openpi/openpi_rbm_sft_config.patch
git apply /path/to/RLinf/sft/openpi/openpi_rbm_sft_config.patch
cp /path/to/RLinf/sft/openpi/robobenchmart_policy.py \
  src/openpi/policies/robobenchmart_policy.py
```

补丁增加 `LeRobotRBMDataConfig`、`pi0_eval_rbm`、`pi05_eval_rbm` 和
`pi05_sft_rbm_pick_to_basket`。SFT 配置使用 π0.5、batch size 32、20,000 steps、AdamW、
gradient clip 1.0、1,000-step warmup 和 `1e-5` learning rate。补丁内的 assets 和初始
checkpoint 绝对路径需要按实际目录修改。

## normalization 与训练

```bash
cd /path/to/openpi
HF_LEROBOT_HOME=/path/to/lerobot_data \
uv run scripts/compute_norm_stats.py --config-name pi05_sft_rbm_pick_to_basket

HF_LEROBOT_HOME=/path/to/lerobot_data \
uv run scripts/train.py pi05_sft_rbm_pick_to_basket \
  --exp-name pick_to_basket_sft_20k_official_ckpt \
  --overwrite
```

训练完成后先使用 `pi05_eval_rbm` 在 JAX 路径评测，再通过 RLinf 的
`convert_openpi_jax_to_python.py` 转为 PyTorch checkpoint。

## 复现边界

本目录公开自定义源码和配置，不包含 H5/LeRobot 数据、π0.5 初始权重、checkpoint、
norm stats、仿真资产和视频。这些内容需遵循上游项目许可单独获取。
