# 项目自定义改动索引

本文区分上游 RLinf/OpenPI/RoboBenchMart 原有能力与本项目新增工作，方便简历评审、
技术面试和代码审查快速定位 SFT、KL-PPO 与 RECAP-lite 的实现证据。

## 1. SFT 与 OpenPI 适配

- `sft/convert_pick_to_basket_h5_to_lerobot.py`：将 Fanta、Nivea、Stars 的 H5 expert
  demonstrations 转成 LeRobot v2.1，包含三视角、15 维 state、13 维 action、任务文本、
  episode metadata 和逐 episode statistics。
- `sft/openpi/robobenchmart_policy.py`：将 RBM 三视角映射为 π0.5 的
  `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`，输出时截取前 13 维 action。
- `sft/openpi/openpi_rbm_sft_config.patch`：增加 `LeRobotRBMDataConfig`、`pi0_eval_rbm`、
  `pi05_eval_rbm`、`pi05_sft_rbm_pick_to_basket`，并提高大 checkpoint 保存可靠性。
- `sft/docs/`：保存 SFT 适配说明和 20k-step 完整训练报告。
- JAX 固定协议下，三个 ID 商品成功率从 43/90 提升到 67/90。

## 2. RoboBenchMart 与 RLinf 接入

- `rlinf/envs/robobenchmart/`：环境 wrapper、task mapping、奖励与 proxy tasks。
- `rlinf/models/embodiment/openpi/`：RBM data config、policy adapter 和 flow rollout。
- `rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py`：JAX → PyTorch 转换。
- `scripts_local/08_rbm_eval_matched.sh`：ID、OOD、Proxy fixed-seed matched evaluation。

## 3. True SFT-reference KL-PPO

- `rlinf/workers/actor/fsdp_actor_worker.py`：冻结初始 SFT reference；在同一 observation、
  action、flow noise 和 timestep 上重算 reference logprob；加入 reference KL、自适应
  beta、zero-drift assertion、SFT rehearsal 和正确的 checkpoint reference 生命周期。
- `rlinf/algorithms/losses.py`：critic-only warmup 与 actor loss 门控。
- `rlinf/utils/embodied_training_safety.py`：最小成功数门控、KL 调整、完整 rollout EV 和
  终止语义安全检查。
- `rlinf/envs/robobenchmart/robobenchmart_env.py`：success-dominant event/progress reward，
  修复重复奖励、首步虚假 progress、done 后 chunk reward 和错误 bootstrap。
- `tests/unit_tests/test_rbm_post_training_safety.py`：训练安全逻辑测试。
- `examples/embodiment/config/rbm_*.yaml`、`scripts_local/06_*`、`07_*`：配置与入口。

## 4. RECAP-lite

- `scripts_local/09_rbm_collect_indomain_rollouts.sh`：三个 ID 任务自主轨迹采集。
- `scripts_local/09_rbm_repair_rollout_schema.py`：第三视角 schema 原子迁移。
- `examples/recap/process/make_recap_lite_advantages.py`：生成 episode/frame 对齐的
  positive/negative advantage sidecar。
- `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py`：RBM 三视角、正负条件、
  conditional dropout 和 positive guidance。
- `rlinf/data/datasets/recap/cfg_model.py`：advantage-preserving、episode-balanced 与
  task/label quota sampling。
- `rlinf/workers/sft/fsdp_cfg_worker.py`：多数据源 CFG-SFT 与 quota cycle。
- `scripts_local/10_*`、`11_*`、`12_*`：标签、训练和评测入口。
- `tests/unit_tests/test_recap_lite_advantages.py`：sidecar 与标签测试。

## 5. 分布式与工程修复

- 修复多 pipeline stage 写入相同 LeRobot shard 导致覆盖。
- writer 使用绝对 root，并自动发现多个 rollout shards。
- PPO 与 SFT 分开 backward，限制 micro batch 并 offload rollout model，解决 7.56 GiB
  峰值分配导致的 OOM。
- checkpoint 保存权重、optimizer、scheduler、warmup、actor update 和 reference 元数据。
- `scripts_local/13_rbm_post_training_preflight.sh` 在训练前检查配置、schema、标签和测试。

## 6. 实验与文档

- `docs/rbm_resume_and_interview_project_summary.md`：综合项目讲解。
- `docs/rbm_true_kl_ppo_and_reward_changes.md`：KL-PPO 与 reward 专题。
- `docs/rbm_recap_lite_changes.md`：RECAP-lite 专题。
- `docs/rbm_post_training_fixes_and_runbook.md`：运行手册。
- `docs/rbm_20260710_failures_analysis.md`：负实验和根因分析。

## 7. 公开仓库边界

仓库不包含数据、模型 checkpoint、TensorBoard、视频和仿真资产。SFT 代码以独立脚本和
OpenPI patch 提供；应用 patch 时需使用兼容 OpenPI 版本并修改本地绝对路径。当前实现是
RECAP-lite，不等同于包含专家 correction 与完整 value pipeline 的全量 RECAP。
