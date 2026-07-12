# RoboBenchMart PickToBasket SFT + KL-PPO Post-Training Pipeline

[中文说明](README.zh-CN.md)

For a detailed Chinese walkthrough of the task, SFT, KL-PPO, RECAP-lite, engineering
fixes, and experimental conclusions, see the
[beginner-oriented project summary](docs/rbm_resume_and_interview_project_summary.md).

This repository is a personal research project that builds an end-to-end post-training pipeline for **OpenPI pi0.5** on **RoboBenchMart PickToBasket**:

```text
PickToBasket SFT
-> JAX checkpoint evaluation
-> JAX-to-PyTorch checkpoint conversion
-> RLinf rollout / evaluation
-> SFT-reference KL-PPO probing
-> failure episode analysis
-> correction-data / RECAP-lite style weighted SFT
-> train / robo / unseen / OOD evaluation
```

The project started from a supervised fine-tuned OpenPI pi0.5 model that improved in-domain PickToBasket performance. The next question was whether closed-loop reinforcement learning could improve both in-domain and out-of-distribution behavior without destroying the SFT policy. This repository documents the engineering work needed to answer that question inside RLinf.

It is not a checkpoint release. Model weights, assets, logs, videos, and datasets are intentionally excluded.

## What This Repository Is

This is a reproducible code and documentation branch for connecting three systems:

```text
OpenPI         pi0.5 model, SFT training, JAX checkpoint format
RoboBenchMart  PickToBasket simulation tasks and evaluation environments
RLinf          PyTorch rollout, PPO actor/value training, distributed workers
```

The main contribution is the glue code and debugging work required to make the full path usable:

1. Train or load an OpenPI pi0.5 SFT checkpoint for RoboBenchMart PickToBasket.
2. Evaluate the JAX SFT policy in the RoboBenchMart environment.
3. Convert the JAX checkpoint to the PyTorch format expected by RLinf.
4. Run matched RLinf evaluation to check whether the converted policy still behaves correctly.
5. Run conservative PPO with logprob, value, old logprob, advantage, and SFT-reference KL.
6. Compare PPO checkpoints against the SFT baseline instead of assuming reward improvement means task improvement.
7. Use failed rollouts as the input for correction-data or RECAP-lite weighted SFT.

## Why This Project Exists

Plain SFT can improve in-domain behavior, but it often does not reliably improve unseen or OOD tasks. Closed-loop RL seems attractive because the policy can interact with the environment and receive task feedback.

In practice, direct PPO on a strong robot SFT policy is fragile:

- The reward can be sparse or misaligned with the final success metric.
- A shaped reward can be exploited by partial behavior such as lifting without placing.
- PPO clip constrains the current update against the rollout policy, but it does not by itself keep the policy close to the original SFT model across many updates.
- Small action or preprocessing mismatches between JAX eval and PyTorch rollout can make PPO optimize the wrong behavior.

This repository therefore treats PPO as a **small, conservative probe**, not as a blind long training run. The more stable next stage is to use PPO/eval failures to build correction data and then run weighted SFT.

## Technical Route

The intended workflow is:

```text
Stage 0: OpenPI / RoboBenchMart SFT
  Train pi0.5 on PickToBasket demonstrations with the OpenPI JAX trainer.

Stage 1: SFT evaluation and checkpoint conversion
  Evaluate the JAX SFT checkpoint, then convert it to RLinf's PyTorch OpenPI format.

Stage 2: Matched RLinf evaluation
  Run deterministic RLinf evaluation on the converted checkpoint and compare it with JAX behavior.

Stage 3: Conservative SFT-reference KL-PPO
  Run small PPO probes with value prediction, old logprobs, advantages, PPO clipping, and reference KL to the SFT policy.

Stage 4: Failure collection
  Save failed episodes, videos, action traces, and task metadata for correction analysis.

Stage 5: RECAP-lite / weighted SFT
  Use successful and corrected failure episodes with advantage- or outcome-based weights.

Stage 6: Evaluation
  Evaluate train / robo / unseen / OOD splits and compare every result against the matched SFT baseline.
```

## Repository Contents

Important files added or modified for RoboBenchMart:

```text
rlinf/envs/robobenchmart/
  RoboBenchMart environment wrapper used by RLinf EnvWorker.

rlinf/envs/robobenchmart/robobenchmart_env.py
  PickToBasket observation mapping, task selection, reset handling, seed alignment, video/debug hooks, and shaped reward.

rlinf/envs/robobenchmart/proxy_tasks.py
  Proxy task definitions used to train on object/layout variations while keeping Nestle/Slam/Duff-style OOD tasks out of PPO training.

rlinf/models/embodiment/openpi/dataconfig/robobenchmart_dataconfig.py
  RLinf-side OpenPI data config for RoboBenchMart observations.

rlinf/models/embodiment/openpi/policies/robobenchmart_policy.py
  Policy adapter for converting RoboBenchMart observations/actions to OpenPI-compatible tensors.

rlinf/workers/actor/fsdp_actor_worker.py
  Adds true SFT-reference KL regularization to embodied PPO.

examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
  Main PickToBasket PPO/eval config.

examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
  Mixed proxy-task PPO config.

examples/embodiment/config/env/robobenchmart_pick_to_basket_*.yaml
  Train, proxy, unseen, and OOD-style environment splits.

scripts_local/06_rbm_ppo_smoke.sh
scripts_local/07_rbm_ppo_proxy_mix.sh
  Local smoke and proxy-mix PPO entrypoints.

scripts_local/08_rbm_collect_jax_action_trace.py
scripts_local/09_compare_rbm_action_traces.py
  Tools for comparing OpenPI JAX policy actions with RLinf PyTorch policy actions.

docs/rbm_stage1_rlinf_ppo_smoke_adaptation.md
  Stage-1 adaptation notes.

docs/rbm_true_kl_ppo_and_reward_changes.md
  Detailed explanation of SFT-reference KL-PPO and PickToBasket reward shaping.
```

## SFT Stage

The OpenPI-side RoboBenchMart SFT config is:

```text
pi05_sft_rbm_pick_to_basket
```

It uses:

```text
model: OpenPI pi0.5
task: RoboBenchMart PickToBasket
action_dim: 13
training length: 20,000 steps in the local experiment
```

The SFT checkpoint remains in JAX/OpenPI checkpoint format. It is evaluated first in the original JAX policy path before conversion. This matters because PPO should start only after the SFT baseline and environment are known to work.

## Checkpoint Conversion

RLinf trains the OpenPI policy through its PyTorch model path, so the SFT checkpoint must be converted:

```bash
python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint_dir /path/to/openpi_jax_sft_checkpoint/19999 \
  --output_path /path/to/rbm_pi05_sft_pytorch \
  --config_name pi05_eval_rbm
```

Use `pi05_eval_rbm` for conversion and evaluation. The SFT training config is `pi05_sft_rbm_pick_to_basket`; the eval/conversion config is `pi05_eval_rbm`.

## PPO Stage

The PPO implementation needs the following quantities:

```text
action
logprob
value
old_logprob
ref_logprob
advantage
KL penalty
```

In this project:

- `action` is sampled by the rollout policy during environment interaction.
- `old_logprob` is the log probability of the sampled action under the rollout policy.
- `value` is predicted by the actor/value model and used for return/advantage estimation.
- `advantage` is computed from rewards and value predictions by the PPO pipeline.
- `logprob` is recomputed by the current actor during PPO update.
- `ref_logprob` is recomputed by a frozen copy of the initial SFT model.
- `KL(current, SFT)` is added to the actor loss to reduce policy drift.

The loss becomes:

```text
loss = PPO actor/value loss + kl_beta * reference_KL(current_logprob, ref_logprob)
```

This is different from PPO's usual `approx_kl`, which only compares the current policy to the rollout old policy. The reference KL compares the current policy to the original SFT policy.

## Reward Design

The PickToBasket reward is event/progress based:

```text
+5.00 * success
+1.00 * first_placed
+0.30 * first_lifted
+0.50 * positive_progress_to_basket
+0.30 * first_placed_static
-0.20 * first_non_target_displacement
```

The design goal is to make reward closer to the final success metric:

- Lifting is useful but should not dominate training.
- Moving toward the basket is useful only when it is real progress.
- Placement and final success should be more important than intermediate motion.
- Non-target object disruption should be discouraged without overwhelming the reward.

## Current Findings

The main finding so far is pragmatic:

```text
SFT is the reliable baseline.
Direct PPO is fragile.
Reward increase does not necessarily mean success-rate increase.
True SFT-reference KL is required before PPO results are meaningful.
The next stronger direction is correction-data / RECAP-lite weighted SFT.
```

In matched experiments, PPO probes did not reliably beat the SFT baseline. This is why the repository documents the PPO path as a controlled post-training probe and debugging tool, rather than presenting PPO as a finished performance improvement.

## What Is Not Included

The repository intentionally does not include:

```text
model checkpoints
training logs
TensorBoard event files
evaluation videos
RoboBenchMart assets
ManiSkill assets
OpenPI cached assets
local virtual environments
private datasets
```

These artifacts are too large, machine-specific, or not appropriate for a public GitHub repository.

## Minimal Reproduction Layout

Expected local layout:

```text
/path/to/RLinf
/path/to/openpi
/path/to/RoboBenchMart
/path/to/ManiSkill
/path/to/rbm_pi05_sft_pytorch
```

Environment variables:

```bash
export PYTHONPATH=/path/to/RLinf:/path/to/RoboBenchMart:/path/to/openpi:${PYTHONPATH:-}
export MODEL_PATH=/path/to/rbm_pi05_sft_pytorch
export OMP_NUM_THREADS=1
```

PPO smoke:

```bash
bash scripts_local/06_rbm_ppo_smoke.sh
```

Proxy-mix PPO:

```bash
bash scripts_local/07_rbm_ppo_proxy_mix.sh
```

Action trace comparison:

```bash
python scripts_local/09_compare_rbm_action_traces.py \
  /path/to/jax_trace.npz \
  /path/to/rlinf_trace_dir \
  --stage-id 0 \
  --atol 1e-4
```

## Project Status

This is a research engineering repository. It is useful for showing the full SFT-to-RL post-training pipeline, the debugging process, and the reasoning behind conservative KL-PPO for embodied policies. It should not be read as a final benchmark checkpoint release.
