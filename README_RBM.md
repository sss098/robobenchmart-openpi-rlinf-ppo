# RoboBenchMart PickToBasket OpenPI/RLinf Adaptation

This branch adapts RLinf's embodied RL pipeline to RoboBenchMart PickToBasket with OpenPI pi0.5 policies.

## What This Project Adds

- RoboBenchMart environment wrapper for RLinf embodied workers.
- PickToBasket task configs for train, proxy, unseen, and OOD-style evaluation splits.
- OpenPI pi0.5 RoboBenchMart data config and policy setup.
- JAX-to-PyTorch checkpoint conversion workflow compatibility.
- Matched RLinf evaluation path for converted OpenPI checkpoints.
- Action trace utilities for comparing official OpenPI JAX server behavior with RLinf PyTorch rollout behavior.
- True SFT-reference KL regularization for embodied PPO.
- Event/progress shaped reward for PickToBasket.

## Main Technical Route

The intended research workflow is:

```text
PickToBasket SFT
-> true SFT-reference KL-PPO probe
-> failure episode correction data
-> RECAP-lite weighted SFT
-> train / robo / unseen / OOD evaluation
```

The PPO stage is treated conservatively. Current experiments show that direct PPO probes did not reliably outperform the SFT baseline, so the recommended next stage is failure correction plus weighted SFT rather than long PPO training.

## Important Files

```text
rlinf/envs/robobenchmart/
rlinf/models/embodiment/openpi/policies/robobenchmart_policy.py
rlinf/models/embodiment/openpi/dataconfig/robobenchmart_dataconfig.py
examples/embodiment/config/rbm_pick_to_basket_ppo_openpi_pi05.yaml
examples/embodiment/config/rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05.yaml
examples/embodiment/config/env/robobenchmart_pick_to_basket_*.yaml
scripts_local/06_rbm_ppo_smoke.sh
scripts_local/07_rbm_ppo_proxy_mix.sh
scripts_local/08_rbm_collect_jax_action_trace.py
scripts_local/09_compare_rbm_action_traces.py
docs/rbm_stage1_rlinf_ppo_smoke_adaptation.md
docs/rbm_true_kl_ppo_and_reward_changes.md
```

## True KL-PPO Fix

RLinf's embodied PPO path originally constrained updates against rollout old log-probabilities, but it did not apply an explicit SFT-reference KL penalty in `EmbodiedFSDPActor`.

This project adds:

```text
ref_logprobs = SFT_reference(forward_inputs)
loss += kl_beta * KL(current_logprobs, ref_logprobs)
```

and logs:

```text
train/actor/ref_kl_loss
train/actor/ref_kl_abs
```

See `docs/rbm_true_kl_ppo_and_reward_changes.md` for details.

## Reward Design

The PickToBasket shaped reward is event/progress based:

```text
+5.00 * success
+1.00 * first_placed
+0.30 * first_lifted
+0.50 * positive_progress_to_basket
+0.30 * first_placed_static
-0.20 * first_non_target_displacement
```

The goal is to reduce reward hacking where the policy only learns to lift or approach the basket without completing the placement.

## What Is Not Included

This repository intentionally does not include:

```text
model checkpoints
training logs
evaluation videos
RoboBenchMart assets
ManiSkill assets
OpenPI cached assets
local virtual environments
```

You must provide your own OpenPI / RoboBenchMart / ManiSkill setup and checkpoints.

## Status

This is an engineering/research adaptation branch, not a final benchmark-winning checkpoint release. The code documents the integration path, debugging process, true KL-PPO correction, and evaluation alignment work.
