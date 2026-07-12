#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch}
DATASET_PATH=${DATASET_PATH:-/root/autodl-tmp/lerobot_data/rbm_dataset}
RBM_PROJECT=${RBM_PROJECT:-/root/autodl-tmp/projects/RoboBenchMart}

cd "$PROJECT"
source .venv_openpi/bin/activate

for path in "$MODEL_PATH" "$DATASET_PATH" "$RBM_PROJECT"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path"
    exit 1
  fi
done

bash -n \
  scripts_local/06_rbm_ppo_smoke.sh \
  scripts_local/07_rbm_ppo_proxy_mix.sh \
  scripts_local/08_rbm_eval_matched.sh \
  scripts_local/10_rbm_recap_prepare_advantages.sh \
  scripts_local/11_rbm_recap_cfg_sft.sh \
  scripts_local/11_rbm_recap_cfg_mixed.sh \
  scripts_local/12_rbm_recap_eval_checkpoint.sh

python -m pytest -q \
  tests/unit_tests/test_rbm_post_training_safety.py \
  tests/unit_tests/test_recap_lite_advantages.py

PROJECT="$PROJECT" EMBODIED_PATH="$PROJECT/examples/embodiment" REPO_PATH="$PROJECT" python - <<'PY'
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

project = os.environ["PROJECT"]
cases = [
    ("examples/embodiment/config", "rbm_pick_to_basket_ppo_openpi_pi05"),
    ("examples/embodiment/config", "rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05"),
    ("examples/recap/cfg/config", "rbm_pick_to_basket_recap_lite_cfg_openpi_pi05"),
]
for config_dir, config_name in cases:
    with initialize_config_dir(config_dir=f"{project}/{config_dir}", version_base=None):
        cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        print(
            config_name,
            "chunk=", cfg.actor.model.openpi.action_chunk,
            "steps=", cfg.actor.model.openpi.num_steps,
        )
        if "env" in cfg:
            rollout_steps = cfg.env.train.max_steps_per_rollout_epoch
            chunk = cfg.actor.model.num_action_chunks
            assert rollout_steps >= chunk and rollout_steps % chunk == 0, (
                f"invalid rollout/chunk combination: {rollout_steps}/{chunk}"
            )
            assert cfg.actor.optim.critic_warmup_steps > 0
            if config_name == "rbm_pick_to_basket_ppo_openpi_pi05":
                assert cfg.actor.optim.critic_warmup_steps == 12
                assert cfg.algorithm.rollout_epoch == 4
                assert cfg.algorithm.actor_update_min_successes == 2
                assert cfg.actor.sft_loss_weight == 1.0
                assert cfg.actor.model.openpi.joint_logprob is False
                assert cfg.env.train.total_num_envs == 3
                assert cfg.env.train.auto_reset is False
                assert cfg.env.train.ignore_terminations is False
                assert cfg.env.train.init_params.sim_backend == "cpu"
                assert len(cfg.env.train.episode_seed_lists) == 3
                samples_per_rollout = (
                    cfg.env.train.total_num_envs
                    * cfg.env.train.max_steps_per_rollout_epoch
                    // cfg.actor.model.num_action_chunks
                    * cfg.algorithm.rollout_epoch
                )
                assert samples_per_rollout % cfg.actor.global_batch_size == 0
                assert cfg.actor.global_batch_size % cfg.actor.micro_batch_size == 0
            expected_micro_batch = (
                1 if config_name == "rbm_pick_to_basket_ppo_openpi_pi05" else 2
            )
            assert cfg.actor.micro_batch_size == expected_micro_batch
            assert cfg.rollout.seed == 42
            assert cfg.algorithm.bootstrap_type == "none"
            assert cfg.algorithm.kl_penalty == "low_var_kl"
            assert cfg.weight_syncer.type == "patch"
            assert cfg.weight_syncer.patch.snapshot_device == "cpu"
            assert cfg.weight_syncer.patch.transport_device == "cpu"
PY

echo "RBM post-training preflight passed."
