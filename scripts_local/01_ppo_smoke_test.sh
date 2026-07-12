#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/projects/RLinf
SFT_CKPT=${SFT_CKPT:-/root/autodl-tmp/checkpoints/openpi_sft/pi05_libero/pi05_base_sft_libero10_official_demo/10000}

cd "$PROJECT"
source .venv_openpi/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

bash scripts_local/00_check_sft_ckpt.sh

echo
echo "Start PPO smoke test..."

bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05 LIBERO \
  actor.model.model_path="$SFT_CKPT" \
  rollout.model.model_path="$SFT_CKPT" \
  runner.logger.experiment_name=pi05_sft_ppo_smoke_test \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  runner.save_interval=1 \
  runner.val_check_interval=1 \
  env.train.total_num_envs=4 \
  env.eval.total_num_envs=10 \
  actor.micro_batch_size=4 \
  actor.global_batch_size=16 \
  algorithm.rollout_epoch=1 \
  algorithm.update_epoch=1 \
  algorithm.eval_rollout_epoch=1

echo
echo "PPO smoke test finished."
