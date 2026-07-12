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
echo "Start conservative PPO training from SFT checkpoint..."

bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05 LIBERO \
  actor.model.model_path="$SFT_CKPT" \
  rollout.model.model_path="$SFT_CKPT" \
  runner.logger.experiment_name=pi05_base_sft_libero10_ppo_from_10000_conservative \
  runner.max_epochs=1000 \
  runner.max_steps=-1 \
  runner.save_interval=20 \
  runner.val_check_interval=20 \
  env.train.total_num_envs=32 \
  env.eval.total_num_envs=100 \
  actor.micro_batch_size=64 \
  actor.global_batch_size=1024 \
  actor.optim.lr=2.0e-6 \
  actor.optim.value_lr=5.0e-5 \
  algorithm.rollout_epoch=4 \
  algorithm.update_epoch=2 \
  algorithm.kl_beta=0.0 \
  algorithm.entropy_bonus=0.0 \
  algorithm.sampling_params.temperature_train=0.8 \
  algorithm.sampling_params.temperature_eval=0.6
