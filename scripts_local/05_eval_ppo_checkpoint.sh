#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/projects/RLinf

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts_local/05_eval_ppo_checkpoint.sh /path/to/global_step_xxx"
  exit 1
fi

PPO_CKPT_DIR="$1"

cd "$PROJECT"
source .venv_openpi/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

if [ ! -d "$PPO_CKPT_DIR" ]; then
  echo "ERROR: PPO checkpoint directory not found: $PPO_CKPT_DIR"
  exit 1
fi

echo "Evaluate PPO checkpoint:"
echo "$PPO_CKPT_DIR"

bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05 LIBERO \
  actor.model.model_path="$PPO_CKPT_DIR" \
  rollout.model.model_path="$PPO_CKPT_DIR" \
  runner.only_eval=True \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  algorithm.eval_rollout_epoch=1 \
  env.eval.total_num_envs=100
