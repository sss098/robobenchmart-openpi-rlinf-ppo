#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
RBM_PROJECT=${RBM_PROJECT:-/path/to/RoboBenchMart}
MODEL_PATH=${MODEL_PATH:-/path/to/rbm_pi05_sft_pytorch}

cd "$PROJECT"
source .venv_openpi/bin/activate

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export PYTHONPATH="$PROJECT:$RBM_PROJECT:${PYTHONPATH:-}"

bash examples/embodiment/run_embodiment.sh rbm_pick_to_basket_ppo_openpi_pi05 LIBERO \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  env.train.project_path="$RBM_PROJECT" \
  env.eval.project_path="$RBM_PROJECT"
