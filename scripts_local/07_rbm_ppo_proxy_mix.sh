#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
RBM_PROJECT=${RBM_PROJECT:-/root/autodl-tmp/projects/RoboBenchMart}
OPENPI_PROJECT=${OPENPI_PROJECT:-/root/autodl-tmp/projects/openpi}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch}
MAX_EPOCHS=${MAX_EPOCHS:-6}
CRITIC_WARMUP_STEPS=${CRITIC_WARMUP_STEPS:-3}

if [ "${ALLOW_PROXY_MIX:-0}" != "1" ]; then
  echo "Refusing to start proxy-mix by default."
  echo "First verify proxy BASE success and the fixed in-domain PPO smoke."
  echo "Then rerun with ALLOW_PROXY_MIX=1."
  exit 2
fi

if (( MAX_EPOCHS <= CRITIC_WARMUP_STEPS )); then
  echo "MAX_EPOCHS ($MAX_EPOCHS) must be greater than CRITIC_WARMUP_STEPS ($CRITIC_WARMUP_STEPS)." >&2
  exit 2
fi

for arg in "$@"; do
  if [[ "$arg" == runner.resume_dir=* ]]; then
    echo "runner.resume_dir is disabled for this smoke script because warmup optimizer checkpoints are incompatible with a full actor optimizer." >&2
    exit 2
  fi
done

cd "$PROJECT"
source .venv_openpi/bin/activate

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=1
export PYTHONPATH="$PROJECT:$RBM_PROJECT:$OPENPI_PROJECT:${PYTHONPATH:-}"

bash examples/embodiment/run_embodiment.sh \
  rbm_pick_to_basket_proxy_mix_ppo_openpi_pi05 \
  LIBERO \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  env.train.project_path="$RBM_PROJECT" \
  env.eval.project_path="$RBM_PROJECT" \
  runner.max_epochs="$MAX_EPOCHS" \
  runner.save_interval=1 \
  runner.val_check_interval=-1 \
  runner.logger.experiment_name="${EXPERIMENT_NAME:-rbm_proxy_mix_ppo_safety_smoke}" \
  actor.optim.critic_warmup_steps="$CRITIC_WARMUP_STEPS" \
  "$@"
