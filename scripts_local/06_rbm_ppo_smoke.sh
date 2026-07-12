#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
RBM_PROJECT=${RBM_PROJECT:-/root/autodl-tmp/projects/RoboBenchMart}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch}
MAX_EPOCHS=${MAX_EPOCHS:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rbm_pick_to_basket_ppo_safety_smoke}
CRITIC_WARMUP_STEPS=${CRITIC_WARMUP_STEPS:-12}
UPDATES_PER_EPOCH=${UPDATES_PER_EPOCH:-12}
BASELINE_DIAGNOSTIC=${BASELINE_DIAGNOSTIC:-0}
RESUME_WARMUP_COMPLETE=0

for arg in "$@"; do
  if [[ "$arg" == runner.resume_dir=* ]]; then
    if [ "${ALLOW_RESUME:-0}" != "1" ]; then
      echo "Set ALLOW_RESUME=1 to acknowledge checkpoint resume." >&2
      echo "Only checkpoints containing actor/trainer_state.json are supported." >&2
      exit 2
    fi
    resume_dir=${arg#runner.resume_dir=}
    trainer_state="$resume_dir/actor/trainer_state.json"
    if [ ! -f "$trainer_state" ]; then
      echo "Missing resume metadata: $trainer_state" >&2
      exit 2
    fi
    if grep -Eq '"critic_warmup_steps"[[:space:]]*:[[:space:]]*0' "$trainer_state"; then
      RESUME_WARMUP_COMPLETE=1
    fi
  fi
done

if [ "$BASELINE_DIAGNOSTIC" = "1" ]; then
  if [ "$MAX_EPOCHS" != "1" ]; then
    echo "BASELINE_DIAGNOSTIC=1 requires MAX_EPOCHS=1." >&2
    exit 2
  fi
elif [ "$RESUME_WARMUP_COMPLETE" != "1" ] \
  && (( MAX_EPOCHS * UPDATES_PER_EPOCH <= CRITIC_WARMUP_STEPS )); then
  echo "Training provides $((MAX_EPOCHS * UPDATES_PER_EPOCH)) optimizer updates, which must exceed CRITIC_WARMUP_STEPS ($CRITIC_WARMUP_STEPS)." >&2
  echo "Otherwise the smoke run never updates the actor policy." >&2
  exit 2
fi

if [ "$BASELINE_DIAGNOSTIC" != "1" ] && [ "${ALLOW_ACTOR_PPO:-0}" != "1" ]; then
  echo "Actor PPO is blocked: the matched training-path BASE diagnostic has not produced a success." >&2
  echo "Do not spend compute until rollout/task metrics are corrected and the BASE gate passes." >&2
  exit 2
fi

cd "$PROJECT"
source .venv_openpi/bin/activate

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=1
export PYTHONPATH="$PROJECT:$RBM_PROJECT:${PYTHONPATH:-}"

bash examples/embodiment/run_embodiment.sh rbm_pick_to_basket_ppo_openpi_pi05 LIBERO \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  env.train.project_path="$RBM_PROJECT" \
  env.eval.project_path="$RBM_PROJECT" \
  runner.max_epochs="$MAX_EPOCHS" \
  runner.save_interval="$SAVE_INTERVAL" \
  runner.val_check_interval=-1 \
  runner.logger.experiment_name="$EXPERIMENT_NAME" \
  actor.optim.critic_warmup_steps="$CRITIC_WARMUP_STEPS" \
  "$@"
