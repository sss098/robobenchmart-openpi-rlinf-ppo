#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
RBM_PROJECT=${RBM_PROJECT:-/root/autodl-tmp/projects/RoboBenchMart}
OPENPI_PROJECT=${OPENPI_PROJECT:-/root/autodl-tmp/projects/openpi}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch}
DATASET_PATH=${DATASET_PATH:-/root/autodl-tmp/lerobot_data/rbm_dataset}
TAG=${TAG:-rbm_recap_lite_sft_pos}
CONFIG=${CONFIG:-rbm_pick_to_basket_recap_lite_cfg_openpi_pi05}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
LR=${LR:-2.0e-7}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
UNCONDITIONAL_PROB=${UNCONDITIONAL_PROB:-0.3}
NUM_WORKERS=${NUM_WORKERS:-0}
AUTO_PREPARE=${AUTO_PREPARE:-1}

cd "$PROJECT"
source .venv_openpi/bin/activate

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export OMP_NUM_THREADS=1
export PYTHONPATH="$PROJECT:$RBM_PROJECT:$OPENPI_PROJECT:${PYTHONPATH:-}"

ADV_PATH="$DATASET_PATH/meta/advantages_${TAG}.parquet"
if [ ! -f "$ADV_PATH" ]; then
  if [ "$AUTO_PREPARE" = "1" ]; then
    DATASET_PATH="$DATASET_PATH" TAG="$TAG" bash scripts_local/10_rbm_recap_prepare_advantages.sh
  else
    echo "ERROR: advantage file not found: $ADV_PATH"
    echo "Run: DATASET_PATH=$DATASET_PATH TAG=$TAG bash scripts_local/10_rbm_recap_prepare_advantages.sh"
    exit 1
  fi
fi

if [ -n "${TRAIN_DATA_PATHS_OVERRIDE:-}" ]; then
  DATA_ARGS=("$TRAIN_DATA_PATHS_OVERRIDE")
else
  DATA_ARGS=(data.train_data_paths.0.dataset_path="$DATASET_PATH")
fi

bash examples/recap/cfg/run_cfg_sft.sh "$CONFIG" \
  actor.model.model_path="$MODEL_PATH" \
  actor.model.openpi.cfgrl_guidance_scale="$GUIDANCE_SCALE" \
  actor.model.openpi.unconditional_prob="$UNCONDITIONAL_PROB" \
  actor.global_batch_size="$GLOBAL_BATCH_SIZE" \
  actor.micro_batch_size="$MICRO_BATCH_SIZE" \
  actor.optim.lr="$LR" \
  actor.optim.total_training_steps="$MAX_STEPS" \
  runner.max_steps="$MAX_STEPS" \
  runner.save_interval="$SAVE_INTERVAL" \
  data.num_workers="$NUM_WORKERS" \
  data.advantage_tag="$TAG" \
  "${DATA_ARGS[@]}" \
  "$@"
