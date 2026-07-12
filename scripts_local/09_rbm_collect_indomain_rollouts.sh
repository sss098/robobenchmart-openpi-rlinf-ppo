#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/lerobot_data/rbm_indomain_rollouts}
EVAL_EPOCHS=${EVAL_EPOCHS:-30}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch}

if [ -e "$OUTPUT_ROOT" ] && find "$OUTPUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: collection output is not empty: $OUTPUT_ROOT" >&2
  echo "Choose a new OUTPUT_ROOT; existing rollout data will not be overwritten." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT"

EVAL_SPLIT=train3 EVAL_EPOCHS="$EVAL_EPOCHS" BASE_MODEL_PATH="$BASE_MODEL_PATH" \
bash scripts_local/08_rbm_eval_matched.sh BASE \
  +env.eval.data_collection.enabled=True \
  +env.eval.data_collection.save_dir="$OUTPUT_ROOT" \
  +env.eval.data_collection.export_format=lerobot \
  +env.eval.data_collection.robot_type=ds_fetch_basket \
  +env.eval.data_collection.fps=10 \
  +env.eval.data_collection.only_success=False \
  +env.eval.data_collection.finalize_interval=0

shard_count=$(find "$OUTPUT_ROOT" -type f -path '*/meta/info.json' | wc -l)
if [ "$shard_count" -ne 3 ]; then
  echo "ERROR: expected 3 task shards, found $shard_count under $OUTPUT_ROOT" >&2
  exit 2
fi

echo "Collected $shard_count in-domain LeRobot shards under: $OUTPUT_ROOT"
