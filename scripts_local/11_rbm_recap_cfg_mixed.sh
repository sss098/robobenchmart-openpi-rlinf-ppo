#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
SFT_DATASET=${SFT_DATASET:-/root/autodl-tmp/lerobot_data/rbm_dataset}
ROLLOUT_DATASET=${ROLLOUT_DATASET:-/root/autodl-tmp/lerobot_data/rbm_indomain_rollouts}
TAG=${TAG:-rbm_recap_indomain_v1}
SFT_WEIGHT=${SFT_WEIGHT:-0.9}
ROLLOUT_WEIGHT=${ROLLOUT_WEIGHT:-0.1}
MAX_STEPS=${MAX_STEPS:-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
LR=${LR:-5.0e-8}
POSITIVE_FRACTION=${POSITIVE_FRACTION:-0.67}
QUOTA_CYCLE_SIZE=${QUOTA_CYCLE_SIZE:-60}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-12}
MIN_EPISODES_PER_SHARD=${MIN_EPISODES_PER_SHARD:-10}

if [ ! -f "$SFT_DATASET/meta/advantages_${TAG}.parquet" ]; then
  echo "ERROR: missing SFT advantage sidecar: $SFT_DATASET/meta/advantages_${TAG}.parquet" >&2
  exit 2
fi

ROLLOUT_SHARDS=()
if [ -f "$ROLLOUT_DATASET/meta/info.json" ]; then
  ROLLOUT_SHARDS=("$ROLLOUT_DATASET")
elif [ -d "$ROLLOUT_DATASET" ]; then
  while IFS= read -r info_file; do
    ROLLOUT_SHARDS+=("$(dirname "$(dirname "$info_file")")")
  done < <(find "$ROLLOUT_DATASET" -type f -path '*/meta/info.json' | sort)
fi
if [ ${#ROLLOUT_SHARDS[@]} -eq 0 ]; then
  echo "ERROR: no complete LeRobot rollout shards under: $ROLLOUT_DATASET" >&2
  echo "Run scripts_local/09_rbm_collect_indomain_rollouts.sh first." >&2
  exit 2
fi

for dataset in "${ROLLOUT_SHARDS[@]}"; do
  advantage_path="$dataset/meta/advantages_${TAG}.parquet"
  if [ ! -f "$advantage_path" ]; then
    echo "ERROR: missing rollout advantage sidecar: $advantage_path" >&2
    echo "Run scripts_local/10_rbm_recap_prepare_advantages.sh for the rollout root." >&2
    exit 2
  fi
done

source "$PROJECT/.venv_openpi/bin/activate"

ROLLOUT_DATASETS="$(IFS=:; echo "${ROLLOUT_SHARDS[*]}")" TAG="$TAG" \
MIN_EPISODES_PER_SHARD="$MIN_EPISODES_PER_SHARD" \
ALLOW_SMALL_ROLLOUT_DATASET="${ALLOW_SMALL_ROLLOUT_DATASET:-0}" python - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

paths = [Path(p) for p in os.environ["ROLLOUT_DATASETS"].split(":")]
minimum = int(os.environ["MIN_EPISODES_PER_SHARD"])
allow_small = os.environ["ALLOW_SMALL_ROLLOUT_DATASET"] == "1"
for path in paths:
    episode_file = path / "meta" / "episodes.jsonl"
    episodes = [json.loads(line) for line in episode_file.read_text().splitlines() if line.strip()]
    lengths = [int(ep["length"]) for ep in episodes]
    parquet_files = sorted((path / "data").rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"ERROR: no parquet frames in {path}")
    columns = pq.read_schema(parquet_files[0]).names
    if "extra_image" not in columns or "extra_view_image" in columns:
        raise SystemExit(
            f"ERROR: incompatible RBM third-view schema in {path}: {columns}. "
            "Run scripts_local/09_rbm_repair_rollout_schema.py."
        )
    if any(length < 50 for length in lengths):
        raise SystemExit(f"ERROR: fragment episode in {path}: min length={min(lengths)}")
    if len(episodes) < minimum and not allow_small:
        raise SystemExit(
            f"ERROR: {path} has {len(episodes)} episodes; require >= {minimum}. Only engineering smoke may set ALLOW_SMALL_ROLLOUT_DATASET=1."
        )
    shard_labels = pd.read_parquet(
        path / "meta" / f"advantages_{os.environ['TAG']}.parquet",
        columns=["episode_index", "advantage"],
    ).groupby("episode_index")["advantage"].any().astype(bool)
    if not allow_small and (not shard_labels.any() or shard_labels.all()):
        raise SystemExit(
            f"ERROR: {path} must contain both successful and failed episodes"
        )
labels = pd.concat(
    [pd.read_parquet(p / "meta" / f"advantages_{os.environ['TAG']}.parquet", columns=["advantage"])["advantage"].astype(bool) for p in paths],
    ignore_index=True,
)
positive = int(labels.sum())
negative = int((~labels).sum())
print(f"rollout advantage frames: positive={positive}, negative={negative}")
if positive == 0 or negative == 0:
    raise SystemExit("ERROR: rollout data must contain both positive and negative labels")
PY

ROLLOUT_SHARD_WEIGHT=$(python -c "print(float('$ROLLOUT_WEIGHT') / int('${#ROLLOUT_SHARDS[@]}'))")
SFT_WEIGHT="$SFT_WEIGHT" ROLLOUT_SHARD_WEIGHT="$ROLLOUT_SHARD_WEIGHT" \
NUM_SHARDS="${#ROLLOUT_SHARDS[@]}" QUOTA_CYCLE_SIZE="$QUOTA_CYCLE_SIZE" \
MAX_STEPS="$MAX_STEPS" GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
POSITIVE_FRACTION="$POSITIVE_FRACTION" python - <<'PY'
import os

cycle = int(os.environ["QUOTA_CYCLE_SIZE"])
steps = int(os.environ["MAX_STEPS"])
batch = int(os.environ["GLOBAL_BATCH_SIZE"])
weights = [float(os.environ["SFT_WEIGHT"])] + [
    float(os.environ["ROLLOUT_SHARD_WEIGHT"])
] * int(os.environ["NUM_SHARDS"])
exact = [weight * cycle for weight in weights]
counts = [round(value) for value in exact]
if any(abs(value - count) > 1e-6 for value, count in zip(exact, counts)):
    raise SystemExit(
        f"ERROR: quota cycle {cycle} does not give integer dataset counts: {exact}"
    )
if sum(counts) != cycle:
    raise SystemExit(f"ERROR: quota counts {counts} do not sum to cycle {cycle}")
if steps * batch % cycle != 0:
    raise SystemExit(
        f"ERROR: MAX_STEPS*GLOBAL_BATCH_SIZE={steps * batch} must be a multiple "
        f"of QUOTA_CYCLE_SIZE={cycle}"
    )
if any(count < 2 for count in counts[1:]):
    raise SystemExit(
        f"ERROR: every rollout shard needs at least 2 slots for positive/negative: {counts}"
    )
positive_fraction = float(os.environ["POSITIVE_FRACTION"])
label_counts = []
for count in counts[1:]:
    positive = min(max(round(count * positive_fraction), 1), count - 1)
    label_counts.append((positive, count - positive))
print(f"quota counts: sft={counts[0]}, rollout/task={counts[1:]}")
print(f"rollout label quotas/task (positive, negative): {label_counts}")
PY
MIXTURE_OVERRIDE="data.train_data_paths=[{dataset_path:$SFT_DATASET,type:sft,weight:$SFT_WEIGHT}"
for dataset in "${ROLLOUT_SHARDS[@]}"; do
  MIXTURE_OVERRIDE+=",{dataset_path:$dataset,type:rollout,weight:$ROLLOUT_SHARD_WEIGHT}"
done
MIXTURE_OVERRIDE+="]"
echo "Mixed RECAP datasets: 1 SFT + ${#ROLLOUT_SHARDS[@]} rollout shard(s)"
echo "Sampling: episode-balanced, positive_fraction=$POSITIVE_FRACTION"
echo "Quota: cycle=$QUOTA_CYCLE_SIZE, global_batch_size=$GLOBAL_BATCH_SIZE"
echo "Training defaults: steps=$MAX_STEPS save_interval=$SAVE_INTERVAL lr=$LR"
if [ "${CHECK_ONLY:-0}" = "1" ]; then
  echo "Mixed RECAP preflight passed."
  echo "$MIXTURE_OVERRIDE"
  exit 0
fi

PROJECT="$PROJECT" \
DATASET_PATH="$SFT_DATASET" \
TAG="$TAG" \
AUTO_PREPARE=0 \
TRAIN_DATA_PATHS_OVERRIDE="$MIXTURE_OVERRIDE" \
MAX_STEPS="$MAX_STEPS" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
LR="$LR" \
bash "$PROJECT/scripts_local/11_rbm_recap_cfg_sft.sh" \
  data.balance_dataset_weights=False \
  data.episode_balanced=True \
  data.positive_fraction="$POSITIVE_FRACTION" \
  data.quota_cycle_size="$QUOTA_CYCLE_SIZE" \
  actor.global_batch_size="$GLOBAL_BATCH_SIZE" \
  "$@"
