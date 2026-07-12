#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-$(pwd)}
DATASET_PATH=${DATASET_PATH:-/root/autodl-tmp/lerobot_data/rbm_dataset}
TAG=${TAG:-rbm_recap_lite_sft_pos}
DATASET_TYPE=${DATASET_TYPE:-sft}
LABEL_MODE=${LABEL_MODE:-auto}

if [ "$DATASET_TYPE" != "sft" ] && [ "$DATASET_TYPE" != "rollout" ] && [ "$DATASET_TYPE" != "correction" ]; then
  echo "ERROR: DATASET_TYPE must be sft, rollout, or correction; got: $DATASET_TYPE" >&2
  exit 2
fi

DATASETS=()
if [ -d "$DATASET_PATH/data" ] && [ -d "$DATASET_PATH/meta" ]; then
  DATASETS=("$DATASET_PATH")
elif [ -d "$DATASET_PATH" ]; then
  while IFS= read -r info_file; do
    DATASETS+=("$(dirname "$(dirname "$info_file")")")
  done < <(find "$DATASET_PATH" -type f -path '*/meta/info.json' | sort)
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "ERROR: incomplete LeRobot dataset: $DATASET_PATH" >&2
  echo "Required directories are missing:" >&2
  echo "  $DATASET_PATH/data" >&2
  echo "  $DATASET_PATH/meta" >&2
  if [ "$DATASET_TYPE" = "rollout" ]; then
    echo >&2
    echo "This command labels an existing rollout dataset; it does not collect rollouts." >&2
    echo "Collect and export Fanta/Nivea/Stars episodes with success labels first." >&2
    echo "Do not create empty data/meta directories: mixed RECAP requires real parquet frames." >&2
  fi
  exit 2
fi

cd "$PROJECT"
source .venv_openpi/bin/activate

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

for dataset in "${DATASETS[@]}"; do
  if ! find "$dataset/data" -type f -name '*.parquet' -print -quit | grep -q .; then
    echo "ERROR: no parquet frames found under $dataset/data" >&2
    exit 2
  fi
  echo "Preparing advantages for dataset shard: $dataset"
  python examples/recap/process/make_recap_lite_advantages.py \
    --dataset-path "$dataset" \
    --tag "$TAG" \
    --dataset-type "$DATASET_TYPE" \
    --label-mode "$LABEL_MODE" \
    --overwrite \
    "$@"
done
