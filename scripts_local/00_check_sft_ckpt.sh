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

echo "PROJECT=$PROJECT"
echo "SFT_CKPT=$SFT_CKPT"

if [ ! -d "$SFT_CKPT" ]; then
  echo "ERROR: SFT checkpoint directory not found: $SFT_CKPT"
  exit 1
fi

if [ ! -f "$SFT_CKPT/model.safetensors" ]; then
  echo "ERROR: model.safetensors not found in $SFT_CKPT"
  exit 1
fi

if [ ! -f "$SFT_CKPT/assets/physical-intelligence/libero/norm_stats.json" ]; then
  echo "ERROR: norm_stats.json not found:"
  echo "$SFT_CKPT/assets/physical-intelligence/libero/norm_stats.json"
  exit 1
fi

if [ ! -e "$SFT_CKPT/physical-intelligence" ]; then
  ln -s assets/physical-intelligence "$SFT_CKPT/physical-intelligence"
  echo "Created symlink: $SFT_CKPT/physical-intelligence -> assets/physical-intelligence"
else
  echo "Symlink or directory already exists: $SFT_CKPT/physical-intelligence"
fi

echo
echo "Check files:"
ls -l "$SFT_CKPT/physical-intelligence/libero/norm_stats.json"
ls -lh "$SFT_CKPT/model.safetensors"

echo
echo "SFT checkpoint check passed."
