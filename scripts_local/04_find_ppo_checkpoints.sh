#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/projects/RLinf
cd "$PROJECT"

LOG_DIR=$(ls -td logs/*libero_10_ppo_openpi_pi05* 2>/dev/null | head -1 || true)

if [ -z "$LOG_DIR" ]; then
  echo "ERROR: no PPO log directory found under $PROJECT/logs"
  exit 1
fi

echo "Latest LOG_DIR=$PROJECT/$LOG_DIR"
echo

echo "Checkpoint directories:"
find "$LOG_DIR" -maxdepth 5 -type d -name 'global_step_*' -print || true

echo
echo "Checkpoint files:"
find "$LOG_DIR" -maxdepth 8 -type f \( -name '*.pt' -o -name '*.safetensors' -o -name '*.bin' \) -print || true
