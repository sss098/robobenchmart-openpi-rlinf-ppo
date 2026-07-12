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

if [ -f "$LOG_DIR/run_embodiment.log" ]; then
  tail -f "$LOG_DIR/run_embodiment.log"
else
  echo "run_embodiment.log not found. Existing files:"
  find "$LOG_DIR" -maxdepth 3 -type f | head -50
fi
