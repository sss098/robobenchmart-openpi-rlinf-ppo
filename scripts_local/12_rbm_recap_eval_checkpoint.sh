#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts_local/12_rbm_recap_eval_checkpoint.sh BASE"
  echo "  bash scripts_local/12_rbm_recap_eval_checkpoint.sh /path/to/global_step_x_or_actor_or_full_weights.pt"
  echo
  echo "Common overrides:"
  echo "  EVAL_SPLIT=train3|ood|proxy EVAL_EPOCHS=30 SAVE_VIDEO=True GUIDANCE_SCALE=1.0 bash scripts_local/12_rbm_recap_eval_checkpoint.sh ..."
  exit 1
fi

PROJECT=${PROJECT:-$(pwd)}
RBM_PROJECT=${RBM_PROJECT:-/root/autodl-tmp/projects/RoboBenchMart}
OPENPI_PROJECT=${OPENPI_PROJECT:-/root/autodl-tmp/projects/openpi}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/autodl-tmp/checkpoints/rbm_pi05_sft_pytorch}
CONFIG=${CONFIG:-rbm_pick_to_basket_ppo_openpi_pi05}
CKPT_INPUT=$1
shift

EVAL_SPLIT=${EVAL_SPLIT:-train3}
if [ "$EVAL_SPLIT" = "train3" ]; then
  PIPELINE_STAGES=3
  TOTAL_NUM_ENVS=3
  EVAL_EPOCHS=${EVAL_EPOCHS:-30}
  ENV_OVERRIDES=(
    'env.eval.episode_seed_start=null'
    'env.eval.episode_seed_lists=[[4,8,12,16,20,24,28,32,36,40,44,52,56,72,76,80,88,96,100,104,108,116,120,124,128,132,136,140,148,152],[0,4,8,12,20,24,32,36,40,44,48,52,56,60,64,72,80,84,88,92,96,100,108,116,120,124,128,132,136,140],[0,4,8,16,20,28,36,40,48,60,68,76,84,88,92,96,104,108,116,120,124,128,132,136,140,144,152,156,160,168]]'
    'env.eval.robot_init_pose_start_seed=null'
  )
elif [ "$EVAL_SPLIT" = "ood" ]; then
  PIPELINE_STAGES=2
  TOTAL_NUM_ENVS=2
  EVAL_EPOCHS=${EVAL_EPOCHS:-30}
  ENV_OVERRIDES=(
    'env.eval.init_params.id=[PickToBasketContNestleEnv,PickToBasketContSlamEnv]'
    "env.eval.init_params.config_dir_path=$RBM_PROJECT/demo_envs/test_unseen_items_pick_to_basket"
    'env.eval.episode_seed_lists=null'
    'env.eval.episode_seed_start=42000'
    'env.eval.robot_init_pose_start_seed=10000'
  )
elif [ "$EVAL_SPLIT" = "proxy" ]; then
  PIPELINE_STAGES=1
  TOTAL_NUM_ENVS=1
  EVAL_EPOCHS=${EVAL_EPOCHS:-30}
  ENV_OVERRIDES=(
    'env.eval.init_params.id=[PickToBasketProxyRandomEnv]'
    "env.eval.init_params.config_dir_path=$RBM_PROJECT/demo_envs/pick_to_basket"
    'env.eval.episode_seed_lists=null'
    'env.eval.episode_seed_start=0'
    'env.eval.robot_init_pose_start_seed=null'
  )
else
  echo "ERROR: EVAL_SPLIT must be train3, ood, or proxy, got: $EVAL_SPLIT"
  exit 1
fi

FULL_WEIGHTS=""
if [ "$CKPT_INPUT" = "BASE" ]; then
  CKPT_ARGS=()
elif [ -f "$CKPT_INPUT" ]; then
  FULL_WEIGHTS=$CKPT_INPUT
elif [ -f "$CKPT_INPUT/model_state_dict/full_weights.pt" ]; then
  FULL_WEIGHTS="$CKPT_INPUT/model_state_dict/full_weights.pt"
elif [ -f "$CKPT_INPUT/actor/model_state_dict/full_weights.pt" ]; then
  FULL_WEIGHTS="$CKPT_INPUT/actor/model_state_dict/full_weights.pt"
else
  echo "ERROR: cannot find full_weights.pt under: $CKPT_INPUT"
  exit 1
fi

if [ -n "$FULL_WEIGHTS" ]; then
  CKPT_ARGS=(runner.ckpt_path="$FULL_WEIGHTS")
fi

GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
GUIDANCE_TYPE=${GUIDANCE_TYPE:-positive}
UNCONDITIONAL_PROB=${UNCONDITIONAL_PROB:-0.3}
SAVE_VIDEO=${SAVE_VIDEO:-False}
VIDEO_DIR=${VIDEO_DIR:-/root/autodl-tmp/projects/RLinf/logs/rbm_recap_eval_videos}

cd "$PROJECT"
source .venv_openpi/bin/activate

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
export OMP_NUM_THREADS=1
export PYTHONPATH="$PROJECT:$RBM_PROJECT:$OPENPI_PROJECT:${PYTHONPATH:-}"

echo "Evaluating CFG checkpoint weights: ${FULL_WEIGHTS:-BASE model (zero-step control)}"
echo "Base OpenPI stats/model path: $BASE_MODEL_PATH"
echo "Eval split: $EVAL_SPLIT, tasks: $TOTAL_NUM_ENVS, trials/task: $EVAL_EPOCHS"

bash examples/embodiment/run_embodiment.sh "$CONFIG" LIBERO \
  runner.only_eval=True \
  runner.max_epochs=1 \
  runner.val_check_interval=-1 \
  "${CKPT_ARGS[@]}" \
  actor.model.model_path="$BASE_MODEL_PATH" \
  rollout.model.model_path="$BASE_MODEL_PATH" \
  actor.model.model_type=cfg_model \
  actor.model.add_value_head=False \
  actor.model.openpi.config_name=pi05_rbm \
  actor.model.openpi.num_images_in_input=3 \
  actor.model.openpi.action_chunk=50 \
  actor.model.openpi.action_env_dim=13 \
  actor.model.openpi.num_steps=10 \
  actor.model.openpi.train_expert_only=True \
  +actor.model.openpi.guidance_type="$GUIDANCE_TYPE" \
  +actor.model.openpi.positive_only_conditional=False \
  actor.model.num_action_chunks=50 \
  actor.model.num_steps=10 \
  +actor.model.openpi.cfgrl_guidance_scale="$GUIDANCE_SCALE" \
  +actor.model.openpi.unconditional_prob="$UNCONDITIONAL_PROB" \
  actor.enable_sft_co_train=False \
  env.train.project_path="$RBM_PROJECT" \
  env.eval.project_path="$RBM_PROJECT" \
  env.eval.total_num_envs="$TOTAL_NUM_ENVS" \
  env.eval.max_steps_per_rollout_epoch=600 \
  env.eval.max_episode_steps=600 \
  env.eval.auto_reset=False \
  env.eval.ignore_terminations=False \
  env.eval.init_params.sim_backend=cpu \
  env.eval.video_cfg.save_video="$SAVE_VIDEO" \
  env.eval.video_cfg.video_base_dir="$VIDEO_DIR" \
  algorithm.eval_rollout_epoch="$EVAL_EPOCHS" \
  algorithm.sampling_params.do_sample=False \
  algorithm.sampling_params.temperature_eval=0.0 \
  rollout.pipeline_stage_num="$PIPELINE_STAGES" \
  "${ENV_OVERRIDES[@]}" \
  "$@"
