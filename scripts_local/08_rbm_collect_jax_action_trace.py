#!/usr/bin/env python3
"""Collect RoboBenchMart action traces from the official JAX websocket server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


def prepare_obs(obs_raw, language_instruction: str, time_step: int) -> dict:
    return {
        "observation/image": obs_raw["sensor_data"]["left_base_camera_link"]["rgb"][0]
        .cpu()
        .numpy(),
        "observation/extra_image": obs_raw["sensor_data"]["right_base_camera_link"]["rgb"][0]
        .cpu()
        .numpy(),
        "observation/wrist_image": obs_raw["sensor_data"]["fetch_hand"]["rgb"][0]
        .cpu()
        .numpy(),
        "observation/state": obs_raw["agent"]["qpos"][0].cpu().numpy(),
        "prompt": language_instruction,
        "time_step": time_step,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbm-root", default="/root/autodl-tmp/projects/RoboBenchMart")
    parser.add_argument("--env-id", default="PickToBasketContFantaEnv")
    parser.add_argument("--scene-dir", default="/root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket")
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--robot-init-pose-start-seed", type=int, default=10000)
    parser.add_argument("--num-traj", type=int, default=3)
    parser.add_argument("--max-horizon", type=int, default=600)
    parser.add_argument("--sim-backend", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rbm_root = Path(args.rbm_root).resolve()
    os.chdir(rbm_root)
    sys.path.insert(0, str(rbm_root))

    import dsynth.envs  # noqa: F401
    import dsynth.robots  # noqa: F401
    from dsynth.web_utils import WebsocketClient

    env = gym.make(
        args.env_id,
        robot_uids="ds_fetch_basket",
        config_dir_path=args.scene_dir,
        num_envs=1,
        control_mode="pd_joint_pos",
        obs_mode="rgb",
        sim_backend=args.sim_backend,
        render_mode="rgb_array",
        enable_shadow=True,
        parallel_in_single_scene=False,
        viewer_camera_configs={"shader_pack": "default"},
        human_render_camera_configs={"shader_pack": "default"},
    )
    client = WebsocketClient(host=args.host, port=args.port)

    raw_chunks = []
    sent_chunks = []
    traj_ids = []
    chunk_steps = []
    successes = []

    for traj_idx in range(args.num_traj):
        seed = args.start_seed + traj_idx
        reset_options = {"reconfigure": True}
        if args.robot_init_pose_start_seed is not None:
            reset_options["robot_init_pose_seed"] = args.robot_init_pose_start_seed + traj_idx
        obs, info = env.reset(seed=seed, options=reset_options)
        language_instruction = env.language_instructions[0]

        t = 0
        while t < args.max_horizon:
            obs_prepared = prepare_obs(obs, language_instruction, t)
            actions = np.asarray(client.infer(obs_prepared)["actions"], dtype=np.float32)
            sent_actions = actions[:, :13].copy()
            sent_actions[:, 8] = 0.0
            sent_actions[:, 9] = 0.0

            raw_chunks.append(actions.astype(np.float32, copy=False))
            sent_chunks.append(sent_actions)
            traj_ids.append(traj_idx)
            chunk_steps.append(t)

            done_any = False
            for action in sent_actions:
                obs, reward, done, trunc, info = env.step(action)
                t += 1
                if bool(done) or bool(trunc) or t >= args.max_horizon:
                    done_any = True
                    break
            if done_any:
                break

        successes.append(bool(info["success"][0].item()))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        raw_actions=np.asarray(raw_chunks, dtype=np.float32),
        actions=np.asarray(sent_chunks, dtype=np.float32),
        traj_ids=np.asarray(traj_ids, dtype=np.int64),
        chunk_steps=np.asarray(chunk_steps, dtype=np.int64),
        successes=np.asarray(successes, dtype=np.bool_),
        env_id=np.asarray(args.env_id),
        start_seed=np.asarray(args.start_seed, dtype=np.int64),
        robot_init_pose_start_seed=np.asarray(args.robot_init_pose_start_seed, dtype=np.int64),
    )
    print(f"saved {len(sent_chunks)} action chunks to {out}")
    print(f"success_rate={np.mean(successes) if successes else 0.0:.4f}")


if __name__ == "__main__":
    main()
