# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import torch
from omegaconf import ListConfig, open_dict

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv


def _ensure_robobenchmart_importable(project_path: str | None) -> None:
    if project_path:
        rbm_path = str(Path(project_path).expanduser().resolve())
        if rbm_path not in sys.path:
            sys.path.insert(0, rbm_path)
        os.chdir(rbm_path)

    # Import side effects register dsynth env ids and robots with gymnasium/ManiSkill.
    import dsynth.envs  # noqa: F401
    import dsynth.robots  # noqa: F401
    import rlinf.envs.robobenchmart.proxy_tasks  # noqa: F401


def _select_robobenchmart_task(cfg, seed_offset: int):
    env_ids = cfg.init_params.id
    if not isinstance(env_ids, (list, tuple, ListConfig)):
        return cfg
    if len(env_ids) == 0:
        raise ValueError("RoboBenchMart init_params.id list must not be empty.")

    selected_env_id = str(env_ids[seed_offset % len(env_ids)])
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        cfg.init_params.id = selected_env_id
        cfg.selected_env_id = selected_env_id
    return cfg


class RoboBenchMartEnv(ManiskillEnv):
    """RLinf env wrapper for RoboBenchMart ManiSkill environments."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        _ensure_robobenchmart_importable(getattr(cfg, "project_path", None))
        cfg = _select_robobenchmart_task(cfg, seed_offset)
        self._rbm_seed_offset = seed_offset
        self._rbm_reset_count = 0
        super().__init__(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
            record_metrics=record_metrics,
        )
        self._non_target_penalty_applied = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._target_lifted_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._target_placed_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._target_placed_static_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prev_basket_proximity = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.info_logging_keys = [
            "is_obj_placed",
            "is_robot_static",
            "is_non_target_produncts_displaced",
            "success",
        ]

    @property
    def instruction(self):
        instructions = getattr(self.env.unwrapped, "language_instructions", None)
        if instructions is None:
            return ["move to shelf and pick target to basket"] * self.num_envs
        return list(instructions)

    def _wrap_obs(self, raw_obs: dict[str, Any], infos=None) -> dict[str, Any]:
        sensor_data = raw_obs["sensor_data"]
        return {
            "main_images": sensor_data["left_base_camera_link"]["rgb"],
            "wrist_images": sensor_data["fetch_hand"]["rgb"],
            "extra_view_images": sensor_data["right_base_camera_link"]["rgb"],
            "states": raw_obs["agent"]["qpos"].to(torch.float32),
            "task_descriptions": self.instruction,
        }

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        full_reset = options is None or "env_idx" not in options
        options = dict(options or {})
        episode_seed_start = getattr(self.cfg, "episode_seed_start", None)
        if episode_seed_start is not None and full_reset:
            seed = self._episode_seed_values(int(episode_seed_start))
        elif seed is None and full_reset:
            seed = self.seed
        if getattr(self.cfg, "reconfigure_on_reset", True):
            options.setdefault("reconfigure", True)

        robot_seed_start = getattr(self.cfg, "robot_init_pose_start_seed", None)
        if robot_seed_start is not None and full_reset:
            options.setdefault(
                "robot_init_pose_seed",
                self._episode_seed_values(int(robot_seed_start)),
            )

        if full_reset:
            self._rbm_reset_count += 1

        extracted_obs, infos = super().reset(seed=seed, options=options)
        self._reset_non_target_penalty_state(options)
        return extracted_obs, infos

    def _reset_non_target_penalty_state(self, options: dict) -> None:
        if not hasattr(self, "_non_target_penalty_applied"):
            return
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._non_target_penalty_applied[env_idx] = False
            self._target_lifted_once[env_idx] = False
            self._target_placed_once[env_idx] = False
            self._target_placed_static_once[env_idx] = False
            self._prev_basket_proximity[env_idx] = 0.0
        else:
            self._non_target_penalty_applied[:] = False
            self._target_lifted_once[:] = False
            self._target_placed_once[:] = False
            self._target_placed_static_once[:] = False
            self._prev_basket_proximity[:] = 0.0

    def _episode_seed_values(self, start_seed: int) -> int | list[int]:
        """Match RoboBenchMart official eval: trajectory seeds increase by episode."""
        start = start_seed + self._rbm_reset_count * self.num_envs
        seeds = [start + i for i in range(self.num_envs)]
        return seeds[0] if self.num_envs == 1 else seeds

    def _calc_step_reward(self, reward, info):
        reward_mode = getattr(self.cfg, "reward_mode", "only_success")
        if reward_mode == "raw":
            return reward.to(torch.float32)
        if reward_mode in {"rbm_shaped", "shaped"}:
            step_reward = self._calc_pick_to_basket_shaped_reward(info)
        else:
            success = info.get("success")
            if success is None:
                return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            step_reward = success.to(torch.float32)

        if getattr(self.cfg, "use_rel_reward", False):
            reward_diff = step_reward - self.prev_step_reward
            self.prev_step_reward = step_reward
            return reward_diff

        self.prev_step_reward = step_reward
        return step_reward

    def _calc_pick_to_basket_shaped_reward(self, info):
        success = info.get("success")
        if success is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        success_f = success.to(torch.float32)
        placed = info.get("is_obj_placed", success).to(torch.bool)
        static = info.get(
            "is_robot_static",
            torch.zeros_like(success, dtype=torch.bool),
        ).to(torch.bool)
        non_target = info.get(
            "is_non_target_produncts_displaced",
            torch.zeros_like(success, dtype=torch.bool),
        ).to(torch.bool)

        lifted_f, basket_proximity = self._target_object_progress_terms()
        lifted = lifted_f.to(torch.bool)

        first_lifted = lifted & ~self._target_lifted_once
        first_placed = placed & ~self._target_placed_once
        placed_static = placed & static
        first_placed_static = placed_static & ~self._target_placed_static_once

        basket_progress = torch.clamp(
            basket_proximity - self._prev_basket_proximity,
            min=0.0,
            max=1.0,
        )
        self._prev_basket_proximity = torch.maximum(
            self._prev_basket_proximity,
            basket_proximity,
        )

        first_non_target_displacement = non_target & ~self._non_target_penalty_applied

        self._target_lifted_once |= lifted
        self._target_placed_once |= placed
        self._target_placed_static_once |= placed_static
        self._non_target_penalty_applied |= non_target

        step_reward = (
            5.00 * success_f
            + 1.00 * first_placed.to(torch.float32)
            + 0.30 * first_lifted.to(torch.float32)
            + 0.50 * basket_progress
            + 0.30 * first_placed_static.to(torch.float32)
            - 0.20 * first_non_target_displacement.to(torch.float32)
        )
        return torch.clamp(step_reward, min=-0.2, max=6.0)

    def _target_object_progress_terms(self):
        lifted = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        proximity = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        env = self.env.unwrapped
        if not all(
            hasattr(env, name)
            for name in ("target_products_df", "actors", "products_initial_poses")
        ):
            return lifted, proximity

        try:
            target_pos = env.calc_target_pose().p
        except Exception:
            return lifted, proximity

        for scene_idx in range(self.num_envs):
            target_products_df = env.target_products_df[
                env.target_products_df["scene_idx"] == scene_idx
            ]
            if len(target_products_df) == 0:
                continue

            distances = []
            lift_flags = []
            for actor_name in target_products_df["actor_name"]:
                actor = env.actors["products"].get(actor_name)
                if actor is None:
                    continue
                pos = actor.pose.p
                distances.append(torch.linalg.norm(pos - target_pos[scene_idx], dim=-1))

                initial_pose = env.products_initial_poses.get(actor_name)
                if initial_pose is not None:
                    initial_z = initial_pose[..., 2]
                    lift_flags.append(pos[..., 2] > initial_z + 0.03)

            if distances:
                min_dist = torch.min(torch.stack(distances).reshape(-1))
                proximity[scene_idx] = torch.clamp(1.0 - min_dist / 1.0, 0.0, 1.0)
            if lift_flags:
                lifted[scene_idx] = torch.stack(lift_flags).any().to(torch.float32)

        return lifted, proximity
