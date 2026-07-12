from collections import defaultdict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.utils.embodied_training_safety import (
    adapt_reference_kl_beta,
    apply_episode_time_limit,
    compose_pick_to_basket_reward,
    compute_cfg_routing_masks,
    compute_rollout_explained_variance,
    count_successful_shaped_reward_trajectories,
    mask_finished_env_step,
    select_episode_seed_values,
    stage_channel_extra,
    validate_reference_kl_zero_drift,
)
from rlinf.workers.env.env_worker import EnvWorker


def test_pipeline_channel_keys_are_stage_specific():
    assert stage_channel_extra("train", 0, "obs") == "train_stage_0_obs"
    assert stage_channel_extra("train", 1, "obs") == "train_stage_1_obs"


def test_explicit_episode_seeds_follow_each_task_json_order():
    seed_lists = [[4, 8, 12], [0, 4, 8]]

    assert (
        select_episode_seed_values(seed_lists, seed_offset=0, reset_count=1, num_envs=1)
        == 8
    )
    assert (
        select_episode_seed_values(seed_lists, seed_offset=1, reset_count=2, num_envs=1)
        == 8
    )


def test_reference_kl_beta_adapts_only_outside_dead_band():
    kwargs = {
        "target_kl": 0.01,
        "beta_min": 1.0e-4,
        "beta_max": 10.0,
        "adaptation_rate": 1.5,
    }
    assert adapt_reference_kl_beta(0.1, 0.03, **kwargs) == pytest.approx(0.15)
    assert adapt_reference_kl_beta(0.1, 0.001, **kwargs) == pytest.approx(0.1 / 1.5)
    assert adapt_reference_kl_beta(0.1, 0.01, **kwargs) == 0.1


def test_configured_max_episode_steps_produces_truncation():
    truncations = torch.tensor([False, False])
    elapsed_steps = torch.tensor([1, 2])

    result = apply_episode_time_limit(truncations, elapsed_steps, 2)

    assert result.tolist() == [False, True]


def test_finished_chunk_env_has_no_repeated_reward_or_done():
    reward, terminations, truncations = mask_finished_env_step(
        reward=torch.tensor([5.0, 1.0]),
        terminations=torch.tensor([True, False]),
        truncations=torch.tensor([False, True]),
        active_mask=torch.tensor([False, True]),
    )

    assert reward.tolist() == [0.0, 1.0]
    assert terminations.tolist() == [False, False]
    assert truncations.tolist() == [False, True]


def test_rbm_initial_potential_gives_no_progress_reward():
    reward = compose_pick_to_basket_reward(
        first_success=torch.tensor([False]),
        first_placed=torch.tensor([False]),
        first_lifted=torch.tensor([False]),
        basket_progress=torch.tensor([0.0]),
        first_placed_static=torch.tensor([False]),
        first_non_target_displacement=torch.tensor([False]),
    )

    assert reward.item() == 0.0


def test_rbm_success_event_is_bounded_and_one_shot_inputs_can_be_zeroed():
    first = compose_pick_to_basket_reward(
        first_success=torch.tensor([True]),
        first_placed=torch.tensor([True]),
        first_lifted=torch.tensor([True]),
        basket_progress=torch.tensor([1.0]),
        first_placed_static=torch.tensor([True]),
        first_non_target_displacement=torch.tensor([False]),
    )
    repeated = compose_pick_to_basket_reward(
        first_success=torch.tensor([False]),
        first_placed=torch.tensor([False]),
        first_lifted=torch.tensor([False]),
        basket_progress=torch.tensor([0.0]),
        first_placed_static=torch.tensor([False]),
        first_non_target_displacement=torch.tensor([False]),
    )

    assert first.item() == pytest.approx(5.4)
    assert repeated.item() == 0.0


def test_success_count_ignores_progress_only_trajectories():
    rewards = torch.zeros(2, 3, 4)
    rewards[0, 0, 0] = 5.0
    rewards[:, 1, :] = 0.05
    rewards[1, 2, 3] = 5.0

    assert count_successful_shaped_reward_trajectories(rewards, 4.5) == 2


def test_reference_kl_zero_drift_guard_rejects_biased_baseline():
    validate_reference_kl_zero_drift(0.01, 0.02)
    with pytest.raises(RuntimeError, match="zero-drift invariant"):
        validate_reference_kl_zero_drift(0.156, 0.02)


def test_rollout_explained_variance_uses_complete_masked_batch():
    returns = torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]])
    values = torch.tensor([[[0.0], [0.8]], [[2.2], [3.0]], [[9.0], [9.0]]])
    mask = torch.tensor([[[True], [True]], [[True], [False]]])

    result = compute_rollout_explained_variance(returns, values, mask)

    assert result == pytest.approx(0.96)


def test_cfg_routes_negative_samples_to_negative_condition():
    routing = compute_cfg_routing_masks(
        torch.tensor([True, False]),
        positive_only_conditional=False,
        unconditional_prob=0.0,
        random_values=torch.ones(2),
    )

    assert routing["positive_conditional_mask"].tolist() == [True, False]
    assert routing["negative_conditional_mask"].tolist() == [False, True]


def test_fixed_episode_metrics_keep_all_pipeline_stages_without_holes():
    worker = object.__new__(EnvWorker)
    worker.stage_num = 3
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "init_params": {"id": ["Fanta", "Nivea", "Stars"]},
                }
            }
        }
    )
    metrics = defaultdict(list)

    for stage_id, success in enumerate((True, False, True)):
        worker.record_env_metrics(
            metrics,
            {"success_once": torch.tensor([success])},
            epoch=0,
            stage_id=stage_id,
        )

    assert [value.item() for value in metrics["success_once"]] == [True, False, True]
    assert metrics["success_once/Fanta"][0].item() is True
    assert metrics["success_once/Nivea"][0].item() is False
    assert metrics["success_once/Stars"][0].item() is True
    assert all(value is not None for values in metrics.values() for value in values)


def test_lerobot_export_truncates_frozen_actions_after_raw_chunk_done():
    collector = object.__new__(CollectEpisode)
    collector.num_envs = 1
    obs = {
        "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_images": np.zeros((4, 4, 3), dtype=np.uint8),
        "extra_view_images": np.zeros((4, 4, 3), dtype=np.uint8),
        "states": np.zeros(13, dtype=np.float32),
        "task_descriptions": ["pick item to basket"],
    }
    buf = {
        "observations": [obs, obs, obs],
        "actions": [np.zeros(13), np.ones(13), np.ones(13) * 2],
        "terminated": [False, False, True],
        "truncated": [False, False, False],
        "infos": [
            {},
            {"_rlinf_step_terminated": True},
            {"_rlinf_step_terminated": False},
            {"_rlinf_step_terminated": False},
        ],
    }

    episode = collector._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)

    assert episode is not None
    assert len(episode) == 1
    assert episode[0]["done"].item() is True
