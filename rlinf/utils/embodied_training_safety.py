import torch


def select_episode_seed_values(
    seed_lists: list[list[int]],
    *,
    seed_offset: int,
    reset_count: int,
    num_envs: int,
) -> int | list[int]:
    """Select the next per-task episode seeds from an explicit evaluation list."""
    if not seed_lists:
        raise ValueError("episode seed lists must not be empty")
    task_index = seed_offset % len(seed_lists)
    task_seeds = seed_lists[task_index]
    start = reset_count * num_envs
    selected = task_seeds[start : start + num_envs]
    if len(selected) != num_envs:
        raise ValueError(
            f"episode seed list {task_index} is exhausted at reset {reset_count}: "
            f"need {num_envs} seeds, have {len(selected)}"
        )
    return selected[0] if num_envs == 1 else selected


def stage_channel_extra(mode: str, stage_id: int, payload: str) -> str:
    """Build a pipeline-stage-specific channel suffix shared by both peers."""
    if mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if stage_id < 0:
        raise ValueError(f"stage_id must be non-negative, got {stage_id}")
    return f"{mode}_stage_{stage_id}_{payload}"


def adapt_reference_kl_beta(
    beta: float,
    measured_kl: float,
    *,
    target_kl: float,
    beta_min: float,
    beta_max: float,
    adaptation_rate: float,
) -> float:
    """Adjust the reference-KL coefficient outside a two-sided dead band."""
    if target_kl <= 0 or adaptation_rate <= 1:
        return beta
    if measured_kl > target_kl * 2:
        return min(beta * adaptation_rate, beta_max)
    if measured_kl < target_kl / 2:
        return max(beta / adaptation_rate, beta_min)
    return beta


def count_successful_shaped_reward_trajectories(
    rewards: torch.Tensor, success_reward_threshold: float
) -> int:
    """Count trajectories containing the one-shot RBM success reward event."""
    if rewards.ndim < 2:
        raise ValueError(f"Expected rollout rewards with at least 2 dims, got {rewards.shape}")
    reduce_dims = tuple(dim for dim in range(rewards.ndim) if dim != 1)
    trajectory_returns = rewards.sum(dim=reduce_dims)
    return int((trajectory_returns >= success_reward_threshold).sum().item())


def validate_reference_kl_zero_drift(
    measured_abs_kl: float, tolerance: float
) -> None:
    """Reject a biased reference KL before the policy's first update."""
    if measured_abs_kl > tolerance:
        raise RuntimeError(
            "Reference-KL zero-drift invariant failed before the first actor "
            f"update: abs_kl={measured_abs_kl:.6f} > tolerance={tolerance:.6f}. "
            "Do not train until reference/current logprob parity is fixed."
        )


def compute_rollout_explained_variance(
    returns: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> float:
    """Compute critic explained variance once over a complete rollout batch."""
    returns = returns.detach().float()
    values = values.detach().float()
    if values.shape[0] == returns.shape[0] + 1:
        values = values[:-1]
    if values.shape != returns.shape:
        values = values.expand_as(returns)
    if loss_mask is not None:
        mask = loss_mask.to(device=returns.device, dtype=torch.bool)
        if mask.shape != returns.shape:
            mask = mask.expand_as(returns)
        returns = returns[mask]
        values = values[mask]
    else:
        returns = returns.reshape(-1)
        values = values.reshape(-1)
    if returns.numel() < 2:
        return float("nan")
    var_returns = torch.var(returns, unbiased=False)
    if not torch.isfinite(var_returns) or var_returns <= 0:
        return float("nan")
    var_error = torch.var(returns - values, unbiased=False)
    if not torch.isfinite(var_error):
        return float("nan")
    return float((1 - var_error / var_returns).item())


def apply_episode_time_limit(
    truncations: torch.Tensor,
    elapsed_steps: torch.Tensor,
    max_episode_steps: int,
) -> torch.Tensor:
    """Apply the configured task horizon to vectorized environments."""
    if max_episode_steps <= 0:
        return truncations
    return torch.logical_or(truncations, elapsed_steps >= max_episode_steps)


def mask_finished_env_step(
    reward: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove rewards and repeated done flags after an env finished a chunk."""
    active_mask = active_mask.to(device=reward.device, dtype=torch.bool)
    return (
        torch.where(active_mask, reward, torch.zeros_like(reward)),
        terminations & active_mask,
        truncations & active_mask,
    )


def compose_pick_to_basket_reward(
    *,
    first_success: torch.Tensor,
    first_placed: torch.Tensor,
    first_lifted: torch.Tensor,
    basket_progress: torch.Tensor,
    first_placed_static: torch.Tensor,
    first_non_target_displacement: torch.Tensor,
) -> torch.Tensor:
    """Compose bounded event/progress reward from already-detected transitions."""
    reward = (
        5.00 * first_success.float()
        + 0.20 * first_placed.float()
        + 0.05 * first_lifted.float()
        + 0.05 * basket_progress
        + 0.10 * first_placed_static.float()
        - 0.10 * first_non_target_displacement.float()
    )
    return torch.clamp(reward, min=-0.1, max=5.4)


def compute_cfg_routing_masks(
    advantage: torch.Tensor,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Route positive, negative, and dropped-condition CFG samples."""
    advantage = advantage.to(dtype=torch.bool)
    if random_values is None:
        random_values = torch.rand(advantage.shape[0], device=advantage.device)
    else:
        random_values = random_values.to(device=advantage.device)

    positive_mask = advantage
    negative_mask = ~positive_mask
    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (random_values > unconditional_prob)
        negative_conditional_mask = torch.zeros_like(positive_mask)
    else:
        guidance_mask = random_values > unconditional_prob
        positive_conditional_mask = positive_mask & guidance_mask
        negative_conditional_mask = negative_mask & guidance_mask

    conditional_mask = positive_conditional_mask | negative_conditional_mask
    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_unconditional_mask": positive_mask & ~positive_conditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "negative_unconditional_mask": negative_mask & ~negative_conditional_mask,
    }
