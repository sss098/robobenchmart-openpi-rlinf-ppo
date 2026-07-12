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

"""CFG-model dataset and dataloader helpers for ReCap training."""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Any, Iterator

import numpy as np
import torch

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation as CFGObservation,
)

from .common import BaseDataLoaderImpl, ReCapMixtureDataset, _safe_hash

logger = logging.getLogger(__name__)


class AdvantagePreservingDataset:
    """Wrapper to preserve advantage through the OpenPI transform pipeline."""

    def __init__(
        self,
        base_dataset: Any,
        transformed_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None = None,
        sampling_type: str = "sft",
    ):
        self.sampling_type = sampling_type
        self._transformed_dataset = transformed_dataset
        self._advantage_by_index = self._build_advantage_index(
            base_dataset, advantages_lookup
        )
        self._base_dataset = base_dataset if self._advantage_by_index is None else None
        self._episode_indices_by_label = self._build_episode_indices(base_dataset)

    def _build_episode_indices(
        self, base_dataset: Any
    ) -> dict[bool, tuple[tuple[int, ...], ...]] | None:
        """Index frames by label and episode for length-unbiased sampling."""
        if self._advantage_by_index is None:
            return None
        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            return None
        episode_indices = hf_dataset["episode_index"]
        grouped: dict[bool, dict[int, list[int]]] = {
            True: defaultdict(list),
            False: defaultdict(list),
        }
        for index, episode_index in enumerate(episode_indices):
            grouped[self._advantage_by_index[index]][int(episode_index)].append(index)
        return {
            label: tuple(tuple(indices) for indices in episodes.values())
            for label, episodes in grouped.items()
        }

    def sample_episode_balanced_index(
        self,
        rng: Any,
        positive_fraction: float,
        forced_label: bool | None = None,
    ) -> int:
        """Sample label, episode, then frame so long failures are not overweighted."""
        if self._episode_indices_by_label is None:
            return int(rng.integers(0, len(self)))
        label = (
            forced_label
            if forced_label is not None
            else bool(rng.random() < positive_fraction)
        )
        episodes = self._episode_indices_by_label[label]
        if not episodes:
            episodes = self._episode_indices_by_label[not label]
        episode = episodes[int(rng.integers(0, len(episodes)))]
        return episode[int(rng.integers(0, len(episode)))]

    @staticmethod
    def _get_hf_dataset(dataset: Any) -> Any:
        current = dataset
        while current is not None:
            if hasattr(current, "hf_dataset"):
                return current.hf_dataset
            if hasattr(current, "_dataset"):
                current = current._dataset
            else:
                return None
        return None

    def _build_advantage_index(
        self,
        base_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None,
    ) -> dict[int, bool] | None:
        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            logger.warning(
                "Cannot access underlying HF dataset, "
                "falling back to per-sample advantage loading (slower)."
            )
            return None

        if advantages_lookup is not None:
            ep_indices = hf_dataset["episode_index"]
            frame_indices = hf_dataset["frame_index"]
            advantage_by_index = {}
            missing_keys = []
            for i in range(len(hf_dataset)):
                key = (int(ep_indices[i]), int(frame_indices[i]))
                if key in advantages_lookup:
                    advantage_by_index[i] = advantages_lookup[key]
                else:
                    missing_keys.append(key)
            if missing_keys:
                raise ValueError(
                    f"[AdvantagePreservingDataset] {len(missing_keys)} samples not found "
                    f"in advantages lookup (first 5: {missing_keys[:5]}). "
                    f"The advantages parquet does not match this dataset. "
                    f"Re-run compute_advantages.py."
                )
            return advantage_by_index

        if "advantage" in hf_dataset.column_names:
            advantages = hf_dataset["advantage"]
            return {i: bool(v) for i, v in enumerate(advantages)}

        raise ValueError(
            "[AdvantagePreservingDataset] No advantage data found: "
            "advantages_lookup is None, and 'advantage' column not in dataset. "
            "Run compute_advantages.py first."
        )

    def __len__(self) -> int:
        return len(self._transformed_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._transformed_dataset[idx]

        if self._advantage_by_index is not None:
            if idx not in self._advantage_by_index:
                raise KeyError(
                    f"[AdvantagePreservingDataset] Index {idx} not found in advantage index. "
                    f"Dataset size: {len(self._transformed_dataset)}, "
                    f"advantage index size: {len(self._advantage_by_index)}."
                )
            sample["advantage"] = self._advantage_by_index[idx]
        else:
            base_sample = self._base_dataset[idx]
            if "advantage" not in base_sample:
                raise KeyError(
                    f"[AdvantagePreservingDataset] 'advantage' key not found in base_sample "
                    f"at index {idx}. Run compute_advantages.py first."
                )
            advantage = base_sample["advantage"]
            if isinstance(advantage, torch.Tensor):
                advantage = bool(advantage.item())
            sample["advantage"] = advantage

        return sample


class CFGDataLoaderImpl(BaseDataLoaderImpl):
    """DataLoader wrapper that yields CFG training tuples."""

    def __iter__(self) -> Iterator[tuple[Any, Any, torch.Tensor]]:
        for batch in self._data_loader:
            observation = CFGObservation.from_dict(batch)
            actions = batch["actions"]

            advantage = batch["advantage"]
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, dtype=torch.bool)

            yield observation, actions, advantage


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithGuidance:
    """Tokenize both original prompt and guidance prompts for CFG models."""

    tokenizer: Any
    discrete_state_input: bool = False

    def __call__(self, data: dict) -> dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)

        positive_prompt = f"{prompt}\nAdvantage: positive"
        negative_prompt = f"{prompt}\nAdvantage: negative"

        positive_tokens, positive_masks = self.tokenizer.tokenize(
            positive_prompt, state
        )
        negative_tokens, negative_masks = self.tokenizer.tokenize(
            negative_prompt, state
        )

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "tokenized_positive_guidance_prompt": positive_tokens,
            "tokenized_positive_guidance_prompt_mask": positive_masks,
            "tokenized_negative_guidance_prompt": negative_tokens,
            "tokenized_negative_guidance_prompt_mask": negative_masks,
        }


class CfgMixtureDataset(ReCapMixtureDataset):
    """Mixture of multiple datasets with weighted sampling for CFG-RL training."""

    mixture_name = "CfgMixtureDataset"

    def __init__(
        self,
        *args: Any,
        episode_balanced: bool = False,
        positive_fraction: float = 0.5,
        quota_cycle_size: int | None = None,
        **kwargs: Any,
    ):
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1]")
        self.episode_balanced = episode_balanced
        self.positive_fraction = positive_fraction
        super().__init__(*args, **kwargs)
        self.quota_cycle_size = quota_cycle_size
        self._quota_schedule = self._build_quota_schedule(quota_cycle_size)
        logger.info(
            "%s sampling: episode_balanced=%s, positive_fraction=%.3f",
            self.mixture_name,
            self.episode_balanced,
            self.positive_fraction,
        )

    def _build_quota_schedule(
        self, cycle_size: int | None
    ) -> tuple[tuple[int, bool | None], ...] | None:
        """Build a spread fixed-count dataset/label schedule for short runs."""
        if cycle_size is None:
            return None
        if cycle_size <= 0:
            raise ValueError("quota_cycle_size must be positive")
        exact_counts = self.dataset_sampling_weights * cycle_size
        counts = np.rint(exact_counts).astype(int)
        if counts.sum() != cycle_size or not np.allclose(exact_counts, counts):
            raise ValueError(
                "quota_cycle_size must make every dataset weight an integer count: "
                f"weights={self.dataset_sampling_weights.tolist()}, cycle={cycle_size}"
            )

        slots: list[tuple[float, int, bool | None]] = []
        for dataset_index, count in enumerate(counts):
            if count <= 0:
                raise ValueError(
                    f"Dataset {dataset_index} has zero quota in cycle {cycle_size}"
                )
            dataset = self.datasets[dataset_index]
            is_rollout = getattr(dataset, "sampling_type", "sft") == "rollout"
            positive_count = int(round(count * self.positive_fraction))
            if is_rollout and count >= 2:
                positive_count = min(max(positive_count, 1), count - 1)
            for occurrence in range(count):
                forced_label = None
                if is_rollout:
                    forced_label = occurrence < positive_count
                slots.append(
                    ((occurrence + 0.5) / count, dataset_index, forced_label)
                )
        slots.sort(key=lambda item: (item[0], item[1]))
        return tuple((dataset_index, label) for _, dataset_index, label in slots)

    def _sample_step(self, index: int) -> tuple[Any, int]:
        forced_label = None
        if self._quota_schedule is None:
            dataset, sample_index = super()._sample_step(index)
        else:
            dataset_index, forced_label = self._quota_schedule[
                index % len(self._quota_schedule)
            ]
            dataset = self.datasets[dataset_index]
            sample_index = 0
        if self.episode_balanced and hasattr(
            dataset, "sample_episode_balanced_index"
        ):
            seed = _safe_hash((self._epoch, index, self.seed, "episode-balanced"))
            rng = np.random.default_rng(seed)
            sample_index = dataset.sample_episode_balanced_index(
                rng, self.positive_fraction, forced_label=forced_label
            )
        return dataset, sample_index
