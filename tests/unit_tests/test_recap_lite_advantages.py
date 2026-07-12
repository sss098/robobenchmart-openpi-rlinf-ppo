import importlib.util
from pathlib import Path

import pandas as pd

from rlinf.data.datasets.recap.cfg_model import CfgMixtureDataset

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "recap"
    / "process"
    / "make_recap_lite_advantages.py"
)
SPEC = importlib.util.spec_from_file_location("make_recap_lite_advantages", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_success_labels_are_broadcast_across_parquet_shards(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shards = [
        pd.DataFrame(
            {
                "episode_index": [7],
                "frame_index": [0],
                "success": [False],
            }
        ),
        pd.DataFrame(
            {
                "episode_index": [7, 8],
                "frame_index": [1, 0],
                "success": [True, None],
            }
        ),
    ]

    frames = []
    for index, shard in enumerate(shards):
        path = data_dir / f"part-{index}.parquet"
        shard.to_parquet(path, index=False)
        frames.append(
            MODULE.read_metadata_frame(
                path,
                dataset_path=tmp_path,
                label_mode="success-column",
                advantage_column="advantage",
                success_column="success",
                correction_positive_column=None,
            )
        )

    result = MODULE.broadcast_episode_success_labels(
        pd.concat(frames, ignore_index=True), "success-column"
    )

    assert result.loc[result["episode_index"] == 7, "advantage"].tolist() == [
        True,
        True,
    ]
    assert result.loc[result["episode_index"] == 8, "advantage"].tolist() == [False]


def test_cfg_episode_balanced_sampling_is_not_frame_length_weighted():
    class EpisodeDataset:
        def __len__(self):
            return 110

        def __getitem__(self, index):
            return {"advantage": index < 10}

        def sample_episode_balanced_index(
            self, rng, positive_fraction, forced_label=None
        ):
            if forced_label is not None:
                return int(rng.integers(0, 10)) if forced_label else int(
                    rng.integers(10, 110)
                )
            return int(rng.integers(0, 10)) if rng.random() < positive_fraction else int(
                rng.integers(10, 110)
            )

    dataset = CfgMixtureDataset(
        datasets=[(EpisodeDataset(), 1.0)],
        episode_balanced=True,
        positive_fraction=0.67,
        balance_dataset_weights=False,
        seed=42,
    )
    labels = [bool(dataset[index]["advantage"]) for index in range(1000)]

    assert 0.62 < sum(labels) / len(labels) < 0.72


def test_cfg_quota_cycle_covers_every_rollout_and_label():
    class QuotaDataset:
        def __init__(self, name, sampling_type):
            self.name = name
            self.sampling_type = sampling_type

        def __len__(self):
            return 100

        def __getitem__(self, index):
            return {"dataset": self.name, "advantage": index == 1}

        def sample_episode_balanced_index(
            self, rng, positive_fraction, forced_label=None
        ):
            del rng, positive_fraction
            return 1 if forced_label is not False else 0

    datasets = [(QuotaDataset("sft", "sft"), 0.9)] + [
        (QuotaDataset(f"task-{index}", "rollout"), 1 / 30) for index in range(3)
    ]
    mixture = CfgMixtureDataset(
        datasets=datasets,
        episode_balanced=True,
        positive_fraction=0.67,
        quota_cycle_size=60,
        balance_dataset_weights=False,
        seed=42,
    )
    samples = [mixture[index] for index in range(60)]

    assert sum(sample["dataset"] == "sft" for sample in samples) == 54
    for index in range(3):
        task_samples = [
            sample for sample in samples if sample["dataset"] == f"task-{index}"
        ]
        assert [sample["advantage"] for sample in task_samples] == [True, False]
