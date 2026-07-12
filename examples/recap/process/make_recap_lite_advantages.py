#!/usr/bin/env python
"""Create RECAP-lite advantage sidecars for LeRobot datasets.

This script writes ``meta/advantages_{tag}.parquet`` without training a value
model. It is intended for the conservative RBM RECAP-lite path:

- SFT/demo/correction datasets can be labeled all positive.
- Autonomous rollout datasets can be labeled by episode success when an
  ``is_success``-like column is available.

The CFG dataloader only requires ``episode_index``, ``frame_index``, and the
boolean ``advantage`` column, but this script also writes provenance columns for
auditing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SUCCESS_COLUMNS = ("is_success", "success", "episode_success")
CORRECTION_COLUMNS = ("is_correction", "correction", "intervention", "is_intervention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write RECAP-lite advantage labels for a LeRobot dataset."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Path to a local LeRobot dataset directory.",
    )
    parser.add_argument(
        "--tag",
        default="rbm_recap_lite_sft_pos",
        help="Output tag. Writes meta/advantages_<tag>.parquet.",
    )
    parser.add_argument(
        "--dataset-type",
        default="sft",
        choices=("sft", "demo", "correction", "rollout"),
        help="Dataset role used by --label-mode auto.",
    )
    parser.add_argument(
        "--label-mode",
        default="auto",
        choices=("auto", "all-positive", "success-column", "advantage-column"),
        help=(
            "How to assign labels. auto uses all-positive for sft/demo/correction "
            "and success-column for rollout."
        ),
    )
    parser.add_argument(
        "--advantage-column",
        default="advantage",
        help="Boolean per-frame label column used by --label-mode advantage-column.",
    )
    parser.add_argument(
        "--success-column",
        default=None,
        help="Episode success column. Defaults to the first known success-like column.",
    )
    parser.add_argument(
        "--correction-positive-column",
        default=None,
        help=(
            "Optional boolean column whose true frames are forced positive "
            "(for intervention/correction datasets)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output sidecar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files.",
    )
    return parser.parse_args()


def resolve_label_mode(label_mode: str, dataset_type: str) -> str:
    if label_mode != "auto":
        return label_mode
    if dataset_type in ("sft", "demo", "correction"):
        return "all-positive"
    return "success-column"


def first_present(available: Iterable[str], candidates: Iterable[str]) -> str | None:
    available_set = set(available)
    for name in candidates:
        if name in available_set:
            return name
    return None


def read_metadata_frame(
    parquet_path: Path,
    *,
    dataset_path: Path,
    label_mode: str,
    advantage_column: str,
    success_column: str | None,
    correction_positive_column: str | None,
) -> pd.DataFrame:
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema_arrow.names)

    required = {"episode_index", "frame_index"}
    missing = required - available
    if missing:
        raise ValueError(f"{parquet_path} missing required columns: {sorted(missing)}")

    columns = ["episode_index", "frame_index"]
    if "task_index" in available:
        columns.append("task_index")

    resolved_success_column = success_column
    if label_mode == "success-column":
        if resolved_success_column is None:
            resolved_success_column = first_present(available, SUCCESS_COLUMNS)
        if resolved_success_column is None:
            raise ValueError(
                f"{parquet_path} has no success column. Tried: {SUCCESS_COLUMNS}"
            )
        columns.append(resolved_success_column)

    if label_mode == "advantage-column":
        if advantage_column not in available:
            raise ValueError(
                f"{parquet_path} missing advantage column {advantage_column!r}"
            )
        columns.append(advantage_column)

    resolved_correction_column = correction_positive_column
    if resolved_correction_column is None:
        resolved_correction_column = first_present(available, CORRECTION_COLUMNS)
    if (
        resolved_correction_column is not None
        and resolved_correction_column in available
    ):
        columns.append(resolved_correction_column)
    else:
        resolved_correction_column = None

    # Preserve order and remove duplicates.
    columns = list(dict.fromkeys(columns))
    df = pq.read_table(parquet_path, columns=columns).to_pandas()

    if label_mode == "all-positive":
        advantage = np.ones(len(df), dtype=bool)
        reason = np.full(len(df), "all_positive", dtype=object)
    elif label_mode == "advantage-column":
        advantage = df[advantage_column].fillna(False).astype(bool).to_numpy()
        reason = np.full(len(df), f"column:{advantage_column}", dtype=object)
    else:
        assert resolved_success_column is not None
        advantage = np.zeros(len(df), dtype=bool)
        reason = np.full(
            len(df), f"episode_success:{resolved_success_column}", dtype=object
        )
        for idx in df.groupby("episode_index", sort=False).groups.values():
            idx_list = list(idx)
            ep_success = bool(
                df.iloc[idx_list][resolved_success_column]
                .fillna(False)
                .astype(bool)
                .any()
            )
            advantage[idx_list] = ep_success

    if resolved_correction_column is not None:
        correction_mask = (
            df[resolved_correction_column].fillna(False).astype(bool).to_numpy()
        )
        advantage = advantage | correction_mask
        reason = np.where(
            correction_mask, f"correction:{resolved_correction_column}", reason
        )

    try:
        source_file = str(parquet_path.relative_to(dataset_path))
    except ValueError:
        source_file = str(parquet_path)

    out = pd.DataFrame(
        {
            "episode_index": df["episode_index"].astype("int64"),
            "frame_index": df["frame_index"].astype("int64"),
            "advantage": advantage.astype(bool),
            "advantage_continuous": np.where(advantage, 1.0, -1.0).astype("float32"),
            "label_reason": reason,
            "source_file": source_file,
        }
    )
    if "task_index" in df.columns:
        out["task_index"] = df["task_index"].astype("int64")
    return out


def update_mixture_config(
    dataset_path: Path,
    *,
    tag: str,
    stats: dict,
    dry_run: bool,
) -> None:
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is not installed; skipping mixture_config.yaml update.")
        return

    path = dataset_path / "mixture_config.yaml"
    if path.exists():
        with open(path, "r") as f:
            mixture = yaml.safe_load(f) or {}
    else:
        mixture = {}

    tags = mixture.setdefault("tags", {})
    tags[tag] = {
        "source": "recap_lite",
        "label_mode": stats["label_mode"],
        "dataset_type": stats["dataset_type"],
        "positive_count": int(stats["positive_count"]),
        "negative_count": int(stats["negative_count"]),
        "total_count": int(stats["total_count"]),
        "positive_rate": float(stats["positive_rate"]),
        "output_file": f"meta/advantages_{tag}.parquet",
    }
    mixture["latest_recap_lite_tag"] = tag

    if dry_run:
        logger.info("Dry run: would update %s", path)
        return

    with open(path, "w") as f:
        yaml.safe_dump(mixture, f, sort_keys=False)
    logger.info("Updated %s", path)


def broadcast_episode_success_labels(
    frame: pd.DataFrame, label_mode: str
) -> pd.DataFrame:
    """Broadcast success across complete episodes after parquet shards are merged."""
    if label_mode != "success-column":
        return frame
    frame = frame.copy()
    frame["advantage"] = frame.groupby("episode_index", sort=False)[
        "advantage"
    ].transform("any")
    frame["advantage_continuous"] = np.where(frame["advantage"], 1.0, -1.0).astype(
        "float32"
    )
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    data_dir = dataset_path / "data"
    meta_dir = dataset_path / "meta"
    if not data_dir.exists():
        raise SystemExit(
            f"Incomplete LeRobot dataset: data directory not found: {data_dir}. "
            "This tool labels an existing dataset; it does not collect rollouts."
        )
    if not meta_dir.exists():
        raise SystemExit(
            f"Incomplete LeRobot dataset: meta directory not found: {meta_dir}"
        )

    label_mode = resolve_label_mode(args.label_mode, args.dataset_type)
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No rollout parquet frames found under {data_dir}")

    output_path = meta_dir / f"advantages_{args.tag}.parquet"
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    logger.info("Dataset: %s", dataset_path)
    logger.info("Parquet files: %d", len(parquet_files))
    logger.info("Label mode: %s", label_mode)

    frames = [
        read_metadata_frame(
            parquet_path,
            dataset_path=dataset_path,
            label_mode=label_mode,
            advantage_column=args.advantage_column,
            success_column=args.success_column,
            correction_positive_column=args.correction_positive_column,
        )
        for parquet_path in parquet_files
    ]
    out_df = pd.concat(frames, ignore_index=True)
    out_df = broadcast_episode_success_labels(out_df, label_mode)
    out_df = out_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    duplicate_mask = out_df.duplicated(["episode_index", "frame_index"], keep=False)
    if duplicate_mask.any():
        duplicates = out_df.loc[duplicate_mask, ["episode_index", "frame_index"]].head()
        raise ValueError(
            "Duplicate (episode_index, frame_index) keys in generated advantages: "
            f"{duplicates.to_dict('records')}"
        )

    positive_count = int(out_df["advantage"].sum())
    total_count = int(len(out_df))
    negative_count = total_count - positive_count
    stats = {
        "dataset_path": str(dataset_path),
        "dataset_type": args.dataset_type,
        "label_mode": label_mode,
        "tag": args.tag,
        "output_path": str(output_path),
        "total_count": total_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / max(1, total_count),
    }

    logger.info("Stats: %s", json.dumps(stats, indent=2))

    if args.dry_run:
        logger.info("Dry run: not writing %s", output_path)
    else:
        out_df.to_parquet(output_path, index=False)
        logger.info("Wrote %s", output_path)

    update_mixture_config(dataset_path, tag=args.tag, stats=stats, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
