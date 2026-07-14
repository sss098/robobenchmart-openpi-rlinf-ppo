#!/usr/bin/env python
"""Convert RoboBenchMart PickToBasket H5 demos to an OpenPI-compatible LeRobot dataset.

The produced dataset matches OpenPI's LeRobotRBMDataConfig:
  image       -> observation/image
  wrist_image -> observation/wrist_image
  extra_image -> observation/extra_image
  state       -> observation/state
  actions     -> actions
  task        -> LeRobot task text, later transformed into prompt
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
import shutil

import h5py
import numpy as np
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm


DEFAULT_INPUTS = (
    "demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5",
    "demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5",
    "demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_stars_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5",
)

FEATURES = {
    "image": {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": [128, 128, 3], "names": ["height", "width", "channel"]},
    "extra_image": {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": [15], "names": ["state"]},
    "actions": {"dtype": "float32", "shape": [13], "names": ["actions"]},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _decode_prompt(traj: h5py.Group) -> str:
    bytes_ds = traj["obs/extra/language_instruction_bytes"][0]
    mask_ds = traj["obs/extra/language_instruction_mask"][0].astype(bool)
    return bytes(np.asarray(bytes_ds)[mask_ds].tolist()).decode("utf-8", errors="ignore")


def _traj_keys(h5_file: h5py.File) -> list[str]:
    return sorted((k for k in h5_file.keys() if k.startswith("traj_")), key=lambda k: int(k.split("_")[1]))


def _validate_traj(traj: h5py.Group, h5_path: Path, traj_key: str) -> int:
    actions = traj["actions"]
    qpos = traj["obs/agent/qpos"]
    left = traj["obs/sensor_data/left_base_camera_link/rgb"]
    right = traj["obs/sensor_data/right_base_camera_link/rgb"]
    wrist = traj["obs/sensor_data/fetch_hand/rgb"]

    horizon = actions.shape[0]
    if actions.shape[1:] != (13,):
        raise ValueError(f"{h5_path}:{traj_key} actions shape {actions.shape}, expected (T, 13)")
    if qpos.shape[1:] != (15,):
        raise ValueError(f"{h5_path}:{traj_key} qpos shape {qpos.shape}, expected (T+1, 15)")
    if qpos.shape[0] < horizon:
        raise ValueError(f"{h5_path}:{traj_key} qpos shorter than actions: {qpos.shape[0]} < {horizon}")
    if left.shape[1:] != (256, 256, 3) or right.shape[1:] != (256, 256, 3):
        raise ValueError(f"{h5_path}:{traj_key} base camera shapes {left.shape}, {right.shape}")
    if wrist.shape[1:] != (128, 128, 3):
        raise ValueError(f"{h5_path}:{traj_key} wrist camera shape {wrist.shape}")
    if left.dtype != np.uint8 or right.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError(f"{h5_path}:{traj_key} RGB images must be uint8")
    return horizon


def _image_record(image: np.ndarray, image_format: str, jpeg_quality: int) -> dict:
    buf = BytesIO()
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if image_format == "png":
        pil_image.save(buf, format="PNG")
    elif image_format == "jpeg":
        pil_image.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    return {"bytes": buf.getvalue(), "path": None}


def _hf_features() -> datasets.Features:
    return datasets.Features(
        {
            "image": datasets.Image(),
            "wrist_image": datasets.Image(),
            "extra_image": datasets.Image(),
            "state": datasets.Sequence(length=15, feature=datasets.Value("float32")),
            "actions": datasets.Sequence(length=13, feature=datasets.Value("float32")),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
        }
    )


def _write_episode_parquet(frame_data: dict[str, list], parquet_path: Path) -> None:
    """Write one episode in the same Arrow schema produced by HF datasets.Image."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "image": frame_data["image"],
            "wrist_image": frame_data["wrist_image"],
            "extra_image": frame_data["extra_image"],
            "state": frame_data["state"],
            "actions": frame_data["actions"],
            "timestamp": frame_data["timestamp"],
            "frame_index": frame_data["frame_index"],
            "episode_index": frame_data["episode_index"],
            "index": frame_data["index"],
            "task_index": frame_data["task_index"],
        },
        schema=_hf_features().arrow_schema,
    )
    pq.write_table(table, parquet_path)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, default=_json_default) + "\n")


def _append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, default=_json_default) + "\n")


def _stats(array: np.ndarray) -> dict:
    array = np.asarray(array)
    if array.ndim == 1:
        axis = 0
        keepdims = True
    else:
        axis = 0
        keepdims = False
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([array.shape[0]], dtype=np.int64),
    }


def _episode_stats(frame_data: dict[str, list]) -> dict:
    stats = {}
    for key in ["state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index"]:
        stats[key] = _stats(np.asarray(frame_data[key]))
    return stats


def _plan_episodes(input_h5s: list[Path], max_episodes: int | None) -> list[tuple[Path, str]]:
    plan: list[tuple[Path, str]] = []
    for h5_path in input_h5s:
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        with h5py.File(h5_path, "r") as h5_file:
            for traj_key in _traj_keys(h5_file):
                plan.append((h5_path, traj_key))
                if max_episodes is not None and len(plan) >= max_episodes:
                    return plan
    return plan


def _initial_info(fps: int) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "ds_fetch_basket",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": FEATURES,
    }


def convert(
    input_h5s: list[Path],
    *,
    repo_id: str,
    root: Path,
    fps: int,
    max_episodes: int | None,
    max_frames_per_episode: int | None,
    overwrite: bool,
    image_format: str,
    jpeg_quality: int,
) -> None:
    output_path = root / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)
    (output_path / "meta").mkdir(parents=True)

    episode_plan = _plan_episodes(input_h5s, max_episodes)
    info = _initial_info(fps)
    task_to_index: dict[str, int] = {}
    global_frame_index = 0
    written_frames = 0

    print(f"Output dataset: {output_path}", flush=True)
    print(f"Planned episodes: {len(episode_plan)}", flush=True)
    print(f"Image encoding: {image_format} quality={jpeg_quality if image_format == 'jpeg' else 'n/a'}", flush=True)
    if max_frames_per_episode is not None:
        print(f"Frame cap per episode: {max_frames_per_episode} (debug only; do not use for final SFT data)", flush=True)

    for episode_index, (h5_path, traj_key) in enumerate(tqdm(episode_plan, desc="episodes", unit="ep")):
        with h5py.File(h5_path, "r") as h5_file:
            traj = h5_file[traj_key]
            horizon = _validate_traj(traj, h5_path, traj_key)
            if max_frames_per_episode is not None:
                horizon = min(horizon, max_frames_per_episode)
            task = _decode_prompt(traj)
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)
                _append_jsonl(output_path / "meta/tasks.jsonl", {"task_index": task_to_index[task], "task": task})
            task_index = task_to_index[task]

            actions = traj["actions"]
            qpos = traj["obs/agent/qpos"]
            left = traj["obs/sensor_data/left_base_camera_link/rgb"]
            right = traj["obs/sensor_data/right_base_camera_link/rgb"]
            wrist = traj["obs/sensor_data/fetch_hand/rgb"]

            frame_data = {
                "image": [],
                "wrist_image": [],
                "extra_image": [],
                "state": [],
                "actions": [],
                "timestamp": [],
                "frame_index": [],
                "episode_index": [],
                "index": [],
                "task_index": [],
            }

            frame_iter = tqdm(range(horizon), desc=f"{h5_path.name}:{traj_key}", unit="frame", leave=False)
            for t in frame_iter:
                frame_data["image"].append(_image_record(left[t], image_format, jpeg_quality))
                frame_data["wrist_image"].append(_image_record(wrist[t], image_format, jpeg_quality))
                frame_data["extra_image"].append(_image_record(right[t], image_format, jpeg_quality))
                frame_data["state"].append(qpos[t].astype(np.float32))
                frame_data["actions"].append(actions[t].astype(np.float32))
                frame_data["timestamp"].append(np.float32(t / fps))
                frame_data["frame_index"].append(np.int64(t))
                frame_data["episode_index"].append(np.int64(episode_index))
                frame_data["index"].append(np.int64(global_frame_index + t))
                frame_data["task_index"].append(np.int64(task_index))

            chunk = episode_index // info["chunks_size"]
            parquet_path = output_path / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
            _write_episode_parquet(frame_data, parquet_path)

            ep_stats = _episode_stats(frame_data)
            _append_jsonl(
                output_path / "meta/episodes.jsonl",
                {"episode_index": episode_index, "tasks": [task], "length": horizon},
            )
            _append_jsonl(
                output_path / "meta/episodes_stats.jsonl",
                {"episode_index": episode_index, "stats": ep_stats},
            )

            global_frame_index += horizon
            written_frames += horizon
            completed_episodes = episode_index + 1
            info["total_episodes"] = completed_episodes
            info["total_frames"] = written_frames
            info["total_tasks"] = len(task_to_index)
            info["total_chunks"] = math.ceil(completed_episodes / info["chunks_size"])
            info["splits"] = {"train": f"0:{completed_episodes}"}
            _write_json(output_path / "meta/info.json", info)
            tqdm.write(f"saved episode {completed_episodes:04d}/{len(episode_plan)} frames={horizon} task={task!r}")

    print("Conversion complete", flush=True)
    print(f"Episodes written: {len(episode_plan)}", flush=True)
    print(f"Frames written: {written_frames}", flush=True)
    print(f"Tasks written: {len(task_to_index)}", flush=True)
    print(f"Output dataset: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        action="append",
        default=None,
        help="Input RoboBenchMart H5. Can be passed multiple times. Defaults to fanta/nivea/stars PickToBasket demos.",
    )
    parser.add_argument("--repo-id", default="rbm_dataset", help="LeRobot repo id written under root.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/lerobot_data"),
        help="LeRobot root directory. OpenPI must later run with HF_LEROBOT_HOME set to this directory.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=None, help="Small smoke-test episode limit.")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Debug-only frame cap. Leave unset for final SFT data.",
    )
    parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_h5s = [Path(p) for p in (args.input_h5 or DEFAULT_INPUTS)]
    convert(
        input_h5s,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        overwrite=args.overwrite,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
