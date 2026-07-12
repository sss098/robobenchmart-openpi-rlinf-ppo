#!/usr/bin/env python3
"""Build temporal contact sheets for the saved RBM OOD evaluation videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def sample_video(path: Path, frame_count: int, size: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total - 1, frame_count, dtype=int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        cv2.putText(
            frame,
            f"{index / max(total - 1, 1):.0%}",
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)
    capture.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--episodes-per-sheet", type=int, default=5)
    parser.add_argument("--thumbnail-size", type=int, default=128)
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=0,
        help="Arrange each episode in a temporal grid; intended with episodes-per-sheet=1.",
    )
    args = parser.parse_args()

    videos = sorted(args.input_dir.glob("*.mp4"), key=lambda path: int(path.stem))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_width = 150
    for start in range(0, len(videos), args.episodes_per_sheet):
        rows = []
        batch = videos[start : start + args.episodes_per_sheet]
        for path in batch:
            frames = sample_video(path, args.frames, args.thumbnail_size)
            if args.grid_columns:
                if len(batch) != 1:
                    raise ValueError("--grid-columns requires --episodes-per-sheet=1")
                columns = args.grid_columns
                blank = np.zeros_like(frames[0])
                padded = frames + [blank] * (-len(frames) % columns)
                grid_rows = [
                    np.hstack(padded[index : index + columns])
                    for index in range(0, len(padded), columns)
                ]
                grid = np.vstack(grid_rows)
                header = np.full((50, grid.shape[1], 3), 28, dtype=np.uint8)
                cv2.putText(
                    header,
                    f"{args.task}  seed {path.stem}",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (100, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                rows.append(np.vstack([header, grid]))
                continue
            label = np.full(
                (args.thumbnail_size, label_width, 3), 28, dtype=np.uint8
            )
            cv2.putText(
                label,
                args.task,
                (7, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                label,
                f"seed {path.stem}",
                (7, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 220, 255),
                1,
                cv2.LINE_AA,
            )
            rows.append(np.hstack([label, *frames]))
        sheet = np.vstack(rows)
        first_seed, last_seed = batch[0].stem, batch[-1].stem
        output = args.output_dir / f"{args.task}_{first_seed}_{last_seed}.jpg"
        if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Failed to write {output}")
        print(output)


if __name__ == "__main__":
    main()
