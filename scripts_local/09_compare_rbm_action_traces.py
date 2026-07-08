#!/usr/bin/env python3
"""Compare RoboBenchMart action traces saved as .npz files or RLinf trace directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_actions(path: Path, stage_id: int | None = None) -> np.ndarray:
    if path.is_dir():
        pattern = f"*stage{stage_id}_*.npz" if stage_id is not None else "*.npz"
        files = sorted(path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no .npz files under {path} matching {pattern}")
        arrays = []
        for file in files:
            data = np.load(file, allow_pickle=False)
            arrays.append(np.asarray(data["actions"], dtype=np.float32))
        return np.concatenate(arrays, axis=0)

    data = np.load(path, allow_pickle=False)
    return np.asarray(data["actions"], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jax_trace", type=Path)
    parser.add_argument("rlinf_trace", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--stage-id", type=int, default=None, help="Filter RLinf trace directory to one pipeline stage.")
    args = parser.parse_args()

    jax = load_actions(args.jax_trace)
    rlinf = load_actions(args.rlinf_trace, stage_id=args.stage_id)
    n = min(jax.shape[0], rlinf.shape[0])
    if n == 0:
        raise RuntimeError("empty trace")

    jax = jax[:n]
    rlinf = rlinf[:n]
    if jax.shape != rlinf.shape:
        print(f"shape mismatch after truncation: jax={jax.shape}, rlinf={rlinf.shape}")
        common_shape = tuple(min(a, b) for a, b in zip(jax.shape, rlinf.shape))
        slices = tuple(slice(0, s) for s in common_shape)
        jax = jax[slices]
        rlinf = rlinf[slices]

    diff = rlinf - jax
    abs_diff = np.abs(diff)
    print(f"compared_shape={jax.shape}")
    print(f"mean_abs={abs_diff.mean():.8f}")
    print(f"max_abs={abs_diff.max():.8f}")
    print(f"rmse={np.sqrt(np.mean(diff ** 2)):.8f}")
    print("per_dim_mean_abs=" + np.array2string(abs_diff.reshape(-1, abs_diff.shape[-1]).mean(axis=0), precision=6))
    print("allclose=" + str(bool(np.allclose(jax, rlinf, atol=args.atol, rtol=0.0))))

    mismatch = np.argwhere(abs_diff > args.atol)
    if len(mismatch):
        idx = tuple(mismatch[0])
        print(f"first_mismatch_idx={idx} jax={jax[idx]:.8f} rlinf={rlinf[idx]:.8f} diff={diff[idx]:.8f}")


if __name__ == "__main__":
    main()
