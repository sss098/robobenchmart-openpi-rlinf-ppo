#!/usr/bin/env python3
"""Repair legacy RBM rollout shards from extra_view_image to extra_image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

OLD_KEY = "extra_view_image"
NEW_KEY = "extra_image"


def _rename_json(value):
    if isinstance(value, dict):
        return {
            str(key).replace(OLD_KEY, NEW_KEY): _rename_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_json(item) for item in value]
    if isinstance(value, str):
        return value.replace(OLD_KEY, NEW_KEY)
    return value


def _atomic_write_json(path: Path) -> None:
    data = _rename_json(json.loads(path.read_text()))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n")
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path) -> None:
    rows = [
        _rename_json(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    os.replace(tmp, path)


def repair_shard(shard: Path) -> tuple[int, int]:
    changed = 0
    checked = 0
    for parquet_path in sorted((shard / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        names = table.column_names
        checked += 1
        if NEW_KEY in names and OLD_KEY in names:
            raise RuntimeError(f"Both {OLD_KEY} and {NEW_KEY} exist in {parquet_path}")
        if OLD_KEY not in names:
            if NEW_KEY not in names:
                raise RuntimeError(f"Neither third-view key exists in {parquet_path}")
            continue
        renamed = [NEW_KEY if name == OLD_KEY else name for name in names]
        tmp = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table.rename_columns(renamed), tmp, compression="zstd")
        os.replace(tmp, parquet_path)
        changed += 1

    for path in (shard / "meta").glob("*.json"):
        _atomic_write_json(path)
    for path in (shard / "meta").glob("*.jsonl"):
        _atomic_write_jsonl(path)
    return changed, checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    shards = sorted(path.parent.parent for path in root.glob("**/meta/info.json"))
    if not shards:
        raise SystemExit(f"No LeRobot shards found under {root}")
    for shard in shards:
        changed, checked = repair_shard(shard)
        print(f"{shard}: repaired={changed}, checked={checked}")


if __name__ == "__main__":
    main()
