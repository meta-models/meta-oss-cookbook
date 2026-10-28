#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create a path-only QLoRA config for the bounded 2,048/256 CoSyn control.

Requires the cap-2048 and cap-256 preparations documented in the README.
Checks their selected split counts and JSONL hashes before writing a new
config. Training settings and the shipped config remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

RECIPE_ROOT = Path(__file__).resolve().parents[1]


def prepare_cosyn_control(work_dir: Path) -> Path:
    """Validate the capped inputs and write a config without overwriting one."""

    work = work_dir.expanduser().resolve()
    config_path = RECIPE_ROOT / "configs" / "muse-glimmer-30b-qlora.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for field, cap, split, count in (
        ("datasets", 2048, "train", 2048),
        ("test_datasets", 256, "validation", 256),
    ):
        directory = work / f"datasets/cap-{cap}/cosyn-circuit"
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        path = directory / f"{split}.jsonl"
        stats = manifest["splits"][split]
        if stats["examples"] != count:
            raise ValueError(
                f"Unexpected {split} count: {stats['examples']}; "
                f"expected {count} in {directory / 'manifest.json'}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != stats["jsonl_sha256"]:
            raise ValueError(f"Changed {split} file: {path}")
        config[field][0]["path"] = str(path)
    config["dataset_prepared_path"] = str(work / "datasets/cache/cosyn-qlora")
    config["output_dir"] = str(work / "checkpoints/cosyn-qlora")
    output_path = work / "configs/cosyn-qlora.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        yaml.safe_dump(config, output, sort_keys=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="WORK directory containing datasets/cap-2048 and datasets/cap-256",
    )
    args = parser.parse_args()
    print(prepare_cosyn_control(args.work_dir))


if __name__ == "__main__":
    main()
