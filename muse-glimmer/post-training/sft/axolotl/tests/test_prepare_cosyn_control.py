# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

RECIPE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = RECIPE_ROOT / "scripts" / "prepare_cosyn_control.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_cosyn_control", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
prepare_cosyn_control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_cosyn_control)

SELECTED_SPLITS = ((2048, "train"), (256, "validation"))


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    work = tmp_path / "work with spaces"
    for cap in (2048, 256):
        directory = work / f"datasets/cap-{cap}/cosyn-circuit"
        directory.mkdir(parents=True)
        splits = {}
        for split, count in (("train", cap), ("validation", min(cap, 821))):
            data = "".join(
                json.dumps({"id": f"{split}-{index}"}) + "\n"
                for index in range(count)
            ).encode()
            (directory / f"{split}.jsonl").write_bytes(data)
            splits[split] = {
                "examples": count,
                "jsonl_sha256": hashlib.sha256(data).hexdigest(),
            }
        (directory / "manifest.json").write_text(
            json.dumps({"splits": splits}), encoding="utf-8"
        )
    return work


def test_only_paths_change_and_inputs_are_preserved(work_dir: Path) -> None:
    source = RECIPE_ROOT / "configs/muse-glimmer-30b-qlora.yaml"
    source_bytes = source.read_bytes()
    expected = yaml.safe_load(source_bytes)
    original = copy.deepcopy(expected)
    inputs = {
        path: path.read_bytes()
        for path in (work_dir / "datasets").glob("cap-*/*/*")
        if path.is_file()
    }
    for (cap, split), field in zip(
        SELECTED_SPLITS, ("datasets", "test_datasets")
    ):
        expected[field][0]["path"] = str(
            work_dir / f"datasets/cap-{cap}/cosyn-circuit/{split}.jsonl"
        )
    expected["dataset_prepared_path"] = str(
        work_dir / "datasets/cache/cosyn-qlora"
    )
    expected["output_dir"] = str(work_dir / "checkpoints/cosyn-qlora")

    output = prepare_cosyn_control.prepare_cosyn_control(work_dir)

    assert output == work_dir / "configs/cosyn-qlora.yaml"
    assert yaml.safe_load(output.read_text()) == expected
    assert source.read_bytes() == source_bytes
    assert yaml.safe_load(source.read_bytes()) == original
    assert all(path.read_bytes() == data for path, data in inputs.items())
    assert not (work_dir / "datasets/cache").exists()
    assert not (work_dir / "checkpoints").exists()


@pytest.mark.parametrize("cap,split", SELECTED_SPLITS)
def test_rejects_unexpected_manifest_count(
    work_dir: Path, cap: int, split: str
) -> None:
    path = work_dir / f"datasets/cap-{cap}/cosyn-circuit/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"][split]["examples"] -= 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=f"Unexpected {split} count"):
        prepare_cosyn_control.prepare_cosyn_control(work_dir)

    assert not (work_dir / "configs").exists()


@pytest.mark.parametrize("cap,split", SELECTED_SPLITS)
def test_rejects_changed_jsonl(work_dir: Path, cap: int, split: str) -> None:
    path = work_dir / f"datasets/cap-{cap}/cosyn-circuit/{split}.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=f"Changed {split} file"):
        prepare_cosyn_control.prepare_cosyn_control(work_dir)

    assert not (work_dir / "configs").exists()


@pytest.mark.parametrize("cap,split", SELECTED_SPLITS)
@pytest.mark.parametrize("missing", ("manifest.json", "jsonl"))
def test_rejects_missing_input(
    work_dir: Path, cap: int, split: str, missing: str
) -> None:
    filename = f"{split}.jsonl" if missing == "jsonl" else missing
    (work_dir / f"datasets/cap-{cap}/cosyn-circuit/{filename}").unlink()

    with pytest.raises(FileNotFoundError):
        prepare_cosyn_control.prepare_cosyn_control(work_dir)

    assert not (work_dir / "configs").exists()


def test_refuses_to_overwrite_existing_config(work_dir: Path) -> None:
    output = prepare_cosyn_control.prepare_cosyn_control(work_dir)
    output.write_text("existing config\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_cosyn_control.prepare_cosyn_control(work_dir)

    assert output.read_text() == "existing config\n"


def test_cli_works_outside_recipe_with_relative_work_dir(
    work_dir: Path, tmp_path: Path
) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--work-dir", work_dir.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    output = work_dir / "configs/cosyn-qlora.yaml"
    assert result.stdout.strip() == str(output)
    assert output.is_file()


def test_cli_requires_work_dir(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--work-dir" in result.stderr
