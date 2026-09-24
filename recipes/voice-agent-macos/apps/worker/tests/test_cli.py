from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from muse_glimmer_worker import cli


def test_console_command_uses_installed_module() -> None:
    args = argparse.Namespace(
        input_device="Built-in Mic",
        output_device="Built-in Output",
        list_devices=True,
        text=True,
        record=False,
        console_log_level="info",
    )
    assert cli._console_command(args) == [
        sys.executable,
        "-m",
        "muse_glimmer_worker",
        "console",
        "--log-level",
        "info",
        "--input-device",
        "Built-in Mic",
        "--output-device",
        "Built-in Output",
        "--list-devices",
        "--text",
    ]


def test_issue_bundle_omits_runtime_log(tmp_path: Path) -> None:
    for name in ("report.json", "events.jsonl", "runtime.log"):
        (tmp_path / name).write_text(name)

    bundle = cli._create_issue_bundle(tmp_path)

    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"report.json", "events.jsonl"}


def test_provider_metrics_redact_nested_artifact_paths(tmp_path: Path) -> None:
    model = tmp_path / "private" / "model.pte"
    config = SimpleNamespace(
        muse_glimmer_api_key="local-secret",
        parakeet_helper_path=tmp_path / "bin" / "parakeet_helper",
        parakeet_model_path=model,
        parakeet_tokenizer_path=tmp_path / "private" / "tokenizer.model",
        parakeet_delegate_data_path=None,
        supertonic_runner_path=tmp_path / "bin" / "supertonic_runner",
        supertonic_pte_path=tmp_path / "private" / "supertonic.pte",
        supertonic_asset_dir=tmp_path / "private" / "assets",
        supertonic_voice_style_path=tmp_path / "private" / "voice.json",
    )

    redacted = cli._redact_payload(
        {"metadata": {"model_name": model}, "details": ["local-secret"]},
        config,
    )

    serialized = str(redacted)
    assert str(tmp_path) not in serialized
    assert "local-secret" not in serialized
    assert ".../model.pte" in serialized
