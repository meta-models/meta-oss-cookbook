from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts import validate_manifests
from scripts.repository import landed_gate_commits


@pytest.fixture
def compatibility() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "artifact-gated",
        "platform": "macos-arm64",
        "executorch": {
            "repository": "https://github.com/pytorch/executorch.git",
            "commit": "20ad5ee43ff53804030899d621590af3daadda53",
            "required_capabilities": [
                "parakeet_persistent_helper",
                "muse_glimmer_dflash_mlx",
                "supports_cancel",
                "supertonic_server_jsonl",
            ],
            "gates": {
                "supertonic_runtime": {
                    "status": "landed",
                    "pull_request": "https://github.com/pytorch/executorch/pull/22063",
                    "commit": "81969a92dd2e5515fa23ccdf9d87346cf3ba2ba2",
                },
                "supports_cancel": {
                    "status": "landed",
                    "pull_request": "https://github.com/pytorch/executorch/pull/22070",
                    "commit": "5bd86e50fcd986999e4c09b82de040a3ba224466",
                },
                "supertonic_server_jsonl": {
                    "status": "landed",
                    "pull_request": "https://github.com/pytorch/executorch/pull/22208",
                    "commit": "20ad5ee43ff53804030899d621590af3daadda53",
                },
            },
        },
        "ready_for_release": False,
    }


def _gates(compatibility: dict[str, object]) -> dict[str, dict[str, object]]:
    executorch = compatibility["executorch"]
    assert isinstance(executorch, dict)
    gates = executorch["gates"]
    assert isinstance(gates, dict)
    return gates  # type: ignore[return-value]


def test_accepts_artifact_gated_capabilities(compatibility: dict[str, object]) -> None:
    validate_manifests._validate_compatibility(compatibility)
    assert landed_gate_commits(compatibility) == (
        "20ad5ee43ff53804030899d621590af3daadda53",
        "5bd86e50fcd986999e4c09b82de040a3ba224466",
        "81969a92dd2e5515fa23ccdf9d87346cf3ba2ba2",
    )


def test_landed_gate_requires_commit(compatibility: dict[str, object]) -> None:
    value = deepcopy(compatibility)
    _gates(value)["supertonic_runtime"]["commit"] = None

    with pytest.raises(RuntimeError, match="landed compatibility gate requires a commit"):
        validate_manifests._validate_compatibility(value)


def test_unlanded_gate_rejects_commit(compatibility: dict[str, object]) -> None:
    value = deepcopy(compatibility)
    gate = _gates(value)["supertonic_server_jsonl"]
    gate["status"] = "pending"
    gate["commit"] = "a" * 40

    with pytest.raises(RuntimeError, match="unlanded compatibility gate cannot have a commit"):
        validate_manifests._validate_compatibility(value)


def test_release_ready_rejects_unlanded_gate(compatibility: dict[str, object]) -> None:
    value = deepcopy(compatibility)
    value["ready_for_release"] = True
    gate = _gates(value)["supertonic_server_jsonl"]
    gate["status"] = "pending"
    gate["commit"] = None

    with pytest.raises(RuntimeError, match="unlanded gates"):
        validate_manifests._validate_compatibility(value)


def test_release_ready_requires_checkout_ancestry_verification(
    compatibility: dict[str, object],
) -> None:
    value = deepcopy(compatibility)
    value["ready_for_release"] = True
    executorch = value["executorch"]
    assert isinstance(executorch, dict)
    executorch["commit"] = "b" * 40
    for gate in _gates(value).values():
        gate["status"] = "landed"
        gate["commit"] = "a" * 40

    with pytest.raises(RuntimeError, match="requires checkout ancestry verification"):
        validate_manifests._validate_compatibility(value)


def test_release_ready_verifies_all_landed_gate_commits(
    compatibility: dict[str, object], tmp_path, monkeypatch
) -> None:
    value = deepcopy(compatibility)
    value["ready_for_release"] = True
    executorch = value["executorch"]
    assert isinstance(executorch, dict)
    executorch["commit"] = "b" * 40
    for index, gate in enumerate(_gates(value).values(), start=1):
        gate["status"] = "landed"
        gate["commit"] = str(index) * 40
    calls = []
    monkeypatch.setattr(
        validate_manifests,
        "validate_executorch_checkout",
        lambda checkout, commit, *, required_ancestors: calls.append(
            (checkout, commit, required_ancestors)
        ),
    )

    validate_manifests._validate_compatibility(value, release_checkout=tmp_path)

    assert calls == [(tmp_path, "b" * 40, ("1" * 40, "2" * 40, "3" * 40))]


@pytest.mark.parametrize(
    "required_role",
    [
        "mlx_metallib",
        "muse_glimmer_tokenizer_config",
        "muse_glimmer_chat_template",
    ],
)
def test_artifact_contract_requires_runtime_sidecars(
    required_role: str, tmp_path, monkeypatch
) -> None:
    manifest = json.loads(validate_manifests.ARTIFACT_LOCK.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        artifact for artifact in manifest["artifacts"] if artifact["role"] != required_role
    ]
    artifact_lock = tmp_path / "artifacts.json"
    artifact_lock.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(validate_manifests, "ARTIFACT_LOCK", artifact_lock)

    with pytest.raises(RuntimeError, match="artifact roles differ from the runtime contract"):
        validate_manifests._validate_artifacts()


def test_mlx_metallib_requires_pinned_submodule_provenance() -> None:
    artifact = {
        "role": "mlx_metallib",
        "source": "executorch",
        "revision": "b" * 40,
        "sha256": "c" * 64,
        "size_bytes": 1,
        "license": "BSD-3-Clause",
    }

    with pytest.raises(RuntimeError, match="MLX metallib provenance"):
        validate_manifests._validate_release_artifacts([artifact], "b" * 40)

    artifact.update(
        source="https://github.com/ml-explore/mlx.git",
        revision="7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247",
        license="MIT",
    )
    validate_manifests._validate_release_artifacts([artifact], "b" * 40)


def test_executorch_artifact_revision_must_match_final_commit() -> None:
    artifact = {
        "role": "supertonic_runner",
        "source": "executorch",
        "revision": "a" * 40,
        "sha256": "c" * 64,
        "size_bytes": 1,
        "license": "BSD-3-Clause",
    }

    with pytest.raises(RuntimeError, match="must match the final compatibility commit"):
        validate_manifests._validate_release_artifacts([artifact], "b" * 40)

    artifact["revision"] = "b" * 40
    validate_manifests._validate_release_artifacts([artifact], "b" * 40)

    artifact["source"] = "other"
    with pytest.raises(RuntimeError, match="runtime artifact has invalid source"):
        validate_manifests._validate_release_artifacts([artifact], "b" * 40)
