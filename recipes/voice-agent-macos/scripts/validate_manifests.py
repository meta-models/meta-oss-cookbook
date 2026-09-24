from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

try:
    import jsonschema
except ModuleNotFoundError:  # Core validation must also work before bootstrap.
    jsonschema = None  # type: ignore[assignment]

from scripts.repository import (
    ARTIFACT_LOCK,
    COMPATIBILITY_LOCK,
    ROOT,
    TOOLCHAIN_LOCK,
    landed_gate_commits,
    validate_executorch_checkout,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_CAPABILITIES = {"supports_cancel", "supertonic_server_jsonl"}
_REQUIRED_GATES = {"supertonic_runtime", "supports_cancel", "supertonic_server_jsonl"}
_GATE_STATUSES = {"landed", "pending", "unsubmitted"}
_EXECUTORCH_REPOSITORY = "https://github.com/pytorch/executorch.git"
_EXECUTORCH_ARTIFACT_ROLES = {
    "parakeet_helper",
    "muse_glimmer_worker",
    "supertonic_runner",
}
_MLX_REPOSITORY = "https://github.com/ml-explore/mlx.git"
_MLX_COMMIT = "7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247"


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"manifest must be a JSON object: {path}")
    return value


def _validate_artifacts() -> None:
    manifest = _load(ARTIFACT_LOCK)
    if manifest.get("schema_version") != 1 or manifest.get("platform") != "macos-arm64":
        raise RuntimeError("artifact manifest has an unsupported schema or platform")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("artifact manifest must contain artifacts")
    roles: set[str] = set()
    required_keys = {
        "role",
        "kind",
        "executable",
        "distribution",
        "license",
        "destination",
        "sensitive",
        "sha256",
    }
    for item in artifacts:
        if not isinstance(item, dict) or not required_keys <= set(item):
            raise RuntimeError("artifact entries must contain all required keys")
        role = item["role"]
        if not isinstance(role, str) or not role or role in roles:
            raise RuntimeError(f"artifact role is empty or duplicated: {role!r}")
        roles.add(role)
        if item["kind"] not in {"file", "directory"} or not isinstance(item["executable"], bool):
            raise RuntimeError(f"invalid artifact shape for {role}")
        if item["kind"] == "directory" and item["executable"]:
            raise RuntimeError(f"artifact directories cannot be executable: {role}")
        if item["distribution"] not in {"build", "download", "user-provided"}:
            raise RuntimeError(f"invalid distribution for {role}")
        destination = item["destination"]
        if not isinstance(destination, str) or not destination.startswith(".local/artifacts/"):
            raise RuntimeError(f"artifact destination escapes .local/artifacts: {role}")
        checksum = item["sha256"]
        if checksum is not None and (
            not isinstance(checksum, str) or not _SHA256.fullmatch(checksum)
        ):
            raise RuntimeError(f"invalid artifact checksum: {role}")
    required_roles = {
        "parakeet_helper",
        "parakeet_model",
        "parakeet_tokenizer",
        "muse_glimmer_worker",
        "muse_glimmer_model",
        "muse_glimmer_tokenizer",
        "muse_glimmer_tokenizer_config",
        "muse_glimmer_chat_template",
        "supertonic_runner",
        "mlx_metallib",
        "supertonic_model",
        "supertonic_assets",
        "supertonic_voice_style",
    }
    if roles != required_roles:
        raise RuntimeError(f"artifact roles differ from the runtime contract: {sorted(roles)}")


def _validate_compatibility(
    compatibility: dict[str, object], *, release_checkout: Path | None = None
) -> None:
    if compatibility.get("schema_version") != 1 or compatibility.get("platform") != "macos-arm64":
        raise RuntimeError("compatibility lock has an unsupported schema or platform")
    executorch = compatibility.get("executorch")
    if not isinstance(executorch, dict):
        raise RuntimeError("compatibility lock has no ExecuTorch configuration")
    if executorch.get("repository") != _EXECUTORCH_REPOSITORY:
        raise RuntimeError("compatibility lock must use the official ExecuTorch repository")
    capabilities = executorch.get("required_capabilities")
    if not isinstance(capabilities, list) or not set(capabilities) >= _REQUIRED_CAPABILITIES:
        raise RuntimeError("compatibility lock omits required runtime capabilities")
    gates = executorch.get("gates")
    if not isinstance(gates, dict) or not set(gates) >= _REQUIRED_GATES:
        raise RuntimeError("compatibility lock omits required upstream gates")
    for name in _REQUIRED_GATES:
        gate = gates[name]
        if not isinstance(gate, dict):
            raise RuntimeError(f"compatibility gate must be an object: {name}")
        status = gate.get("status")
        commit = gate.get("commit")
        pull_request = gate.get("pull_request")
        if status not in _GATE_STATUSES:
            raise RuntimeError(f"compatibility gate has invalid status: {name}")
        if pull_request is not None and (
            not isinstance(pull_request, str)
            or not pull_request.startswith("https://github.com/pytorch/executorch/pull/")
        ):
            raise RuntimeError(f"compatibility gate has invalid pull request: {name}")
        if status == "landed":
            if not isinstance(commit, str) or not _GIT_COMMIT.fullmatch(commit):
                raise RuntimeError(f"landed compatibility gate requires a commit: {name}")
        elif commit is not None:
            raise RuntimeError(f"unlanded compatibility gate cannot have a commit: {name}")

    final_commit = executorch.get("commit")
    if final_commit is not None and (
        not isinstance(final_commit, str) or not _GIT_COMMIT.fullmatch(final_commit)
    ):
        raise RuntimeError("compatibility lock has an invalid final ExecuTorch commit")
    if compatibility.get("ready_for_release"):
        if final_commit is None:
            raise RuntimeError("release-ready compatibility requires an immutable commit")
        unlanded = sorted(name for name in _REQUIRED_GATES if gates[name]["status"] != "landed")
        if unlanded:
            raise RuntimeError(f"release-ready compatibility has unlanded gates: {unlanded}")
        if release_checkout is None:
            raise RuntimeError(
                "release-ready compatibility requires checkout ancestry verification"
            )
        validate_executorch_checkout(
            release_checkout,
            final_commit,
            required_ancestors=landed_gate_commits(compatibility),
        )


def _validate_release_artifacts(artifacts: object, final_executorch_commit: object) -> None:
    if not isinstance(artifacts, list) or not isinstance(final_executorch_commit, str):
        raise RuntimeError("release-ready artifact validation requires an ExecuTorch commit")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("release-ready artifact entries must be objects")
        role = artifact.get("role")
        if not all(artifact.get(field) for field in ("source", "revision", "sha256")):
            raise RuntimeError(f"release-ready artifact provenance is incomplete: {role}")
        if artifact.get("size_bytes") is None or str(artifact.get("license", "")).startswith(
            "See "
        ):
            raise RuntimeError(f"release-ready artifact metadata is incomplete: {role}")
        source = artifact.get("source")
        if role in _EXECUTORCH_ARTIFACT_ROLES and source != "executorch":
            raise RuntimeError(f"ExecuTorch runtime artifact has invalid source: {role}")
        if source == "executorch" and artifact.get("revision") != final_executorch_commit:
            raise RuntimeError(
                f"ExecuTorch-built artifact must match the final compatibility commit: {role}"
            )
        if role == "mlx_metallib" and (
            source != _MLX_REPOSITORY
            or artifact.get("revision") != _MLX_COMMIT
            or artifact.get("license") != "MIT"
        ):
            raise RuntimeError("MLX metallib provenance must match the pinned MLX submodule")


def main() -> int:
    _validate_artifacts()
    schema = _load(ROOT / "artifacts/manifest.schema.json")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise RuntimeError("artifact schema must use JSON Schema 2020-12")
    if jsonschema is not None:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(_load(ARTIFACT_LOCK))

    compatibility = _load(COMPATIBILITY_LOCK)
    release_checkout = None
    if compatibility.get("ready_for_release"):
        release_checkout = (
            Path(os.environ.get("GLIMMER_EXECUTORCH_ROOT", ROOT / ".local/src/executorch"))
            .expanduser()
            .resolve()
        )
    _validate_compatibility(compatibility, release_checkout=release_checkout)
    if compatibility.get("ready_for_release"):
        _validate_release_artifacts(
            _load(ARTIFACT_LOCK)["artifacts"],
            compatibility["executorch"].get("commit"),
        )

    toolchain = _load(TOOLCHAIN_LOCK)
    if toolchain.get("platform") != {"system": "Darwin", "machine": "arm64"}:
        raise RuntimeError("toolchain platform must be macOS arm64")
    print("Manifest validation passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"validate-manifests: {error}", file=sys.stderr)
        raise SystemExit(1) from error
