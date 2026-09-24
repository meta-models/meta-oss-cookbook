from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.repository import load_valid_receipt, relative_local_path  # noqa: E402

MODEL_ID = "muse-glimmer-k-quant-17G-128K-text-dflash-metal"


def _server_environment(checkout: Path) -> dict[str, str]:
    pythonpath = os.pathsep.join(
        value for value in (str(checkout / "src"), os.environ.get("PYTHONPATH", "")) if value
    )
    environment = {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": os.environ.get("HOME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": pythonpath,
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "TRANSFORMERS_OFFLINE": "1",
    }
    if ssl_cert_file := os.environ.get("SSL_CERT_FILE"):
        environment["SSL_CERT_FILE"] = ssl_cert_file
    return environment


def main() -> None:
    receipt = load_valid_receipt()
    artifacts = receipt["artifacts"]
    checkout = Path(receipt["executorch_checkout"]).resolve()
    worker = relative_local_path(artifacts["muse_glimmer_worker"]["path"])
    model = relative_local_path(artifacts["muse_glimmer_model"]["path"])
    tokenizer = relative_local_path(artifacts["muse_glimmer_tokenizer"]["path"])
    tokenizer_root = tokenizer.parent

    environment = _server_environment(checkout)
    command = [
        sys.executable,
        "-m",
        "executorch.examples.models.muse_glimmer.serving.serve",
        "--model-path",
        str(model),
        "--tokenizer-path",
        str(tokenizer),
        "--hf-tokenizer",
        str(tokenizer_root),
        "--worker-bin",
        str(worker),
        "--model-id",
        MODEL_ID,
        "--artifact-mode",
        "dflash",
        "--max-context",
        "131072",
        "--tool-parser",
        "none",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    os.chdir(checkout)
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
