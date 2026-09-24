from __future__ import annotations

import subprocess
import sys

import pytest

from scripts import dev_stack
from scripts.repository import (
    CREDENTIAL_FILE,
    PREPARED_RECEIPT,
    ROOT,
    load_valid_receipt,
)

pytestmark = pytest.mark.e2e


def test_running_prepared_stack_generation_cancellation_and_privacy() -> None:
    if not PREPARED_RECEIPT.is_file():
        pytest.skip("local artifacts are not prepared")
    if dev_stack.status() != 0:
        pytest.skip("prepared Muse Glimmer stack is not running; run `make dev` first")

    api_key, api_secret = CREDENTIAL_FILE.read_text(encoding="utf-8").strip().split(": ", 1)
    worker_environment = dev_stack._services(load_valid_receipt(), api_key, api_secret)[
        -1
    ].environment
    subprocess.run(
        [sys.executable, "-m", "scripts.llm_readiness"], cwd=ROOT, check=True, timeout=240
    )
    subprocess.run(
        [str(ROOT / ".venv/bin/muse-glimmer-diagnostics"), "doctor"],
        cwd=ROOT,
        env=worker_environment,
        check=True,
        timeout=360,
    )
    subprocess.run(
        [sys.executable, "-m", "scripts.privacy_audit"], cwd=ROOT, check=True, timeout=60
    )
