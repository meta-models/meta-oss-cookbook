from __future__ import annotations

import subprocess

from scripts.repository import ROOT


def test_dev_and_dev_up_execute_up_once() -> None:
    dev = subprocess.run(
        ["make", "-n", "dev"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    dev_up = subprocess.run(
        ["make", "-n", "dev", "up"], cwd=ROOT, check=True, capture_output=True, text=True
    )

    command = ".venv/bin/python -m scripts.dev_stack up"
    assert dev.stdout.count(command) == 1
    assert dev_up.stdout.count(command) == 1
