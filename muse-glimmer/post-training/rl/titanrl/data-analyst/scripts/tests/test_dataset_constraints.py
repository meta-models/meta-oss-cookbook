"""Test version constraints extracted from declared dataset requirements."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "dataset_constraints.py"
SPEC = importlib.util.spec_from_file_location(
    "data_analyst_dataset_constraints", SCRIPT
)
constraints = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(constraints)


def test_active_constraints_preserve_bounds_without_installing_extras() -> None:
    result = constraints.active_constraints(
        [
            "dill>=0.3.0,<0.4.1",
            "fsspec[http]>=2023.1.0,<=2026.2.0",
            "huggingface-hub>=0.25.0,<2.0",
            "pytest; extra == 'test'",
            "old-python-only; python_version < '3.9'",
        ]
    )
    assert set(result) == {
        "dill<0.4.1,>=0.3.0",
        "fsspec<=2026.2.0,>=2023.1.0",
        "huggingface-hub<2.0,>=0.25.0",
    }


def test_constraints_are_deduplicated_and_deterministic() -> None:
    assert constraints.active_constraints(["numpy>=1.17", "numpy>=1.17"]) == [
        "numpy>=1.17"
    ]
