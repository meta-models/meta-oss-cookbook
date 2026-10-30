"""Carry installed datasets dependency bounds into the nightly RL resolve."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement


def active_constraints(requirements: list[str]) -> list[str]:
    constraints = []
    for spec in requirements:
        requirement = Requirement(spec)
        if requirement.marker is not None and not requirement.marker.evaluate(
            {"extra": ""}
        ):
            continue
        # Constraints limit versions; extras were installed by the base resolve.
        requirement.extras = set()
        requirement.marker = None
        constraints.append(str(requirement))
    return sorted(set(constraints))


def main(output: str) -> None:
    requirements = importlib.metadata.requires("datasets") or []
    Path(output).write_text("\n".join(active_constraints(requirements)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
