from __future__ import annotations

import pytest

from scripts.bootstrap import _require_version, _version_tuple


def test_version_parser_accepts_common_tool_output() -> None:
    assert _version_tuple("Python 3.13.2") == (3, 13, 2)
    assert _version_tuple("v22.12.0") == (22, 12, 0)
    assert _version_tuple("livekit-server version 1.13.5") == (1, 13, 5)


def test_version_range_is_enforced() -> None:
    _require_version("node", "v22.12.0", ">=22.12.0,<23")
    with pytest.raises(RuntimeError, match="does not satisfy"):
        _require_version("node", "v24.0.0", ">=22.12.0,<23")
