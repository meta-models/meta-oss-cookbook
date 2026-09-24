from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request

_BASE_URL = "http://127.0.0.1:8000"
_MODEL_ID = "muse-glimmer-k-quant-17G-128K-text-dflash-metal"


def _body(*, stream: bool, max_tokens: int) -> bytes:
    return json.dumps(
        {
            "model": _MODEL_ID,
            "messages": [{"role": "user", "content": "Reply with the word ready."}],
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_strength": "low"},
        }
    ).encode()


def _generation() -> None:
    request = urllib.request.Request(
        f"{_BASE_URL}/v1/chat/completions",
        data=_body(stream=False, max_tokens=2),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    if not payload.get("choices"):
        raise RuntimeError("LLM readiness generation returned no choices")


def _disconnect_stream() -> None:
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=90)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=_body(stream=True, max_tokens=128),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"LLM cancellation probe returned HTTP {response.status}")
    deadline = time.monotonic() + 90
    observed_content = False
    while time.monotonic() < deadline:
        line = response.readline()
        if not line:
            break
        if line.startswith(b"data:") and b"choices" in line:
            observed_content = True
            break
    connection.close()
    if not observed_content:
        raise RuntimeError("LLM cancellation probe received no streamed output")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generation, cancellation, and reuse")
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    args = parser.parse_args()
    _generation()
    _disconnect_stream()
    time.sleep(args.settle_seconds)
    _generation()
    print("MuseGlimmer generation, cancellation, and reuse probe passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, urllib.error.URLError) as error:
        raise SystemExit(f"llm-readiness: {error}") from error
