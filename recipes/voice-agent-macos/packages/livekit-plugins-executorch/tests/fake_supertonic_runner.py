#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import struct
import sys
import time
import wave
from pathlib import Path


def _args() -> list[str]:
    return sys.argv[1:]


def _send(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _write_wav(path: Path, *, channels: int = 1, rate: int = 44100, width: int = 2) -> None:
    samples = 4410
    if width == 2:
        payload = struct.pack(f"<{samples * channels}h", *([1000] * samples * channels))
    else:
        payload = b"\x80" * samples * channels
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(payload)


def main() -> int:
    argv = _args()
    capture_argv = os.getenv("FAKE_SUPERTONIC_ARGV_CAPTURE")
    if capture_argv:
        Path(capture_argv).write_text("\n".join(argv), encoding="utf-8")
    if "--server_jsonl=true" not in argv:
        print("server mode required", file=sys.stderr, flush=True)
        return 2

    mode = os.getenv("FAKE_SUPERTONIC_MODE", "success")
    if mode == "ready_timeout":
        time.sleep(60)
        return 0
    if mode == "bad_ready":
        _send({"type": "ready", "protocol_version": 2})
        return 3
    if mode == "stderr_crash":
        print("model load exploded", file=sys.stderr, flush=True)
        return 17

    _send(
        {
            "type": "ready",
            "protocol_version": 1,
            "sample_rate": 44100,
            "load_seconds": 0.01,
            "warmup_seconds": 0.02,
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        capture_request = os.getenv("FAKE_SUPERTONIC_REQUEST_CAPTURE")
        if capture_request and request.get("type") == "synthesize":
            with Path(capture_request).open("a", encoding="utf-8") as output:
                output.write(json.dumps(request, separators=(",", ":")) + "\n")
        if request.get("type") == "shutdown":
            if mode == "ignore_shutdown":
                continue
            _send({"type": "stopped"})
            return 0
        if request.get("type") != "synthesize":
            _send({"type": "error", "id": request.get("id"), "message": "bad request"})
            continue
        if mode == "sleep":
            time.sleep(60)
            continue
        if mode == "ignore_terminate":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)
            continue
        if mode == "error":
            print("voice style is invalid", file=sys.stderr, flush=True)
            _send(
                {
                    "type": "error",
                    "id": request["id"],
                    "message": "voice style is invalid",
                }
            )
            continue
        if mode == "wrong_id":
            request["id"] += 1
        output_path = Path(request["output"])
        if mode == "malformed":
            output_path.write_bytes(b"not a wav")
        elif mode == "stereo":
            _write_wav(output_path, channels=2)
        elif mode == "wrong_rate":
            _write_wav(output_path, rate=24000)
        elif mode == "wrong_width":
            _write_wav(output_path, width=1)
        elif mode != "missing":
            _write_wav(output_path)
        _send(
            {
                "type": "result",
                "id": request["id"],
                "output": request["output"],
                "samples": 4410,
                "audio_seconds": 0.1,
                "synthesis_seconds": 0.01,
                "rtf": 0.1,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
