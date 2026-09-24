#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import struct
import sys
import time


def send(message: dict[str, object], payload: bytes | None = None) -> None:
    sys.stdout.buffer.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
    if payload is not None:
        sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def main() -> int:
    mode = os.environ.get("FAKE_HELPER_MODE", "stt")
    if mode == "timeout":
        time.sleep(60)
        return 0
    if mode == "stderr_crash":
        print("model load exploded", file=sys.stderr, flush=True)
        return 17
    if mode == "malformed_ready":
        sys.stdout.buffer.write(b"not-json\n")
        sys.stdout.buffer.flush()
        return 0
    if mode == "oversized":
        send({"type": "ready", "version": 1})
        send({"type": "audio_chunk", "version": 1, "payload_byte_count": 999999})
        return 0

    if mode.startswith("tts"):
        send(
            {
                "type": "ready",
                "version": 1,
                "sample_rate": 24000,
                "channel_count": 1,
                "encoding": "f32le",
            }
        )
    else:
        send({"type": "ready", "version": 1})

    active_request_id: str | None = None
    for raw_line in sys.stdin.buffer:
        request = json.loads(raw_line)
        request_type = request.get("type")
        if request_type == "shutdown":
            if mode == "ignore_shutdown":
                continue
            return 0
        if request_type == "transcribe":
            audio = request["audio"]
            payload = sys.stdin.buffer.read(audio["payload_byte_count"])
            if mode == "eof":
                return 3
            if mode == "stt_slow":
                time.sleep(60)
                continue
            if audio != {
                "encoding": "f32le",
                "sample_rate": 16000,
                "channel_count": 1,
                "payload_byte_count": len(payload),
            }:
                send(
                    {
                        "type": "error",
                        "version": 1,
                        "request_id": request["request_id"],
                        "message": "invalid audio descriptor",
                    }
                )
                continue
            if mode == "stt_bad_result":
                send(
                    {
                        "type": "result",
                        "version": 1,
                        "request_id": request["request_id"],
                        "text": 42,
                    }
                )
                continue
            if mode == "stt_error":
                send(
                    {
                        "type": "error",
                        "version": 1,
                        "request_id": request["request_id"],
                        "message": "bad audio",
                        "details": "fake failure",
                    }
                )
                continue
            samples = struct.unpack(f"<{len(payload) // 4}f", payload)
            send(
                {
                    "type": "status",
                    "version": 1,
                    "request_id": request["request_id"],
                    "phase": "running_encoder",
                    "message": "Running encoder...",
                }
            )
            send(
                {
                    "type": "result",
                    "version": 1,
                    "request_id": request["request_id"],
                    "text": ",".join(f"{sample:.3f}" for sample in samples[:4]),
                    "audio_descriptor": audio,
                }
            )
            continue
        if request_type == "synthesize":
            request_id = request["request_id"]
            active_request_id = request_id
            if (
                request.get("voice") != "voice.pt"
                or request.get("temperature") != 0.25
                or request.get("max_new_tokens") != 321
            ):
                send(
                    {
                        "type": "error",
                        "version": 1,
                        "request_id": request_id,
                        "message": "invalid synthesis options",
                    }
                )
                continue
            if mode == "tts_error":
                send(
                    {
                        "type": "error",
                        "version": 1,
                        "request_id": request_id,
                        "message": "voice missing",
                    }
                )
                continue
            if mode == "tts_cancel_timeout":
                time.sleep(60)
                continue
            if mode == "tts_slow":
                time.sleep(60)
                continue
            if mode == "tts_wait_cancel":
                continue
            if mode == "tts_finish_cancel_race":
                send(
                    {
                        "type": "result",
                        "version": 1,
                        "request_id": request_id,
                        "cancelled": False,
                        "sample_count": 0,
                    }
                )
                active_request_id = None
                continue
            chunks = [(-1.5, -1.0, -0.5, 0.0), (0.5, 1.0, 1.5)]
            for chunk in chunks:
                if mode == "tts_progressive":
                    chunk = chunk * 300
                payload = struct.pack(f"<{len(chunk)}f", *chunk)
                send(
                    {
                        "type": "audio_chunk",
                        "version": 1,
                        "request_id": request_id,
                        "payload_byte_count": len(payload),
                    },
                    payload,
                )
            send(
                {
                    "type": "result",
                    "version": 1,
                    "request_id": request_id,
                    "cancelled": "no" if mode == "tts_bad_result" else False,
                    "sample_count": 7,
                    "request": request,
                }
            )
            continue
        if request_type == "cancel" and request["request_id"] == active_request_id:
            send(
                {
                    "type": "result",
                    "version": 1,
                    "request_id": request["request_id"],
                    "cancelled": True,
                    "sample_count": 0,
                }
            )
            active_request_id = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
