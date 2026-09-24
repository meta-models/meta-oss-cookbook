from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
import wave
import zipfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit, urlunsplit

from livekit import rtc
from livekit.agents import APIConnectOptions, llm
from livekit.agents.utils.audio import AudioByteStream

from .agent import main as worker_main
from .config import REASONING_STRENGTH, GlimmerConfig, LocalProviders, create_local_providers
from .lifecycle import ProviderCleanup

logger = logging.getLogger("museglimmer-cli")

_APP_NAME = "MuseGlimmer-VoiceAgent"
_REPORT_SCHEMA_VERSION = 1
_DEFAULT_TIMEOUT = 300.0


class JsonlReporter:
    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.events_path = report_dir / "events.jsonl"
        self._stream: TextIO = self.events_path.open("w", encoding="utf-8")

    def emit(self, event: str, **fields: object) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        self._stream.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        self._stream.flush()
        message = str(fields.get("message", event))
        if event.endswith("failed"):
            logger.error("%s: %s", event, message)
        else:
            logger.info("%s: %s", event, message)

    def close(self) -> None:
        self._stream.close()


class StageTimer:
    def __init__(self, reporter: JsonlReporter, stage: str, durations: dict[str, float]) -> None:
        self._reporter = reporter
        self._stage = stage
        self._durations = durations
        self._started = 0.0

    def __enter__(self) -> StageTimer:
        self._started = time.perf_counter()
        self._reporter.emit("stage_started", stage=self._stage, message=self._stage)
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        duration = time.perf_counter() - self._started
        self._durations[self._stage] = duration
        if exc is None:
            self._reporter.emit(
                "stage_completed",
                stage=self._stage,
                duration_seconds=round(duration, 6),
                message=self._stage,
            )
        else:
            self._reporter.emit(
                "stage_failed",
                stage=self._stage,
                duration_seconds=round(duration, 6),
                error_type=type(exc).__name__,
                message=str(exc),
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glimmer_cli.py",
        description="Diagnose and exercise the MuseGlimmer-VoiceAgent pipeline.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser(
        "doctor",
        help="Validate artifacts, native helper startup, and MuseGlimmer HTTP readiness.",
    )
    doctor.add_argument("--report-dir", type=Path)
    doctor.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT)

    pipeline = commands.add_parser(
        "pipeline",
        help="Run a PCM WAV through Parakeet, MuseGlimmer, and Supertonic.",
    )
    pipeline.add_argument("input_wav", type=Path)
    pipeline.add_argument("--output-wav", type=Path)
    pipeline.add_argument("--report-dir", type=Path)
    pipeline.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT)
    pipeline.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output WAV after a successful run.",
    )
    pipeline.add_argument(
        "--include-content",
        action="store_true",
        help="Include transcript and response text in the issue report.",
    )

    console = commands.add_parser(
        "console",
        help="Run the LiveKit microphone/speaker console with the same providers.",
    )
    console.add_argument("--input-device")
    console.add_argument("--output-device")
    console.add_argument("--list-devices", action="store_true")
    console.add_argument("--text", action="store_true")
    console.add_argument("--record", action="store_true")
    console.add_argument(
        "--console-log-level",
        choices=("trace", "debug", "info", "warn", "error", "critical"),
        default="debug",
        help="Log level passed to the LiveKit console process.",
    )
    return parser


class RedactingFormatter(logging.Formatter):
    def __init__(self, fmt: str) -> None:
        super().__init__(fmt)
        self.config: GlimmerConfig | None = None

    def format(self, record: logging.LogRecord) -> str:
        return _redact_text(super().format(record), self.config)


def _attach_runtime_log(report_dir: Path) -> logging.FileHandler:
    handler = logging.FileHandler(report_dir / "runtime.log", encoding="utf-8")
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _set_runtime_log_config(handler: logging.FileHandler, config: GlimmerConfig) -> None:
    formatter = handler.formatter
    if isinstance(formatter, RedactingFormatter):
        formatter.config = config


def _detach_runtime_log(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def _prepare_report_dir(command: str, requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        requested = Path.cwd() / "museglimmer-reports" / f"{stamp}-{command}-{uuid.uuid4().hex[:8]}"
    path = requested.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"report directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _system_snapshot() -> dict[str, object]:
    return {
        "app": _APP_NAME,
        "schema_version": _REPORT_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "packages": {
            "livekit-agents": _package_version("livekit-agents"),
            "livekit-plugins-executorch": _package_version("livekit-plugins-executorch"),
            "livekit-plugins-openai": _package_version("livekit-plugins-openai"),
        },
    }


def _artifact(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    stat = path.stat()
    return {
        "path": f".../{path.name}",
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "executable": os.access(path, os.X_OK),
    }


def _safe_url(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _redacted_config(config: GlimmerConfig) -> dict[str, object]:
    return {
        "agent_name": config.agent_name,
        "language": config.language,
        "muse_glimmer": {
            "base_url": _safe_url(config.muse_glimmer_base_url),
            "model_id": config.muse_glimmer_model_id,
            "temperature": config.muse_glimmer_temperature,
            "max_tokens": config.muse_glimmer_max_tokens,
            "reasoning_strength": REASONING_STRENGTH,
            "api_key": "<redacted>",
        },
        "parakeet": {
            "helper": _artifact(config.parakeet_helper_path),
            "model": _artifact(config.parakeet_model_path),
            "tokenizer": _artifact(config.parakeet_tokenizer_path),
            "delegate_data": _artifact(config.parakeet_delegate_data_path),
        },
        "supertonic": {
            "runner": _artifact(config.supertonic_runner_path),
            "pte": _artifact(config.supertonic_pte_path),
            "asset_dir": _artifact(config.supertonic_asset_dir),
            "voice_style": _artifact(config.supertonic_voice_style_path),
            "speed": config.supertonic_speed,
            "seed": config.supertonic_seed,
        },
    }


def _read_pcm_wav(path: Path) -> tuple[list[rtc.AudioFrame], dict[str, object]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"input WAV does not exist: {source}")
    with wave.open(str(source), "rb") as input_wav:
        channels = input_wav.getnchannels()
        sample_rate = input_wav.getframerate()
        sample_width = input_wav.getsampwidth()
        frame_count = input_wav.getnframes()
        compression = input_wav.getcomptype()
        payload = input_wav.readframes(frame_count)
    if compression != "NONE" or sample_width != 2:
        raise ValueError("input must be uncompressed signed PCM16 WAV")
    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError("input WAV must contain non-empty audio with a valid format")

    byte_stream = AudioByteStream(
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=max(1, sample_rate // 10),
    )
    frames = [*byte_stream.push(payload), *byte_stream.flush()]
    return frames, {
        "path": f".../{source.name}",
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frame_count,
        "duration_seconds": frame_count / sample_rate,
        "size_bytes": source.stat().st_size,
    }


async def _write_synthesized_wav(
    stream: Any, output_path: Path, *, force: bool
) -> dict[str, object]:
    target = output_path.expanduser().resolve()
    if target.exists() and not force:
        raise ValueError(f"output WAV already exists; pass --force to replace it: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    sample_rate: int | None = None
    channels: int | None = None
    sample_count = 0
    event_count = 0
    request_id: str | None = None
    output_wav: wave.Wave_write | None = None
    try:
        async with stream:
            async for event in stream:
                frame = event.frame
                if event.request_id != request_id:
                    if output_wav is not None:
                        output_wav.close()
                        partial.unlink(missing_ok=True)
                    request_id = event.request_id
                    sample_rate = frame.sample_rate
                    channels = frame.num_channels
                    sample_count = 0
                    event_count = 0
                    output_wav = wave.open(str(partial), "wb")  # noqa: SIM115
                    output_wav.setnchannels(channels)
                    output_wav.setsampwidth(2)
                    output_wav.setframerate(sample_rate)
                elif frame.sample_rate != sample_rate or frame.num_channels != channels:
                    raise RuntimeError("TTS changed audio format during one synthesis attempt")
                if output_wav is None:
                    raise RuntimeError("TTS stream did not initialize an output attempt")
                output_wav.writeframesraw(frame.data.tobytes())
                sample_count += frame.samples_per_channel
                event_count += 1
        if output_wav is not None:
            output_wav.close()
            output_wav = None
        if sample_rate is None or channels is None or sample_count == 0:
            raise RuntimeError("TTS returned no audio")
        if target.exists() and not force:
            raise ValueError(
                f"output WAV appeared during synthesis; refusing to replace it: {target}"
            )
        partial.replace(target)
    except BaseException:
        if output_wav is not None:
            output_wav.close()
        partial.unlink(missing_ok=True)
        raise
    return {
        "path": f".../{target.name}",
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": 2,
        "samples_per_channel": sample_count,
        "duration_seconds": sample_count / sample_rate,
        "events": event_count,
        "request_id": request_id,
        "size_bytes": target.stat().st_size,
    }


async def _probe_synthesized_audio(stream: Any) -> dict[str, object]:
    sample_count = 0
    event_count = 0
    request_id: str | None = None
    async with stream:
        async for event in stream:
            frame = event.frame
            if frame.sample_rate != 44100 or frame.num_channels != 1:
                raise RuntimeError("Supertonic must return 44.1 kHz mono audio")
            if request_id is None:
                request_id = event.request_id
            elif event.request_id != request_id:
                raise RuntimeError("Supertonic changed request ID during the doctor probe")
            sample_count += frame.samples_per_channel
            event_count += 1
    if request_id is None or sample_count <= 0:
        raise RuntimeError("Supertonic returned no audio during the doctor probe")
    return {
        "sample_rate": 44100,
        "channels": 1,
        "samples_per_channel": sample_count,
        "duration_seconds": sample_count / 44100,
        "events": event_count,
        "request_id": request_id,
    }


def _http_json(url: str, api_key: str, timeout: float) -> object:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HTTP readiness request failed for {_safe_url(url)}: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"HTTP readiness response was not JSON: {_safe_url(url)}") from exc


def _provider_cleanup(providers: LocalProviders) -> ProviderCleanup:
    return ProviderCleanup(
        llm=providers.llm,
        parakeet=providers.parakeet,
        supertonic=providers.supertonic,
        logger=logger,
    )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _create_issue_bundle(report_dir: Path) -> Path:
    bundle = report_dir / "issue-report.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("report.json", "events.jsonl"):
            path = report_dir / name
            if path.exists():
                archive.write(path, arcname=name)
    return bundle


def _attach_provider_reporting(
    providers: LocalProviders,
    reporter: JsonlReporter,
    config: GlimmerConfig,
) -> None:
    def metrics(stage: str, event: object) -> None:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else str(event)
        reporter.emit(
            "provider_metrics",
            stage=stage,
            metrics=_redact_payload(payload, config),
            message=stage,
        )

    def error(stage: str, event: object) -> None:
        exception = getattr(event, "error", RuntimeError(str(event)))
        reporter.emit(
            "provider_error",
            stage=stage,
            recoverable=bool(getattr(event, "recoverable", False)),
            error_type=type(exception).__name__,
            message=_redact_text(str(exception), config),
        )

    for stage, provider in (
        ("stt", providers.parakeet),
        ("llm", providers.llm),
        ("tts", providers.supertonic),
    ):
        if not hasattr(provider, "on"):
            continue
        provider.on("metrics_collected", lambda event, stage=stage: metrics(stage, event))
        provider.on("error", lambda event, stage=stage: error(stage, event))


def _redact_payload(value: object, config: GlimmerConfig | None) -> object:
    if isinstance(value, str):
        return _redact_text(value, config)
    if isinstance(value, dict):
        return {
            _redact_text(str(key), config): _redact_payload(item, config)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_payload(item, config) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value), config)


def _redact_text(value: str, config: GlimmerConfig | None) -> str:
    replacements: dict[str, str] = {str(Path.home()): "~"}
    for secret_name in ("LIVEKIT_API_SECRET", "MUSE_GLIMMER_API_KEY"):
        secret = os.getenv(secret_name, "")
        if secret:
            replacements[secret] = "<redacted>"
    if config is not None:
        replacements[config.muse_glimmer_api_key] = "<redacted>"
        for path in (
            config.parakeet_helper_path,
            config.parakeet_model_path,
            config.parakeet_tokenizer_path,
            config.parakeet_delegate_data_path,
            config.supertonic_runner_path,
            config.supertonic_pte_path,
            config.supertonic_asset_dir,
            config.supertonic_voice_style_path,
        ):
            if path is not None:
                replacements[str(path)] = f".../{path.name}"
    for original, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if original:
            value = value.replace(original, replacement)
    return value


def _failure(exc: BaseException, config: GlimmerConfig | None) -> dict[str, object]:
    return {
        "type": type(exc).__name__,
        "message": _redact_text(str(exc), config),
        "traceback": _redact_text("".join(traceback.format_exception(exc)), config),
    }


async def _run_doctor(args: argparse.Namespace) -> int:
    report_dir = _prepare_report_dir("doctor", args.report_dir)
    runtime_handler = _attach_runtime_log(report_dir)
    reporter = JsonlReporter(report_dir)
    durations: dict[str, float] = {}
    cleanup: ProviderCleanup | None = None
    config: GlimmerConfig | None = None
    summary: dict[str, object] = {
        "command": "doctor",
        "status": "failed",
        "system": _system_snapshot(),
        "durations_seconds": durations,
    }
    exit_code = 1
    try:
        with StageTimer(reporter, "configuration", durations):
            config = GlimmerConfig.from_env()
            _set_runtime_log_config(runtime_handler, config)
            summary["configuration"] = _redacted_config(config)

        with StageTimer(reporter, "muse_glimmer_http", durations):
            base = config.muse_glimmer_base_url.removesuffix("/v1")
            health, models = await asyncio.gather(
                asyncio.to_thread(
                    _http_json, f"{base}/health", config.muse_glimmer_api_key, args.timeout
                ),
                asyncio.to_thread(
                    _http_json,
                    f"{config.muse_glimmer_base_url}/models",
                    config.muse_glimmer_api_key,
                    args.timeout,
                ),
            )
            if not isinstance(health, dict) or health.get("status") != "ok":
                raise RuntimeError(f"MuseGlimmer health check did not return status=ok: {health!r}")
            if not isinstance(models, dict) or not isinstance(models.get("data"), list):
                raise RuntimeError(f"MuseGlimmer models response is invalid: {models!r}")
            model_ids = {item.get("id") for item in models["data"] if isinstance(item, dict)}
            if config.muse_glimmer_model_id not in model_ids:
                raise RuntimeError(
                    f"MuseGlimmer model is missing from /v1/models: {config.muse_glimmer_model_id}"
                )
            summary["muse_glimmer_http"] = {"health": health, "models": models}

        providers = create_local_providers(config)
        cleanup = _provider_cleanup(providers)
        _attach_provider_reporting(providers, reporter, config)
        connect_options = APIConnectOptions(max_retry=0, timeout=args.timeout)

        with StageTimer(reporter, "parakeet_startup", durations):
            async with asyncio.timeout(args.timeout):
                await providers.parakeet.start()

        with StageTimer(reporter, "supertonic_synthesis", durations):
            async with asyncio.timeout(args.timeout):
                first = await _probe_synthesized_audio(
                    providers.supertonic.synthesize(
                        "Hello from Glimmer.", conn_options=connect_options
                    )
                )
                process = providers.supertonic._process
                first_pid = process.pid if process is not None else None
                second = await _probe_synthesized_audio(
                    providers.supertonic.synthesize(
                        "The warm voice process is reusable.", conn_options=connect_options
                    )
                )
                if (
                    first_pid is None
                    or providers.supertonic._process is None
                    or providers.supertonic._process.pid != first_pid
                ):
                    raise RuntimeError("Supertonic did not reuse one warm process")
                summary["supertonic_probe"] = {
                    "process_reused": True,
                    "utterances": [first, second],
                }

        summary["status"] = "passed"
        exit_code = 0
        reporter.emit("doctor_completed", message="all deployment checks passed")
    except Exception as exc:
        error = _failure(exc, config)
        summary["error"] = error
        reporter.emit("doctor_failed", error_type=type(exc).__name__, message=error["message"])
    finally:
        if cleanup is not None:
            await cleanup.close()
        _write_json(report_dir / "report.json", summary)
        reporter.close()
        _detach_runtime_log(runtime_handler)
        bundle = _create_issue_bundle(report_dir)
        print(
            json.dumps(
                {"status": summary["status"], "report_dir": str(report_dir), "bundle": str(bundle)}
            )
        )
    return exit_code


async def _run_pipeline(args: argparse.Namespace) -> int:
    report_dir = _prepare_report_dir("pipeline", args.report_dir)
    runtime_handler = _attach_runtime_log(report_dir)
    reporter = JsonlReporter(report_dir)
    durations: dict[str, float] = {}
    cleanup: ProviderCleanup | None = None
    config: GlimmerConfig | None = None
    output_path = (args.output_wav or (report_dir / "response.wav")).expanduser().resolve()
    summary: dict[str, object] = {
        "command": "pipeline",
        "status": "failed",
        "system": _system_snapshot(),
        "durations_seconds": durations,
        "content_included": bool(args.include_content),
    }
    exit_code = 1
    transcript = ""
    response_text = ""
    try:
        with StageTimer(reporter, "configuration", durations):
            config = GlimmerConfig.from_env()
            _set_runtime_log_config(runtime_handler, config)
            summary["configuration"] = _redacted_config(config)

        with StageTimer(reporter, "input_wav", durations):
            input_frames, input_metadata = _read_pcm_wav(args.input_wav)
            summary["input_audio"] = input_metadata

        with StageTimer(reporter, "provider_startup", durations):
            providers = create_local_providers(config)
            cleanup = _provider_cleanup(providers)
            _attach_provider_reporting(providers, reporter, config)
            async with asyncio.timeout(args.timeout):
                await providers.parakeet.start()

        connect_options = APIConnectOptions(max_retry=0, timeout=args.timeout)
        with StageTimer(reporter, "stt", durations):
            async with asyncio.timeout(args.timeout):
                speech = await providers.parakeet.recognize(
                    input_frames,
                    language=config.language,
                    conn_options=connect_options,
                )
            if not speech.alternatives:
                raise RuntimeError("Parakeet returned no transcript alternatives")
            transcript = speech.alternatives[0].text.strip()
            if not transcript:
                raise RuntimeError("Parakeet returned an empty transcript")
            summary["transcript_chars"] = len(transcript)

        with StageTimer(reporter, "llm", durations):
            chat_context = llm.ChatContext()
            chat_context.add_message(role="system", content=config.instructions)
            chat_context.add_message(role="user", content=transcript)
            async with asyncio.timeout(args.timeout):
                completion = await providers.llm.chat(
                    chat_ctx=chat_context,
                    conn_options=connect_options,
                ).collect()
            if completion.tool_calls:
                raise RuntimeError("MuseGlimmer returned tool calls in the no-tools CLI pipeline")
            response_text = completion.text.strip()
            if not response_text:
                raise RuntimeError("MuseGlimmer returned an empty response")
            summary["response_chars"] = len(response_text)
            if completion.usage is not None:
                summary["llm_usage"] = completion.usage.model_dump(mode="json")

        with StageTimer(reporter, "tts", durations):
            async with asyncio.timeout(args.timeout):
                audio_metadata = await _write_synthesized_wav(
                    providers.supertonic.synthesize(response_text, conn_options=connect_options),
                    output_path,
                    force=args.force,
                )
            summary["output_audio"] = audio_metadata

        if args.include_content:
            summary["transcript"] = transcript
            summary["response_text"] = response_text
        summary["status"] = "passed"
        exit_code = 0
        reporter.emit("pipeline_completed", message="end-to-end pipeline passed")
    except Exception as exc:
        error = _failure(exc, config)
        summary["error"] = error
        reporter.emit("pipeline_failed", error_type=type(exc).__name__, message=error["message"])
    finally:
        if cleanup is not None:
            await cleanup.close()
        _write_json(report_dir / "report.json", summary)
        reporter.close()
        _detach_runtime_log(runtime_handler)
        bundle = _create_issue_bundle(report_dir)
        result: dict[str, object] = {
            "status": summary["status"],
            "report_dir": str(report_dir),
            "bundle": str(bundle),
        }
        if exit_code == 0:
            result.update(
                {
                    "transcript": transcript,
                    "response": response_text,
                    "output_wav": str(output_path),
                }
            )
        print(json.dumps(result, indent=2, ensure_ascii=True))
    return exit_code


def _console_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "muse_glimmer_worker",
        "console",
        "--log-level",
        args.console_log_level,
    ]
    if args.input_device:
        command.extend(("--input-device", args.input_device))
    if args.output_device:
        command.extend(("--output-device", args.output_device))
    if args.list_devices:
        command.append("--list-devices")
    if args.text:
        command.append("--text")
    if args.record:
        command.append("--record")
    return command


def _exec_console(args: argparse.Namespace) -> int:
    command = _console_command(args)
    os.execv(sys.executable, command)
    return 127


def run_worker() -> None:
    worker_main()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if getattr(args, "timeout", 1.0) <= 0:
        parser.error("--timeout must be positive")
    if args.command == "console":
        return _exec_console(args)
    try:
        if args.command == "doctor":
            return asyncio.run(_run_doctor(args))
        if args.command == "pipeline":
            return asyncio.run(_run_pipeline(args))
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
