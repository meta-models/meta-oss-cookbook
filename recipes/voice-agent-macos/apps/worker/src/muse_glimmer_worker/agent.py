from __future__ import annotations

import logging
import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    CloseEvent,
    ErrorEvent,
    JobContext,
    JobProcess,
    SessionUsageUpdatedEvent,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    cli,
)
from livekit.plugins import silero

from .config import GlimmerConfig, create_providers
from .lifecycle import ProviderCleanup

logger = logging.getLogger("muse-glimmer-worker")
_VAD_KEY = "glimmer_vad"


class GlimmerAgent(Agent):
    def __init__(self, *, instructions: str) -> None:
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user briefly and ask how you can help.",
            allow_interruptions=True,
        )


def setup_process(proc: JobProcess) -> None:
    proc.userdata[_VAD_KEY] = silero.VAD.load()


server = AgentServer(setup_fnc=setup_process, host="127.0.0.1")


@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext) -> None:
    config = GlimmerConfig.from_env()
    ctx.log_context_fields = {"room": ctx.room.name, "agent": config.agent_name}
    providers = create_providers(
        config,
        voice_activity_detector=ctx.proc.userdata[_VAD_KEY],
    )
    cleanup = ProviderCleanup(
        llm=providers.llm,
        parakeet=providers.parakeet,
        supertonic=providers.supertonic,
        logger=logger,
    )
    ctx.add_shutdown_callback(cleanup.close)

    try:
        await providers.parakeet.start()
        session: AgentSession[None] = AgentSession(
            stt=providers.session_stt,
            llm=providers.llm,
            tts=providers.session_tts,
            turn_handling=TurnHandlingOptions(
                interruption={
                    "enabled": True,
                    "resume_false_interruption": True,
                    "false_interruption_timeout": 1.0,
                },
                preemptive_generation={"enabled": False},
            ),
            tts_text_transforms=["filter_markdown", "filter_emoji"],
        )
        _attach_session_logging(session)
        await session.start(
            agent=GlimmerAgent(instructions=config.instructions),
            room=ctx.room,
        )
    except BaseException:
        await cleanup.close()
        raise


def _attach_session_logging(session: AgentSession[None]) -> None:
    @session.on("user_state_changed")
    def _on_user_state_changed(event: UserStateChangedEvent) -> None:
        logger.info("USER: %s -> %s", event.old_state, event.new_state)

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: UserInputTranscribedEvent) -> None:
        if event.is_final:
            logger.info("STT: final transcript received (%d characters)", len(event.transcript))
        logger.debug("STT: final=%s transcript=%r", event.is_final, event.transcript)

    @session.on("user_transcription_timeout")
    def _on_user_transcription_timeout(event: UserTranscriptionTimeoutEvent) -> None:
        logger.warning(
            "STT: no transcript after %.2fs of VAD-detected speech",
            event.speech_duration,
        )

    @session.on("agent_state_changed")
    def _on_agent_state_changed(event: AgentStateChangedEvent) -> None:
        logger.info("AGENT: %s -> %s", event.old_state, event.new_state)

    @session.on("error")
    def _on_error(event: ErrorEvent) -> None:
        logger.error("PIPELINE ERROR: %s", event.model_dump(mode="json"))

    @session.on("close")
    def _on_close(event: CloseEvent) -> None:
        logger.info("SESSION CLOSED: reason=%s error=%s", event.reason.value, event.error)

    last_usage: str | None = None

    @session.on("session_usage_updated")
    def _on_usage_updated(event: SessionUsageUpdatedEvent) -> None:
        nonlocal last_usage
        snapshot = repr(event.usage)
        if snapshot == last_usage:
            return
        last_usage = snapshot
        logger.debug("Glimmer session usage changed: %s", event.usage)


def main() -> None:
    config = GlimmerConfig.from_env()
    os.environ["LIVEKIT_URL"] = config.livekit_url
    os.environ["LIVEKIT_API_KEY"] = config.livekit_api_key
    os.environ["LIVEKIT_API_SECRET"] = config.livekit_api_secret
    cli.run_app(server)


if __name__ == "__main__":
    main()
