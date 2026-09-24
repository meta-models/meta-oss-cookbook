from __future__ import annotations

import asyncio
import logging

from muse_glimmer_worker.lifecycle import ProviderCleanup


class Closeable:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    async def aclose(self) -> None:
        self.calls += 1
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error


async def test_cleanup_attempts_every_provider_exactly_once(
    caplog,
) -> None:
    model = Closeable(error=RuntimeError("llm close failed"))
    parakeet = Closeable()
    supertonic = Closeable()
    cleanup = ProviderCleanup(
        llm=model,
        parakeet=parakeet,
        supertonic=supertonic,
        logger=logging.getLogger("cleanup-test"),
    )

    with caplog.at_level(logging.INFO):
        await asyncio.gather(cleanup.close(), cleanup.close(), cleanup.close())

    assert cleanup.closed
    assert (model.calls, parakeet.calls, supertonic.calls) == (1, 1, 1)
    assert "Closing LLM provider" in caplog.text
    assert "Failed to close LLM provider: llm close failed" in caplog.text
    assert "Closed Parakeet provider" in caplog.text
    assert "Closed Supertonic provider" in caplog.text


async def test_cleanup_survives_caller_cancellation() -> None:
    release = asyncio.Event()

    class BlockingCloseable(Closeable):
        async def aclose(self) -> None:
            self.calls += 1
            await release.wait()

    model = BlockingCloseable()
    parakeet = BlockingCloseable()
    supertonic = BlockingCloseable()
    cleanup = ProviderCleanup(
        llm=model,
        parakeet=parakeet,
        supertonic=supertonic,
        logger=logging.getLogger("cleanup-cancellation-test"),
    )
    first = asyncio.create_task(cleanup.close())
    await asyncio.sleep(0)
    first.cancel()
    release.set()

    await first
    await cleanup.close()

    assert cleanup.closed
    assert (model.calls, parakeet.calls, supertonic.calls) == (1, 1, 1)
