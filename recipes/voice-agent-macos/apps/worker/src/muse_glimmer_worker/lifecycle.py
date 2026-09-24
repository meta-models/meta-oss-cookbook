from __future__ import annotations

import asyncio
import logging
from typing import Protocol


class AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


class ProviderCleanup:
    """Close every local provider once and let repeated callers await the result."""

    def __init__(
        self,
        *,
        llm: AsyncCloseable,
        parakeet: AsyncCloseable,
        supertonic: AsyncCloseable,
        logger: logging.Logger,
    ) -> None:
        self._providers = (
            ("LLM", llm),
            ("Parakeet", parakeet),
            ("Supertonic", supertonic),
        )
        self._logger = logger
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._cleanup_task is not None and self._cleanup_task.done()

    async def close(self) -> None:
        async with self._lock:
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(
                    self._close_providers(), name="muse-glimmer-provider-cleanup"
                )
            cleanup_task = self._cleanup_task

        # Shutdown must finish even if its original caller is cancelled repeatedly.
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        cleanup_task.result()

    async def _close_providers(self) -> None:
        for name, _ in self._providers:
            self._logger.info("Closing %s provider", name)
        results = await asyncio.gather(
            *(provider.aclose() for _, provider in self._providers),
            return_exceptions=True,
        )
        for (name, _), result in zip(self._providers, results, strict=True):
            if isinstance(result, BaseException):
                self._logger.error(
                    "Failed to close %s provider: %s",
                    name,
                    result,
                )
            else:
                self._logger.info("Closed %s provider", name)
