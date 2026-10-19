"""Micro-batching: run requests that arrive close together as one model batch.

The first queued request starts a batch; the batcher then waits at most `max_wait_ms` for more,
up to `max_batch` images, and runs them in a single ONNX Runtime call. Phase 5b measured INT8
throughput rising from 88 img/s (batch 1) to 117 img/s (batch 32) on the laptop.

Only inference is batched. Preprocessing runs per request, in parallel, before queueing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np

PredictBatch = Callable[[np.ndarray], np.ndarray]  # (N, 3, H, W) -> (N, 6)


class MicroBatcher:
    def __init__(
        self,
        predict_batch: PredictBatch,
        max_batch: int = 8,
        max_wait_ms: float = 5.0,
        on_batch: Callable[[int], None] | None = None,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self.predict_batch = predict_batch
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.on_batch = on_batch
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        while self._queue is not None and not self._queue.empty():
            _, future = self._queue.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("service shutting down"))

    async def predict(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """Probabilities for one image (3, H, W), and the size of the batch it ran in."""
        if self._queue is None:
            raise RuntimeError("MicroBatcher.start() was not called")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((x, future))
        return await future

    async def _collect(self) -> list[tuple[np.ndarray, asyncio.Future]]:
        loop = asyncio.get_running_loop()
        items = [await self._queue.get()]
        deadline = loop.time() + self.max_wait
        while len(items) < self.max_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                items.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return items

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = await self._collect()
            batch = np.stack([x for x, _ in items])
            try:
                # The model call blocks, so it runs in a worker thread, not on the event loop.
                probs = await loop.run_in_executor(None, self.predict_batch, batch)
            except Exception as exc:  # every waiting request gets the error, none hangs
                for _, future in items:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for (_, future), p in zip(items, probs, strict=True):
                if not future.done():
                    future.set_result((p, len(items)))
            if self.on_batch is not None:
                self.on_batch(len(items))
