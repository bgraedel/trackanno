"""Background clip prefetch.

The cache is an uncompressed memmap, so "loading" a clip is really "is this 1 MB
already in the OS page cache". Touching upcoming clips on a worker thread turns
almost every navigation into a warm read; numpy releases the GIL during the copy,
so a plain daemon thread is enough - no Qt threading required."""

from __future__ import annotations

import threading
from collections import deque

__all__ = ["Prefetcher"]


class Prefetcher:
    def __init__(self, store, depth: int = 8):
        self.store = store
        self.depth = max(0, depth)
        self._queue: deque[int] = deque()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._stop = False
        self._thread: threading.Thread | None = None
        if self.depth:
            self._thread = threading.Thread(target=self._run, name="trackanno-prefetch", daemon=True)
            self._thread.start()

    def around(self, order: list[int], position: int) -> None:
        """Queue the clips the annotator is most likely to ask for next."""
        if not self.depth:
            return
        wanted = [
            order[p]
            for p in range(position + 1, min(position + 1 + self.depth, len(order)))
        ]
        if position > 0:
            wanted.append(order[position - 1])
        with self._lock:
            self._queue.clear()
            self._queue.extend(wanted)
        self._wake.set()

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop and WAIT for the worker to leave the clip cache.

        Joining is not optional: the caller unmaps the memory this thread reads,
        and a thread caught inside ``touch()`` then segfaults the process. False
        if it did not stop in time, so the caller can decline to close the
        store."""
        self._stop = True
        self._wake.set()
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while not self._stop:
                with self._lock:
                    if not self._queue:
                        break
                    index = self._queue.popleft()
                try:
                    self.store.touch(index)
                except Exception:
                    pass  # a prefetch failure is never worth interrupting the UI
