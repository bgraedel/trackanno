"""Background prefetch: touch upcoming clips so they are in the OS page cache."""

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
        """Stop and join the worker before the cache is unmapped; False on timeout."""
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
                    pass  # prefetch failures are harmless
