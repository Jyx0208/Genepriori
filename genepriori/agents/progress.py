from __future__ import annotations

import re
import time
from contextlib import contextmanager
from threading import Event, Thread
from typing import Callable, Iterator, Optional

WAIT_UPDATE_SECONDS = 10


class DebateProgress:
    def __init__(self, emit: Optional[Callable[[str], None]], *, secret: Optional[str] = None):
        self.emit = emit
        self.secret = secret

    def __call__(self, message: str) -> None:
        if self.emit is not None:
            if self.secret:
                message = message.replace(self.secret, "[redacted]")
            self.emit(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", message))

    @contextmanager
    def waiting(self, message: str) -> Iterator[None]:
        if self.emit is None:
            yield
            return

        self(message)
        done = Event()
        started = time.monotonic()

        def update() -> None:
            while not done.wait(WAIT_UPDATE_SECONDS):
                self(f"{message} (elapsed {time.monotonic() - started:.0f}s)")

        thread = Thread(target=update, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()
