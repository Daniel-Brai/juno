import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Protocol

from juno.config import Durations
from juno.domain.session import Session
from juno.ports import Clock


class TurnDriver(Protocol):
    async def next_message(self, session: Session, /) -> str | None: ...


class ContinueDriver:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    async def next_message(self, session: Session, /) -> str:
        remaining = session.remaining(self._clock.now())
        return f"Continue working. {Durations.format(remaining)} remaining."


class ScriptDriver:
    def __init__(self, messages: Iterable[str]) -> None:
        self._messages = deque(messages)

    async def next_message(self, _session: Session, /) -> str | None:
        return self._messages.popleft() if self._messages else None


class HumanDriver:
    def __init__(self) -> None:
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()

    async def next_message(self, _session: Session, /) -> str | None:
        return await self.inbox.get()
