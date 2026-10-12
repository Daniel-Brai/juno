from typing import Any, ClassVar

import structlog

from juno.domain.events import (
    CacheMissed,
    CheckpointTaken,
    Event,
    ModelCalled,
    ResourcePressure,
    SessionEnded,
    SubmissionRegistered,
    TurnContinued,
)
from juno.domain.values import Checkpoint
from juno.ports import Checkpointer, Clock, UnitOfWorkFactory
from juno.service.bus import MessageBus
from juno.service.context import RunContext


class EventRecorder:
    def __init__(self, units: UnitOfWorkFactory) -> None:
        self._units = units

    def attach(self, bus: MessageBus) -> None:
        bus.subscribe(Event, self.record)

    async def record(self, event: Event) -> None:
        async with self._units() as uow:
            await uow.events.add(event)
            await uow.commit()


class CheckpointRecorder:
    TRIGGERS: tuple[type[Event], ...] = (SubmissionRegistered, TurnContinued, SessionEnded)

    def __init__(self, checkpointer: Checkpointer, context: RunContext) -> None:
        self._checkpointer = checkpointer
        self._units = context.units
        self._bus = context.bus
        self._clock = context.clock
        self._session = context.session

    def attach(self) -> None:
        for trigger in self.TRIGGERS:
            self._bus.subscribe(trigger, self.checkpoint)

    async def checkpoint(self, event: Event) -> None:
        index = len(self._session.messages) - 1
        commit = await self._checkpointer.checkpoint(f"{type(event).__name__} at message {index}")

        if commit is None:
            return

        checkpoint = Checkpoint(commit=commit, message_index=index)

        async with self._units() as uow:
            await uow.checkpoints.add(self._session.id, checkpoint)
            await uow.commit()

        await self._bus.publish(
            CheckpointTaken(
                session_id=self._session.id, at=self._clock.now(), checkpoint=checkpoint
            )
        )


class CacheMonitor:
    WARN_AFTER: ClassVar[int] = 2
    CACHEABLE_TOKENS: ClassVar[int] = 4_096

    def __init__(self, bus: MessageBus, clock: Clock) -> None:
        self._bus = bus
        self._clock = clock
        self._calls = 0
        self._misses = 0

    def attach(self) -> None:
        self._bus.subscribe(ModelCalled, self.observe)

    async def observe(self, event: ModelCalled) -> None:
        self._calls += 1
        usage = event.usage
        prompt = (
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
        )

        if self._calls == 1 or usage.cache_read_input_tokens:
            self._misses = 0
            return

        if prompt < self.CACHEABLE_TOKENS:
            return

        self._misses += 1

        if self._misses == self.WARN_AFTER:
            await self._bus.publish(
                CacheMissed(
                    session_id=event.session_id,
                    at=self._clock.now(),
                    message_index=event.message_index,
                    misses=self._misses,
                )
            )


class LogRecorder:
    TEXT_LIMIT: ClassVar[int] = 200
    WARNINGS: ClassVar[tuple[type[Event], ...]] = (CacheMissed, ResourcePressure)

    def __init__(self, logger: Any = None) -> None:
        self._log = logger or structlog.get_logger("juno")

    def attach(self, bus: MessageBus) -> None:
        bus.subscribe(Event, self.record)

    async def record(self, event: Event) -> None:
        log = self._log.warning if isinstance(event, self.WARNINGS) else self._log.info
        log(type(event).__name__, **self.fields(event))

    @classmethod
    def fields(cls, event: Event) -> dict[str, Any]:
        data = event.model_dump(mode="json", exclude={"session_id"})
        return {key: cls._shorten(value) for key, value in data.items()}

    @classmethod
    def _shorten(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) > cls.TEXT_LIMIT:
            return f"{value[: cls.TEXT_LIMIT]}... (+{len(value) - cls.TEXT_LIMIT} chars)"

        if isinstance(value, dict):
            return {key: cls._shorten(item) for key, item in value.items()}

        if isinstance(value, list):
            return [cls._shorten(item) for item in value]

        return value
