from juno.domain.events import (
    CheckpointTaken,
    Event,
    SessionEnded,
    SubmissionRegistered,
    TurnContinued,
)
from juno.domain.session import Session
from juno.domain.values import Checkpoint
from juno.ports import Checkpointer, Clock, UnitOfWorkFactory
from juno.service.bus import MessageBus


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

    def __init__(
        self,
        checkpointer: Checkpointer,
        units: UnitOfWorkFactory,
        bus: MessageBus,
        clock: Clock,
        session: Session,
    ) -> None:
        self._checkpointer = checkpointer
        self._units = units
        self._bus = bus
        self._clock = clock
        self._session = session

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
