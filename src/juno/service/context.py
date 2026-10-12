from dataclasses import dataclass

from juno.domain.session import Session
from juno.ports import Clock, UnitOfWorkFactory
from juno.service.bus import MessageBus


@dataclass(frozen=True, slots=True)
class RunContext:
    session: Session
    clock: Clock
    units: UnitOfWorkFactory
    bus: MessageBus
