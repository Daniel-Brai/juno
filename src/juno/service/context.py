from dataclasses import dataclass

from juno.domain.session import Session
from juno.ports import Clock, TransactionFactory
from juno.service.bus import MessageBus


@dataclass(frozen=True, slots=True)
class RunContext:
    session: Session
    clock: Clock
    transactions: TransactionFactory
    bus: MessageBus
