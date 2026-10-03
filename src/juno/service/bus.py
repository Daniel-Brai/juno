import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from juno.domain.events import Event

type Handler[E: Event] = Callable[[E], Awaitable[None]]

logger = logging.getLogger(__name__)


class MessageBus:
    def __init__(self) -> None:
        self.failures = 0
        self._handlers: defaultdict[type[Event], list[Handler[Any]]] = defaultdict(list)

    def subscribe[E: Event](self, event_type: type[E], handler: Handler[E]) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        for event_type in type(event).__mro__:
            for handler in self._handlers.get(event_type, ()):
                try:
                    await handler(event)
                except Exception:
                    self.failures += 1
                    logger.exception("event handler failed for %s", type(event).__name__)
