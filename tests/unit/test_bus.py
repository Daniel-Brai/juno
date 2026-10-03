import pytest

from juno.domain.events import Event, TurnContinued
from juno.service.bus import MessageBus

pytestmark = pytest.mark.anyio

EVENT = TurnContinued(session_id="s1", at=0.0, remaining_s=60)


class Collector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)


class TestMessageBus:
    async def test_delivers_to_exact_and_base_subscribers(self) -> None:
        bus, exact, everything = MessageBus(), Collector(), Collector()
        bus.subscribe(TurnContinued, exact)
        bus.subscribe(Event, everything)

        await bus.publish(EVENT)

        assert exact.events == [EVENT]
        assert everything.events == [EVENT]

    async def test_a_failing_handler_does_not_stop_the_others(self) -> None:
        bus, after = MessageBus(), Collector()

        async def broken(event: TurnContinued) -> None:
            raise RuntimeError("subscriber bug")

        bus.subscribe(TurnContinued, broken)
        bus.subscribe(TurnContinued, after)

        await bus.publish(EVENT)

        assert after.events == [EVENT]
        assert bus.failures == 1
