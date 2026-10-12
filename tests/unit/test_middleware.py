import pytest

from fakes.clock import FakeClock
from juno.domain.events import Event, ToolFinished, ToolStarted
from juno.domain.values import ToolCall, ToolResult
from juno.service.bus import MessageBus
from juno.service.context import RunContext
from juno.tools.middleware import ContainErrors, EmitEvents, TruncateOutput

pytestmark = pytest.mark.anyio

CALL = ToolCall(id="toolu_1", name="bash", args={"command": "ls"})


class Fixed:
    def __init__(self, content: str, clock: FakeClock | None = None, takes_s: float = 0) -> None:
        self._content = content
        self._clock = clock
        self._takes_s = takes_s

    async def __call__(self, call: ToolCall) -> ToolResult:
        if self._clock is not None:
            self._clock.advance(self._takes_s)

        return ToolResult(call_id=call.id, content=self._content)


class TestTruncateOutput:
    async def test_short_output_is_untouched(self) -> None:
        result = await TruncateOutput(limit=10)(CALL, Fixed("0123456789"))

        assert (result.content, result.truncated) == ("0123456789", False)

    async def test_long_output_keeps_head_and_tail(self) -> None:
        result = await TruncateOutput(limit=10)(CALL, Fixed("abcde" + "x" * 50 + "vwxyz"))

        assert result.truncated
        assert result.content == "abcde\n[... 50 characters omitted ...]\nvwxyz"


class TestContainErrors:
    async def test_exceptions_become_error_results(self) -> None:
        async def explode(call: ToolCall) -> ToolResult:
            raise OSError("disk full")

        result = await ContainErrors()(CALL, explode)

        assert result.is_error
        assert result.content == "tool failed: OSError: disk full"


class TestEmitEvents:
    async def test_publishes_start_and_finish_with_duration(
        self, context: RunContext, clock: FakeClock, bus: MessageBus
    ) -> None:
        seen: list[Event] = []

        async def record(event: Event) -> None:
            seen.append(event)

        bus.subscribe(Event, record)

        result = await EmitEvents(context)(CALL, Fixed("ok", clock, takes_s=12))

        assert [type(event) for event in seen] == [ToolStarted, ToolFinished]
        assert result.duration_s == 12
        assert seen[1] == ToolFinished(session_id=context.session.id, at=clock.now(), result=result)
