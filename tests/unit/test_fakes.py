import pytest

from fakes.clock import FakeClock
from fakes.llm import Reply, ScriptedLlm, ScriptExhausted
from fakes.probe import FakeProbe
from juno.domain.values import ModelReply, ModelRequest

pytestmark = pytest.mark.anyio


class TestScriptedLlm:
    async def test_replies_in_order_and_records_requests(self, model_request: ModelRequest) -> None:
        llm = ScriptedLlm([Reply.text("one"), Reply.text("two")])

        first = await llm.complete(model_request)
        second = await llm.complete(model_request)

        assert [first.content[0]["text"], second.content[0]["text"]] == ["one", "two"]
        assert llm.requests == [model_request, model_request]
        assert llm.remaining == 0

    async def test_raises_loudly_when_the_script_runs_out(
        self, model_request: ModelRequest
    ) -> None:
        llm = ScriptedLlm([])

        with pytest.raises(ScriptExhausted):
            await llm.complete(model_request)

    async def test_injected_fault_does_not_consume_a_turn(
        self, model_request: ModelRequest
    ) -> None:
        llm = ScriptedLlm([Reply.text("after retry")]).fail_at(0, ConnectionError("dropped"))

        with pytest.raises(ConnectionError):
            await llm.complete(model_request)

        reply = await llm.complete(model_request)

        assert reply.content[0]["text"] == "after retry"

    async def test_callable_turns_see_the_request(self, model_request: ModelRequest) -> None:
        def echo(request: ModelRequest) -> ModelReply:
            return Reply.text(request.system)

        reply = await ScriptedLlm([echo]).complete(model_request)

        assert reply.content[0]["text"] == model_request.system

    async def test_rejects_tools_the_request_did_not_offer(
        self, model_request: ModelRequest
    ) -> None:
        llm = ScriptedLlm([Reply.tools(("web_fetch", {"url": "https://example.com"}))])

        with pytest.raises(AssertionError, match="web_fetch"):
            await llm.complete(model_request)


class TestFakeClock:
    async def test_sleep_advances_time_without_waiting(self, clock: FakeClock) -> None:
        start = clock.now()

        await clock.sleep(3_600)

        assert clock.now() == start + 3_600
        assert clock.sleeps == [3_600]

    def test_never_goes_backwards(self, clock: FakeClock) -> None:
        with pytest.raises(ValueError):
            clock.advance(-1)


class TestFakeProbe:
    def test_near_oom_raises_memory_use(self, probe: FakeProbe) -> None:
        probe.near_oom(0.97)

        assert probe.sample().ram_used_fraction == pytest.approx(0.97, abs=1e-6)
        assert probe.samples == 1


class TestSystemProbe:
    def test_reports_memory_and_no_gpus_without_a_driver(self) -> None:
        from juno.adapters.system import SystemProbe

        probe = SystemProbe()
        snapshot = probe.sample()

        assert snapshot.ram_total_bytes > 0
        assert probe.has_gpus or snapshot.gpus == ()
