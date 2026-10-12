from pathlib import Path

import pytest

from fakes.clock import FakeClock
from fakes.probe import FakeProbe
from juno.domain.commands import CancelJob
from juno.domain.events import CacheMissed, Event, ModelCalled, ResourcePressure, ToolFinished
from juno.domain.values import Job, ToolResult, Usage
from juno.service.bus import MessageBus
from juno.service.context import RunContext
from juno.service.recorders import CacheMonitor, LogRecorder
from juno.service.watchdog import MemoryWatchdog

pytestmark = pytest.mark.anyio


class Seen:
    def __init__(self, bus: MessageBus) -> None:
        self.events: list[Event] = []
        bus.subscribe(Event, self.record)

    async def record(self, event: Event) -> None:
        self.events.append(event)


class Jobs:
    def __init__(self, largest: Job | None) -> None:
        self.largest = largest
        self.cancelled: list[str] = []

    async def largest_running(self) -> Job | None:
        return self.largest

    async def cancel(self, args: CancelJob) -> str:
        self.cancelled.append(args.job_id)
        return f"cancelled {args.job_id}"


JOB = Job(
    id="job-1",
    name="train",
    command="python train.py",
    pid=1,
    pgid=1,
    log_path=Path("a"),
    exit_path=Path("b"),
)


class TestMemoryWatchdog:
    def watchdog(self, probe: FakeProbe, jobs: Jobs | None, context: RunContext) -> MemoryWatchdog:
        return MemoryWatchdog(probe, jobs, context, limit=0.95, interval_s=0.01)

    async def test_does_nothing_below_the_limit(
        self, probe: FakeProbe, context: RunContext
    ) -> None:
        jobs = Jobs(JOB)
        seen = Seen(context.bus)

        assert await self.watchdog(probe, jobs, context).check() is None
        assert jobs.cancelled == [] and seen.events == []

    async def test_stops_the_largest_job_when_memory_runs_out(
        self, probe: FakeProbe, context: RunContext
    ) -> None:
        probe.near_oom(0.97)
        jobs = Jobs(JOB)
        seen = Seen(context.bus)

        victim = await self.watchdog(probe, jobs, context).check()

        assert victim == JOB
        assert jobs.cancelled == ["job-1"]
        [event] = seen.events
        assert isinstance(event, ResourcePressure)
        assert event.reason == "RAM at 97%; stopped job-1 (train)"

    async def test_reports_pressure_even_with_no_job_to_stop(
        self, probe: FakeProbe, context: RunContext
    ) -> None:
        probe.near_oom(0.99)
        seen = Seen(context.bus)

        await self.watchdog(probe, None, context).check()

        assert [type(event) for event in seen.events] == [ResourcePressure]


class TestCacheMonitor:
    def call(self, index: int, cached: int, prompt: int = 5_000) -> ModelCalled:
        return ModelCalled(
            session_id="s1",
            at=0.0,
            message_index=index,
            usage=Usage(input_tokens=prompt, cache_read_input_tokens=cached),
            duration_s=1.0,
        )

    async def test_warns_once_after_two_misses_in_a_row(self) -> None:
        bus = MessageBus()
        seen = Seen(bus)
        CacheMonitor(bus, FakeClock()).attach()

        for index, cached in enumerate([0, 0, 0, 0]):
            await bus.publish(self.call(index, cached))

        missed = [event for event in seen.events if isinstance(event, CacheMissed)]
        assert [(event.message_index, event.misses) for event in missed] == [(2, 2)]

    async def test_a_hit_resets_the_count(self) -> None:
        bus = MessageBus()
        seen = Seen(bus)
        CacheMonitor(bus, FakeClock()).attach()

        for index, cached in enumerate([0, 0, 500, 0]):
            await bus.publish(self.call(index, cached))

        assert not any(isinstance(event, CacheMissed) for event in seen.events)

    async def test_prompts_too_small_to_cache_are_not_misses(self) -> None:
        bus = MessageBus()
        seen = Seen(bus)
        CacheMonitor(bus, FakeClock()).attach()

        for index in range(5):
            await bus.publish(self.call(index, cached=0, prompt=1_900))

        assert not any(isinstance(event, CacheMissed) for event in seen.events)


class TestLogRecorder:
    def test_long_text_is_shortened(self) -> None:
        event = ToolFinished(
            session_id="s1", at=1.0, result=ToolResult(call_id="c", content="x" * 1_000)
        )

        fields = LogRecorder.fields(event)

        assert "session_id" not in fields
        assert fields["result"]["content"] == "x" * 200 + "... (+800 chars)"
