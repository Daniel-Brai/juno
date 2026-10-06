from collections.abc import Sequence
from typing import Any

from fakes.clock import FakeClock
from fakes.llm import ScriptedLlm, Turn
from fakes.probe import FakeProbe
from juno.bootstrap import Adapters, Harness
from juno.config import BudgetSettings, Flags, HarnessConfig, ModelSettings, SandboxSettings
from juno.domain.events import Event
from juno.domain.values import ContentBlock, Role, RunPaths, Task
from juno.ports import Sandbox


class ScriptedRun:
    def __init__(self, harness: Harness, llm: ScriptedLlm) -> None:
        self.harness = harness
        self.llm = llm
        self.events: list[Event] = []
        harness.bus.subscribe(Event, self.record)

    async def record(self, event: Event) -> None:
        self.events.append(event)

    def tool_results(self) -> list[ContentBlock]:
        return [
            block
            for message in self.harness.session.messages
            if message.role is Role.USER and isinstance(message.content, list)
            for block in message.content
            if block.get("type") == "tool_result"
        ]

    def transcript(self) -> list[dict[str, Any]]:
        return [message.model_dump() for message in self.harness.session.messages]


class RunFactory:
    def __init__(self, task: Task, paths: RunPaths, clock: FakeClock, probe: FakeProbe) -> None:
        self.task = task
        self.paths = paths
        self.clock = clock
        self.probe = probe
        self._open: list[Harness] = []

    async def start(self, turns: Sequence[Turn], **options: Any) -> ScriptedRun:
        return await self._open_run(Harness.start, turns, **options)

    async def resume(self, turns: Sequence[Turn], **options: Any) -> ScriptedRun:
        return await self._open_run(Harness.resume, turns, **options)

    async def close_all(self) -> None:
        for harness in self._open:
            await harness.close()

        self._open.clear()

    async def close(self, run: ScriptedRun) -> None:
        await run.harness.close()
        self._open.remove(run.harness)

    async def _open_run(
        self,
        opener: Any,
        turns: Sequence[Turn],
        flags: Flags | None = None,
        model: ModelSettings | None = None,
        budget: BudgetSettings | None = None,
        paths: RunPaths | None = None,
        clock: FakeClock | None = None,
        probe: FakeProbe | None = None,
        sandbox: Sandbox | None = None,
    ) -> ScriptedRun:
        llm = ScriptedLlm(turns)
        harness: Harness = await opener(
            config=HarnessConfig(
                sandbox=SandboxSettings(kind="local"),
                flags=flags or Flags(),
                model=model or ModelSettings(),
                budget=budget or BudgetSettings(),
            ),
            task=self.task,
            paths=paths or self.paths,
            adapters=Adapters(
                llm=llm,
                clock=clock or self.clock,
                probe=probe or self.probe,
                sandbox=sandbox,
            ),
        )
        self._open.append(harness)
        return ScriptedRun(harness, llm)
