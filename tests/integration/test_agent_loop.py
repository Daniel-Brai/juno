import sys

import pytest

from fakes.clock import FakeClock
from fakes.llm import Reply, ScriptedLlm, Turn
from fakes.probe import FakeProbe
from juno.bootstrap import Harness
from juno.config import Flags, HarnessConfig, SandboxSettings
from juno.domain.events import Event, SessionEnded, SessionStarted, ToolFinished
from juno.domain.session import State
from juno.domain.values import ModelReply, ModelRequest, Role, RunPaths, Task

pytestmark = pytest.mark.anyio

TRAIN_SCRIPT = """\
from pathlib import Path

Path("submission.csv").write_text("id,EAP\\n1,0.6\\n")
print("val log_loss 0.62")
"""


class Run:
    def __init__(
        self,
        turns: list[Turn],
        task: Task,
        paths: RunPaths,
        clock: FakeClock,
        probe: FakeProbe,
        flags: Flags | None = None,
    ) -> None:
        self.llm = ScriptedLlm(turns)
        self.events: list[Event] = []
        self.harness = Harness.build(
            config=HarnessConfig(sandbox=SandboxSettings(kind="local"), flags=flags or Flags()),
            task=task,
            paths=paths,
            llm=self.llm,
            clock=clock,
            probe=probe,
        )
        self.harness.bus.subscribe(Event, self.record)

    async def record(self, event: Event) -> None:
        self.events.append(event)

    def tool_results(self) -> list[dict[str, object]]:
        return [
            block
            for message in self.harness.session.messages
            if message.role is Role.USER and isinstance(message.content, list)
            for block in message.content
        ]


class TestScriptedSession:
    async def test_ten_turns_end_with_the_best_validation_submission(
        self, task: Task, paths: RunPaths, clock: FakeClock, probe: FakeProbe
    ) -> None:
        def finish(request: ModelRequest) -> ModelReply:
            clock.advance(7 * 24 * 3_600)
            return Reply.text("Out of ideas within budget.")

        turns: list[Turn] = [
            Reply.tools(("bash", {"command": f"ls {task.data_dir}"})),
            Reply.tools(
                ("editor", {"action": "create", "path": "train.py", "file_text": TRAIN_SCRIPT})
            ),
            Reply.tools(("bash", {"command": f"{sys.executable} train.py"})),
            Reply.tools(("submission_register", {"path": "submission.csv", "val_score": 0.62})),
            Reply.text("Baseline registered."),
            Reply.tools(
                (
                    "job_submit",
                    {
                        "command": "sleep 0.2 && printf 'id,EAP\\n1,0.7\\n' > v2.csv",
                        "name": "tfidf",
                    },
                )
            ),
            Reply.tools(("job_wait", {"job_id": "job-1", "timeout_s": 10})),
            Reply.tools(("system", {})),
            Reply.tools(
                ("submission_register", {"path": "v2.csv", "val_score": 0.48, "notes": "tfidf"})
            ),
            finish,
        ]
        run = Run(turns, task, paths, clock, probe)

        best = await run.harness.run()

        assert best is not None and best.val_score == 0.48 and best.notes == "tfidf"
        assert best.path.parent == paths.submissions
        assert run.harness.session.state is State.DONE
        assert (len(run.llm.requests), run.llm.remaining) == (10, 0)

        results = [block for block in run.tool_results() if block.get("type") == "tool_result"]
        assert len(results) == 8
        assert not any(block["is_error"] for block in results)
        assert "train.csv" in str(results[0]["content"])
        assert "val log_loss 0.62" in str(results[2]["content"])

        continues = [
            message.content
            for message in run.harness.session.messages
            if isinstance(message.content, str) and message.content.startswith("Continue working.")
        ]
        assert continues == ["Continue working. 1d remaining."]

        assert isinstance(run.events[0], SessionStarted)
        assert isinstance(run.events[-1], SessionEnded) and run.events[-1].state is State.DONE
        assert sum(isinstance(event, ToolFinished) for event in run.events) == 8

    async def test_every_request_shares_one_cacheable_prefix(
        self, task: Task, paths: RunPaths, clock: FakeClock, probe: FakeProbe
    ) -> None:
        def finish(request: ModelRequest) -> ModelReply:
            clock.advance(7 * 24 * 3_600)
            return Reply.text("done")

        run = Run(
            [Reply.tools(("system", {})), Reply.text("thinking"), finish], task, paths, clock, probe
        )

        await run.harness.run()

        first, *rest = run.llm.requests
        assert all(
            request.system == first.system and request.tools == first.tools for request in rest
        )
        assert all(
            later.messages[: len(earlier.messages)] == earlier.messages
            for earlier, later in zip(run.llm.requests, rest, strict=False)
        )


class TestEndings:
    async def test_a_refusal_fails_the_session(
        self, task: Task, paths: RunPaths, clock: FakeClock, probe: FakeProbe
    ) -> None:
        run = Run([Reply.refusal()], task, paths, clock, probe)

        assert await run.harness.run() is None
        assert run.harness.session.state is State.FAILED

    async def test_a_spent_budget_ends_without_calling_the_model(
        self, task: Task, paths: RunPaths, clock: FakeClock, probe: FakeProbe
    ) -> None:
        run = Run([], task, paths, clock, probe)
        clock.advance(run.harness.session.deadline - clock.now())

        await run.harness.run()

        assert run.harness.session.state is State.DONE
        assert run.llm.requests == []

    async def test_bad_tool_calls_come_back_as_errors_and_the_run_goes_on(
        self, task: Task, paths: RunPaths, clock: FakeClock, probe: FakeProbe
    ) -> None:
        def finish(request: ModelRequest) -> ModelReply:
            clock.advance(7 * 24 * 3_600)
            return Reply.text("done")

        run = Run(
            [Reply.tools(("editor", {"action": "explode", "path": "x"})), finish],
            task,
            paths,
            clock,
            probe,
        )

        await run.harness.run()

        [result] = [block for block in run.tool_results() if block.get("type") == "tool_result"]
        assert result["is_error"] is True
        assert str(result["content"]).startswith("invalid input: action:")
        assert run.harness.session.state is State.DONE


class TestFlags:
    @pytest.mark.parametrize(
        ("overrides", "missing"),
        [
            ({"tools.system": False}, {"system"}),
            (
                {"tools.jobs": False},
                {"job_submit", "job_wait", "job_queue", "job_info", "job_cancel"},
            ),
            ({"tools.submission": False}, {"submission_register", "submission_list"}),
        ],
    )
    def test_tool_flags_remove_their_tools(
        self,
        task: Task,
        paths: RunPaths,
        clock: FakeClock,
        probe: FakeProbe,
        overrides: dict[str, bool],
        missing: set[str],
    ) -> None:
        run = Run([], task, paths, clock, probe, flags=Flags().with_overrides(overrides))

        names = set(run.harness.registry.names)

        assert not names & missing
        assert {"bash", "editor"} <= names
