import asyncio
import subprocess
import sys

import pytest

from fakes.clock import FakeClock
from fakes.llm import Reply, Turn
from fakes.run import RunFactory
from juno.config import BudgetSettings, Flags
from juno.domain.events import (
    BudgetSpent,
    Event,
    SessionEnded,
    SessionStarted,
    ToolFinished,
    ToolStarted,
)
from juno.domain.session import Limit, State
from juno.domain.values import ModelReply, ModelRequest, RunPaths, StopReason, Task, Usage
from juno.ports import ModelRejected
from juno.service.loop import AgentLoop

pytestmark = pytest.mark.anyio

TRAIN_SCRIPT = """\
from pathlib import Path

Path("submission.csv").write_text("id,EAP\\n1,0.6\\n")
print("val log_loss 0.62")
"""


class Finish:
    def __init__(self, clock: FakeClock, text: str = "done") -> None:
        self._clock = clock
        self._text = text

    def __call__(self, request: ModelRequest) -> ModelReply:
        self._clock.advance(7 * 24 * 3_600)
        return Reply.text(self._text)


class TestScriptedSession:
    async def test_ten_turns_end_with_the_best_validation_submission(
        self, runs: RunFactory, task: Task, paths: RunPaths, clock: FakeClock
    ) -> None:
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
            Finish(clock, "Out of ideas within budget."),
        ]
        run = await runs.start(turns)

        best = await run.harness.run()

        assert best is not None and best.val_score == 0.48 and best.notes == "tfidf"
        assert best.path.parent == paths.submissions
        assert run.harness.session.state is State.DONE
        assert (len(run.llm.requests), run.llm.remaining) == (10, 0)

        results = run.tool_results()
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
        assert isinstance(run.events[-1], SessionEnded)
        assert run.events[-1].state is State.DONE and run.events[-1].handler_failures == 0
        assert sum(isinstance(event, ToolFinished) for event in run.events) == 8

    async def test_the_run_is_recorded_in_the_database_and_git(
        self, runs: RunFactory, paths: RunPaths, clock: FakeClock
    ) -> None:
        turns: list[Turn] = [
            Reply.tools(
                ("editor", {"action": "create", "path": "sub.csv", "file_text": "id,p\n1,0.5\n"})
            ),
            Reply.tools(("submission_register", {"path": "sub.csv", "val_score": 0.5})),
            Finish(clock),
        ]
        run = await runs.start(turns)

        await run.harness.run()

        async with run.harness.store.unit_of_work() as uow:
            session_id = run.harness.session.id
            records = await uow.messages.all(session_id)
            checkpoints = await uow.checkpoints.all(session_id)
            event_types = await uow.events.types(session_id)
            stored = await uow.sessions.get(session_id)

        assert [record.message for record in records] == run.harness.session.messages
        assert all(r.request_hash for r in records if r.message.role == "assistant")
        assert stored is not None and stored.state is State.DONE
        assert event_types[0] == "SessionStarted" and "SessionEnded" in event_types
        assert "CheckpointTaken" in event_types

        completed = await asyncio.to_thread(
            subprocess.run,
            ["git", "log", "--format=%H"],
            cwd=paths.workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        log = completed.stdout.split()
        assert {checkpoint.commit for checkpoint in checkpoints} <= set(log)
        assert checkpoints

    async def test_every_request_shares_one_cacheable_prefix(
        self, runs: RunFactory, clock: FakeClock
    ) -> None:
        run = await runs.start([Reply.tools(("system", {})), Reply.text("thinking"), Finish(clock)])

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
    async def test_a_refusal_fails_the_session(self, runs: RunFactory) -> None:
        run = await runs.start([Reply.refusal()])

        assert await run.harness.run() is None
        assert run.harness.session.state is State.FAILED

    async def test_a_spent_budget_ends_without_calling_the_model(
        self, runs: RunFactory, clock: FakeClock
    ) -> None:
        run = await runs.start([])
        clock.advance(run.harness.session.deadline - clock.now())

        await run.harness.run()

        assert run.harness.session.state is State.DONE
        assert run.llm.requests == []

    async def test_a_rejected_request_fails_the_run_with_its_reason(self, runs: RunFactory) -> None:
        run = await runs.start([Reply.text("never sent")])
        run.llm.fail_at(
            0, ModelRejected("anthropic rejected the request with 401: invalid x-api-key")
        )

        assert await run.harness.run() is None

        session = run.harness.session
        assert session.state is State.FAILED
        assert session.failure == "anthropic rejected the request with 401: invalid x-api-key"
        [ended] = [event for event in run.events if isinstance(event, SessionEnded)]
        assert ended.failure == session.failure

    async def test_reaching_the_cost_cap_wraps_up_like_the_deadline(self, runs: RunFactory) -> None:
        spend = Usage(input_tokens=1_000, cost_usd=0.03)
        turns: list[Turn] = [Reply.text("one", usage=spend), Reply.text("two", usage=spend)]
        run = await runs.start(
            [*turns, Reply.text("never sent")], budget=BudgetSettings(max_cost=0.05)
        )

        await run.harness.run()

        [spent] = [event for event in run.events if isinstance(event, BudgetSpent)]
        assert spent.limit is Limit.COST
        assert spent.cost_usd == pytest.approx(0.06)
        assert run.harness.session.state is State.DONE
        assert (len(run.llm.requests), run.llm.remaining) == (2, 1)

    async def test_bad_tool_calls_come_back_as_errors_and_the_run_goes_on(
        self, runs: RunFactory, clock: FakeClock
    ) -> None:
        run = await runs.start(
            [Reply.tools(("editor", {"action": "explode", "path": "x"})), Finish(clock)]
        )

        await run.harness.run()

        [result] = run.tool_results()
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
    async def test_tool_flags_remove_their_tools(
        self, runs: RunFactory, overrides: dict[str, bool], missing: set[str]
    ) -> None:
        run = await runs.start([], flags=Flags().with_overrides(overrides))

        names = set(run.harness.registry.names)

        assert not names & missing
        assert {"bash", "editor"} <= names


class TestParallelTools:
    async def test_read_only_calls_in_one_turn_run_together(
        self, runs: RunFactory, clock: FakeClock
    ) -> None:
        turns: list[Turn] = [
            Reply.tools(("job_submit", {"command": "sleep 30", "name": "a"})),
            Reply.tools(("job_submit", {"command": "sleep 30", "name": "b"})),
            Reply.tools(
                ("job_wait", {"job_id": "job-1", "timeout_s": 1.0}),
                ("job_wait", {"job_id": "job-2", "timeout_s": 1.0}),
            ),
            Finish(clock),
        ]
        run = await runs.start(turns)
        loop = asyncio.get_running_loop()
        timings: list[float] = []

        async def time_waits(event: Event) -> None:
            if isinstance(event, ToolStarted | ToolFinished):
                timings.append(loop.time())

        run.harness.bus.subscribe(Event, time_waits)

        await run.harness.run()

        waits = run.tool_results()[2:4]
        assert all("still running" in str(block["content"]) for block in waits)
        assert timings[-1] - timings[-4] < 1.6

    async def test_a_batch_with_a_writing_tool_runs_in_order(
        self, runs: RunFactory, paths: RunPaths, clock: FakeClock
    ) -> None:
        run = await runs.start(
            [
                Reply.tools(
                    ("editor", {"action": "create", "path": "a.txt", "file_text": "one\n"}),
                    ("editor", {"action": "view", "path": "a.txt"}),
                ),
                Finish(clock),
            ]
        )

        await run.harness.run()

        created, viewed = run.tool_results()
        assert created["content"] == "created a.txt"
        assert viewed["content"] == "     1\tone"


class TestTruncatedReplies:
    async def test_tool_calls_cut_off_by_the_output_limit_do_not_run(
        self, runs: RunFactory, paths: RunPaths, clock: FakeClock
    ) -> None:
        cut = Reply.tools(
            ("editor", {"action": "create", "path": "half.py", "file_text": "print('par"})
        ).model_copy(update={"stop_reason": StopReason.MAX_TOKENS})
        run = await runs.start([cut, Finish(clock)])

        await run.harness.run()

        [result] = run.tool_results()
        assert result["is_error"] is True
        assert result["content"] == AgentLoop.CUT_OFF
        assert not (paths.workspace / "half.py").exists()
