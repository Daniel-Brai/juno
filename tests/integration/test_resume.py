import asyncio
import os
import signal
import time
from pathlib import Path

import pytest

from fakes.clock import FakeClock
from fakes.crash import Crash, CrashingProbe, CrashingSandbox
from fakes.llm import Reply, Turn
from fakes.run import RunFactory, ScriptedRun
from juno.adapters.stores.sqlite import SqliteStore
from juno.config import BudgetSettings, Flags, ModelSettings
from juno.domain.events import SessionResumed
from juno.domain.session import State
from juno.domain.values import ModelReply, ModelRequest, RunPaths
from juno.service.journal import Journal, ReplayDivergence, SetupChanged

pytestmark = pytest.mark.anyio


class Script:
    TURNS = (
        Reply.tools(("system", {})),
        Reply.tools(("editor", {"action": "create", "path": "notes.md", "file_text": "plan\n"})),
        Reply.tools(
            ("editor", {"action": "create", "path": "sub.csv", "file_text": "id,p\n1,0.5\n"})
        ),
        Reply.tools(("submission_register", {"path": "sub.csv", "val_score": 0.5})),
        Reply.text("Baseline in place."),
        Reply.tools(("editor", {"action": "view", "path": "notes.md"})),
    )

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def turns(self) -> list[Turn]:
        return [*self.TURNS, self.finish]

    def finish(self, request: ModelRequest) -> ModelReply:
        self._clock.advance(7 * 24 * 3_600)
        return Reply.text("done")


class Lab:
    def __init__(self, runs: RunFactory, tmp_path: Path) -> None:
        self._runs = runs
        self._tmp_path = tmp_path

    def paths(self, name: str) -> RunPaths:
        return RunPaths(root=self._tmp_path / name)

    async def uninterrupted(self) -> ScriptedRun:
        clock = FakeClock(start=1_000.0)
        run = await self._runs.start(Script(clock).turns(), paths=self.paths("clean"), clock=clock)
        await run.harness.run()
        return run

    async def crashed(self, name: str, crash_at: int, clock: FakeClock | None = None) -> RunPaths:
        clock = clock or FakeClock(start=1_000.0)
        paths = self.paths(name)
        first = await self._runs.start(Script(clock).turns(), paths=paths, clock=clock)
        first.llm.fail_at(crash_at, Crash("killed"))

        with pytest.raises(Crash):
            await first.harness.run()

        await self._runs.close(first)
        return paths

    async def crash_then_resume(self, crash_at: int) -> ScriptedRun:
        clock = FakeClock(start=1_000.0)
        paths = await self.crashed(f"crash-{crash_at}", crash_at, clock)
        resumed = await self._runs.resume(
            Script(clock).turns()[crash_at:], paths=paths, clock=clock
        )
        await resumed.harness.run()
        return resumed


@pytest.fixture
def lab(runs: RunFactory, tmp_path: Path) -> Lab:
    return Lab(runs, tmp_path)


class StoreRows:
    async def rewrite_first_message(self, paths: RunPaths, content: str) -> None:
        store = await SqliteStore.open(paths.root / "run.db")

        async with store.unit_of_work() as uow:
            session = await uow.sessions.latest()
            assert session is not None
            [first, *_] = await uow.messages.all(session.id)
            changed = first.model_copy(
                update={"message": first.message.model_copy(update={"content": content})}
            )
            await store.connection.execute(
                "UPDATE messages SET data_json = ? WHERE session_id = ? AND idx = 0",
                (changed.model_dump_json(), session.id),
            )
            await uow.commit()

        await store.close()


@pytest.fixture
def store_rows() -> StoreRows:
    return StoreRows()


class TestDowntime:
    async def test_time_lost_to_a_crash_does_not_count(
        self, runs: RunFactory, lab: Lab, clock: FakeClock
    ) -> None:
        paths = lab.paths("downtime")
        turns: list[Turn] = [
            Reply.tools(("system", {})),
            Reply.tools(("system", {})),
            Script(clock).finish,
        ]
        first = await runs.start(turns, paths=paths)
        deadline = first.harness.session.deadline
        first.llm.fail_at(1, Crash("killed"))

        with pytest.raises(Crash):
            await first.harness.run()

        await runs.close(first)
        clock.advance(2 * 3_600)
        resumed = await runs.resume(turns[1:], paths=paths)
        await resumed.harness.run()

        session = resumed.harness.session
        assert session.paused_s == 7_200
        assert session.deadline == deadline + 7_200

        [resumed_event] = [e for e in resumed.events if isinstance(e, SessionResumed)]
        assert resumed_event.downtime_s == 7_200

        before, after = [str(block["content"]) for block in resumed.tool_results()]
        assert (
            before.splitlines()[-2:] == after.splitlines()[-2:] == ["Elapsed: 0s", "Remaining: 1d"]
        )


class TestKillAndResume:
    @pytest.mark.parametrize("crash_at", range(len(Script.TURNS) + 1))
    async def test_resuming_after_a_crash_matches_an_uninterrupted_run(
        self, lab: Lab, crash_at: int
    ) -> None:
        clean = await lab.uninterrupted()
        resumed = await lab.crash_then_resume(crash_at)

        assert resumed.transcript() == clean.transcript()
        assert resumed.harness.session.state is State.DONE
        assert resumed.llm.remaining == 0

        best = await resumed.harness.loop.run(resumed.harness.session)
        assert best is not None and best.val_score == 0.5

    async def test_paid_model_calls_are_never_repeated(self, lab: Lab) -> None:
        resumed = await lab.crash_then_resume(crash_at=4)

        assert len(resumed.llm.requests) == len(Script.TURNS) + 1 - 4


class TestInterruptedTools:
    async def test_an_idempotent_tool_is_run_again(
        self, runs: RunFactory, lab: Lab, clock: FakeClock
    ) -> None:
        clean = await lab.uninterrupted()
        paths = lab.paths("probe-crash")
        script = Script(clock).turns()
        first = await runs.start(script, paths=paths, probe=CrashingProbe())

        with pytest.raises(Crash):
            await first.harness.run()

        await runs.close(first)
        resumed = await runs.resume(script[1:], paths=paths)
        await resumed.harness.run()

        assert resumed.transcript() == clean.transcript()

    async def test_a_non_idempotent_tool_is_reported_as_interrupted(
        self, runs: RunFactory, lab: Lab, clock: FakeClock
    ) -> None:
        paths = lab.paths("bash-crash")
        script: list[Turn] = [Reply.tools(("bash", {"command": "echo hi"})), Script(clock).finish]
        first = await runs.start(script, paths=paths, sandbox=CrashingSandbox(paths.workspace))

        with pytest.raises(Crash):
            await first.harness.run()

        await runs.close(first)
        resumed = await runs.resume(script[1:], paths=paths)
        await resumed.harness.run()

        [result] = resumed.tool_results()
        assert result == {
            "type": "tool_result",
            "tool_use_id": result["tool_use_id"],
            "content": Journal.INTERRUPTED,
            "is_error": True,
        }
        assert resumed.harness.session.state is State.DONE


class TestRefusedResumes:
    @pytest.mark.parametrize(
        ("options", "changed"),
        [
            ({"flags": Flags().with_overrides({"tools.jobs": False})}, ["flags.tools.jobs"]),
            ({"model": ModelSettings(name="claude-sonnet-5-5")}, ["model.name"]),
            ({"model": ModelSettings(effort="max")}, ["model.effort"]),
            ({"budget": BudgetSettings(bash_timeout=60)}, ["budget.bash_timeout"]),
        ],
        ids=["flag", "model", "effort", "budget"],
    )
    async def test_changed_settings_are_named(
        self, runs: RunFactory, lab: Lab, options: dict[str, object], changed: list[str]
    ) -> None:
        paths = await lab.crashed("changed", crash_at=2)

        with pytest.raises(SetupChanged) as refused:
            await runs.resume([], paths=paths, **options)

        assert refused.value.changes == changed

    async def test_a_tampered_history_is_refused(
        self, runs: RunFactory, lab: Lab, store_rows: StoreRows
    ) -> None:
        paths = await lab.crashed("tampered", crash_at=3)
        await store_rows.rewrite_first_message(paths, "Solve a different task.")

        with pytest.raises(ReplayDivergence) as refused:
            await runs.resume([], paths=paths)

        assert refused.value.index == 1

    async def test_nothing_to_resume_is_an_error(self, runs: RunFactory, lab: Lab) -> None:
        with pytest.raises(FileNotFoundError, match="no run to resume"):
            await runs.resume([], paths=lab.paths("empty"))

    async def test_a_run_folder_holds_one_run(self, runs: RunFactory, lab: Lab) -> None:
        clean = await lab.uninterrupted()
        await runs.close(clean)

        with pytest.raises(FileExistsError, match="resume it or use a new folder"):
            await runs.start([], paths=lab.paths("clean"))

    async def test_a_finished_run_resumes_as_finished(self, lab: Lab, runs: RunFactory) -> None:
        clean = await lab.uninterrupted()
        await runs.close(clean)

        again = await runs.resume([], paths=lab.paths("clean"))
        best = await again.harness.run()

        assert again.harness.session.state is State.DONE
        assert again.llm.requests == []
        assert best is not None and best.val_score == 0.5


class TestJobsAcrossRestarts:
    async def run_until_crash(self, runs: RunFactory, paths: RunPaths, command: str) -> ScriptedRun:
        first = await runs.start(
            [Reply.tools(("job_submit", {"command": command, "name": "train"}))], paths=paths
        )

        with pytest.raises(Exception, match="no scripted reply"):
            await first.harness.run()

        await runs.close(first)
        return first

    async def test_a_job_that_outlives_the_harness_is_picked_up(
        self, runs: RunFactory, lab: Lab, clock: FakeClock
    ) -> None:
        paths = lab.paths("job-survives")
        await self.run_until_crash(runs, paths, "sleep 0.3; echo finished")
        script: list[Turn] = [
            Reply.tools(("job_wait", {"job_id": "job-1", "timeout_s": 10})),
            Script(clock).finish,
        ]

        resumed = await runs.resume(script, paths=paths)
        await resumed.harness.run()

        result = resumed.tool_results()[-1]
        assert str(result["content"]).startswith("job-1 (train): succeeded, exit code 0")
        assert "finished" in str(result["content"])

    async def test_a_job_that_died_with_the_machine_is_lost(
        self, runs: RunFactory, lab: Lab, clock: FakeClock
    ) -> None:
        paths = lab.paths("job-lost")
        await self.run_until_crash(runs, paths, "sleep 30")
        log = paths.logs / "job-1.log"
        assert log.exists()

        resumed = await runs.resume(
            [Reply.tools(("job_queue", {})), Script(clock).finish], paths=paths
        )

        async with resumed.harness.store.unit_of_work() as uow:
            [job] = await uow.jobs.all(resumed.harness.session.id)

        os.killpg(job.pgid, signal.SIGKILL)
        await resumed.harness.run()

        result = resumed.tool_results()[-1]
        assert "job-1 (train): lost" in str(result["content"])


class TestDowntimeWithJobs:
    async def crash_with_job(
        self, runs: RunFactory, lab: Lab, name: str, command: str, clock: FakeClock
    ) -> RunPaths:
        paths = lab.paths(name)
        first = await runs.start(
            [Reply.tools(("job_submit", {"command": command, "name": "train"}))],
            paths=paths,
            clock=clock,
        )

        with pytest.raises(Exception, match="no scripted reply"):
            await first.harness.run()

        await runs.close(first)
        return paths

    async def test_time_a_job_used_during_downtime_still_counts(
        self, runs: RunFactory, lab: Lab
    ) -> None:
        clock = FakeClock(start=time.time())
        paths = await self.crash_with_job(runs, lab, "job-finished", "sleep 0.3", clock)
        await asyncio.sleep(0.6)
        clock.advance(3_600)
        finish = Script(clock).finish

        resumed = await runs.resume([finish], paths=paths, clock=clock)
        await resumed.harness.run()

        [event] = [e for e in resumed.events if isinstance(e, SessionResumed)]
        assert event.downtime_s == pytest.approx(3_600, abs=1)
        assert 0.2 < event.jobs_busy_s < 1.5
        assert event.paused_s == pytest.approx(event.downtime_s - event.jobs_busy_s)

    async def test_a_job_busy_for_the_whole_downtime_pauses_nothing(
        self, runs: RunFactory, lab: Lab
    ) -> None:
        clock = FakeClock(start=time.time())
        paths = await self.crash_with_job(runs, lab, "job-running", "sleep 30", clock)
        clock.advance(3_600)

        resumed = await runs.resume([Script(clock).finish], paths=paths, clock=clock)
        deadline = resumed.harness.session.deadline
        await resumed.harness.run()

        [event] = [e for e in resumed.events if isinstance(e, SessionResumed)]
        assert event.paused_s == 0
        assert resumed.harness.session.paused_s == 0
        assert resumed.harness.session.deadline == deadline
