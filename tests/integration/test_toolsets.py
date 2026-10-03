import pytest

from fakes.clock import FakeClock
from fakes.probe import FakeProbe
from juno.adapters.local_sandbox import LocalSandbox
from juno.adapters.memory import InMemorySubmissionRepo
from juno.domain.commands import (
    CancelJob,
    InspectJob,
    ListJobs,
    ListSubmissions,
    QuerySystem,
    RegisterSubmission,
    RunBash,
    SubmitJob,
    WaitJob,
)
from juno.domain.events import Event, SubmissionRegistered
from juno.domain.values import Direction, RunPaths
from juno.service.bus import MessageBus
from juno.tools.bash import BashTools
from juno.tools.jobs import JobTools
from juno.tools.registry import ToolError
from juno.tools.submissions import SubmissionTools
from juno.tools.system import SystemTools
from juno.tools.workspace import Workspace

pytestmark = pytest.mark.anyio


@pytest.fixture
def sandbox(paths: RunPaths) -> LocalSandbox:
    return LocalSandbox(paths.workspace)


class TestBashTools:
    async def test_reports_exit_code_and_output(
        self, sandbox: LocalSandbox, clock: FakeClock
    ) -> None:
        bash = BashTools(sandbox, clock, deadline=clock.now() + 60, max_timeout_s=10)

        output = await bash.run(RunBash(command="echo ready"))

        assert output.startswith("exit code 0")
        assert output.endswith("ready\n")

    async def test_refuses_to_run_past_the_deadline(
        self, sandbox: LocalSandbox, clock: FakeClock
    ) -> None:
        bash = BashTools(sandbox, clock, deadline=clock.now(), max_timeout_s=10)

        with pytest.raises(ToolError, match="budget is spent"):
            await bash.run(RunBash(command="echo late"))


class TestJobTools:
    @pytest.fixture
    def jobs(
        self, sandbox: LocalSandbox, workspace: Workspace, paths: RunPaths, clock: FakeClock
    ) -> JobTools:
        return JobTools(
            sandbox, workspace, paths.logs, clock, deadline=clock.now() + 600, max_running=1
        )

    async def test_submit_then_wait_shows_the_log(self, jobs: JobTools) -> None:
        started = await jobs.submit(SubmitJob(command="echo epoch 1; echo epoch 2", name="train"))
        finished = await jobs.wait(WaitJob(job_id="job-1", timeout_s=10))

        assert started.startswith("started job-1 (train)")
        assert finished.splitlines()[0].startswith("job-1 (train): succeeded, exit code 0")
        assert finished.splitlines()[-1] == "epoch 2"

    async def test_wait_returns_while_the_job_still_runs(self, jobs: JobTools) -> None:
        await jobs.submit(SubmitJob(command="sleep 30", name="slow"))

        output = await jobs.wait(WaitJob(job_id="job-1", timeout_s=0.3))

        assert output.startswith("job-1 still running")
        await jobs.cancel_all()

    async def test_running_jobs_are_capped(self, jobs: JobTools) -> None:
        await jobs.submit(SubmitJob(command="sleep 30", name="first"))

        with pytest.raises(ToolError, match="already running"):
            await jobs.submit(SubmitJob(command="sleep 30", name="second"))

        await jobs.cancel_all()

    async def test_cancel_stops_the_job(self, jobs: JobTools) -> None:
        await jobs.submit(SubmitJob(command="sleep 30", name="slow"))

        assert await jobs.cancel(CancelJob(job_id="job-1")) == "cancelled job-1"
        assert "cancelled" in await jobs.queue(ListJobs())
        assert jobs.running() == []

    async def test_info_and_unknown_jobs(self, jobs: JobTools) -> None:
        await jobs.submit(SubmitJob(command="echo done", name="quick"))
        await jobs.wait(WaitJob(job_id="job-1", timeout_s=10))

        assert "command: echo done" in await jobs.info(InspectJob(job_id="job-1"))

        with pytest.raises(ToolError, match="no job"):
            await jobs.info(InspectJob(job_id="job-9"))


class TestSubmissionTools:
    @pytest.fixture
    def bus(self) -> MessageBus:
        return MessageBus()

    @pytest.fixture
    def tools(
        self, workspace: Workspace, paths: RunPaths, clock: FakeClock, bus: MessageBus
    ) -> SubmissionTools:
        repo = InMemorySubmissionRepo()
        return SubmissionTools(
            repo, workspace, paths.submissions, "s1", Direction.MINIMIZE, clock, bus
        )

    async def test_registers_copies_and_tracks_the_best(
        self, tools: SubmissionTools, workspace: Workspace, paths: RunPaths, bus: MessageBus
    ) -> None:
        seen: list[Event] = []

        async def record(event: Event) -> None:
            seen.append(event)

        bus.subscribe(SubmissionRegistered, record)
        (workspace.root / "a.csv").write_text("id,p\n1,0.5\n")
        (workspace.root / "b.csv").write_text("id,p\n1,0.4\n")

        await tools.register(RegisterSubmission(path="a.csv", val_score=0.7))
        second = await tools.register(
            RegisterSubmission(path="b.csv", val_score=0.5, notes="tfidf")
        )

        assert second == "registered submission 2 with validation score 0.5; best so far 0.5"
        assert len(list(paths.submissions.iterdir())) == 2
        assert len(seen) == 2
        assert await tools.list(ListSubmissions()) == "1. 0.7\n2. 0.5 (best) tfidf"

    async def test_the_same_file_is_registered_once(
        self, tools: SubmissionTools, workspace: Workspace
    ) -> None:
        (workspace.root / "a.csv").write_text("id,p\n1,0.5\n")

        await tools.register(RegisterSubmission(path="a.csv", val_score=0.7))
        again = await tools.register(RegisterSubmission(path="a.csv", val_score=0.1))

        assert again == "a.csv is already registered with validation score 0.7"

    @pytest.mark.parametrize(
        ("path", "message"), [("missing.csv", "not a file"), ("../x.csv", "outside")]
    )
    async def test_bad_paths_are_refused(
        self, tools: SubmissionTools, path: str, message: str
    ) -> None:
        with pytest.raises(ToolError, match=message):
            await tools.register(RegisterSubmission(path=path, val_score=0.1))


class TestSystemTools:
    async def test_reports_resources_and_time(self, probe: FakeProbe, clock: FakeClock) -> None:
        system = SystemTools(probe, clock, started_at=clock.now(), deadline=clock.now() + 7_200)
        clock.advance(1_800)

        output = await system.run(QuerySystem())

        assert output.splitlines() == [
            "CPU: 10%",
            "RAM: 8.0 of 64.0 GiB (12%)",
            "GPU 0: 0% busy, 0.0 of 80.0 GiB VRAM",
            "Elapsed: 30m",
            "Remaining: 1h 30m",
        ]
