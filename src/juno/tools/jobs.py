import asyncio
from pathlib import Path
from typing import Any, ClassVar

from juno.config import Durations
from juno.domain.commands import CancelJob, InspectJob, ListJobs, SubmitJob, WaitJob
from juno.domain.values import Job, JobState
from juno.ports import Clock, Sandbox
from juno.tools.registry import Tool, ToolError
from juno.tools.workspace import Workspace


class JobTools:
    POLL_S: ClassVar[float] = 0.2
    TAIL_BYTES: ClassVar[int] = 65_536

    def __init__(
        self,
        sandbox: Sandbox,
        workspace: Workspace,
        logs_dir: Path,
        clock: Clock,
        deadline: float,
        max_running: int,
    ) -> None:
        self._sandbox = sandbox
        self._workspace = workspace
        self._logs_dir = logs_dir
        self._clock = clock
        self._deadline = deadline
        self._max_running = max_running
        self._jobs: dict[str, Job] = {}
        self._names: dict[str, str] = {}
        self._started: dict[str, float] = {}

    def tools(self) -> list[Tool[Any]]:
        return [
            Tool(
                name="job_submit",
                description="Start a shell command in the background, e.g. a training run. Output goes to a log file.",
                input_model=SubmitJob,
                handler=self.submit,
            ),
            Tool(
                name="job_wait",
                description="Wait until a job finishes or the timeout passes, then show its log tail.",
                input_model=WaitJob,
                handler=self.wait,
                idempotent=True,
            ),
            Tool(
                name="job_queue",
                description="List every job with its state and runtime.",
                input_model=ListJobs,
                handler=self.queue,
                idempotent=True,
            ),
            Tool(
                name="job_info",
                description="Show one job's state, exit code, runtime and the end of its log.",
                input_model=InspectJob,
                handler=self.info,
                idempotent=True,
            ),
            Tool(
                name="job_cancel",
                description="Stop a job and every process it started.",
                input_model=CancelJob,
                handler=self.cancel,
                idempotent=True,
            ),
        ]

    async def submit(self, args: SubmitJob) -> str:
        if len(self.running()) >= self._max_running:
            raise ToolError(
                f"{self._max_running} jobs are already running; wait for or cancel one first"
            )

        job_id = f"job-{len(self._jobs) + 1}"
        spawned = await self._sandbox.spawn(
            args.command, self._logs_dir / f"{job_id}.log", self._logs_dir / f"{job_id}.exit"
        )
        self._jobs[job_id] = spawned.model_copy(update={"id": job_id})
        self._names[job_id] = args.name
        self._started[job_id] = self._clock.now()
        return f"started {job_id} ({args.name}), log at {spawned.log_path}"

    async def wait(self, args: WaitJob) -> str:
        job = self._refresh(self._get(args.job_id))
        timeout = min(args.timeout_s, max(self._deadline - self._clock.now(), 0))

        try:
            async with asyncio.timeout(timeout):
                while job.state is JobState.RUNNING:
                    await asyncio.sleep(self.POLL_S)
                    job = self._refresh(job)
        except TimeoutError:
            return f"{job.id} still running after {Durations.format(timeout)}\n{await self._tail(job, 20)}"

        return f"{self._describe(job)}\n{await self._tail(job, 20)}"

    async def queue(self, _: ListJobs) -> str:
        jobs = [self._refresh(job) for job in list(self._jobs.values())]
        return "\n".join(self._describe(job) for job in jobs) or "no jobs yet"

    async def info(self, args: InspectJob) -> str:
        job = self._refresh(self._get(args.job_id))
        return f"{self._describe(job)}\ncommand: {job.command}\nlog: {job.log_path}\n{await self._tail(job, args.tail_lines)}"

    async def cancel(self, args: CancelJob) -> str:
        job = self._refresh(self._get(args.job_id))

        if job.state is not JobState.RUNNING:
            return f"{job.id} already {job.state}"

        await self._sandbox.kill_group(job.pgid)
        self._jobs[job.id] = job.model_copy(update={"state": JobState.CANCELLED})
        return f"cancelled {job.id}"

    async def cancel_all(self) -> None:
        for job in self.running():
            await self.cancel(CancelJob(job_id=job.id))

    def running(self) -> list[Job]:
        return [
            job
            for job in map(self._refresh, list(self._jobs.values()))
            if job.state is JobState.RUNNING
        ]

    def _get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)

        if job is None:
            raise ToolError(f"no job {job_id!r}; job_queue lists them")

        return job

    def _refresh(self, job: Job) -> Job:
        if job.state is not JobState.RUNNING or (code := self._sandbox.poll(job)) is None:
            return job

        state = JobState.SUCCEEDED if code == 0 else JobState.FAILED
        self._jobs[job.id] = job.model_copy(update={"state": state, "exit_code": code})
        return self._jobs[job.id]

    def _describe(self, job: Job) -> str:
        runtime = Durations.format(self._clock.now() - self._started[job.id])
        exit_code = f", exit code {job.exit_code}" if job.exit_code is not None else ""
        return f"{job.id} ({self._names[job.id]}): {job.state}{exit_code}, started {runtime} ago"

    async def _tail(self, job: Job, lines: int) -> str:
        return await asyncio.to_thread(self._read_tail, job.log_path, lines) if lines else ""

    def _read_tail(self, path: Path, lines: int) -> str:
        if not path.exists():
            return ""

        with path.open("rb") as log:
            size = log.seek(0, 2)
            log.seek(max(size - self.TAIL_BYTES, 0))
            text = log.read().decode(errors="replace")

        return "\n".join(text.splitlines()[-lines:])
