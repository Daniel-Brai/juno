import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from juno.config import Durations
from juno.domain.commands import CancelJob, InspectJob, ListJobs, SubmitJob, WaitJob
from juno.domain.session import Session
from juno.domain.values import Job, JobState
from juno.ports import Clock, Sandbox, UnitOfWorkFactory
from juno.tools.registry import Tool, ToolError


class JobTools:
    POLL_S: ClassVar[float] = 0.2
    TAIL_BYTES: ClassVar[int] = 65_536

    def __init__(
        self,
        sandbox: Sandbox,
        units: UnitOfWorkFactory,
        session: Session,
        logs_dir: Path,
        clock: Clock,
        max_running: int,
        jobs: Sequence[Job] = (),
    ) -> None:
        self._sandbox = sandbox
        self._units = units
        self._session = session
        self._logs_dir = logs_dir
        self._clock = clock
        self._max_running = max_running
        self._jobs = {job.id: job for job in jobs}

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
                read_only=True,
            ),
            Tool(
                name="job_queue",
                description="List every job with its state and runtime.",
                input_model=ListJobs,
                handler=self.queue,
                read_only=True,
            ),
            Tool(
                name="job_info",
                description="Show one job's state, exit code, runtime and the end of its log.",
                input_model=InspectJob,
                handler=self.info,
                read_only=True,
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
        if len(await self.running()) >= self._max_running:
            raise ToolError(
                f"{self._max_running} jobs are already running; wait for or cancel one first"
            )

        job_id = f"job-{len(self._jobs) + 1}"
        spawned = await self._sandbox.spawn(
            args.command, self._logs_dir / f"{job_id}.log", self._logs_dir / f"{job_id}.exit"
        )
        job = spawned.model_copy(
            update={"id": job_id, "name": args.name, "started_at": self._clock.now()}
        )
        await self._save(job)
        return f"started {job.id} ({job.name}), log at {job.log_path}"

    async def wait(self, args: WaitJob) -> str:
        job = await self._refresh(self._get(args.job_id))
        timeout = min(args.timeout_s, self._session.remaining(self._clock.now()))

        try:
            async with asyncio.timeout(timeout):
                while job.state is JobState.RUNNING:
                    await asyncio.sleep(self.POLL_S)
                    job = await self._refresh(job)
        except TimeoutError:
            return f"{job.id} still running after {Durations.format(timeout)}\n{await self._tail(job, 20)}"

        return f"{self._describe(job)}\n{await self._tail(job, 20)}"

    async def queue(self, _: ListJobs) -> str:
        jobs = [await self._refresh(job) for job in list(self._jobs.values())]
        return "\n".join(self._describe(job) for job in jobs) or "no jobs yet"

    async def info(self, args: InspectJob) -> str:
        job = await self._refresh(self._get(args.job_id))
        return (
            f"{self._describe(job)}\ncommand: {job.command}\nlog: {job.log_path}\n"
            f"{await self._tail(job, args.tail_lines)}"
        )

    async def cancel(self, args: CancelJob) -> str:
        job = await self._refresh(self._get(args.job_id))

        if job.state is not JobState.RUNNING:
            return f"{job.id} already {job.state}"

        await self._sandbox.kill_group(job.pgid)
        await self._save(job.model_copy(update={"state": JobState.CANCELLED}))
        return f"cancelled {job.id}"

    async def cancel_all(self) -> None:
        for job in await self.running():
            await self.cancel(CancelJob(job_id=job.id))

    async def running(self) -> list[Job]:
        jobs = [await self._refresh(job) for job in list(self._jobs.values())]
        return [job for job in jobs if job.state is JobState.RUNNING]

    async def largest_running(self) -> Job | None:
        running = await self.running()
        return max(running, key=self._sandbox.memory_bytes, default=None)

    def busy_seconds(self, start: float, end: float) -> float:
        spans = sorted(
            (max(start, job.started_at), min(end, self._last_seen(job, end)))
            for job in self._jobs.values()
            if job.state is JobState.RUNNING
        )
        busy, reached = 0.0, start

        for begin, finish in spans:
            begin = max(begin, reached)

            if finish > begin:
                busy += finish - begin

            reached = max(reached, finish)

        return busy

    def _last_seen(self, job: Job, now: float) -> float:
        if job.exit_path.exists():
            return job.exit_path.stat().st_mtime

        if self._sandbox.is_alive(job):
            return now

        return job.log_path.stat().st_mtime if job.log_path.exists() else job.started_at

    def _get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)

        if job is None:
            raise ToolError(f"no job {job_id!r}; job_queue lists them")

        return job

    async def _refresh(self, job: Job) -> Job:
        if job.state is not JobState.RUNNING:
            return job

        code = self._sandbox.poll(job)

        if code is None and not self._sandbox.is_alive(job):
            code = self._sandbox.poll(job)

            if code is None:
                return await self._save(job.model_copy(update={"state": JobState.LOST}))

        if code is None:
            return job

        state = JobState.SUCCEEDED if code == 0 else JobState.FAILED
        return await self._save(job.model_copy(update={"state": state, "exit_code": code}))

    async def _save(self, job: Job) -> Job:
        async with self._units() as uow:
            await uow.jobs.save(self._session.id, job)
            await uow.commit()

        self._jobs[job.id] = job
        return job

    def _describe(self, job: Job) -> str:
        runtime = Durations.format(self._clock.now() - job.started_at)
        exit_code = f", exit code {job.exit_code}" if job.exit_code is not None else ""
        return f"{job.id} ({job.name}): {job.state}{exit_code}, started {runtime} ago"

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
