import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from juno.eval.experiment import Experiment, RunPlan
from juno.eval.grading import Grader, GradingError, RunGrade
from juno.eval.runs import RunFolder


class Launcher(Protocol):
    async def start(self, plan: RunPlan, gpus: tuple[int, ...]) -> int: ...

    async def resume(self, plan: RunPlan) -> int: ...


class JunoProcess:
    LOG: ClassVar[str] = "runner.log"
    STOP_WAIT_S: ClassVar[float] = 10.0

    def __init__(self, executable: Sequence[str] = (sys.executable, "-m", "juno")) -> None:
        self._executable = list(executable)

    def start_argv(self, plan: RunPlan, gpus: tuple[int, ...]) -> list[str]:
        argv = [
            *self._executable,
            "run",
            str(plan.task.path),
            "--runs",
            str(plan.folder.parent),
            "--name",
            plan.folder.name,
            "--budget",
            f"{plan.budget}s",
        ]

        for key, value in plan.flags.items():
            argv += ["--flag", f"{key}={str(value).lower()}"]

        settings = dict(plan.settings)

        if plan.max_cost is not None:
            settings.setdefault("budget.max_cost", plan.max_cost)

        if gpus:
            settings["sandbox.gpus"] = list(gpus)

        for key, value in settings.items():
            argv += ["--set", f"{key}={json.dumps(value)}"]

        return argv

    def resume_argv(self, plan: RunPlan) -> list[str]:
        return [*self._executable, "resume", str(plan.folder)]

    async def start(self, plan: RunPlan, gpus: tuple[int, ...]) -> int:
        return await self._spawn(plan, self.start_argv(plan, gpus))

    async def resume(self, plan: RunPlan) -> int:
        return await self._spawn(plan, self.resume_argv(plan))

    async def _spawn(self, plan: RunPlan, argv: list[str]) -> int:
        plan.folder.mkdir(parents=True, exist_ok=True)

        with (plan.folder / self.LOG).open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                process_group=0,
            )

            try:
                return await process.wait()
            except asyncio.CancelledError:
                await self._stop(process)
                raise

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, sig)

            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.STOP_WAIT_S):
                    await process.wait()
                    return


class GpuPool:
    def __init__(self, gpus: Sequence[int]) -> None:
        self._free = set(gpus)
        self._changed = asyncio.Condition()

    async def take(self, count: int) -> tuple[int, ...]:
        async with self._changed:
            await self._changed.wait_for(lambda: len(self._free) >= count)
            chosen = tuple(sorted(self._free)[:count])
            self._free -= set(chosen)
            return chosen

    async def take_exact(self, gpus: Sequence[int]) -> tuple[int, ...]:
        async with self._changed:
            await self._changed.wait_for(lambda: self._free.issuperset(gpus))
            self._free -= set(gpus)
            return tuple(gpus)

    async def give(self, gpus: Sequence[int]) -> None:
        async with self._changed:
            self._free |= set(gpus)
            self._changed.notify_all()


class RunOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan: RunPlan
    state: str | None
    attempts: int
    exit_codes: tuple[int, ...] = ()
    grade: RunGrade | None = None
    grade_error: str | None = None
    stopped: bool = False


class EvalRunner:
    MANIFEST: ClassVar[str] = "experiment.toml"

    def __init__(
        self,
        experiment: Experiment,
        launcher: Launcher,
        *,
        grader: Grader | None = None,
        spend_poll_s: float = 10.0,
    ) -> None:
        self._experiment = experiment
        self._launcher = launcher
        self._grader = grader or Grader()
        self._spend_poll_s = spend_poll_s
        self._slots = asyncio.Semaphore(experiment.experiment.parallel)
        self._gpus = GpuPool(experiment.experiment.gpus)
        self._stop = asyncio.Event()
        self._plans = experiment.plans()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def spent(self) -> float:
        total = 0.0

        for plan in self._plans:
            folder = RunFolder(plan.folder)
            session = folder.session()

            if session is not None:
                total += folder.usage(session.id).cost_usd

        return total

    async def run(self, manifest: Path | None = None) -> list[RunOutcome]:
        self._experiment.folder.mkdir(parents=True, exist_ok=True)

        if manifest is not None:
            shutil.copyfile(manifest, self._experiment.folder / self.MANIFEST)

        await self._check_spend()

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self.execute(plan)) for plan in self._plans]
            group.create_task(self._watch_spend(tasks))

        return [task.result() for task in tasks]

    async def execute(self, plan: RunPlan) -> RunOutcome:
        folder = RunFolder(plan.folder)
        session = folder.session()

        if session is not None and session.terminal:
            return self.graded(plan, session.state, attempts=0)

        codes: list[int] = []

        async with self._slots:
            gpus = await self._take_gpus(session.setup if session else None)

            try:
                while len(codes) <= self._experiment.experiment.retries:
                    await self._check_spend()

                    if self.stopped:
                        break

                    launch = (
                        self._launcher.start(plan, gpus)
                        if session is None
                        else self._launcher.resume(plan)
                    )
                    code = await self._unless_stopped(launch)
                    session = folder.session()

                    if code is None:
                        break

                    codes.append(code)

                    if session is not None and session.terminal:
                        break
            finally:
                await self._gpus.give(gpus)

        state = session.state if session is not None else None
        terminal = session is not None and session.terminal
        outcome = self.graded(plan, state, attempts=len(codes))
        return outcome.model_copy(
            update={"exit_codes": tuple(codes), "stopped": self.stopped and not terminal}
        )

    def graded(self, plan: RunPlan, state: str | None, *, attempts: int) -> RunOutcome:
        outcome = RunOutcome(plan=plan, state=state, attempts=attempts)

        if state is None:
            return outcome

        folder = RunFolder(plan.folder)

        try:
            grade = self._grader.grade(folder, plan.task.answers)
        except GradingError as error:
            return outcome.model_copy(update={"grade_error": str(error)})

        self._grader.save(folder, grade)
        return outcome.model_copy(update={"grade": grade})

    async def _unless_stopped(self, launch: Coroutine[Any, Any, int]) -> int | None:
        running = asyncio.ensure_future(launch)
        stopping = asyncio.ensure_future(self._stop.wait())

        try:
            await asyncio.wait({running, stopping}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopping.cancel()

        if running.done():
            return running.result()

        running.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await running

        return None

    async def _check_spend(self) -> None:
        limit = self._experiment.experiment.max_total_cost

        if limit is not None and not self.stopped and await asyncio.to_thread(self.spent) >= limit:
            self._stop.set()

    async def _watch_spend(self, tasks: Sequence[asyncio.Task[RunOutcome]]) -> None:
        if self._experiment.experiment.max_total_cost is None:
            return

        while not self.stopped and not all(task.done() for task in tasks):
            await self._check_spend()
            await asyncio.wait(tasks, timeout=self._spend_poll_s)

    async def _take_gpus(self, setup: dict[str, Any] | None) -> tuple[int, ...]:
        if not self._experiment.experiment.gpus_per_run:
            return ()

        if setup is not None:
            return await self._gpus.take_exact(setup.get("sandbox", {}).get("gpus", ()))

        return await self._gpus.take(self._experiment.experiment.gpus_per_run)
