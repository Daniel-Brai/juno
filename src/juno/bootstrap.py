import asyncio
import contextlib
import os
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from juno.adapters.git import GitCheckpointer
from juno.adapters.llm.anthropic import AnthropicClient, AnthropicPrices
from juno.adapters.llm.openrouter import OpenRouterClient
from juno.adapters.sandboxes.local import LocalSandbox
from juno.adapters.sandboxes.podman import PodmanSandbox
from juno.adapters.stores.sqlite import SqliteStore
from juno.adapters.system import SystemClock, SystemProbe
from juno.config import HarnessConfig
from juno.domain.session import Session
from juno.domain.values import Job, RunPaths, Submission, Task, Usage
from juno.ports import Checkpointer, Clock, LlmClient, ResourceProbe, Sandbox
from juno.service.bus import MessageBus
from juno.service.context import RunContext
from juno.service.journal import Journal, SetupChanged
from juno.service.loop import AgentLoop
from juno.service.prompts import Prompts
from juno.service.recorders import CacheMonitor, CheckpointRecorder, EventRecorder, LogRecorder
from juno.service.watchdog import MemoryWatchdog
from juno.tools.bash import BashTools
from juno.tools.editor import EditorTools
from juno.tools.jobs import JobTools
from juno.tools.middleware import ContainErrors, EmitEvents, TruncateOutput
from juno.tools.registry import ToolRegistry
from juno.tools.submissions import SubmissionTools
from juno.tools.system import SystemTools
from juno.tools.workspace import Workspace


@dataclass(frozen=True, slots=True)
class Adapters:
    llm: LlmClient | None = None
    clock: Clock | None = None
    probe: ResourceProbe | None = None
    sandbox: Sandbox | None = None
    checkpointer: Checkpointer | None = None


@dataclass(frozen=True, slots=True)
class RunReport:
    session: Session
    best: Submission | None
    jobs: list[Job]
    checkpoints: int
    usage: Usage


@dataclass(frozen=True, slots=True)
class Harness:
    DATABASE = "run.db"

    session: Session
    loop: AgentLoop
    registry: ToolRegistry
    bus: MessageBus
    store: SqliteStore
    sandbox: Sandbox
    llm: LlmClient
    watchdog: MemoryWatchdog
    jobs: JobTools | None

    @classmethod
    async def start(
        cls, *, config: HarnessConfig, task: Task, paths: RunPaths, adapters: Adapters
    ) -> Harness:
        clock = adapters.clock or SystemClock()
        now = clock.now()
        session = Session(
            task_id=task.id,
            started_at=now,
            deadline=now + config.budget.total,
            cost_limit=config.budget.max_cost,
            setup=cls.setup_of(config),
        )
        store = await cls.open_store(paths)

        try:
            async with store.unit_of_work() as uow:
                existing = await uow.sessions.latest()

            if existing is not None:
                raise FileExistsError(
                    f"{paths.root} already holds run {existing.id}; resume it or use a new folder"
                )

            return await cls._assemble(config, task, paths, adapters, clock, store, session, [])
        except BaseException:
            await store.close()
            raise

    @classmethod
    async def resume(
        cls, *, config: HarnessConfig, task: Task, paths: RunPaths, adapters: Adapters
    ) -> Harness:
        store = await cls.open_store(paths)

        try:
            async with store.unit_of_work() as uow:
                session = await uow.sessions.latest()
                jobs = await uow.jobs.all(session.id) if session is not None else []

            if session is None:
                raise FileNotFoundError(f"no run to resume in {paths.root}")

            if changes := session.setup_changes(cls.setup_of(config)):
                raise SetupChanged(changes)

            harness = await cls._assemble(
                config, task, paths, adapters, adapters.clock or SystemClock(), store, session, jobs
            )
            await harness.loop.journal.restore(session)
            return harness
        except BaseException:
            await store.close()
            raise

    @classmethod
    async def open_store(cls, paths: RunPaths) -> SqliteStore:
        for path in paths.all():
            path.mkdir(parents=True, exist_ok=True)

        return await SqliteStore.open(paths.root / cls.DATABASE)

    @classmethod
    async def report(cls, paths: RunPaths) -> RunReport:
        if not (paths.root / cls.DATABASE).is_file():
            raise FileNotFoundError(f"{paths.root} is not a juno run folder")

        store = await SqliteStore.open(paths.root / cls.DATABASE)

        try:
            async with store.unit_of_work() as uow:
                session = await uow.sessions.latest()

                if session is None:
                    raise FileNotFoundError(f"no run in {paths.root}")

                records = await uow.messages.all(session.id)
                return RunReport(
                    session=session,
                    best=await uow.submissions.best(session.id),
                    jobs=await uow.jobs.all(session.id),
                    checkpoints=len(await uow.checkpoints.all(session.id)),
                    usage=sum((r.usage for r in records if r.usage is not None), Usage()),
                )
        finally:
            await store.close()

    @staticmethod
    def setup_of(config: HarnessConfig) -> dict[str, Any]:
        return config.model_dump(mode="json")

    @staticmethod
    def llm_for(config: HarnessConfig, session: Session) -> LlmClient:
        model = config.model

        if model.provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")

            if not key:
                raise RuntimeError("set OPENROUTER_API_KEY to use the openrouter provider")

            return OpenRouterClient(model, api_key=key, session_id=session.id)

        if config.budget.max_cost is not None and not AnthropicPrices.knows(model.name):
            raise ValueError(
                f"budget.max_cost needs a price for {model.name}, and juno has none; "
                "drop the cost cap or use a priced model"
            )

        return AnthropicClient(model, compaction=config.flags.compaction)

    @staticmethod
    def sandbox_for(
        config: HarnessConfig, paths: RunPaths, task: Task, session: Session
    ) -> Sandbox:
        if config.sandbox.kind == "local":
            return LocalSandbox(paths.workspace)

        return PodmanSandbox(
            config.sandbox,
            name=f"juno-{session.id[-12:]}",
            workspace=paths.workspace,
            logs=paths.logs,
            data=task.data_dir,
        )

    @classmethod
    async def stop(cls, *, task: Task, paths: RunPaths, adapters: Adapters) -> Session:
        report = await cls.report(paths)
        session = report.session
        config = HarnessConfig(**session.setup)
        sandbox = adapters.sandbox or cls.sandbox_for(config, paths, task, session)

        for job in report.jobs:
            if job.state == "running":
                with contextlib.suppress(Exception):
                    await sandbox.kill_group(job.pgid)

        await sandbox.stop()
        return session

    async def run(self) -> Submission | None:
        watcher = asyncio.create_task(self.watchdog.watch())

        try:
            best = await self.loop.run(self.session)
        finally:
            watcher.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await watcher

        if self.session.terminal:
            await self.sandbox.stop()

        return best

    async def close(self) -> None:
        await self.llm.close()
        await self.store.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    @classmethod
    async def _assemble(
        cls,
        config: HarnessConfig,
        task: Task,
        paths: RunPaths,
        adapters: Adapters,
        clock: Clock,
        store: SqliteStore,
        session: Session,
        stored_jobs: list[Job],
    ) -> Harness:
        units = store.unit_of_work
        sandbox = adapters.sandbox or cls.sandbox_for(config, paths, task, session)
        await sandbox.start()

        probe = adapters.probe or SystemProbe()
        workspace = Workspace(paths.workspace)
        bus = MessageBus()
        context = RunContext(session=session, clock=clock, units=units, bus=bus)
        registry = ToolRegistry(middleware=[EmitEvents(context), ContainErrors(), TruncateOutput()])
        registry.register(
            BashTools(sandbox, context, max_timeout_s=config.budget.bash_timeout),
            EditorTools(workspace),
        )

        jobs: JobTools | None = None
        flags = config.flags.tools

        if flags.jobs:
            jobs = JobTools(
                sandbox,
                context,
                logs_dir=paths.logs,
                max_running=config.budget.max_jobs,
                jobs=stored_jobs,
            )
            registry.register(jobs)

        if flags.system:
            registry.register(SystemTools(probe, context))

        if flags.submission:
            registry.register(
                SubmissionTools(
                    workspace, context, store_dir=paths.submissions, direction=task.direction
                )
            )

        seen_task = (
            task.model_copy(update={"data_dir": PodmanSandbox.DATA})
            if isinstance(sandbox, PodmanSandbox)
            else task
        )
        prompts = Prompts(seen_task, config.budget.total)
        journal = Journal(units, registry=registry, prompts=prompts, clock=clock)

        EventRecorder(units).attach(bus)
        LogRecorder().attach(bus)
        CacheMonitor(bus, clock).attach()
        CheckpointRecorder(
            adapters.checkpointer or GitCheckpointer(paths.workspace), context
        ).attach()

        llm = adapters.llm or cls.llm_for(config, session)
        watchdog = MemoryWatchdog(
            probe,
            jobs,
            context,
            limit=config.watchdog.memory_fraction,
            interval_s=config.watchdog.interval,
        )
        loop = AgentLoop(llm, journal, clock=clock, bus=bus, prompts=prompts, jobs=jobs)

        return cls(
            session=session,
            loop=loop,
            registry=registry,
            bus=bus,
            store=store,
            sandbox=sandbox,
            llm=llm,
            watchdog=watchdog,
            jobs=jobs,
        )
