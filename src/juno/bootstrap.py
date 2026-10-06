from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from juno.adapters.git import GitCheckpointer
from juno.adapters.local_sandbox import LocalSandbox
from juno.adapters.sqlite_store import SqliteStore
from juno.adapters.system import PsutilProbe, SystemClock
from juno.config import HarnessConfig
from juno.domain.session import Session
from juno.domain.values import Job, RunPaths, Submission, Task
from juno.ports import Checkpointer, Clock, LlmClient, ResourceProbe, Sandbox
from juno.service.bus import MessageBus
from juno.service.journal import Journal, SetupChanged
from juno.service.loop import AgentLoop
from juno.service.prompts import Prompts
from juno.service.recorders import CheckpointRecorder, EventRecorder
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
    llm: LlmClient
    clock: Clock | None = None
    probe: ResourceProbe | None = None
    sandbox: Sandbox | None = None
    checkpointer: Checkpointer | None = None


@dataclass(frozen=True, slots=True)
class Harness:
    DATABASE = "run.db"

    session: Session
    loop: AgentLoop
    registry: ToolRegistry
    bus: MessageBus
    store: SqliteStore
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

    @staticmethod
    def setup_of(config: HarnessConfig) -> dict[str, Any]:
        return config.model_dump(mode="json")

    async def run(self) -> Submission | None:
        return await self.loop.run(self.session)

    async def close(self) -> None:
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
        sandbox = adapters.sandbox or cls.sandbox_for(config, paths)
        workspace = Workspace(paths.workspace)
        bus = MessageBus()
        registry = ToolRegistry(
            middleware=[EmitEvents(bus, clock, session.id), ContainErrors(), TruncateOutput()]
        )
        registry.register(
            BashTools(sandbox, clock, session, config.budget.bash_timeout),
            EditorTools(workspace),
        )

        jobs: JobTools | None = None
        flags = config.flags.tools

        if flags.jobs:
            jobs = JobTools(
                sandbox, units, session, paths.logs, clock, config.budget.max_jobs, stored_jobs
            )
            registry.register(jobs)

        if flags.system:
            registry.register(SystemTools(adapters.probe or PsutilProbe(), clock, session))

        if flags.submission:
            registry.register(
                SubmissionTools(
                    units, workspace, paths.submissions, session.id, task.direction, clock, bus
                )
            )

        prompts = Prompts(task, config.budget.total)
        journal = Journal(units, registry, prompts, clock)
        EventRecorder(units).attach(bus)
        CheckpointRecorder(
            adapters.checkpointer or GitCheckpointer(paths.workspace), units, bus, clock, session
        ).attach()

        loop = AgentLoop(adapters.llm, journal, clock, bus, prompts, jobs)
        return cls(session=session, loop=loop, registry=registry, bus=bus, store=store, jobs=jobs)

    @staticmethod
    def sandbox_for(config: HarnessConfig, paths: RunPaths) -> Sandbox:
        if config.sandbox.kind == "local":
            return LocalSandbox(paths.workspace)

        raise NotImplementedError(
            "the podman sandbox arrives in phase 3; set sandbox.kind = 'local'"
        )
