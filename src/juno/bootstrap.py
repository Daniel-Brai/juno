import asyncio
import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Protocol, Self

from juno.adapters.git import GitCheckpointer
from juno.adapters.llm.anthropic import AnthropicClient, AnthropicPrices
from juno.adapters.llm.openrouter import OpenRouterClient
from juno.adapters.sandboxes.local import LocalSandbox
from juno.adapters.sandboxes.podman import PodmanSandbox
from juno.adapters.stores.sqlite import SqliteStore
from juno.adapters.system import SystemClock, SystemProbe
from juno.config import HarnessConfig
from juno.domain.events import Event, SessionStarted, TurnContinued
from juno.domain.session import Session
from juno.domain.values import Job, RunPaths, Submission, Task, Usage
from juno.ports import Approver, Checkpointer, Clock, LlmClient, ResourceProbe, Sandbox
from juno.service.bus import MessageBus
from juno.service.context import RunContext
from juno.service.drivers import ContinueDriver, TurnDriver
from juno.service.journal import Journal, SetupChanged
from juno.service.loop import AgentLoop
from juno.service.prompts import ChatPrompts, Prompts, TaskPrompts
from juno.service.recorders import CacheMonitor, CheckpointRecorder, EventRecorder, LogRecorder
from juno.service.watchdog import MemoryWatchdog
from juno.tools.approvals import Approvals
from juno.tools.bash import BashTools
from juno.tools.editor import EditorTools
from juno.tools.jobs import JobTools
from juno.tools.middleware import ContainErrors, EmitEvents, TruncateOutput
from juno.tools.permissions import Permissions
from juno.tools.registry import Middleware, ToolRegistry, Toolset
from juno.tools.search import SearchTools
from juno.tools.submissions import SubmissionTools
from juno.tools.system import SystemTools
from juno.tools.todo import TodoTools
from juno.tools.workspace import Workspace


@dataclass(frozen=True, slots=True)
class Adapters:
    llm: LlmClient | None = None
    clock: Clock | None = None
    probe: ResourceProbe | None = None
    sandbox: Sandbox | None = None
    checkpointer: Checkpointer | None = None


@dataclass(frozen=True, slots=True)
class ChatSetup:
    driver: TurnDriver
    approver: Approver
    permissions: Permissions
    instructions: str | None = None
    log: bool = False


@dataclass(frozen=True, slots=True)
class Opened:
    config: HarnessConfig
    paths: RunPaths
    adapters: Adapters
    clock: Clock
    store: SqliteStore
    session: Session
    jobs: list[Job] = field(default_factory=list)


class Profile(Protocol):
    task: Task | None
    todo: TodoTools | None
    chat: bool
    log: bool

    def middleware(
        self, context: RunContext, read_only: Callable[[str], bool], /
    ) -> list[Middleware]: ...

    def toolsets(
        self, opened: Opened, context: RunContext, workspace: Workspace, /
    ) -> list[Toolset]: ...

    def prompts(self, opened: Opened, sandbox: Sandbox, /) -> Prompts: ...

    def driver(self, clock: Clock, /) -> TurnDriver: ...

    def checkpointing(self, paths: RunPaths, /) -> tuple[Checkpointer, tuple[type[Event], ...]]: ...


class TaskProfile:
    def __init__(self, task: Task) -> None:
        self.task: Task | None = task
        self.todo: TodoTools | None = None
        self.chat = False
        self.log = True
        self._task = task

    def middleware(
        self, _context: RunContext, _read_only: Callable[[str], bool], /
    ) -> list[Middleware]:
        return []

    def toolsets(
        self, opened: Opened, context: RunContext, workspace: Workspace, /
    ) -> list[Toolset]:
        if not opened.config.flags.tools.submission:
            return []

        submissions = SubmissionTools(
            workspace, context, store_dir=opened.paths.submissions, direction=self._task.direction
        )
        return [submissions]

    def prompts(self, opened: Opened, sandbox: Sandbox, /) -> Prompts:
        seen = (
            self._task.model_copy(update={"data_dir": PodmanSandbox.DATA})
            if isinstance(sandbox, PodmanSandbox)
            else self._task
        )
        return TaskPrompts(seen, opened.config.budget.total)

    def driver(self, clock: Clock, /) -> TurnDriver:
        return ContinueDriver(clock)

    def checkpointing(self, paths: RunPaths, /) -> tuple[Checkpointer, tuple[type[Event], ...]]:
        return GitCheckpointer(paths.workspace), CheckpointRecorder.TRIGGERS


class ChatProfile:
    CHECKPOINTS: ClassVar[str] = "checkpoints.git"

    def __init__(self, setup: ChatSetup) -> None:
        self.task: Task | None = None
        self.todo: TodoTools | None = TodoTools()
        self.chat = True
        self.log = setup.log
        self._setup = setup

    def middleware(
        self, context: RunContext, read_only: Callable[[str], bool], /
    ) -> list[Middleware]:
        approvals = Approvals(
            self._setup.permissions, self._setup.approver, context, read_only=read_only
        )
        return [approvals]

    def toolsets(
        self, _opened: Opened, _context: RunContext, workspace: Workspace, /
    ) -> list[Toolset]:
        return [SearchTools(workspace), *([self.todo] if self.todo else [])]

    def prompts(self, opened: Opened, _sandbox: Sandbox, /) -> Prompts:
        return ChatPrompts(str(opened.paths.workspace), self._setup.instructions)

    def driver(self, _clock: Clock, /) -> TurnDriver:
        return self._setup.driver

    def checkpointing(self, paths: RunPaths, /) -> tuple[Checkpointer, tuple[type[Event], ...]]:
        shadow = GitCheckpointer(paths.workspace, git_dir=paths.root / self.CHECKPOINTS)
        return shadow, (SessionStarted, TurnContinued)


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
    todo: TodoTools | None = None
    checkpointer: Checkpointer | None = None
    chat: bool = False

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
            async with store.transaction() as tx:
                existing = await tx.sessions.latest()

            if existing is not None:
                raise FileExistsError(
                    f"{paths.root} already holds run {existing.id}; resume it or use a new folder"
                )

            opened = Opened(config, paths, adapters, clock, store, session)
            return await cls._assemble(opened, TaskProfile(task))
        except BaseException:
            await store.close()
            raise

    @classmethod
    async def start_chat(
        cls, *, config: HarnessConfig, chat: ChatSetup, paths: RunPaths, adapters: Adapters
    ) -> Harness:
        clock = adapters.clock or SystemClock()
        project = cls._project(paths)
        session = Session(
            id=paths.root.name,
            task_id=project.name,
            started_at=clock.now(),
            cost_limit=config.budget.max_cost,
            setup={
                **cls.setup_of(config),
                "chat": {"project": str(project), "instructions": chat.instructions},
            },
        )
        store = await cls.open_store(paths)

        try:
            opened = Opened(config, paths, adapters, clock, store, session)
            return await cls._assemble(opened, ChatProfile(chat))
        except BaseException:
            await store.close()
            raise

    @classmethod
    async def resume_chat(
        cls, *, config: HarnessConfig, chat: ChatSetup, paths: RunPaths, adapters: Adapters
    ) -> Harness:
        store = await cls.open_store(paths)

        try:
            async with store.transaction() as tx:
                session = await tx.sessions.latest()
                jobs = await tx.jobs.all(session.id) if session is not None else []

            if session is None:
                raise FileNotFoundError(f"no session to resume in {paths.root}")

            stored = session.setup.get("chat", {}).get("instructions")
            chat = ChatSetup(
                driver=chat.driver,
                approver=chat.approver,
                permissions=chat.permissions,
                instructions=stored,
                log=chat.log,
            )
            clock = adapters.clock or SystemClock()
            opened = Opened(config, paths, adapters, clock, store, session, jobs)
            harness = await cls._assemble(opened, ChatProfile(chat))
            await harness.loop.journal.restore(session)

            if harness.todo is not None:
                harness.todo.items = TodoTools.from_history(session.messages).items

            return harness
        except BaseException:
            await store.close()
            raise

    @classmethod
    async def resume(
        cls, *, config: HarnessConfig, task: Task, paths: RunPaths, adapters: Adapters
    ) -> Harness:
        store = await cls.open_store(paths)

        try:
            async with store.transaction() as tx:
                session = await tx.sessions.latest()
                jobs = await tx.jobs.all(session.id) if session is not None else []

            if session is None:
                raise FileNotFoundError(f"no run to resume in {paths.root}")

            if changes := session.setup_changes(cls.setup_of(config)):
                raise SetupChanged(changes)

            clock = adapters.clock or SystemClock()
            opened = Opened(config, paths, adapters, clock, store, session, jobs)
            harness = await cls._assemble(opened, TaskProfile(task))
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
            async with store.transaction() as tx:
                session = await tx.sessions.latest()

                if session is None:
                    raise FileNotFoundError(f"no run in {paths.root}")

                records = await tx.messages.all(session.id)
                return RunReport(
                    session=session,
                    best=await tx.submissions.best(session.id),
                    jobs=await tx.jobs.all(session.id),
                    checkpoints=len(await tx.checkpoints.all(session.id)),
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
        config: HarnessConfig, *, paths: RunPaths, task: Task | None, session: Session
    ) -> Sandbox:
        if config.sandbox.kind == "local":
            return LocalSandbox(paths.workspace)

        return PodmanSandbox(
            config.sandbox,
            name=f"juno-{session.id[-12:]}",
            paths=paths,
            data=task.data_dir if task is not None else None,
        )

    @classmethod
    async def stop(cls, *, task: Task, paths: RunPaths, adapters: Adapters) -> Session:
        report = await cls.report(paths)
        session = report.session
        config = HarnessConfig(**session.setup)
        sandbox = adapters.sandbox or cls.sandbox_for(
            config, paths=paths, task=task, session=session
        )

        for job in report.jobs:
            if job.state == "running":
                with contextlib.suppress(Exception):
                    await sandbox.kill_group(job.pgid)

        await sandbox.stop()
        return session

    async def run(self) -> Submission | None:
        watcher = asyncio.create_task(self.watchdog.watch())

        try:
            best = await self.loop.run()
        finally:
            watcher.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await watcher

        if self.session.terminal or self.chat:
            await self.sandbox.stop()

        return best

    async def close(self) -> None:
        for client in {id(self.llm): self.llm, id(self.loop.llm): self.loop.llm}.values():
            await client.close()

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

    @staticmethod
    def _project(paths: RunPaths) -> Path:
        if paths.project is None:
            raise ValueError("a chat session needs a project folder")

        return paths.project

    @classmethod
    async def _assemble(cls, opened: Opened, profile: Profile) -> Harness:
        config, paths, adapters, clock = opened.config, opened.paths, opened.adapters, opened.clock
        transactions = opened.store.transaction
        sandbox = adapters.sandbox or cls.sandbox_for(
            config, paths=paths, task=profile.task, session=opened.session
        )
        await sandbox.start()

        probe = adapters.probe or SystemProbe()
        workspace = Workspace(paths.workspace)
        bus = MessageBus()
        context = RunContext(
            session=opened.session, clock=clock, transactions=transactions, bus=bus
        )
        registry = ToolRegistry(
            middleware=[
                EmitEvents(context),
                ContainErrors(),
                *profile.middleware(context, lambda name: registry.is_read_only(name)),
                TruncateOutput(),
            ]
        )
        registry.register(
            BashTools(sandbox, context, max_timeout_s=config.budget.bash_timeout),
            EditorTools(workspace),
            *profile.toolsets(opened, context, workspace),
        )
        jobs = (
            JobTools(
                sandbox,
                context,
                logs_dir=paths.logs,
                max_running=config.budget.max_jobs,
                jobs=opened.jobs,
            )
            if config.flags.tools.jobs
            else None
        )

        if jobs is not None:
            registry.register(jobs)

        if config.flags.tools.system:
            registry.register(SystemTools(probe, context))

        prompts = profile.prompts(opened, sandbox)
        journal = Journal(transactions, registry=registry, prompts=prompts, clock=clock)
        EventRecorder(transactions).attach(bus)

        if profile.log:
            LogRecorder().attach(bus)

        CacheMonitor(bus, clock).attach()
        default_checkpointer, triggers = profile.checkpointing(paths)
        checkpointer = adapters.checkpointer or default_checkpointer
        CheckpointRecorder(checkpointer, context, triggers=triggers).attach()

        llm = adapters.llm or cls.llm_for(config, opened.session)
        watchdog = MemoryWatchdog(
            probe,
            jobs,
            context,
            limit=config.watchdog.memory_fraction,
            interval_s=config.watchdog.interval,
        )
        loop = AgentLoop(
            llm,
            journal,
            context,
            prompts=prompts,
            driver=profile.driver(clock),
            jobs=jobs,
        )

        return cls(
            session=opened.session,
            loop=loop,
            registry=registry,
            bus=bus,
            store=opened.store,
            sandbox=sandbox,
            llm=llm,
            watchdog=watchdog,
            jobs=jobs,
            todo=profile.todo,
            checkpointer=checkpointer,
            chat=profile.chat,
        )
