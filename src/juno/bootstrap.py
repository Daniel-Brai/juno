from dataclasses import dataclass

from juno.adapters.local_sandbox import LocalSandbox
from juno.adapters.memory import InMemorySubmissionRepo
from juno.adapters.system import PsutilProbe, SystemClock
from juno.config import HarnessConfig
from juno.domain.session import Session
from juno.domain.values import RunPaths, Submission, Task
from juno.ports import Clock, LlmClient, ResourceProbe, Sandbox, SubmissionRepo
from juno.service.bus import MessageBus
from juno.service.loop import AgentLoop
from juno.service.prompts import Prompts
from juno.tools.bash import BashTools
from juno.tools.editor import EditorTools
from juno.tools.jobs import JobTools
from juno.tools.middleware import ContainErrors, EmitEvents, TruncateOutput
from juno.tools.registry import ToolRegistry
from juno.tools.submissions import SubmissionTools
from juno.tools.system import SystemTools
from juno.tools.workspace import Workspace


@dataclass(frozen=True, slots=True)
class Harness:
    session: Session
    loop: AgentLoop
    registry: ToolRegistry
    bus: MessageBus
    jobs: JobTools | None

    @classmethod
    def build(
        cls,
        *,
        config: HarnessConfig,
        task: Task,
        paths: RunPaths,
        llm: LlmClient,
        clock: Clock | None = None,
        probe: ResourceProbe | None = None,
        sandbox: Sandbox | None = None,
        submissions: SubmissionRepo | None = None,
        session: Session | None = None,
    ) -> Harness:
        clock = clock or SystemClock()
        submissions = submissions or InMemorySubmissionRepo()
        sandbox = sandbox or cls._sandbox(config, paths)

        for path in paths.all():
            path.mkdir(parents=True, exist_ok=True)

        now = clock.now()
        session = session or Session(
            task_id=task.id, started_at=now, deadline=now + config.budget.total
        )
        workspace = Workspace(paths.workspace)
        bus = MessageBus()
        registry = ToolRegistry(
            middleware=[EmitEvents(bus, clock, session.id), ContainErrors(), TruncateOutput()]
        )
        registry.register(
            BashTools(sandbox, clock, session.deadline, config.budget.bash_timeout),
            EditorTools(workspace),
        )

        jobs: JobTools | None = None
        flags = config.flags.tools

        if flags.jobs:
            jobs = JobTools(
                sandbox, workspace, paths.logs, clock, session.deadline, config.budget.max_jobs
            )
            registry.register(jobs)

        if flags.system:
            registry.register(
                SystemTools(probe or PsutilProbe(), clock, session.started_at, session.deadline)
            )

        if flags.submission:
            registry.register(
                SubmissionTools(
                    submissions,
                    workspace,
                    paths.submissions,
                    session.id,
                    task.direction,
                    clock,
                    bus,
                )
            )

        loop = AgentLoop(
            llm=llm,
            registry=registry,
            clock=clock,
            bus=bus,
            prompts=Prompts(task, config.budget.total),
            submissions=submissions,
            jobs=jobs,
        )
        return cls(session=session, loop=loop, registry=registry, bus=bus, jobs=jobs)

    async def run(self) -> Submission | None:
        return await self.loop.run(self.session)

    @staticmethod
    def _sandbox(config: HarnessConfig, paths: RunPaths) -> Sandbox:
        if config.sandbox.kind == "local":
            return LocalSandbox(paths.workspace)

        raise NotImplementedError(
            "the podman sandbox arrives in phase 3; set sandbox.kind = 'local'"
        )
