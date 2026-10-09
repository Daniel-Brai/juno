from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from juno.domain.events import Event
from juno.domain.session import Session
from juno.domain.values import (
    Checkpoint,
    CommandResult,
    Job,
    MessageRecord,
    ModelReply,
    ModelRequest,
    ResourceSnapshot,
    Submission,
    ToolCall,
    ToolCallRecord,
    ToolCallStatus,
    ToolResult,
)


class LlmClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelReply: ...

    async def close(self) -> None: ...


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class ResourceProbe(Protocol):
    def sample(self) -> ResourceSnapshot: ...


class Sandbox(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def exec(self, command: str, timeout_s: float) -> CommandResult: ...

    async def spawn(self, command: str, log_path: Path, exit_path: Path) -> Job: ...

    def poll(self, job: Job) -> int | None: ...

    def is_alive(self, job: Job) -> bool: ...

    def memory_bytes(self, job: Job) -> int: ...

    async def kill_group(self, pgid: int) -> None: ...


class Checkpointer(Protocol):
    async def checkpoint(self, message: str) -> str | None: ...


class SessionRepo(Protocol):
    async def save(self, session: Session) -> None: ...

    async def get(self, session_id: str) -> Session | None: ...

    async def latest(self) -> Session | None: ...


class MessageRepo(Protocol):
    async def append(self, session_id: str, record: MessageRecord) -> None: ...

    async def all(self, session_id: str) -> list[MessageRecord]: ...


class ToolCallRepo(Protocol):
    async def record_intent(self, session_id: str, call: ToolCall) -> None: ...

    async def record_result(
        self, session_id: str, result: ToolResult, status: ToolCallStatus = ToolCallStatus.DONE
    ) -> None: ...

    async def find(self, session_id: str, call_id: str) -> ToolCallRecord | None: ...


class JobRepo(Protocol):
    async def save(self, session_id: str, job: Job) -> None: ...

    async def all(self, session_id: str) -> list[Job]: ...


class SubmissionRepo(Protocol):
    async def add(self, session_id: str, submission: Submission) -> None: ...

    async def all(self, session_id: str) -> list[Submission]: ...

    async def best(self, session_id: str) -> Submission | None: ...


class CheckpointRepo(Protocol):
    async def add(self, session_id: str, checkpoint: Checkpoint) -> None: ...

    async def all(self, session_id: str) -> list[Checkpoint]: ...


class EventRepo(Protocol):
    async def add(self, event: Event) -> None: ...

    async def count(self, session_id: str) -> int: ...


class UnitOfWork(Protocol):
    @property
    def sessions(self) -> SessionRepo: ...

    @property
    def messages(self) -> MessageRepo: ...

    @property
    def tool_calls(self) -> ToolCallRepo: ...

    @property
    def jobs(self) -> JobRepo: ...

    @property
    def submissions(self) -> SubmissionRepo: ...

    @property
    def checkpoints(self) -> CheckpointRepo: ...

    @property
    def events(self) -> EventRepo: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


type UnitOfWorkFactory = Callable[[], UnitOfWork]
