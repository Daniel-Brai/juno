from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from juno.domain.values import (
    CommandResult,
    Job,
    Message,
    ModelReply,
    ModelRequest,
    ResourceSnapshot,
    Submission,
    ToolCall,
    ToolCallRecord,
    ToolResult,
)


class LlmClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelReply: ...


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class ResourceProbe(Protocol):
    def sample(self) -> ResourceSnapshot: ...


class Sandbox(Protocol):
    async def exec(self, command: str, timeout_s: float) -> CommandResult: ...

    async def spawn(self, command: str, log_path: Path, exit_path: Path) -> Job: ...

    def poll(self, job: Job) -> int | None: ...

    async def kill_group(self, pgid: int) -> None: ...


class Checkpointer(Protocol):
    async def checkpoint(self, message: str) -> str: ...


class MessageRepo(Protocol):
    async def append(
        self, session_id: str, index: int, message: Message, request_hash: str | None = None
    ) -> None: ...

    async def all(self, session_id: str) -> list[Message]: ...


class ToolCallRepo(Protocol):
    async def record_intent(self, session_id: str, call: ToolCall) -> None: ...

    async def record_result(self, session_id: str, result: ToolResult) -> None: ...

    async def find(self, session_id: str, call_id: str) -> ToolCallRecord | None: ...


class JobRepo(Protocol):
    async def add(self, session_id: str, job: Job) -> None: ...

    async def update(self, job: Job) -> None: ...

    async def running(self, session_id: str) -> list[Job]: ...


class SubmissionRepo(Protocol):
    async def add(self, session_id: str, submission: Submission) -> None: ...

    async def all(self, session_id: str) -> list[Submission]: ...

    async def best(self, session_id: str) -> Submission | None: ...


class UnitOfWork(Protocol):
    @property
    def messages(self) -> MessageRepo: ...

    @property
    def tool_calls(self) -> ToolCallRepo: ...

    @property
    def jobs(self) -> JobRepo: ...

    @property
    def submissions(self) -> SubmissionRepo: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...
