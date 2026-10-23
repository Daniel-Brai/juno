import asyncio
import copy
import hashlib
from collections.abc import Sequence
from typing import ClassVar

from juno.domain.session import Session
from juno.domain.values import (
    Message,
    MessageRecord,
    ModelRequest,
    Role,
    Submission,
    ToolCall,
    ToolCallStatus,
    ToolResult,
    Usage,
)
from juno.ports import Clock, TransactionFactory
from juno.service.prompts import Prompts
from juno.tools.registry import ToolRegistry


class ResumeRefused(Exception):
    pass


class SetupChanged(ResumeRefused):
    def __init__(self, changes: list[str]) -> None:
        super().__init__(
            f"the run was started with different settings for {', '.join(changes)}; "
            "resume it with its original settings or start a new run"
        )
        self.changes = changes


class ReplayDivergence(ResumeRefused):
    def __init__(self, index: int) -> None:
        super().__init__(
            f"message {index} came from a different prompt, tool set or history; "
            "refusing to resume a run whose setup changed"
        )
        self.index = index


class HistoryHash:
    def __init__(self, header: str) -> None:
        self._digest = hashlib.sha256(header.encode())

    @property
    def value(self) -> str:
        return self._digest.hexdigest()

    def add(self, message: Message) -> None:
        self._digest.update(b"\n" + message.model_dump_json().encode())

    def copy(self) -> HistoryHash:
        clone = copy.copy(self)
        clone._digest = self._digest.copy()
        return clone


class Journal:
    INTERRUPTED: ClassVar[str] = (
        "interrupted by a harness restart before this call finished; "
        "check the workspace before retrying it"
    )
    STOPPED: ClassVar[str] = "stopped by the user before it finished"

    def __init__(
        self,
        transactions: TransactionFactory,
        *,
        registry: ToolRegistry,
        prompts: Prompts,
        clock: Clock,
    ) -> None:
        self._transactions = transactions
        self._clock = clock
        self._registry = registry
        self._prompts = prompts
        self._persisted = 0
        self._history = self._empty_history()

    @property
    def fingerprint(self) -> str:
        return self._history.value

    def request(self, messages: Sequence[Message]) -> ModelRequest:
        return ModelRequest(
            system=self._prompts.system(),
            tools=self._registry.specs(),
            messages=list(messages),
        )

    async def save(
        self, session: Session, request_hash: str | None = None, usage: Usage | None = None
    ) -> None:
        session.touch(self._clock.now())
        history = self._history.copy()

        async with self._transactions() as tx:
            await tx.sessions.save(session)

            for index in range(self._persisted, len(session.messages)):
                message = session.messages[index]
                history.add(message)
                is_reply = message.role is Role.ASSISTANT
                await tx.messages.append(
                    session.id,
                    MessageRecord(
                        index=index,
                        message=message,
                        request_hash=request_hash if is_reply else None,
                        usage=usage if is_reply else None,
                    ),
                )

            await tx.commit()

        self._persisted = len(session.messages)
        self._history = history

    async def restore(self, session: Session) -> Session:
        async with self._transactions() as tx:
            records = await tx.messages.all(session.id)

        history = self._empty_history()

        for record in records:
            if record.request_hash is not None and record.request_hash != history.value:
                raise ReplayDivergence(record.index)

            history.add(record.message)

        session.messages = [record.message for record in records]
        self._persisted = len(records)
        self._history = history
        return session

    async def run_tools(self, session_id: str, calls: Sequence[ToolCall]) -> list[ToolResult]:
        if len(calls) < 2 or not all(self._registry.is_read_only(call.name) for call in calls):
            return [await self._run_tool(session_id, call) for call in calls]

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._run_tool(session_id, call)) for call in calls]

        return [task.result() for task in tasks]

    async def stop_tools(self, session_id: str, calls: Sequence[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []

        for call in calls:
            async with self._transactions() as tx:
                record = await tx.tool_calls.find(session_id, call.id)

                if record is not None and record.result is not None:
                    results.append(record.result)
                    continue

                stopped = ToolResult(call_id=call.id, content=self.STOPPED, is_error=True)

                if record is None:
                    await tx.tool_calls.record_intent(session_id, call)

                await tx.tool_calls.record_result(session_id, stopped, ToolCallStatus.INTERRUPTED)
                await tx.commit()

            results.append(stopped)

        return results

    async def best(self, session_id: str) -> Submission | None:
        async with self._transactions() as tx:
            return await tx.submissions.best(session_id)

    async def _run_tool(self, session_id: str, call: ToolCall) -> ToolResult:
        async with self._transactions() as tx:
            record = await tx.tool_calls.find(session_id, call.id)

        if record is not None and record.result is not None:
            return record.result

        if record is not None and not self._registry.is_idempotent(call.name):
            interrupted = ToolResult(call_id=call.id, content=self.INTERRUPTED, is_error=True)
            await self._finish(session_id, interrupted, ToolCallStatus.INTERRUPTED)
            return interrupted

        async with self._transactions() as tx:
            await tx.tool_calls.record_intent(session_id, call)
            await tx.commit()

        result = await self._registry.dispatch(call)
        await self._finish(session_id, result, ToolCallStatus.DONE)
        return result

    def _empty_history(self) -> HistoryHash:
        return HistoryHash(self.request([]).model_dump_json(exclude={"messages"}))

    async def _finish(self, session_id: str, result: ToolResult, status: ToolCallStatus) -> None:
        async with self._transactions() as tx:
            await tx.tool_calls.record_result(session_id, result, status)
            await tx.commit()
