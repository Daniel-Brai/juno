from collections.abc import Sequence
from typing import ClassVar, Protocol

from juno.domain.events import (
    ModelCalled,
    SessionEnded,
    SessionResumed,
    SessionStarted,
    TurnContinued,
)
from juno.domain.session import Session, State, Trigger
from juno.domain.values import StopReason, Submission, ToolCall, ToolResult
from juno.ports import Clock, LlmClient
from juno.service.bus import MessageBus
from juno.service.journal import Journal
from juno.service.prompts import Prompts


class Jobs(Protocol):
    async def cancel_all(self) -> None: ...

    def busy_seconds(self, start: float, end: float) -> float: ...


class AgentLoop:
    CUT_OFF: ClassVar[str] = (
        "this tool call was cut off by the output limit and did not run; "
        "resend it with less content per call"
    )

    def __init__(
        self,
        llm: LlmClient,
        journal: Journal,
        clock: Clock,
        bus: MessageBus,
        prompts: Prompts,
        jobs: Jobs | None = None,
    ) -> None:
        self._llm = llm
        self.journal = journal
        self._clock = clock
        self._bus = bus
        self._prompts = prompts
        self._jobs = jobs

    async def run(self, session: Session) -> Submission | None:
        if session.terminal:
            return await self.journal.best(session.id)

        if session.state is State.STARTING:
            await self._start(session)
        else:
            await self._recover(session)

        while not session.terminal:
            if session.state is State.WRAPPING_UP:
                await self._wrap_up(session)
            elif session.remaining(self._clock.now()) <= 0:
                session.fire(Trigger.BUDGET_SPENT)
                await self.journal.save(session)
            else:
                await self._turn(session)

        await self._bus.publish(
            SessionEnded(
                session_id=session.id,
                at=self._clock.now(),
                state=session.state,
                usage=session.usage,
                handler_failures=self._bus.failures,
            )
        )
        return await self.journal.best(session.id)

    async def _start(self, session: Session) -> None:
        session.append_user(self._prompts.task_message())
        session.fire(Trigger.START)
        await self.journal.save(session)
        await self._bus.publish(
            SessionStarted(
                session_id=session.id,
                at=self._clock.now(),
                task_id=session.task_id,
                deadline=session.deadline,
            )
        )

    async def _recover(self, session: Session) -> None:
        now = self._clock.now()
        downtime = session.downtime(now)
        busy = 0.0

        if self._jobs is not None and session.last_active_at is not None and downtime:
            busy = self._jobs.busy_seconds(session.last_active_at, now)

        paused = session.pause_for_downtime(now, busy)
        await self.journal.save(session)
        await self._bus.publish(
            SessionResumed(
                session_id=session.id,
                at=now,
                downtime_s=downtime,
                jobs_busy_s=busy,
                paused_s=paused,
                deadline=session.deadline,
            )
        )

        if session.state is State.AWAITING_CONTINUE:
            await self._continue(session)
        elif calls := session.pending_tool_calls():
            await self._finish_tools(session, calls)

    async def _turn(self, session: Session) -> None:
        request = self.journal.request(session.messages)
        fingerprint = self.journal.fingerprint
        started = self._clock.now()
        reply = await self._llm.complete(request)
        session.append_assistant(reply)
        calls = reply.tool_calls()

        if calls:
            session.fire(Trigger.TOOL_USE)
        elif reply.stop_reason is StopReason.REFUSAL:
            session.fire(Trigger.FATAL)
        else:
            session.fire(Trigger.END_TURN)

        await self.journal.save(session, fingerprint, reply.usage)
        await self._bus.publish(
            ModelCalled(
                session_id=session.id,
                at=self._clock.now(),
                message_index=len(session.messages) - 1,
                usage=reply.usage,
                duration_s=self._clock.now() - started,
            )
        )

        if calls and reply.stop_reason is StopReason.MAX_TOKENS:
            session.append_tool_results([self.cut_off(call) for call in calls])
            await self.journal.save(session)
        elif calls:
            await self._finish_tools(session, calls)
        elif session.state is State.AWAITING_CONTINUE:
            await self._continue(session)

    @classmethod
    def cut_off(cls, call: ToolCall) -> ToolResult:
        return ToolResult(call_id=call.id, content=cls.CUT_OFF, is_error=True)

    async def _finish_tools(self, session: Session, calls: Sequence[ToolCall]) -> None:
        session.append_tool_results(await self.journal.run_tools(session.id, calls))
        await self.journal.save(session)

    async def _continue(self, session: Session) -> None:
        remaining = session.remaining(self._clock.now())

        if remaining <= 0:
            session.fire(Trigger.BUDGET_SPENT)
            await self.journal.save(session)
            return

        session.append_user(self._prompts.continue_message(remaining))
        session.fire(Trigger.CONTINUE)
        await self.journal.save(session)
        await self._bus.publish(
            TurnContinued(session_id=session.id, at=self._clock.now(), remaining_s=remaining)
        )

    async def _wrap_up(self, session: Session) -> None:
        if self._jobs is not None:
            await self._jobs.cancel_all()

        session.fire(Trigger.FINISHED)
        await self.journal.save(session)
