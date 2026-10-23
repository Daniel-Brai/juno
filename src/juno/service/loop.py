from collections.abc import Sequence
from typing import ClassVar, Protocol

from juno.domain.events import (
    BudgetSpent,
    ModelCalled,
    SessionEnded,
    SessionResumed,
    SessionStarted,
    TurnContinued,
)
from juno.domain.session import Session, State, Trigger
from juno.domain.values import (
    ContentBlock,
    ModelReply,
    Role,
    StopReason,
    Submission,
    ToolCall,
    ToolResult,
)
from juno.ports import LlmClient, ModelRejected
from juno.service.context import RunContext
from juno.service.drivers import TurnDriver
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
    STOPPED: ClassVar[str] = "(stopped by the user before replying)"

    def __init__(
        self,
        llm: LlmClient,
        journal: Journal,
        context: RunContext,
        *,
        prompts: Prompts,
        driver: TurnDriver,
        jobs: Jobs | None = None,
    ) -> None:
        self.llm = llm
        self.journal = journal
        self._session = context.session
        self._clock = context.clock
        self._bus = context.bus
        self._prompts = prompts
        self._driver = driver
        self._jobs = jobs

    async def run(self) -> Submission | None:
        session = self._session

        if session.terminal:
            return await self.journal.best(session.id)

        if session.state is not State.STARTING:
            await self._recover(session)
        elif not await self._start(session):
            return None

        while not session.terminal:
            if session.state is State.WRAPPING_UP:
                await self._wrap_up(session)
            elif limit := session.limit_reached(self._clock.now()):
                session.fire(Trigger.BUDGET_SPENT)
                await self.journal.save(session)
                await self._bus.publish(
                    BudgetSpent(
                        session_id=session.id,
                        at=self._clock.now(),
                        limit=limit,
                        cost_usd=session.usage.cost_usd,
                    )
                )
            elif session.state is State.AWAITING_CONTINUE:
                if not await self._continue(session):
                    return await self.journal.best(session.id)
            else:
                await self._turn(session)

        await self._bus.publish(
            SessionEnded(
                session_id=session.id,
                at=self._clock.now(),
                state=session.state,
                failure=session.failure,
                usage=session.usage,
                handler_failures=self._bus.failures,
            )
        )
        return await self.journal.best(session.id)

    async def _start(self, session: Session) -> bool:
        opening = self._prompts.opening() or await self._driver.next_message(session)

        if opening is None:
            return False

        session.append_user(opening)
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
        return True

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

        if calls := session.pending_tool_calls():
            await self._finish_tools(session, calls)

    async def _turn(self, session: Session) -> None:
        request = self.journal.request(session.messages)
        fingerprint = self.journal.fingerprint
        started = self._clock.now()

        try:
            reply = await self.llm.complete(request)
        except ModelRejected as error:
            session.fail(str(error))
            await self.journal.save(session)
            return

        session.append_assistant(reply)
        calls = reply.tool_calls()

        if calls:
            session.fire(Trigger.TOOL_USE)
        elif reply.stop_reason is StopReason.REFUSAL:
            session.fail("the model refused to continue")
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

    async def interrupt(self) -> None:
        session = self._session

        if session.terminal or session.state is not State.RUNNING:
            return

        if calls := session.pending_tool_calls():
            session.append_tool_results(await self.journal.stop_tools(session.id, calls))

        if session.messages and session.messages[-1].role is Role.USER:
            text: ContentBlock = {"type": "text", "text": self.STOPPED}
            session.append_assistant(ModelReply(content=[text], stop_reason=StopReason.END_TURN))

        session.fire(Trigger.END_TURN)
        await self.journal.save(session)

    @classmethod
    def cut_off(cls, call: ToolCall) -> ToolResult:
        return ToolResult(call_id=call.id, content=cls.CUT_OFF, is_error=True)

    async def _finish_tools(self, session: Session, calls: Sequence[ToolCall]) -> None:
        session.append_tool_results(await self.journal.run_tools(session.id, calls))
        await self.journal.save(session)

    async def _continue(self, session: Session) -> bool:
        message = await self._driver.next_message(session)

        if message is None:
            return False

        session.append_user(message)
        session.fire(Trigger.CONTINUE)
        await self.journal.save(session)
        remaining = None if session.open_ended else session.remaining(self._clock.now())
        await self._bus.publish(
            TurnContinued(session_id=session.id, at=self._clock.now(), remaining_s=remaining)
        )
        return True

    async def _wrap_up(self, session: Session) -> None:
        if self._jobs is not None:
            await self._jobs.cancel_all()

        session.fire(Trigger.FINISHED)
        await self.journal.save(session)
