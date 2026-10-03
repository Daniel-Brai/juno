from typing import Protocol

from juno.domain.events import ModelCalled, SessionEnded, SessionStarted, TurnContinued
from juno.domain.session import Session, State, Trigger
from juno.domain.values import ModelRequest, StopReason, Submission
from juno.ports import Clock, LlmClient, SubmissionRepo
from juno.service.bus import MessageBus
from juno.service.prompts import Prompts
from juno.tools.registry import ToolRegistry


class Jobs(Protocol):
    async def cancel_all(self) -> None: ...


class AgentLoop:
    def __init__(
        self,
        llm: LlmClient,
        registry: ToolRegistry,
        clock: Clock,
        bus: MessageBus,
        prompts: Prompts,
        submissions: SubmissionRepo,
        jobs: Jobs | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._clock = clock
        self._bus = bus
        self._prompts = prompts
        self._submissions = submissions
        self._jobs = jobs

    async def run(self, session: Session) -> Submission | None:
        if session.state is State.STARTING:
            await self._start(session)

        while not session.terminal:
            if session.state is State.WRAPPING_UP:
                await self._wrap_up(session)
            elif session.remaining(self._clock.now()) <= 0:
                session.fire(Trigger.BUDGET_SPENT)
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
        return await self._submissions.best(session.id)

    async def _start(self, session: Session) -> None:
        session.append_user(self._prompts.task_message())
        session.fire(Trigger.START)
        await self._bus.publish(
            SessionStarted(
                session_id=session.id,
                at=self._clock.now(),
                task_id=session.task_id,
                deadline=session.deadline,
            )
        )

    async def _turn(self, session: Session) -> None:
        started = self._clock.now()
        reply = await self._llm.complete(self._request(session))
        session.append_assistant(reply)
        await self._bus.publish(
            ModelCalled(
                session_id=session.id,
                at=self._clock.now(),
                message_index=len(session.messages) - 1,
                usage=reply.usage,
                duration_s=self._clock.now() - started,
            )
        )

        if calls := reply.tool_calls():
            session.fire(Trigger.TOOL_USE)
            session.append_tool_results(await self._registry.dispatch_all(calls))
        elif reply.stop_reason is StopReason.REFUSAL:
            session.fire(Trigger.FATAL)
        else:
            await self._continue(session)

    async def _continue(self, session: Session) -> None:
        remaining = session.remaining(self._clock.now())
        session.fire(Trigger.END_TURN)

        if remaining <= 0:
            session.fire(Trigger.BUDGET_SPENT)
            return

        session.append_user(self._prompts.continue_message(remaining))
        session.fire(Trigger.CONTINUE)
        await self._bus.publish(
            TurnContinued(session_id=session.id, at=self._clock.now(), remaining_s=remaining)
        )

    async def _wrap_up(self, session: Session) -> None:
        if self._jobs is not None:
            await self._jobs.cancel_all()

        session.fire(Trigger.FINISHED)

    def _request(self, session: Session) -> ModelRequest:
        return ModelRequest(
            system=self._prompts.system(),
            tools=self._registry.specs(),
            messages=list(session.messages),
        )
