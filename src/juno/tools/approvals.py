from collections.abc import Callable

from juno.domain.events import ApprovalDecided
from juno.domain.values import Answer, ToolCall, ToolResult
from juno.ports import Approver
from juno.service.context import RunContext
from juno.tools.permissions import Action, Permissions
from juno.tools.registry import Dispatch


class RulesOnly:
    def __init__(self, *, yes: bool = False) -> None:
        self._yes = yes

    async def approve(self, _call: ToolCall, _reason: str, /) -> Answer:
        return Answer.ONCE if self._yes else Answer.DENY


class Approvals:
    def __init__(
        self,
        permissions: Permissions,
        approver: Approver,
        context: RunContext,
        *,
        read_only: Callable[[str], bool],
    ) -> None:
        self._permissions = permissions
        self._approver = approver
        self._bus = context.bus
        self._clock = context.clock
        self._session_id = context.session.id
        self._read_only = read_only

    async def __call__(self, call: ToolCall, proceed: Dispatch) -> ToolResult:
        action, reason = self._permissions.decide(call, read_only=self._read_only(call.name))
        asked = action is Action.ASK

        if asked:
            answer = await self._approver.approve(call, reason)

            if answer is Answer.ALWAYS:
                self._permissions.remember(call)

            action = Action.DENY if answer is Answer.DENY else Action.ALLOW
            reason = "the user denied it" if answer is Answer.DENY else "the user allowed it"

        await self._bus.publish(
            ApprovalDecided(
                session_id=self._session_id,
                at=self._clock.now(),
                call=call,
                allowed=action is Action.ALLOW,
                asked=asked,
                reason=reason,
            )
        )

        if action is Action.DENY:
            return ToolResult(call_id=call.id, content=f"not run: {reason}", is_error=True)

        return await proceed(call)
