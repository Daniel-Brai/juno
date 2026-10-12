from juno.domain.events import ToolFinished, ToolStarted
from juno.domain.values import ToolCall, ToolResult
from juno.service.context import RunContext
from juno.tools.registry import Dispatch


class EmitEvents:
    def __init__(self, context: RunContext) -> None:
        self._bus = context.bus
        self._clock = context.clock
        self._session_id = context.session.id

    async def __call__(self, call: ToolCall, proceed: Dispatch) -> ToolResult:
        started = self._clock.now()
        await self._bus.publish(ToolStarted(session_id=self._session_id, at=started, call=call))

        result = await proceed(call)
        finished = self._clock.now()

        if not result.duration_s:
            result = result.model_copy(update={"duration_s": finished - started})

        await self._bus.publish(
            ToolFinished(session_id=self._session_id, at=finished, result=result)
        )
        return result


class ContainErrors:
    async def __call__(self, call: ToolCall, proceed: Dispatch) -> ToolResult:
        try:
            return await proceed(call)
        except Exception as error:
            return ToolResult(
                call_id=call.id,
                content=f"tool failed: {type(error).__name__}: {error}",
                is_error=True,
            )


class TruncateOutput:
    def __init__(self, limit: int = 30_000) -> None:
        self._half = limit // 2

    async def __call__(self, call: ToolCall, proceed: Dispatch) -> ToolResult:
        result = await proceed(call)
        content = result.content

        if len(content) <= 2 * self._half:
            return result

        omitted = len(content) - 2 * self._half
        shortened = f"{content[: self._half]}\n[... {omitted} characters omitted ...]\n{content[-self._half :]}"
        return result.model_copy(update={"content": shortened, "truncated": True})
