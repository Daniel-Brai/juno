from collections import deque
from collections.abc import Callable, Iterable
from typing import Any, Self
from uuid import uuid7

from juno.domain.values import ContentBlock, ModelReply, ModelRequest, StopReason, Usage

type Turn = ModelReply | Callable[[ModelRequest], ModelReply]


class ScriptExhausted(AssertionError):
    pass


class Reply:
    @staticmethod
    def text(text: str, usage: Usage | None = None) -> ModelReply:
        return ModelReply(
            content=[{"type": "text", "text": text}],
            stop_reason=StopReason.END_TURN,
            usage=usage or Usage(),
        )

    @staticmethod
    def tools(*calls: tuple[str, dict[str, Any]], usage: Usage | None = None) -> ModelReply:
        content: list[ContentBlock] = [
            {"type": "tool_use", "id": f"toolu_{uuid7().hex[:12]}", "name": name, "input": args}
            for name, args in calls
        ]
        return ModelReply(content=content, stop_reason=StopReason.TOOL_USE, usage=usage or Usage())

    @staticmethod
    def refusal() -> ModelReply:
        return ModelReply(content=[], stop_reason=StopReason.REFUSAL)


class ScriptedLlm:
    def __init__(self, turns: Iterable[Turn]) -> None:
        self.requests: list[ModelRequest] = []
        self._turns = deque(turns)
        self._faults: dict[int, BaseException] = {}

    @property
    def remaining(self) -> int:
        return len(self._turns)

    def fail_at(self, request_index: int, error: BaseException) -> Self:
        self._faults[request_index] = error
        return self

    async def complete(self, request: ModelRequest) -> ModelReply:
        index = len(self.requests)
        self.requests.append(request)

        if (error := self._faults.pop(index, None)) is not None:
            raise error

        if not self._turns:
            raise ScriptExhausted(f"no scripted reply for request {index}")

        turn = self._turns.popleft()
        reply = turn if isinstance(turn, ModelReply) else turn(request)
        self._check_tools(request, reply)
        return reply

    @staticmethod
    def _check_tools(request: ModelRequest, reply: ModelReply) -> None:
        offered = {tool.name for tool in request.tools}
        unknown = sorted({call.name for call in reply.tool_calls()} - offered)

        if unknown:
            raise AssertionError(f"scripted reply calls tools the request did not offer: {unknown}")
