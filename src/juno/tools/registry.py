from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from juno.domain.commands import Command
from juno.domain.values import ToolCall, ToolResult, ToolSpec

type Dispatch = Callable[[ToolCall], Awaitable[ToolResult]]


class ToolError(Exception):
    pass


class Middleware(Protocol):
    async def __call__(self, call: ToolCall, proceed: Dispatch) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class Tool[C: Command]:
    name: str
    description: str
    input_model: type[C]
    handler: Callable[[C], Awaitable[str]]
    idempotent: bool = False
    read_only: bool = False
    schema: dict[str, Any] | None = None

    @property
    def spec(self) -> ToolSpec:
        schema = (
            dict(self.schema) if self.schema is not None else self.input_model.model_json_schema()
        )
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, input_schema=schema)

    async def run(self, call: ToolCall) -> ToolResult:
        try:
            args = self.input_model.model_validate(call.args)
        except ValidationError as error:
            return self._error(call, f"invalid input: {self._describe(error)}")

        try:
            return ToolResult(call_id=call.id, content=await self.handler(args))
        except ToolError as error:
            return self._error(call, str(error))

    @staticmethod
    def _describe(error: ValidationError) -> str:
        return "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or 'input'}: {issue['msg']}"
            for issue in error.errors(include_url=False)
        )

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(call_id=call.id, content=message, is_error=True)


class Toolset(Protocol):
    def tools(self) -> Sequence[Tool[Any]]: ...


class ToolRegistry:
    def __init__(self, middleware: Sequence[Middleware] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self._specs: list[ToolSpec] = []
        self._dispatch: Dispatch = self._run

        for layer in reversed(middleware):
            self._dispatch = self._wrap(layer, self._dispatch)

    def register(self, *toolsets: Toolset) -> None:
        for toolset in toolsets:
            for tool in toolset.tools():
                if tool.name in self._tools:
                    raise ValueError(f"tool {tool.name!r} is already registered")

                self._tools[tool.name] = tool
                self._specs.append(tool.spec)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def is_idempotent(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and (tool.idempotent or tool.read_only)

    def is_read_only(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.read_only

    async def dispatch(self, call: ToolCall) -> ToolResult:
        return await self._dispatch(call)

    async def dispatch_all(self, calls: Iterable[ToolCall]) -> list[ToolResult]:
        return [await self.dispatch(call) for call in calls]

    async def _run(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(call_id=call.id, content=f"unknown tool {call.name!r}", is_error=True)

        return await tool.run(call)

    @staticmethod
    def _wrap(layer: Middleware, inner: Dispatch) -> Dispatch:
        async def dispatch(call: ToolCall) -> ToolResult:
            return await layer(call, inner)

        return dispatch
