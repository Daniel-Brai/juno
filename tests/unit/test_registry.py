import pytest

from juno.domain.commands import QuerySystem, RunBash
from juno.domain.values import ToolCall, ToolResult
from juno.tools.registry import Dispatch, Tool, ToolError, ToolRegistry

pytestmark = pytest.mark.anyio


class EchoTools:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tools(self) -> list[Tool[RunBash]]:
        return [
            Tool(name="echo", description="Echo", input_model=RunBash, handler=self.echo),
            Tool(name="fail", description="Fail", input_model=RunBash, handler=self.fail),
            Tool(name="crash", description="Crash", input_model=RunBash, handler=self.crash),
        ]

    async def echo(self, args: RunBash) -> str:
        self.calls.append(args.command)
        return args.command

    async def fail(self, args: RunBash) -> str:
        raise ToolError(f"cannot {args.command}")

    async def crash(self, args: RunBash) -> str:
        raise RuntimeError("boom")


class LookTools:
    def tools(self) -> list[Tool[QuerySystem]]:
        return [
            Tool(
                name="look",
                description="Look",
                input_model=QuerySystem,
                handler=self.look,
                idempotent=True,
            )
        ]

    async def look(self, _: QuerySystem) -> str:
        return "nothing changed"


class Recorder:
    def __init__(self, name: str, log: list[str]) -> None:
        self._name = name
        self._log = log

    async def __call__(self, call: ToolCall, proceed: Dispatch) -> ToolResult:
        self._log.append(f"{self._name}:in")
        result = await proceed(call)
        self._log.append(f"{self._name}:out")
        return result


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTools())
    return registry


class TestSpecs:
    def test_schema_comes_from_the_command_model(self, registry: ToolRegistry) -> None:
        spec = registry.specs()[0]

        assert spec.name == "echo"
        assert "title" not in spec.input_schema
        assert spec.input_schema["required"] == ["command"]
        assert spec.input_schema["additionalProperties"] is False

    def test_order_follows_registration(self, registry: ToolRegistry) -> None:
        assert registry.names == ["echo", "fail", "crash"]

    def test_duplicate_names_are_rejected(self, registry: ToolRegistry) -> None:
        with pytest.raises(ValueError, match="echo"):
            registry.register(EchoTools())

    def test_idempotence_is_reported_per_tool(self, registry: ToolRegistry) -> None:
        registry.register(LookTools())

        assert registry.is_idempotent("look")
        assert not registry.is_idempotent("echo")
        assert not registry.is_idempotent("missing")


class TestDispatch:
    async def test_runs_the_handler_with_validated_input(self, registry: ToolRegistry) -> None:
        result = await registry.dispatch(ToolCall(id="1", name="echo", args={"command": "ls"}))

        assert result == ToolResult(call_id="1", content="ls")

    async def test_unknown_tools_become_error_results(self, registry: ToolRegistry) -> None:
        result = await registry.dispatch(ToolCall(id="1", name="nope"))

        assert result.is_error
        assert "nope" in result.content

    async def test_invalid_input_names_the_field(self, registry: ToolRegistry) -> None:
        result = await registry.dispatch(ToolCall(id="1", name="echo", args={"command": ""}))

        assert result.is_error
        assert result.content.startswith("invalid input: command:")

    async def test_tool_errors_become_error_results(self, registry: ToolRegistry) -> None:
        result = await registry.dispatch(ToolCall(id="1", name="fail", args={"command": "x"}))

        assert result == ToolResult(call_id="1", content="cannot x", is_error=True)

    async def test_unexpected_errors_propagate_without_middleware(
        self, registry: ToolRegistry
    ) -> None:
        with pytest.raises(RuntimeError):
            await registry.dispatch(ToolCall(id="1", name="crash", args={"command": "x"}))

    async def test_middleware_wraps_in_the_given_order(self) -> None:
        log: list[str] = []
        registry = ToolRegistry(middleware=[Recorder("outer", log), Recorder("inner", log)])
        registry.register(EchoTools())

        await registry.dispatch(ToolCall(id="1", name="echo", args={"command": "ls"}))

        assert log == ["outer:in", "inner:in", "inner:out", "outer:out"]

    async def test_dispatch_all_keeps_call_order(self) -> None:
        tools = EchoTools()
        registry = ToolRegistry()
        registry.register(tools)
        calls = [ToolCall(id=str(n), name="echo", args={"command": f"c{n}"}) for n in range(3)]

        results = await registry.dispatch_all(calls)

        assert [result.call_id for result in results] == ["0", "1", "2"]
        assert tools.calls == ["c0", "c1", "c2"]
