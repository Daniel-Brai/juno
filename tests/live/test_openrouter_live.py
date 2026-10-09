import os

import pytest

from juno.adapters.llm.openrouter import OpenRouterClient
from juno.domain.values import Message, ModelRequest, Role, StopReason, ToolSpec

pytestmark = [pytest.mark.anyio, pytest.mark.live_api]

MODEL = os.environ.get("JUNO_LIVE_MODEL", "deepseek/deepseek-v4-flash")
BASH = ToolSpec(
    name="bash",
    description="Run a shell command",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)


class TestOpenRouterLive:
    async def test_a_real_model_calls_a_tool_and_reports_usage(self) -> None:
        key = os.environ.get("OPENROUTER_API_KEY")

        if not key:
            pytest.skip("set OPENROUTER_API_KEY")

        client = OpenRouterClient(key, MODEL, "low", 2_000)
        request = ModelRequest(
            system="You are testing a tool connection. Use the tool exactly as asked.",
            tools=[BASH],
            messages=[Message(role=Role.USER, content="Run the command `echo ready` with bash.")],
        )

        reply = await client.complete(request)
        await client.close()

        [call] = reply.tool_calls()
        assert reply.stop_reason is StopReason.TOOL_USE
        assert call.name == "bash" and "echo" in call.args["command"]
        assert reply.usage.input_tokens + reply.usage.cache_read_input_tokens > 0
        assert reply.usage.output_tokens > 0
