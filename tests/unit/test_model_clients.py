import json
from typing import Any

import httpx
import httpx2
import pytest
import respx
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage

from juno.adapters.llm.anthropic import AnthropicClient, AnthropicPrices
from juno.adapters.llm.openrouter import OpenRouterClient, OpenRouterError
from juno.config import ModelSettings
from juno.domain.values import Message, ModelRequest, Role, StopReason, ToolSpec, Usage
from juno.ports import ModelRejected

pytestmark = pytest.mark.anyio

TOOL = ToolSpec(name="bash", description="Run a command", input_schema={"type": "object"})
HISTORY = [
    Message(role=Role.USER, content="Solve the task."),
    Message(
        role=Role.ASSISTANT,
        content=[
            {
                "type": "openrouter_reasoning",
                "details": [{"type": "reasoning.text", "text": "plan"}],
            },
            {"type": "text", "text": "Listing files."},
            {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"command": "ls"}},
        ],
    ),
    Message(
        role=Role.USER,
        content=[
            {"type": "tool_result", "tool_use_id": "call_1", "content": "data/", "is_error": False}
        ],
    ),
]
REQUEST = ModelRequest(system="Be an engineer.", tools=[TOOL], messages=HISTORY)
OPUS = ModelSettings()
FLASH = ModelSettings(provider="openrouter", name="deepseek/deepseek-v4-flash", max_tokens=8_000)


class TestAnthropicClient:
    def test_request_turns_on_caching_thinking_and_fallbacks(self) -> None:
        params = AnthropicClient(OPUS).params(REQUEST)

        assert params["model"] == "claude-opus-5-5"
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}
        assert params["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert params["fallbacks"] == "default"
        assert params["betas"] == ["server-side-fallback-2026-07-01"]
        assert params["tools"][0]["eager_input_streaming"] is True
        assert params["messages"][0] == {"role": "user", "content": "Solve the task."}
        assert "context_management" not in params

    def test_compaction_adds_its_beta(self) -> None:
        params = AnthropicClient(OPUS, compaction=True).params(REQUEST)

        assert "compact-2026-01-12" in params["betas"]
        assert params["context_management"] == {"edits": [{"type": "compact_20260112"}]}

    def test_reply_keeps_blocks_and_counts_cache_tokens(self) -> None:
        message = BetaMessage.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 30,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 4_000,
                },
            }
        )

        reply = AnthropicClient.reply(message)

        assert reply.stop_reason is StopReason.TOOL_USE
        assert reply.content[0] == {"type": "thinking", "thinking": "", "signature": "sig"}
        assert [call.name for call in reply.tool_calls()] == ["bash"]
        assert reply.usage.model_copy(update={"cost_usd": 0}) == Usage(
            input_tokens=12, output_tokens=30, cache_read_input_tokens=4_000
        )
        assert reply.usage.cost_usd == pytest.approx((12 * 4 + 30 * 20 + 4_000 * 0.2) / 1e6)


class TestAnthropicRejections:
    async def test_a_bad_key_raises_model_rejected(self) -> None:
        paths: list[str] = []

        def reject(request: httpx2.Request) -> httpx2.Response:
            paths.append(request.url.path)
            error = {"type": "authentication_error", "message": "invalid x-api-key"}
            return httpx2.Response(401, json={"type": "error", "error": error})

        http = httpx2.AsyncClient(transport=httpx2.MockTransport(reject))
        sdk = AsyncAnthropic(api_key="bad", max_retries=0, http_client=http)
        client = AnthropicClient(OPUS, client=sdk)

        with pytest.raises(ModelRejected, match="anthropic rejected the request with 401"):
            await client.complete(REQUEST)

        await client.close()
        assert paths == ["/v1/messages"]


class TestAnthropicPrices:
    def test_cache_writes_cost_double_input_for_the_hour_ttl(self) -> None:
        usage = Usage(input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)

        assert AnthropicPrices.cost("claude-sonnet-5-5", usage) == pytest.approx(2 + 4)

    def test_haiku_charges_more_above_100k_prompt_tokens(self) -> None:
        small = Usage(input_tokens=100_000, output_tokens=1_000)
        large = Usage(input_tokens=100_001, output_tokens=1_000)

        assert AnthropicPrices.cost("claude-haiku-5-5", small) == pytest.approx(0.0105)
        assert AnthropicPrices.cost("claude-haiku-5-5", large) == pytest.approx(0.0525005)


class TestOpenRouterTranslation:
    def test_history_becomes_chat_messages(self) -> None:
        payload = OpenRouterClient(FLASH, api_key="key").payload(REQUEST)

        system, user, assistant, tool = payload["messages"]
        assert system == {"role": "system", "content": "Be an engineer."}
        assert user == {"role": "user", "content": "Solve the task."}
        assert assistant["content"] == "Listing files."
        assert assistant["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
            }
        ]
        assert assistant["reasoning_details"] == [{"type": "reasoning.text", "text": "plan"}]
        assert tool == {"role": "tool", "tool_call_id": "call_1", "content": "data/"}
        assert payload["tools"][0]["function"]["parameters"] == {"type": "object"}
        assert payload["reasoning"] == {"effort": "high"}
        assert "cache_control" not in payload

    def test_anthropic_models_ask_for_caching(self) -> None:
        payload = OpenRouterClient(
            ModelSettings(provider="openrouter", name="anthropic/claude-opus-5.5"), api_key="key"
        ).payload(REQUEST)

        assert payload["cache_control"] == {"type": "ephemeral"}

    def test_a_reply_becomes_blocks_with_usage_and_cost(self) -> None:
        body = self.body(
            message={
                "role": "assistant",
                "content": "Running it.",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "x"}],
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "pwd"}'},
                    }
                ],
            },
            finish="tool_calls",
        )

        reply = OpenRouterClient.reply(body)

        assert reply.stop_reason is StopReason.TOOL_USE
        assert reply.content[0] == {
            "type": "openrouter_reasoning",
            "details": [{"type": "reasoning.encrypted", "data": "x"}],
        }
        assert reply.tool_calls()[0].args == {"command": "pwd"}
        assert reply.usage == Usage(
            input_tokens=60,
            output_tokens=20,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=30,
            cost_usd=0.0012,
        )

    def test_unparseable_arguments_are_kept_for_validation(self) -> None:
        body = self.body(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{oops"},
                    }
                ],
            },
            finish="tool_calls",
        )

        assert OpenRouterClient.reply(body).tool_calls()[0].args == {"unparsed_arguments": "{oops"}

    @pytest.mark.parametrize(
        ("finish", "stop"),
        [
            ("stop", StopReason.END_TURN),
            ("length", StopReason.MAX_TOKENS),
            ("content_filter", StopReason.REFUSAL),
        ],
    )
    def test_finish_reasons_map_to_stop_reasons(self, finish: str, stop: StopReason) -> None:
        body = self.body(message={"role": "assistant", "content": "done"}, finish=finish)

        assert OpenRouterClient.reply(body).stop_reason is stop

    def test_provider_errors_raise(self) -> None:
        body = self.body(message={"role": "assistant", "content": None}, finish="error")

        with pytest.raises(OpenRouterError):
            OpenRouterClient.reply(body)

    @staticmethod
    def body(message: dict[str, Any], finish: str) -> dict[str, Any]:
        return {
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
                "cost": 0.0012,
            },
        }


class TestOpenRouterTransport:
    @respx.mock
    async def test_retries_rate_limits_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(OpenRouterClient, "BACKOFF_S", 0.0)
        route = respx.post(OpenRouterClient.URL).mock(
            side_effect=[
                httpx.Response(429, json={"error": "slow down"}),
                httpx.Response(
                    200,
                    json=TestOpenRouterTranslation.body(
                        {"role": "assistant", "content": "hi"}, "stop"
                    ),
                ),
            ]
        )
        client = OpenRouterClient(FLASH, api_key="secret")

        reply = await client.complete(REQUEST)
        await client.close()

        assert reply.content == [{"type": "text", "text": "hi"}]
        assert route.call_count == 2
        assert route.calls.last.request.headers["Authorization"] == "Bearer secret"

    @respx.mock
    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
    async def test_rejections_are_not_retried(self, status: int) -> None:
        body = {"error": {"message": "No auth credentials found", "code": status}}
        route = respx.post(OpenRouterClient.URL).mock(
            return_value=httpx.Response(status, json=body)
        )
        client = OpenRouterClient(FLASH, api_key="secret")

        with pytest.raises(ModelRejected, match=f"rejected the request with {status}: No auth"):
            await client.complete(REQUEST)

        await client.close()
        assert route.call_count == 1

    @respx.mock
    async def test_a_rejection_inside_a_200_body_is_fatal_too(self) -> None:
        body = {"error": {"message": "Insufficient credits", "code": 402}}
        respx.post(OpenRouterClient.URL).mock(return_value=httpx.Response(200, json=body))
        client = OpenRouterClient(FLASH, api_key="secret")

        with pytest.raises(ModelRejected, match="402: Insufficient credits"):
            await client.complete(REQUEST)

        await client.close()

    @respx.mock
    async def test_other_client_errors_raise_without_retrying(self) -> None:
        route = respx.post(OpenRouterClient.URL).mock(return_value=httpx.Response(422, json={}))
        client = OpenRouterClient(FLASH, api_key="secret")

        with pytest.raises(httpx.HTTPStatusError):
            await client.complete(REQUEST)

        await client.close()
        assert route.call_count == 1

    @respx.mock
    async def test_error_bodies_raise(self) -> None:
        respx.post(OpenRouterClient.URL).mock(
            return_value=httpx.Response(200, json={"error": {"message": "no credits"}})
        )
        client = OpenRouterClient(FLASH, api_key="secret")

        with pytest.raises(OpenRouterError, match="no credits"):
            await client.complete(REQUEST)

        await client.close()
