from typing import Any, ClassVar

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage

from juno.domain.values import ModelReply, ModelRequest, StopReason, Usage


class AnthropicClient:
    FALLBACK_BETA: ClassVar[str] = "server-side-fallback-2026-07-01"
    COMPACTION_BETA: ClassVar[str] = "compact-2026-01-12"
    CACHE_TTL: ClassVar[str] = "1h"

    def __init__(
        self,
        model: str,
        effort: str,
        max_tokens: int,
        compaction: bool = False,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._compaction = compaction
        self._client = client or AsyncAnthropic(max_retries=5)

    def params(self, request: ModelRequest) -> dict[str, Any]:
        betas = [self.FALLBACK_BETA]
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": request.system,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "tools": [
                {**tool.model_dump(mode="json"), "eager_input_streaming": True}
                for tool in request.tools
            ],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
            "cache_control": {"type": "ephemeral", "ttl": self.CACHE_TTL},
            "fallbacks": "default",
        }

        if self._compaction:
            betas.append(self.COMPACTION_BETA)
            params["context_management"] = {"edits": [{"type": "compact_20260112"}]}

        params["betas"] = betas
        return params

    async def complete(self, request: ModelRequest) -> ModelReply:
        async with self._client.beta.messages.stream(**self.params(request)) as stream:
            message = await stream.get_final_message()

        return self.reply(message)

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def reply(message: BetaMessage) -> ModelReply:
        usage = message.usage
        stop = message.stop_reason or StopReason.END_TURN

        return ModelReply(
            content=[block.model_dump(mode="json", exclude_none=True) for block in message.content],
            stop_reason=StopReason(stop) if stop in StopReason else StopReason.END_TURN,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            ),
        )
