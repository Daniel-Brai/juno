from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from anthropic import (
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from anthropic.types.beta import BetaMessage

from juno.config import ModelSettings
from juno.domain.values import ModelReply, ModelRequest, StopReason, Usage
from juno.ports import ModelRejected


@dataclass(frozen=True, slots=True)
class Price:
    input: float
    output: float
    cache_read: float
    large_prompt: Price | None = None
    large_above: int = 100_000


class AnthropicPrices:
    PER_MILLION: ClassVar[Mapping[str, Price]] = {
        "claude-fable-5-1": Price(input=10, output=50, cache_read=0.25),
        "claude-fable-5": Price(input=10, output=50, cache_read=1),
        "claude-opus-5-5": Price(input=4, output=20, cache_read=0.2),
        "claude-opus-5": Price(input=5, output=25, cache_read=0.5),
        "claude-sonnet-5-5": Price(input=2, output=10, cache_read=0.2),
        "claude-haiku-5-5": Price(
            input=0.1,
            output=0.5,
            cache_read=0.01,
            large_prompt=Price(input=0.5, output=2.5, cache_read=0.05),
        ),
    }
    CACHE_WRITE: ClassVar[float] = 2.0

    @classmethod
    def knows(cls, model: str) -> bool:
        return model in cls.PER_MILLION

    @classmethod
    def cost(cls, model: str, usage: Usage) -> float:
        price = cls.PER_MILLION.get(model)

        if price is None:
            return 0.0

        prompt = (
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
        )

        if price.large_prompt is not None and prompt > price.large_above:
            price = price.large_prompt

        dollars = (
            usage.input_tokens * price.input
            + usage.cache_creation_input_tokens * price.input * cls.CACHE_WRITE
            + usage.cache_read_input_tokens * price.cache_read
            + usage.output_tokens * price.output
        )
        return dollars / 1_000_000


class AnthropicClient:
    FALLBACK_BETA: ClassVar[str] = "server-side-fallback-2026-07-01"
    COMPACTION_BETA: ClassVar[str] = "compact-2026-01-12"
    CACHE_TTL: ClassVar[str] = "1h"

    def __init__(
        self,
        settings: ModelSettings,
        *,
        compaction: bool = False,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._settings = settings
        self._compaction = compaction
        self._client = client or AsyncAnthropic(max_retries=5)

    def params(self, request: ModelRequest) -> dict[str, Any]:
        betas = [self.FALLBACK_BETA]
        params: dict[str, Any] = {
            "model": self._settings.name,
            "max_tokens": self._settings.max_tokens,
            "system": request.system,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "tools": [
                {**tool.model_dump(mode="json"), "eager_input_streaming": True}
                for tool in request.tools
            ],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._settings.effort},
            "cache_control": {"type": "ephemeral", "ttl": self.CACHE_TTL},
            "fallbacks": "default",
        }

        if self._compaction:
            betas.append(self.COMPACTION_BETA)
            params["context_management"] = {"edits": [{"type": "compact_20260112"}]}

        params["betas"] = betas
        return params

    async def complete(self, request: ModelRequest) -> ModelReply:
        try:
            async with self._client.beta.messages.stream(**self.params(request)) as stream:
                message = await stream.get_final_message()
        except (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            BadRequestError,
        ) as error:
            raise ModelRejected(
                f"anthropic rejected the request with {error.status_code}: {error.message}"
            ) from error

        return self.reply(message)

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def reply(message: BetaMessage) -> ModelReply:
        stop = message.stop_reason or StopReason.END_TURN
        usage = Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
        )

        return ModelReply(
            content=[block.model_dump(mode="json", exclude_none=True) for block in message.content],
            stop_reason=StopReason(stop) if stop in StopReason else StopReason.END_TURN,
            usage=usage.model_copy(update={"cost_usd": AnthropicPrices.cost(message.model, usage)}),
        )
