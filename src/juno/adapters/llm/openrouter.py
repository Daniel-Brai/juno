import asyncio
import json
from typing import Any, ClassVar

import httpx

from juno.config import ModelSettings
from juno.domain.values import (
    ContentBlock,
    Message,
    ModelReply,
    ModelRequest,
    Role,
    StopReason,
    ToolSpec,
    Usage,
)


class OpenRouterError(Exception):
    pass


class OpenRouterClient:
    URL: ClassVar[str] = "https://openrouter.ai/api/v1/chat/completions"
    REASONING: ClassVar[str] = "openrouter_reasoning"
    RETRY_STATUSES: ClassVar[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
    ATTEMPTS: ClassVar[int] = 5
    BACKOFF_S: ClassVar[float] = 2.0
    STOP_REASONS: ClassVar[dict[str, StopReason]] = {
        "stop": StopReason.END_TURN,
        "tool_calls": StopReason.TOOL_USE,
        "length": StopReason.MAX_TOKENS,
        "content_filter": StopReason.REFUSAL,
    }

    def __init__(
        self,
        settings: ModelSettings,
        *,
        api_key: str,
        session_id: str | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._session_id = session_id
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30))
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "X-OpenRouter-Title": "juno",
        }

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.name,
            "max_tokens": self._settings.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                *(item for message in request.messages for item in self.to_chat(message)),
            ],
            "tools": [self.to_function(tool) for tool in request.tools],
            "reasoning": {"effort": self._settings.effort},
        }

        if self._session_id is not None:
            payload["session_id"] = self._session_id

        if self._settings.name.startswith("anthropic/"):
            payload["cache_control"] = {"type": "ephemeral"}

        return payload

    async def complete(self, request: ModelRequest) -> ModelReply:
        body = await self._post(self.payload(request))

        if "error" in body:
            raise OpenRouterError(f"openrouter error: {body['error']}")

        return self.reply(body)

    async def close(self) -> None:
        await self._http.aclose()

    @classmethod
    def to_chat(cls, message: Message) -> list[dict[str, Any]]:
        if isinstance(message.content, str):
            return [{"role": message.role, "content": message.content}]

        if message.role is Role.USER:
            return cls._user_parts(message.content)

        return [cls._assistant(message.content)]

    @staticmethod
    def to_function(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @classmethod
    def reply(cls, body: dict[str, Any]) -> ModelReply:
        choice = body["choices"][0]
        message = choice["message"]
        content: list[ContentBlock] = []

        if details := message.get("reasoning_details"):
            content.append({"type": cls.REASONING, "details": details})

        if text := message.get("content"):
            content.append({"type": "text", "text": text})

        content += [cls._tool_use(call) for call in message.get("tool_calls") or []]
        finish = choice.get("finish_reason")

        if finish == "error":
            raise OpenRouterError(f"provider error: {choice.get('native_finish_reason')}")

        return ModelReply(
            content=content,
            stop_reason=cls.STOP_REASONS.get(finish or "stop", StopReason.END_TURN),
            usage=cls._usage(body.get("usage") or {}),
        )

    @classmethod
    def _user_parts(cls, blocks: list[ContentBlock]) -> list[dict[str, Any]]:
        results = [
            {"role": "tool", "tool_call_id": block["tool_use_id"], "content": str(block["content"])}
            for block in blocks
            if block.get("type") == "tool_result"
        ]
        texts = [block["text"] for block in blocks if block.get("type") == "text"]
        return results + ([{"role": "user", "content": "\n\n".join(texts)}] if texts else [])

    @classmethod
    def _assistant(cls, blocks: list[ContentBlock]) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(block["text"] for block in blocks if block.get("type") == "text")
            or None,
        }
        calls = [
            {
                "id": block["id"],
                "type": "function",
                "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
            }
            for block in blocks
            if block.get("type") == "tool_use"
        ]
        details = [
            detail
            for block in blocks
            if block.get("type") == cls.REASONING
            for detail in block["details"]
        ]

        if calls:
            message["tool_calls"] = calls

        if details:
            message["reasoning_details"] = details

        return message

    @staticmethod
    def _tool_use(call: dict[str, Any]) -> ContentBlock:
        function = call["function"]
        raw = function.get("arguments") or "{}"

        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            arguments = {"unparsed_arguments": raw}

        return {
            "type": "tool_use",
            "id": call["id"],
            "name": function["name"],
            "input": arguments if isinstance(arguments, dict) else {"value": arguments},
        }

    @staticmethod
    def _usage(usage: dict[str, Any]) -> Usage:
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        written = details.get("cache_write_tokens") or 0

        return Usage(
            input_tokens=max(0, (usage.get("prompt_tokens") or 0) - cached - written),
            output_tokens=usage.get("completion_tokens") or 0,
            cache_creation_input_tokens=written,
            cache_read_input_tokens=cached,
            cost_usd=float(usage.get("cost") or 0.0),
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.ATTEMPTS):
            try:
                response = await self._http.post(self.URL, json=payload, headers=self._headers)
            except httpx.TransportError:
                if attempt == self.ATTEMPTS - 1:
                    raise
            else:
                if response.status_code not in self.RETRY_STATUSES:
                    response.raise_for_status()
                    body: dict[str, Any] = response.json()
                    return body

                if attempt == self.ATTEMPTS - 1:
                    response.raise_for_status()

            await asyncio.sleep(self.BACKOFF_S * 2**attempt)

        raise OpenRouterError("unreachable: retries exhausted")
