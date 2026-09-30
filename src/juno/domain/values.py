from enum import StrEnum
from pathlib import Path
from typing import Any, Self
from uuid import uuid7

from pydantic import BaseModel, ConfigDict, Field

type ContentBlock = dict[str, Any]


class Value(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


class Usage(Value):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


class Message(Value):
    role: Role
    content: str | list[ContentBlock]


class ToolSpec(Value):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(Value):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_block(cls, block: ContentBlock) -> Self:
        return cls(id=block["id"], name=block["name"], args=block.get("input", {}))


class ToolResult(Value):
    call_id: str
    content: str
    is_error: bool = False
    truncated: bool = False
    duration_s: float = 0.0
    peak_rss_bytes: int | None = None
    peak_vram_bytes: int | None = None

    def to_block(self) -> ContentBlock:
        return {
            "type": "tool_result",
            "tool_use_id": self.call_id,
            "content": self.content,
            "is_error": self.is_error,
        }


class ToolCallStatus(StrEnum):
    INTENT = "intent"
    DONE = "done"
    INTERRUPTED = "interrupted"


class ToolCallRecord(Value):
    call: ToolCall
    status: ToolCallStatus
    result: ToolResult | None = None


class ModelRequest(Value):
    system: str
    tools: list[ToolSpec] = Field(default_factory=list)
    messages: list[Message]


class ModelReply(Value):
    content: list[ContentBlock]
    stop_reason: StopReason
    usage: Usage = Field(default_factory=Usage)

    def tool_calls(self) -> list[ToolCall]:
        return [ToolCall.from_block(block) for block in self.content if block["type"] == "tool_use"]


class ModelStep(Value):
    message_index: int
    request_hash: str
    reply: ModelReply


class JobState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"


class Job(Value):
    id: str = Field(default_factory=lambda: uuid7().hex)
    command: str
    pid: int
    pgid: int
    log_path: Path
    exit_path: Path
    state: JobState = JobState.RUNNING
    exit_code: int | None = None


class Direction(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Submission(Value):
    id: str = Field(default_factory=lambda: uuid7().hex)
    path: Path
    sha256: str
    val_score: float
    direction: Direction
    notes: str = ""
    created_at: float

    def beats(self, other: Submission) -> bool:
        if self.direction is Direction.MINIMIZE:
            return self.val_score < other.val_score

        return self.val_score > other.val_score


class Checkpoint(Value):
    commit: str
    message_index: int


class CommandResult(Value):
    exit_code: int | None
    output: str
    truncated: bool = False
    timed_out: bool = False
    duration_s: float = 0.0


class GpuSnapshot(Value):
    index: int
    utilization_percent: float
    vram_used_bytes: int
    vram_total_bytes: int


class ResourceSnapshot(Value):
    cpu_percent: float
    ram_used_bytes: int
    ram_total_bytes: int
    gpus: tuple[GpuSnapshot, ...] = ()

    @property
    def ram_used_fraction(self) -> float:
        return self.ram_used_bytes / self.ram_total_bytes if self.ram_total_bytes else 0.0
