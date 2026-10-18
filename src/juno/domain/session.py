from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid7

from pydantic import BaseModel, ConfigDict, Field

from juno.domain.values import ContentBlock, Message, ModelReply, Role, ToolCall, ToolResult, Usage


class State(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    AWAITING_CONTINUE = "awaiting_continue"
    WRAPPING_UP = "wrapping_up"
    DONE = "done"
    FAILED = "failed"


class Limit(StrEnum):
    TIME = "time"
    COST = "cost"


class Trigger(StrEnum):
    START = "start"
    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    CONTINUE = "continue"
    BUDGET_SPENT = "budget_spent"
    FINISHED = "finished"
    FATAL = "fatal"


class InvalidTransition(Exception):
    def __init__(self, state: State, trigger: Trigger) -> None:
        super().__init__(f"cannot apply {trigger} in state {state}")
        self.state = state
        self.trigger = trigger


class InvalidHistory(Exception):
    pass


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    TRANSITIONS: ClassVar[Mapping[tuple[State, Trigger], State]] = MappingProxyType(
        {
            (State.STARTING, Trigger.START): State.RUNNING,
            (State.RUNNING, Trigger.TOOL_USE): State.RUNNING,
            (State.RUNNING, Trigger.END_TURN): State.AWAITING_CONTINUE,
            (State.AWAITING_CONTINUE, Trigger.CONTINUE): State.RUNNING,
            (State.RUNNING, Trigger.BUDGET_SPENT): State.WRAPPING_UP,
            (State.AWAITING_CONTINUE, Trigger.BUDGET_SPENT): State.WRAPPING_UP,
            (State.WRAPPING_UP, Trigger.FINISHED): State.DONE,
        }
    )
    TERMINAL: ClassVar[frozenset[State]] = frozenset({State.DONE, State.FAILED})

    id: str = Field(default_factory=lambda: uuid7().hex)
    task_id: str
    seed: int = 0
    started_at: float
    deadline: float
    cost_limit: float | None = None
    state: State = State.STARTING
    failure: str | None = None
    last_active_at: float | None = None
    paused_s: float = 0.0
    setup: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def terminal(self) -> bool:
        return self.state in self.TERMINAL

    def remaining(self, now: float) -> float:
        return max(0.0, self.deadline - now)

    def limit_reached(self, now: float) -> Limit | None:
        if self.remaining(now) <= 0:
            return Limit.TIME

        if self.cost_limit is not None and self.usage.cost_usd >= self.cost_limit:
            return Limit.COST

        return None

    def elapsed(self, now: float) -> float:
        end = self.last_active_at if self.terminal and self.last_active_at is not None else now
        return max(0.0, end - self.started_at - self.paused_s)

    def touch(self, now: float) -> None:
        self.last_active_at = now

    def downtime(self, now: float) -> float:
        if self.terminal or self.last_active_at is None:
            return 0.0

        return max(0.0, now - self.last_active_at)

    def pause_for_downtime(self, now: float, busy_s: float = 0.0) -> float:
        pause = max(0.0, self.downtime(now) - busy_s)
        self.deadline += pause
        self.paused_s += pause

        if not self.terminal:
            self.last_active_at = now

        return pause

    def setup_changes(self, setup: Mapping[str, Any]) -> list[str]:
        return sorted(self._differences(self.setup, setup))

    @classmethod
    def _differences(
        cls, old: Mapping[str, Any], new: Mapping[str, Any], prefix: str = ""
    ) -> set[str]:
        changed: set[str] = set()

        for key in old.keys() | new.keys():
            before, after = old.get(key), new.get(key)

            if isinstance(before, Mapping) and isinstance(after, Mapping):
                changed |= cls._differences(before, after, f"{prefix}{key}.")
            elif before != after:
                changed.add(f"{prefix}{key}")

        return changed

    def fail(self, reason: str) -> None:
        self.fire(Trigger.FATAL)
        self.failure = reason

    def fire(self, trigger: Trigger) -> State:
        if self.terminal:
            raise InvalidTransition(self.state, trigger)

        if trigger is Trigger.FATAL:
            self.state = State.FAILED
            return self.state

        next_state = self.TRANSITIONS.get((self.state, trigger))

        if next_state is None:
            raise InvalidTransition(self.state, trigger)

        self.state = next_state
        return self.state

    def pending_tool_calls(self) -> list[ToolCall]:
        if not self.messages or self.messages[-1].role is not Role.ASSISTANT:
            return []

        content = self.messages[-1].content

        if isinstance(content, str):
            return []

        return [ToolCall.from_block(block) for block in content if block["type"] == "tool_use"]

    def append_user(self, content: str | list[ContentBlock]) -> Message:
        return self._append(Message(role=Role.USER, content=content))

    def append_assistant(self, reply: ModelReply) -> Message:
        message = self._append(Message(role=Role.ASSISTANT, content=reply.content))
        self.usage = self.usage + reply.usage
        return message

    def append_tool_results(self, results: Sequence[ToolResult]) -> Message:
        expected = {call.id for call in self.pending_tool_calls()}
        received = {result.call_id for result in results}

        if not expected or expected != received:
            raise InvalidHistory(
                f"tool results {sorted(received)} do not match calls {sorted(expected)}"
            )

        return self.append_user([result.to_block() for result in results])

    def _append(self, message: Message) -> Message:
        previous = self.messages[-1].role if self.messages else Role.ASSISTANT

        if message.role is previous:
            raise InvalidHistory(f"two {message.role} messages in a row")

        self.messages.append(message)
        return message
