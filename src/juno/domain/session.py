from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar
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
    state: State = State.STARTING
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def terminal(self) -> bool:
        return self.state in self.TERMINAL

    def remaining(self, now: float) -> float:
        return max(0.0, self.deadline - now)

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
