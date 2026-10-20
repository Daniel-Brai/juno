from collections.abc import Sequence
from typing import ClassVar

from pydantic import ValidationError

from juno.domain.commands import TodoItem, UpdateTodos
from juno.domain.values import Message, Role
from juno.tools.registry import Tool


class TodoTools:
    MARKS: ClassVar[dict[str, str]] = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}

    def __init__(self, items: Sequence[TodoItem] = ()) -> None:
        self.items = list(items)

    @classmethod
    def from_history(cls, messages: Sequence[Message]) -> TodoTools:
        for message in reversed(messages):
            if message.role is not Role.ASSISTANT or isinstance(message.content, str):
                continue

            for block in reversed(message.content):
                if block.get("type") == "tool_use" and block.get("name") == "todo":
                    try:
                        return cls(UpdateTodos.model_validate(block.get("input")).items)
                    except ValidationError:
                        continue

        return cls()

    def tools(self) -> list[Tool[UpdateTodos]]:
        return [
            Tool(
                name="todo",
                description=(
                    "Replace the session's task list. Send every item each time, marking one "
                    "in_progress while you work on it and done when it is finished."
                ),
                input_model=UpdateTodos,
                handler=self.update,
                read_only=True,
            )
        ]

    async def update(self, args: UpdateTodos) -> str:
        self.items = list(args.items)
        return self.render() or "the list is empty"

    def render(self) -> str:
        return "\n".join(f"{self.MARKS[item.status]} {item.text}" for item in self.items)
