import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, get_args

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message as UiMessage
from textual.widgets import Collapsible, Markdown, Static, TextArea
from textual.worker import Worker

from juno.bootstrap import Adapters, ChatSetup, Harness
from juno.config import HarnessConfig
from juno.domain.events import (
    BudgetSpent,
    CacheMissed,
    Event,
    ModelCalled,
    ResourcePressure,
    SessionEnded,
    ToolFinished,
    ToolStarted,
)
from juno.domain.session import Session
from juno.domain.values import Answer, Message, Role, RunPaths, ToolCall, ToolResult, Usage
from juno.projects import ProjectSessions
from juno.service.drivers import HumanDriver
from juno.tools.permissions import Mode, Permissions


@dataclass
class ChatLauncher:
    config: HarnessConfig
    sessions: ProjectSessions
    permissions: Permissions
    adapters: Adapters = field(default_factory=Adapters)

    async def open(self, setup: ChatSetup, paths: RunPaths, *, resume: bool) -> Harness:
        opener = Harness.resume_chat if resume else Harness.start_chat
        return await opener(config=self.config, chat=setup, paths=paths, adapters=self.adapters)


class Composer(TextArea):
    class Submitted(UiMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            self.clear()
            return

        if event.key in {"shift+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return

        await super()._on_key(event)


class ToolBlock(Collapsible):
    BODY_LIMIT: ClassVar[int] = 4_000

    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self._body = Static(self.describe(call), classes="tool-body", markup=False)
        super().__init__(self._body, title=f"… {self.title_for(call)}", collapsed=True)

    def finish(self, result: ToolResult) -> None:
        mark = "✗" if result.is_error else "✓"
        self.title = f"{mark} {self.title_for(self.call)}"
        text = result.content

        if len(text) > self.BODY_LIMIT:
            text = f"{text[: self.BODY_LIMIT]}\n… {len(result.content) - self.BODY_LIMIT} more characters"

        self._body.update(f"{self.describe(self.call)}\n\n{text}".strip())

        if self.call.name == "todo":
            self.collapsed = False

    @staticmethod
    def title_for(call: ToolCall) -> str:
        args = call.args

        match call.name:
            case "bash":
                return f"$ {str(args.get('command', '')).splitlines()[0][:100]}"
            case "editor":
                return f"{args.get('action', 'edit')} {args.get('path', '')}"
            case "read" | "glob":
                return f"{call.name} {args.get('path') or args.get('pattern', '')}"
            case "grep":
                return f"grep {args.get('pattern', '')}"
            case _:
                return call.name

    @staticmethod
    def describe(call: ToolCall) -> str:
        args = call.args

        if call.name == "editor" and args.get("action") == "str_replace":
            removed = [f"- {line}" for line in str(args.get("old_str", "")).splitlines()]
            added = [f"+ {line}" for line in str(args.get("new_str", "")).splitlines()]
            return "\n".join(removed + added)

        if call.name == "bash":
            return f"$ {args.get('command', '')}"

        return "" if call.name in {"read", "glob", "grep", "todo"} else json.dumps(args)[:400]


class ApprovalPrompt(Static, can_focus=True):
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("y", "answer('once')", "Allow once"),
        ("a", "answer('always')", "Always allow"),
        ("n", "answer('deny')", "Deny"),
    ]

    def __init__(self, call: ToolCall, reason: str, answer: asyncio.Future[Answer]) -> None:
        self._answer = answer
        what = ToolBlock.title_for(call)
        super().__init__(
            f"Allow {what}?  ({reason})\n[y] once   [a] always   [n] no",
            classes="approval",
            markup=False,
        )

    def action_answer(self, choice: str) -> None:
        if not self._answer.done():
            self._answer.set_result(Answer(choice))


class TuiApprover:
    def __init__(self, app: JunoApp) -> None:
        self._app = app

    async def approve(self, call: ToolCall, reason: str, /) -> Answer:
        return await self._app.ask(call, reason)


class Transcript:
    @classmethod
    def write(cls, session: Session, target: str, default: Path) -> Path:
        path = Path(target).expanduser() if target else default
        path.write_text(cls.markdown(session))
        return path

    @staticmethod
    def markdown(session: Session) -> str:
        parts = [f"# juno session {session.id}"]

        for message in session.messages:
            parts += Transcript._message(message)

        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _message(message: Message) -> list[str]:
        if isinstance(message.content, str):
            speaker = "You" if message.role is Role.USER else "juno"
            return [f"## {speaker}\n\n{message.content}"]

        parts: list[str] = []

        for block in message.content:
            match block.get("type"):
                case "text":
                    parts.append(f"## juno\n\n{block.get('text', '')}")
                case "tool_use":
                    call = ToolCall.from_block(block)
                    parts.append(f"**{ToolBlock.title_for(call)}**")
                case "tool_result":
                    content = str(block.get("content", ""))[:2_000]
                    parts.append(f"```\n{content}\n```")

        return parts


class JunoApp(App[None]):
    TITLE = "juno"
    CSS = """
    #transcript { height: 1fr; padding: 0 1; }
    .user { margin: 1 0 0 0; color: $accent; text-style: bold; }
    .assistant { margin: 0; }
    .notice { color: $text-muted; margin: 0 0 0 2; }
    .warning { color: $warning; margin: 0 0 0 2; }
    .approval { border: round $warning; padding: 0 1; margin: 1 0; }
    .approval:focus { border: round $accent; }
    ToolBlock { margin: 0; padding: 0; border: none; }
    #composer { height: auto; max-height: 10; min-height: 3; border: round $primary; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    """
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "interrupt", "Stop", priority=True),
        Binding("ctrl+o", "toggle_last", "Fold"),
        Binding("ctrl+d", "quit", "Quit"),
    ]
    COMMAND_HELP: ClassVar[str] = (
        "Enter sends, Shift+Enter or Ctrl+J adds a line, Esc stops the current turn, "
        "Ctrl+O folds the last tool, Ctrl+D quits.\n"
        "/model NAME · /effort LEVEL · /mode ask|accept-edits|plan|yolo · /cost · /todo · "
        "/undo · /export [PATH] · /clear · /quit"
    )

    def __init__(self, launcher: ChatLauncher, *, resume: RunPaths | None = None) -> None:
        super().__init__()
        self.launcher = launcher
        self.driver = HumanDriver()
        self.harness: Harness | None = None
        self.paths = resume
        self._resume = resume is not None
        self._worker: Worker[None] | None = None
        self._blocks: dict[str, ToolBlock] = {}
        self._context_tokens = 0
        self._activity = "starting"
        self._note: str | None = None
        self._prompt: ApprovalPrompt | None = None
        self.commands: dict[str, Callable[[str], Awaitable[None]]] = {
            "help": self._help,
            "model": self._model,
            "effort": self._effort,
            "mode": self._mode,
            "cost": self._cost,
            "todo": self._todo,
            "undo": self._undo,
            "export": self._export,
            "clear": self._clear,
            "quit": self._quit,
        }

    @property
    def busy(self) -> bool:
        return self.harness is not None and not self.driver.waiting

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield Composer(id="composer", soft_wrap=True, show_line_numbers=False)
        yield Static(id="status")

    async def on_mount(self) -> None:
        await self._open(self.paths, resume=self._resume)
        self.query_one(Composer).focus()

    async def on_unmount(self) -> None:
        await self._close()

    async def ask(self, call: ToolCall, reason: str) -> Answer:
        answer: asyncio.Future[Answer] = asyncio.get_running_loop().create_future()
        prompt = ApprovalPrompt(call, reason, answer)
        self._prompt = prompt
        await self._transcript.mount(prompt)
        self._transcript.scroll_end(animate=False)
        prompt.focus()
        self._set_activity("waiting for your approval")

        try:
            choice = await answer
        finally:
            self._prompt = None
            await prompt.remove()
            self.query_one(Composer).focus()

        self._set_activity(f"running {call.name}")
        return choice

    async def on_composer_submitted(self, message: Composer.Submitted) -> None:
        text = message.text.strip()

        if not text:
            return

        if text.startswith("/"):
            name, _, argument = text[1:].partition(" ")
            command = self.commands.get(name)

            if command is None:
                await self.notice(f"no command /{name}; /help lists them")
            else:
                await command(argument.strip())

            return

        await self._transcript.mount(Static(text, classes="user", markup=False))
        self._transcript.scroll_end(animate=False)

        if self.busy:
            await self.notice("queued; juno reads it when the current turn ends")

        sent = f"{self._note}\n\n{text}" if self._note else text
        self._note = None
        self.driver.inbox.put_nowait(sent)
        self._set_activity("thinking")

    async def action_interrupt(self) -> None:
        if self.harness is None or not self.busy:
            return

        await self._stop_worker()
        await self.harness.loop.interrupt()

        for block in self._blocks.values():
            if block.title.startswith("…"):
                block.title = f"■ {ToolBlock.title_for(block.call)}"

        await self.notice("stopped; what next?")
        self._run()

    def action_toggle_last(self) -> None:
        blocks = list(self.query(ToolBlock))

        if blocks:
            blocks[-1].collapsed = not blocks[-1].collapsed

    async def action_quit(self) -> None:
        await self._close()
        self.exit()

    async def notice(self, text: str, *, warning: bool = False) -> None:
        widget = Static(text, classes="warning" if warning else "notice", markup=False)
        await self._transcript.mount(widget)
        self._transcript.scroll_end(animate=False)

    @property
    def _transcript(self) -> VerticalScroll:
        return self.query_one("#transcript", VerticalScroll)

    async def _open(self, paths: RunPaths | None, *, resume: bool) -> None:
        setup = ChatSetup(
            driver=self.driver,
            approver=TuiApprover(self),
            permissions=self.launcher.permissions,
            instructions=self.launcher.sessions.instructions(),
        )
        self.paths = paths or self.launcher.sessions.new()
        self.harness = await self.launcher.open(setup, self.paths, resume=resume)
        self.harness.bus.subscribe(Event, self._on_event)

        if resume:
            await self._replay(self.harness.session)

        self._run()

    def _run(self) -> None:
        self._set_activity("ready")
        self._worker = self.run_worker(self._drive(), group="loop", exit_on_error=False)

    async def _drive(self) -> None:
        if self.harness is None:
            return

        try:
            await self.harness.run()
        except Exception as error:
            await self.notice(f"the turn failed: {type(error).__name__}: {error}", warning=True)

    async def _stop_worker(self) -> None:
        if self._worker is None or self._worker.is_finished:
            return

        self._worker.cancel()

        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self._worker.wait()

    async def _close(self) -> None:
        if self.harness is None:
            return

        if self.busy:
            await self._stop_worker()
            await self.harness.loop.interrupt()
            self._run()

        self.driver.inbox.put_nowait(None)

        if self._worker is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._worker.wait()

        await self.harness.close()
        self.harness = None

    async def _on_event(self, event: Event) -> None:
        match event:
            case ModelCalled():
                await self._show_reply(event)
            case ToolStarted():
                block = ToolBlock(event.call)
                self._blocks[event.call.id] = block
                await self._transcript.mount(block)
                self._set_activity(f"running {event.call.name}")
            case ToolFinished():
                if finished := self._blocks.get(event.result.call_id):
                    finished.finish(event.result)

                self._set_activity("thinking")
            case BudgetSpent():
                await self.notice(f"the {event.limit} budget is spent", warning=True)
            case SessionEnded() if event.failure:
                await self.notice(f"the session failed: {event.failure}", warning=True)
            case CacheMissed() | ResourcePressure():
                await self.notice(type(event).__name__, warning=True)
            case _:
                pass

        self._transcript.scroll_end(animate=False)

    async def _show_reply(self, event: ModelCalled) -> None:
        if self.harness is None:
            return

        usage = event.usage
        self._context_tokens = (
            usage.input_tokens
            + usage.cache_read_input_tokens
            + usage.cache_creation_input_tokens
            + usage.output_tokens
        )
        message = self.harness.session.messages[event.message_index]

        if text := self._text_of(message):
            await self._transcript.mount(Markdown(text, classes="assistant"))

        calls = not isinstance(message.content, str) and any(
            block.get("type") == "tool_use" for block in message.content
        )
        self._set_activity("working" if calls else "ready")

    async def _replay(self, session: Session) -> None:
        for message in session.messages:
            if isinstance(message.content, str):
                classes = "user" if message.role is Role.USER else "assistant"
                await self._transcript.mount(Static(message.content, classes=classes, markup=False))
                continue

            if text := self._text_of(message):
                await self._transcript.mount(Markdown(text, classes="assistant"))

            for block in message.content:
                if block.get("type") == "tool_use":
                    call = ToolCall.from_block(block)
                    self._blocks[call.id] = ToolBlock(call)
                    await self._transcript.mount(self._blocks[call.id])
                elif block.get("type") == "tool_result" and (
                    tool := self._blocks.get(str(block.get("tool_use_id")))
                ):
                    tool.finish(
                        ToolResult(
                            call_id=tool.call.id,
                            content=str(block.get("content", "")),
                            is_error=bool(block.get("is_error")),
                        )
                    )

        await self.notice(f"continuing session {session.id}")

    @staticmethod
    def _text_of(message: Message) -> str:
        if message.role is not Role.ASSISTANT:
            return ""

        if isinstance(message.content, str):
            return message.content

        texts = [str(b.get("text", "")) for b in message.content if b.get("type") == "text"]
        return "\n\n".join(text for text in texts if text.strip())

    def _set_activity(self, activity: str) -> None:
        self._activity = activity
        self._refresh_status()

    def _refresh_status(self) -> None:
        usage = self.harness.session.usage if self.harness else Usage()
        parts = [
            self.launcher.config.model.name,
            str(self.launcher.permissions.mode),
            f"{self._context_tokens:,} tokens in context",
        ]

        if usage.cost_usd:
            parts.append(f"${usage.cost_usd:.4f}")

        parts.append(self._activity)
        self.query_one("#status", Static).update(" · ".join(parts))

    async def _help(self, _: str) -> None:
        await self.notice(self.COMMAND_HELP)

    async def _model(self, name: str) -> None:
        await self._switch({"name": name} if name else {}, f"model is {name}")

    async def _effort(self, level: str) -> None:
        levels = get_args(get_args(self.launcher.config.model.model_fields["effort"].annotation))
        allowed = levels or ("low", "medium", "high", "xhigh", "max")

        if level not in allowed:
            await self.notice(f"/effort takes one of {', '.join(allowed)}")
            return

        await self._switch({"effort": level}, f"effort is {level}")

    async def _switch(self, update: dict[str, Any], done: str) -> None:
        if not update or self.harness is None:
            await self.notice(f"now using {self.launcher.config.model.name}")
            return

        model = self.launcher.config.model.model_copy(update=update)
        self.launcher.config = self.launcher.config.model_copy(update={"model": model})
        previous = self.harness.loop.llm

        try:
            self.harness.loop.llm = Harness.llm_for(self.launcher.config, self.harness.session)
        except (RuntimeError, ValueError) as error:
            await self.notice(str(error), warning=True)
            return

        if previous is not self.harness.llm:
            await previous.close()

        await self.notice(f"{done}; it applies from the next turn")
        self._refresh_status()

    async def _mode(self, mode: str) -> None:
        try:
            self.launcher.permissions.mode = Mode(mode)
        except ValueError:
            await self.notice(f"/mode takes one of {', '.join(item.value for item in Mode)}")
            return

        await self.notice(f"permission mode is {mode}")
        self._refresh_status()

    async def _cost(self, _: str) -> None:
        usage = self.harness.session.usage if self.harness else Usage()
        lines = [
            f"input {usage.input_tokens:,} · cache writes {usage.cache_creation_input_tokens:,} · "
            f"cache reads {usage.cache_read_input_tokens:,} · output {usage.output_tokens:,}",
            f"cost ${usage.cost_usd:.4f}" if usage.cost_usd else "no cost reported yet",
        ]
        await self.notice("\n".join(lines))

    async def _todo(self, _: str) -> None:
        todo = self.harness.todo if self.harness else None
        await self.notice((todo.render() if todo else "") or "the todo list is empty")

    async def _undo(self, _: str) -> None:
        if self.harness is None or self.busy:
            await self.notice("wait for the turn to end, or press Esc, before /undo")
            return

        async with self.harness.store.transaction() as tx:
            checkpoints = await tx.checkpoints.all(self.harness.session.id)

        if not checkpoints or self.harness.checkpointer is None:
            await self.notice("there is no checkpoint to go back to")
            return

        changed = await self.harness.checkpointer.restore(checkpoints[-1].commit)

        if not changed:
            await self.notice("nothing changed since the last checkpoint")
            return

        self._note = (
            "(I undid the changes from your last turn, restoring: " + ", ".join(changed) + ")"
        )
        await self.notice(f"restored {len(changed)} files: {', '.join(changed)}")

    async def _export(self, target: str) -> None:
        if self.harness is None or self.paths is None:
            return

        path = await asyncio.to_thread(
            Transcript.write, self.harness.session, target, self.paths.root / "transcript.md"
        )
        await self.notice(f"wrote {path}")

    async def _clear(self, _: str) -> None:
        await self._close()
        await self._transcript.remove_children()
        self._blocks.clear()
        self._context_tokens = 0
        self.driver = HumanDriver()
        await self._open(None, resume=False)
        await self.notice("started a new session")

    async def _quit(self, _: str) -> None:
        await self.action_quit()
