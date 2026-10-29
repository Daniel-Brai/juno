import asyncio
import contextlib
import json
import time
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, get_args

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as UiMessage
from textual.widget import Widget
from textual.widgets import Collapsible, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker

from juno.bootstrap import Adapters, ChatSetup, Harness
from juno.config import Durations, HarnessConfig, McpServer, ModelNames, ModelSettings
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
from juno.domain.values import (
    Answer,
    Message,
    ModelCard,
    Role,
    RunPaths,
    ToolCall,
    ToolResult,
    Usage,
)
from juno.entrypoints.attachments import Attachments
from juno.eval.runs import RunFolder
from juno.ports import ModelRejected
from juno.projects import ProjectSessions
from juno.service.compaction import Compactor
from juno.service.drivers import HumanDriver
from juno.tools.permissions import Mode, Permissions


@dataclass
class ChatLauncher:
    config: HarnessConfig
    sessions: ProjectSessions
    permissions: Permissions
    adapters: Adapters = field(default_factory=Adapters)
    mcp: Mapping[str, McpServer] = field(default_factory=dict)

    async def open(self, setup: ChatSetup, paths: RunPaths, *, resume: bool) -> Harness:
        opener = Harness.resume_chat if resume else Harness.start_chat
        return await opener(config=self.config, chat=setup, paths=paths, adapters=self.adapters)


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    about: str
    run: Callable[[str], Awaitable[None]]
    hint: str = ""
    choices: Callable[[], list[str]] | None = None
    current: Callable[[], str] | None = None

    @property
    def needs_argument(self) -> bool:
        return bool(self.hint) and not self.hint.startswith("[")

    def label(self) -> Text:
        label = Text(f"/{self.name}", style="bold")

        if self.hint:
            label.append(f" {self.hint}", style="italic")

        label.append(f"  {self.about}", style="dim")
        return label


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    label: Text
    submit: bool


class CommandMenu(OptionList):
    LIMIT: ClassVar[int] = 40

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.matches: list[Completion] = []
        self.display = False

    def offer(self, matches: list[Completion]) -> None:
        self.matches = matches[: self.LIMIT]
        self.clear_options()
        self.add_options([Option(match.label) for match in self.matches])
        self.display = bool(self.matches)
        self.highlighted = 0 if self.matches else None

    def close(self) -> None:
        self.matches = []
        self.display = False

    @property
    def chosen(self) -> Completion | None:
        if not self.display or self.highlighted is None:
            return None

        return self.matches[self.highlighted]

    @classmethod
    def completions(cls, text: str, commands: dict[str, SlashCommand]) -> list[Completion]:
        if not text.startswith("/") or "\n" in text:
            return []

        name, space, argument = text[1:].partition(" ")

        if not space:
            return [
                Completion(f"/{command.name} ", command.label(), submit=not command.needs_argument)
                for command in commands.values()
                if command.name.startswith(name)
            ]

        command = commands.get(name)

        if command is None or command.choices is None or " " in argument:
            return []

        current = command.current() if command.current else None
        return [
            Completion(f"/{name} {choice}", cls._choice(choice, current), submit=True)
            for choice in command.choices()
            if argument.lower() in choice.lower()
        ]

    @staticmethod
    def _choice(choice: str, current: str | None) -> Text:
        label = Text(choice)

        if choice == current:
            label.append("  current", style="dim")

        return label


class SessionPicker(OptionList):
    PREVIEW: ClassVar[int] = 60

    def __init__(self, sessions: list[RunPaths]) -> None:
        super().__init__(*(self._option(paths) for paths in sessions), classes="picker")
        self.sessions = {paths.root.name: paths for paths in sessions}

    def _option(self, paths: RunPaths) -> Option:
        folder = RunFolder(paths.root)
        session = folder.session()
        count, first = folder.opening(session.id) if session else (0, "")
        age = Durations.format(time.time() - session.started_at) if session else "?"
        preview = Mentions.typed(first).splitlines()[0] if first.strip() else "(no messages)"
        label = Text(f"{age} ago", style="dim")
        label.append(f"  {count} messages  ", style="dim")
        label.append(preview[: self.PREVIEW])
        return Option(label, id=paths.root.name)


class Composer(TextArea):
    class Submitted(UiMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event: events.Key) -> None:
        menu = self.screen.query_one(CommandMenu)

        if menu.display and event.key in {"up", "down", "tab", "enter"}:
            event.stop()
            event.prevent_default()
            await self._steer(menu, event.key)
            return

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

    def complete(self, text: str) -> None:
        self.text = text
        self.move_cursor(self.document.end)

    async def _steer(self, menu: CommandMenu, key: str) -> None:
        if key == "up":
            menu.action_cursor_up()
        elif key == "down":
            menu.action_cursor_down()
        elif (chosen := menu.chosen) is not None:
            self.complete(chosen.text)
            menu.close()

            if key == "enter" and chosen.submit:
                self.post_message(self.Submitted(self.text))
                self.clear()


class Glyphs:
    NARROW: ClassVar[dict[str, str]] = {
        "\u2705": "\u2713",
        "\u2714": "\u2713",
        "\u274c": "\u2717",
        "\u274e": "\u2717",
        "\u2b50": "\u2605",
        "\U0001f31f": "\u2605",
        "\u2728": "*",
        "\u26a1": "!",
    }
    VARIATION: ClassVar[str] = "\ufe0f"

    @classmethod
    def narrow(cls, text: str) -> str:
        return "".join(cls._one(char) for char in text.replace(cls.VARIATION, ""))

    @classmethod
    def _one(cls, char: str) -> str:
        if char in cls.NARROW:
            return cls.NARROW[char]

        if cell_len(char) == 2 and unicodedata.category(char) == "So":
            return ""

        return char


class ToolLabels:
    QUIET: ClassVar[frozenset[str]] = frozenset({"read", "glob", "grep", "todo"})

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def title(self, call: ToolCall) -> str:
        args = call.args

        match call.name:
            case "bash":
                return f"$ {str(args.get('command', '')).splitlines()[0][:100]}"
            case "editor":
                return f"{args.get('action', 'edit')} {self._path(args.get('path'))}"
            case "read" | "glob":
                return f"{call.name} {self._path(args.get('path')) or args.get('pattern', '')}"
            case "grep":
                return f"grep {args.get('pattern', '')}"
            case _:
                return call.name

    def headline(self, call: ToolCall) -> tuple[str, str]:
        args = call.args

        match call.name:
            case "bash":
                return "Bash", str(args.get("command", "")).splitlines()[0][:100]
            case "editor":
                action = str(args.get("action", "edit"))
                verbs = {
                    "str_replace": "Edit",
                    "create": "Write",
                    "view": "View",
                    "insert": "Insert",
                }
                return verbs.get(action, "Edit"), self._path(args.get("path"))
            case "read":
                return "Read", self._path(args.get("path"))
            case "glob" | "grep":
                return call.name.capitalize(), str(args.get("pattern", ""))
            case "todo":
                return "Update todos", ""
            case _:
                return call.name, ""

    def body(self, call: ToolCall) -> str:
        args = call.args

        if call.name == "editor" and args.get("action") == "str_replace":
            removed = [f"- {line}" for line in str(args.get("old_str", "")).splitlines()]
            added = [f"+ {line}" for line in str(args.get("new_str", "")).splitlines()]
            return "\n".join(removed + added)

        if call.name == "bash":
            return f"$ {args.get('command', '')}"

        return "" if call.name in self.QUIET else json.dumps(args)[:400]

    def _path(self, raw: object) -> str:
        if not isinstance(raw, str) or not raw:
            return ""

        path = Path(raw)

        if path.is_absolute() and path.is_relative_to(self._root):
            return str(path.relative_to(self._root)) or "."

        return raw


class ToolBlock(Vertical):
    BODY_LIMIT: ClassVar[int] = 4_000
    SUMMARY: ClassVar[int] = 90
    RUNNING: ClassVar[Style] = Style(color="#8A8A8A")
    OK: ClassVar[Style] = Style(color="#4EBA65")
    FAILED: ClassVar[Style] = Style(color="#E5534B")

    def __init__(self, call: ToolCall, labels: ToolLabels) -> None:
        super().__init__(classes="tool")
        self.call = call
        self.labels = labels
        self.running = True
        self._header = Static(self._headline(self.RUNNING))
        self._body = Static(labels.body(call), markup=False)
        self._output = Collapsible(
            self._body,
            title="\u23bf running",
            collapsed=True,
            collapsed_symbol="\u25b8",
            expanded_symbol="\u25be",
        )

    def compose(self) -> ComposeResult:
        yield self._header
        yield self._output

    def toggle(self) -> None:
        self._output.collapsed = not self._output.collapsed

    def finish(self, result: ToolResult) -> None:
        self.running = False
        self._header.update(self._headline(self.FAILED if result.is_error else self.OK))
        self._output.title = f"\u23bf {self.summary(result.content)}"
        text = "" if self.fits_title(result.content) else result.content

        if len(text) > self.BODY_LIMIT:
            text = f"{text[: self.BODY_LIMIT]}\n… {len(result.content) - self.BODY_LIMIT} more characters"

        self._body.update(Glyphs.narrow(f"{self.labels.body(self.call)}\n\n{text}".strip()))

        if self.call.name == "todo":
            self._output.collapsed = False

    def stopped(self) -> None:
        if self.running:
            self.running = False
            self._header.update(self._headline(self.RUNNING, dot="\u25a0"))
            self._output.title = "\u23bf stopped"

    @classmethod
    def fits_title(cls, content: str) -> bool:
        return "\n" not in content.strip() and len(content.strip()) <= cls.SUMMARY

    @classmethod
    def summary(cls, content: str) -> str:
        lines = [line for line in content.splitlines() if line.strip()]

        if not lines:
            return "no output"

        first = Glyphs.narrow(lines[0].strip())[: cls.SUMMARY]
        return f"{first}  +{len(lines) - 1} lines" if len(lines) > 1 else first

    def _headline(self, style: Style, *, dot: str = "\u25cf") -> Text:
        name, argument = self.labels.headline(self.call)
        line = Text(f"{dot} ", style)
        line.append(name, Style(bold=True))

        if argument:
            line.append(f"({argument})", Style(color="#B5B5B5"))

        return line


class Reply(Horizontal):
    def __init__(self, text: str) -> None:
        super().__init__(classes="reply")
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("\u25cf", classes="bullet")
        yield Markdown(self._text)


class ActivityLine(Static):
    TICK_S: ClassVar[float] = 0.25
    LIT: ClassVar[Style] = Style(color="#E8855E", bold=True)
    DIM: ClassVar[Style] = Style(color="#6B6B6B")

    def __init__(self) -> None:
        super().__init__(classes="activity")
        self.label = "Thinking"
        self._started = time.monotonic()
        self._tick = 0

    def on_mount(self) -> None:
        self._draw()
        self.set_interval(self.TICK_S, self._advance)

    def describe(self, label: str) -> None:
        self.label = label
        self._draw()

    def _advance(self) -> None:
        self._tick += 1
        self._draw()

    def _draw(self) -> None:
        text = Text()

        for dot in range(3):
            text.append("•", self.LIT if dot == self._tick % 3 else self.DIM)

        text.append(f"  {self.label}…", Style(color="#B5B5B5"))
        text.append(f"  {Durations.format(time.monotonic() - self._started)}", self.DIM)
        self.update(text)


class Mentions:
    STYLE: ClassVar[Style] = Style(color="#6CB6FF", bold=True, underline=True)
    FILES: ClassVar[str] = "\n\n<file path="

    @classmethod
    def prompt(cls, text: str) -> Text:
        line = Text("> ", Style(color="#8A8A8A"))
        line.append_text(cls.highlight(text))
        return line

    @classmethod
    def highlight(cls, text: str) -> Text:
        styled = Text()

        for piece in text.split(" "):
            if styled:
                styled.append(" ")

            if piece.startswith("@") and len(piece) > 1:
                styled.append(piece, cls.STYLE)
            else:
                styled.append(piece)

        return styled

    @classmethod
    def paths(cls, lead: str, names: list[str]) -> Text:
        styled = Text(f"{lead} ")

        for index, name in enumerate(names):
            if index:
                styled.append(", ")

            styled.append(name, cls.STYLE)

        return styled

    @classmethod
    def typed(cls, sent: str) -> str:
        return sent.split(cls.FILES, 1)[0]


class ApprovalPrompt(Static, can_focus=True):
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("y", "answer('once')", "Allow once"),
        ("a", "answer('always')", "Always allow"),
        ("n", "answer('deny')", "Deny"),
    ]

    def __init__(self, what: str, reason: str, answer: asyncio.Future[Answer]) -> None:
        self._answer = answer
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
    def __init__(self, labels: ToolLabels) -> None:
        self._labels = labels

    def write(self, session: Session, target: str, default: Path) -> Path:
        path = Path(target).expanduser() if target else default
        path.write_text(self.markdown(session))
        return path

    def markdown(self, session: Session) -> str:
        parts = [f"# juno session {session.id}"]

        for message in session.messages:
            parts += self._message(message)

        return "\n\n".join(parts) + "\n"

    def _message(self, message: Message) -> list[str]:
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
                    parts.append(f"**{self._labels.title(call)}**")
                case "tool_result":
                    content = str(block.get("content", ""))[:2_000]
                    parts.append(f"```\n{content}\n```")

        return parts


class JunoApp(App[None]):
    TITLE = "juno"
    CSS = """
    #transcript { height: 1fr; padding: 0 1; }
    .user { margin: 1 0 0 0; padding: 0 1; background: $boost; color: $text; }
    .reply { height: auto; margin: 1 0 0 0; }
    .bullet { width: 2; color: $text; }
    .reply Markdown { margin: 0; padding: 0; }
    .tool { height: auto; margin: 1 0 0 0; }
    .tool Collapsible { border: none; padding: 0 0 0 2; margin: 0; background: transparent; }
    .tool CollapsibleTitle { color: $text-muted; padding: 0; }
    .notice { color: $text-muted; margin: 0 0 0 2; }
    .warning { color: $warning; margin: 0 0 0 2; }
    .approval { border: round $warning; padding: 0 1; margin: 1 0; }
    .activity { margin: 1 0 0 2; }
    .picker { height: auto; max-height: 12; border: round $accent; margin: 1 0; }
    .approval:focus { border: round $accent; }
    #menu { height: auto; max-height: 12; border: round $panel; margin: 0 1; }
    #composer { height: auto; max-height: 10; min-height: 3; border: round $primary; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    """
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "interrupt", "Stop", priority=True),
        Binding("ctrl+o", "toggle_last", "Fold"),
        Binding("ctrl+d", "quit", "Quit"),
    ]
    PROVIDERS: ClassVar[tuple[str, ...]] = ("anthropic", "openrouter")
    SETTLE_TRIES: ClassVar[int] = 100
    SETTLE_S: ClassVar[float] = 0.05
    KEYS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Enter", "send; picks the highlighted entry when the menu is open"),
        ("Shift+Enter, Ctrl+J", "new line"),
        ("/", "commands; Tab completes, Up and Down move"),
        ("@path", "attach a file from the project"),
        ("Esc", "close the menu, or stop the current turn"),
        ("Ctrl+O", "fold or unfold the last tool"),
        ("Ctrl+D", "quit"),
    )

    def __init__(self, launcher: ChatLauncher, *, resume: RunPaths | None = None) -> None:
        super().__init__()

        self.launcher = launcher
        self.labels = ToolLabels(launcher.sessions.project)
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
        self._line: ActivityLine | None = None
        self._models: list[ModelCard] = []

        listed = [
            SlashCommand("help", "list keys and commands", self._help),
            SlashCommand(
                "model",
                "switch model from the next turn",
                self._model,
                "NAME",
                choices=lambda: [card.name for card in self._models],
                current=lambda: self.launcher.config.model.name,
            ),
            SlashCommand(
                "provider",
                "talk to models directly or through OpenRouter",
                self._provider,
                "NAME",
                choices=lambda: list(self.PROVIDERS),
                current=lambda: self.launcher.config.model.provider,
            ),
            SlashCommand(
                "effort",
                "change reasoning effort",
                self._effort,
                "LEVEL",
                choices=self._efforts,
                current=lambda: self.launcher.config.model.effort,
            ),
            SlashCommand(
                "mode",
                "how much juno may do without asking",
                self._mode,
                "MODE",
                choices=lambda: [mode.value for mode in Mode],
                current=lambda: self.launcher.permissions.mode.value,
            ),
            SlashCommand("cost", "tokens and spend so far", self._cost),
            SlashCommand("todo", "show the task list", self._todo),
            SlashCommand("undo", "restore files from before the last turn", self._undo),
            SlashCommand("export", "write the transcript as Markdown", self._export, "[PATH]"),
            SlashCommand("resume", "pick an earlier session", self._pick_session, "[ID]"),
            SlashCommand("compact", "summarize, then continue fresh", self._compact, "[FOCUS]"),
            SlashCommand("clear", "start a new session", self._clear),
            SlashCommand("quit", "close juno", self._quit),
        ]
        self.commands = {command.name: command for command in listed}

    @property
    def busy(self) -> bool:
        return self.harness is not None and not self.driver.waiting

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield CommandMenu(id="menu")
        yield Composer(id="composer", soft_wrap=True, show_line_numbers=False)
        yield Static(id="status")

    async def on_mount(self) -> None:
        await self._open(self.paths, resume=self._resume)
        self.query_one(Composer).focus()

    async def on_unmount(self) -> None:
        await self._close()

    async def ask(self, call: ToolCall, reason: str) -> Answer:
        answer: asyncio.Future[Answer] = asyncio.get_running_loop().create_future()
        prompt = ApprovalPrompt(self.labels.title(call), reason, answer)
        self._prompt = prompt
        await self._add(prompt)
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
                await self.notice(f"no command /{name}; type / to see them")
            else:
                await command.run(argument.strip())

            return

        await self._add(Static(Mentions.prompt(text), classes="user"))
        self._transcript.scroll_end(animate=False)
        expanded = await asyncio.to_thread(Attachments(self.launcher.sessions.project).expand, text)

        if expanded.attached:
            await self.notice(Mentions.paths("attached", expanded.attached))

        if expanded.skipped:
            await self.notice(f"not attached: {'; '.join(expanded.skipped)}", warning=True)

        if self.busy:
            await self.notice("queued; juno reads it when the current turn ends")

        if self._line is None:
            self._line = ActivityLine()
            await self._transcript.mount(self._line)
            self._transcript.scroll_end(animate=False)

        sent = f"{self._note}\n\n{expanded.text}" if self._note else expanded.text
        self._note = None
        self.driver.inbox.put_nowait(sent)
        self._set_activity("thinking")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if not isinstance(event.text_area, Composer):
            return

        menu = self.query_one(CommandMenu)
        matches = CommandMenu.completions(event.text_area.text, self.commands)

        if matches:
            menu.offer(matches)
        else:
            menu.close()

    async def action_interrupt(self) -> None:
        menu = self.query_one(CommandMenu)

        if menu.display:
            menu.close()
            return

        if pickers := list(self.query(SessionPicker)):
            for picker in pickers:
                await picker.remove()

            self.query_one(Composer).focus()
            return

        if self.harness is None or not self.busy:
            return

        await self._stop_worker()
        await self.harness.loop.interrupt()

        for block in self._blocks.values():
            block.stopped()

        self._set_activity("ready")
        await self.notice("stopped; what next?")
        self._run()

    def action_toggle_last(self) -> None:
        blocks = list(self.query(ToolBlock))

        if blocks:
            blocks[-1].toggle()

    async def action_quit(self) -> None:
        await self._close()
        self.exit()

    async def notice(self, text: RenderableType, *, warning: bool = False) -> None:
        widget = Static(text, classes="warning" if warning else "notice", markup=False)
        await self._add(widget)
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
            mcp=self.launcher.mcp,
        )
        self.paths = paths or self.launcher.sessions.new()
        self.harness = await self.launcher.open(setup, self.paths, resume=resume)
        self.harness.bus.subscribe(Event, self._on_event)

        if resume:
            await self._replay(self.harness.session)

        for warning in self.harness.warnings:
            await self.notice(warning, warning=True)

        self._run()

        if not self._models:
            self.run_worker(self._load_models(), group="models", exit_on_error=False)

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
                block = ToolBlock(event.call, self.labels)
                self._blocks[event.call.id] = block
                await self._add(block)
                self._set_activity(f"running {event.call.name}")
            case ToolFinished():
                if finished := self._blocks.get(event.result.call_id):
                    finished.finish(event.result)

                self._set_activity("thinking")
            case BudgetSpent():
                await self.notice(f"the {event.limit} budget is spent", warning=True)
            case SessionEnded() if event.failure:
                await self.notice(f"the session failed: {event.failure}", warning=True)
            case CacheMissed():
                await self.notice(
                    f"the prompt cache missed {event.misses} times in a row, "
                    "so these turns cost more than usual",
                    warning=True,
                )
            case ResourcePressure():
                await self.notice(event.reason, warning=True)
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
            await self._add(Reply(Glyphs.narrow(text)))

        calls = not isinstance(message.content, str) and any(
            block.get("type") == "tool_use" for block in message.content
        )
        self._set_activity("working" if calls else "ready")

        if not calls and (fullness := self._fullness()) is not None:
            self.run_worker(self._auto_compact(fullness), group="compact", exit_on_error=False)

    async def _replay(self, session: Session) -> None:
        for message in session.messages:
            if isinstance(message.content, str) and message.role is Role.USER:
                typed = Mentions.typed(message.content)
                await self._add(Static(Mentions.prompt(typed), classes="user"))
                continue

            if isinstance(message.content, str):
                await self._add(Reply(Glyphs.narrow(message.content)))
                continue

            if text := self._text_of(message):
                await self._add(Reply(Glyphs.narrow(text)))

            for block in message.content:
                if block.get("type") == "tool_use":
                    call = ToolCall.from_block(block)
                    self._blocks[call.id] = ToolBlock(call, self.labels)
                    await self._add(self._blocks[call.id])
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

        if self._line is None:
            return

        if activity == "ready":
            self._line.remove()
            self._line = None
        elif activity.startswith("running "):
            self._line.describe(f"Running {activity.removeprefix('running ')}")
        else:
            self._line.describe(activity.capitalize())

    async def _add(self, widget: Widget) -> None:
        await self._transcript.mount(widget, before=self._line)
        self._transcript.scroll_end(animate=False)

    def _refresh_status(self) -> None:
        usage = self.harness.session.usage if self.harness else Usage()
        parts = [
            f"{self.launcher.config.model.name} via {self.launcher.config.model.provider}",
            str(self.launcher.permissions.mode),
            f"{self._context_tokens:,} tokens in context",
        ]

        if usage.cost_usd:
            parts.append(f"${usage.cost_usd:.4f}")

        self.query_one("#status", Static).update(" · ".join(parts))

    async def _help(self, _: str) -> None:
        keys = Table.grid(padding=(0, 2))
        keys.add_column(style="bold")
        keys.add_column(style="dim")

        for key, action in self.KEYS:
            keys.add_row(key, action)

        commands = Table.grid(padding=(0, 2))
        commands.add_column(style="bold")
        commands.add_column(style="italic")
        commands.add_column(style="dim")

        for command in self.commands.values():
            commands.add_row(f"/{command.name}", command.hint, command.about)

        title = Style(bold=True, underline=True)
        await self.notice(
            Group(Text("Keys", title), keys, Text(""), Text("Commands", title), commands)
        )

    async def _model(self, name: str) -> None:
        await self._switch({"name": name} if name else {}, f"model is {name}")

    async def _effort(self, level: str) -> None:
        allowed = self._efforts()

        if level not in allowed:
            await self.notice(f"/effort takes one of {', '.join(allowed)}")
            return

        await self._switch({"effort": level}, f"effort is {level}")

    async def _switch(self, update: dict[str, Any], done: str) -> None:
        if not update or self.harness is None:
            await self.notice(f"now using {self.launcher.config.model.name}")
            return

        model = self.launcher.config.model.model_copy(update=update)
        config = self.launcher.config.model_copy(update={"model": model})
        previous = self.harness.loop.llm

        try:
            self.harness.loop.llm = Harness.llm_for(config, self.harness.session)
        except (RuntimeError, ValueError) as error:
            await self.notice(str(error), warning=True)
            return

        self.launcher.config = config

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
            Transcript(self.labels).write,
            self.harness.session,
            target,
            self.paths.root / "transcript.md",
        )
        await self.notice(f"wrote {path}")

    async def _clear(self, _: str) -> None:
        await self._reopen(None)
        await self.notice("started a new session")

    async def _pick_session(self, session_id: str) -> None:
        sessions = self.launcher.sessions

        if session_id:
            try:
                await self._reopen(sessions.find(session_id))
            except LookupError as error:
                await self.notice(str(error), warning=True)

            return

        current = self.paths.root.name if self.paths else None
        earlier = [paths for paths in reversed(sessions.all()) if paths.root.name != current]

        if not earlier:
            await self.notice("there is no earlier session in this project")
            return

        picker = await asyncio.to_thread(SessionPicker, earlier)
        await self._add(picker)
        picker.focus()

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if isinstance(event.option_list, SessionPicker) and event.option.id:
            await self._reopen(event.option_list.sessions[event.option.id])

    async def _reopen(self, paths: RunPaths | None) -> None:
        await self._close()
        await self._transcript.remove_children()
        self._blocks.clear()
        self._context_tokens = 0
        self.driver = HumanDriver()
        await self._open(paths, resume=paths is not None)
        self.query_one(Composer).focus()

    async def _provider(self, name: str) -> None:
        if name not in self.PROVIDERS:
            await self.notice(f"/provider takes one of {', '.join(self.PROVIDERS)}")
            return

        update: dict[str, Any] = {"provider": name}
        current = self.launcher.config.model.name

        if name == "anthropic" and not ModelNames.is_anthropic(current):
            update["name"] = ModelSettings().name

        await self._switch(update, f"provider is {name}, model {update.get('name', current)}")

        if self.launcher.config.model.provider == name:
            self._models = []
            self.run_worker(self._load_models(), group="models", exit_on_error=False)

    def _efforts(self) -> list[str]:
        return list(get_args(ModelSettings.model_fields["effort"].annotation))

    async def _load_models(self) -> None:
        if self.harness is not None:
            self._models = await self.harness.loop.llm.models()

    def _fullness(self) -> float | None:
        threshold = self.launcher.config.model.compact_at
        current = ModelNames.openrouter(self.launcher.config.model.name)
        window = next(
            (card.context for card in self._models if ModelNames.openrouter(card.name) == current),
            None,
        )

        if threshold is None or not window:
            return None

        fullness = self._context_tokens / window
        return fullness if fullness >= threshold else None

    async def _auto_compact(self, fullness: float) -> None:
        for _ in range(self.SETTLE_TRIES):
            if not self.busy:
                break

            await asyncio.sleep(self.SETTLE_S)

        await self._compact("", reason=f"the context was {fullness:.0%} full, so juno compacted it")

    async def _compact(self, focus: str, reason: str = "") -> None:
        if self.harness is None or self.busy:
            await self.notice("wait for the turn to end, or press Esc, before /compact")
            return

        await self.notice("summarizing this session…")
        compactor = Compactor(self.harness.loop.llm, self.harness.loop.journal)

        try:
            summary = await compactor.summarize(self.harness.session, focus=focus)
        except (ValueError, ModelRejected) as error:
            await self.notice(f"could not compact: {error}", warning=True)
            return

        earlier = self.harness.session.id
        await self._reopen(None)
        self._note = f"(This continues session {earlier[:12]}. A summary of it:)\n\n{summary}"
        block = Collapsible(
            Markdown(Glyphs.narrow(summary)), title="summary of the earlier session"
        )
        await self._add(block)
        if reason:
            await self.notice(reason)

        await self.notice("compacted; the summary goes with your next message")

    async def _quit(self, _: str) -> None:
        await self.action_quit()
