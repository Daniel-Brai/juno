import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from rich.console import Console
from textual.pilot import Pilot
from textual.widgets import Static

from fakes.clock import FakeClock
from fakes.llm import Reply, ScriptedLlm, Turn
from fakes.probe import FakeProbe
from juno.bootstrap import Adapters
from juno.config import HarnessConfig, SandboxSettings
from juno.domain.session import State
from juno.domain.values import ModelReply, ModelRequest, Role
from juno.entrypoints.tui import (
    ActivityLine,
    ApprovalPrompt,
    ChatLauncher,
    CommandMenu,
    Composer,
    Glyphs,
    JunoApp,
    SessionPicker,
)
from juno.projects import ProjectSessions
from juno.service.loop import AgentLoop
from juno.tools.permissions import Permissions

pytestmark = pytest.mark.anyio

FIX = {
    "action": "str_replace",
    "path": "calc.py",
    "old_str": "return a - b",
    "new_str": "return a + b",
}


class StallingLlm(ScriptedLlm):
    def __init__(self, turns: list[Turn]) -> None:
        super().__init__(turns)
        self.stalled = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelReply:
        if not self.requests:
            self.requests.append(request)
            self.stalled.set()
            await asyncio.Event().wait()

        return await super().complete(request)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "calc"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


@pytest.fixture
def sessions(project: Path, tmp_path: Path) -> ProjectSessions:
    return ProjectSessions(project, data_home=tmp_path / "data")


class Chat:
    def __init__(self, sessions: ProjectSessions) -> None:
        self.sessions = sessions

    def app(self, llm: ScriptedLlm, *, resume: bool = False) -> JunoApp:
        launcher = ChatLauncher(
            config=HarnessConfig(sandbox=SandboxSettings(kind="local")),
            sessions=self.sessions,
            permissions=Permissions.load(self.sessions.project, self.sessions.root / "user"),
            adapters=Adapters(llm=llm, clock=FakeClock(), probe=FakeProbe()),
        )
        return JunoApp(launcher, resume=self.sessions.latest() if resume else None)

    @staticmethod
    async def send(pilot: Pilot[None], text: str) -> None:
        pilot.app.query_one(Composer).insert(text)
        await pilot.press("enter")

    @staticmethod
    async def until(pilot: Pilot[None], condition: Callable[[], bool]) -> None:
        for _ in range(200):
            if condition():
                return

            await pilot.pause(0.02)

        raise AssertionError("the app never reached the expected state")

    @staticmethod
    def messages(app: JunoApp) -> int:
        return len(app.harness.session.messages) if app.harness else 0

    @staticmethod
    def idle(app: JunoApp) -> bool:
        return app.harness is not None and not app.busy

    @staticmethod
    def shown(app: JunoApp) -> str:
        console = Console(width=120, color_system=None, highlight=False)

        with console.capture() as captured:
            for widget in app.query("#transcript > *"):
                console.print(widget.content if isinstance(widget, Static) else widget.render())

        return captured.get()


@pytest.fixture
def chat(sessions: ProjectSessions) -> Chat:
    return Chat(sessions)


class TestTui:
    async def test_a_sent_message_gets_an_answer(self, chat: Chat) -> None:
        app = chat.app(ScriptedLlm([Reply.text("Hello from juno.")]))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "hi")
            await chat.until(pilot, lambda: chat.messages(app) == 2)
            await chat.until(pilot, lambda: chat.idle(app))

            assert "hi" in chat.shown(app)
            assert not app.query(ActivityLine)
            assert app.harness is not None
            assert app.harness.session.state is State.AWAITING_CONTINUE

    @pytest.mark.parametrize(("key", "fixed"), [("y", True), ("n", False)])
    async def test_an_edit_waits_for_approval(
        self, chat: Chat, project: Path, key: str, fixed: bool
    ) -> None:
        app = chat.app(ScriptedLlm([Reply.tools(("editor", FIX)), Reply.text("done")]))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "fix add")
            await chat.until(pilot, lambda: bool(app.query(ApprovalPrompt)))

            assert "a - b" in (project / "calc.py").read_text()

            await pilot.press(key)
            await chat.until(pilot, lambda: chat.idle(app))

        assert ("a + b" in (project / "calc.py").read_text()) is fixed

    async def test_always_saves_a_project_rule(self, chat: Chat, project: Path) -> None:
        app = chat.app(ScriptedLlm([Reply.tools(("editor", FIX)), Reply.text("done")]))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "fix add")
            await chat.until(pilot, lambda: bool(app.query(ApprovalPrompt)))
            await pilot.press("a")
            await chat.until(pilot, lambda: chat.idle(app))

        rules = (project / ".juno" / "permissions.toml").read_text()
        assert 'tool = "editor"' in rules and '"calc.py"' in rules

    async def test_escape_stops_a_turn_and_the_next_message_goes_through(self, chat: Chat) -> None:
        llm = StallingLlm([Reply.text("Second answer.")])
        app = chat.app(llm)

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "think hard")
            await chat.until(pilot, llm.stalled.is_set)
            await pilot.press("escape")
            await chat.until(pilot, lambda: chat.idle(app))

            assert app.harness is not None
            session = app.harness.session
            assert session.messages[-1].content == [{"type": "text", "text": AgentLoop.STOPPED}]
            assert "stopped; what next?" in chat.shown(app)

            await chat.send(pilot, "try again")
            await chat.until(pilot, lambda: len(session.messages) == 4)

            assert session.messages[2].content == "try again"

    async def test_escape_during_an_approval_stops_the_call(
        self, chat: Chat, project: Path
    ) -> None:
        app = chat.app(ScriptedLlm([Reply.tools(("editor", FIX))]))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "fix add")
            await chat.until(pilot, lambda: bool(app.query(ApprovalPrompt)))
            await pilot.press("escape")
            await chat.until(pilot, lambda: chat.idle(app))

            assert not app.query(ApprovalPrompt)
            assert app.harness is not None
            stopped = app.harness.session.messages[2].content
            assert isinstance(stopped, list)
            assert stopped[0]["content"] == "stopped by the user before it finished"

        assert "a - b" in (project / "calc.py").read_text()

    async def test_a_session_resumes_with_its_history(
        self, chat: Chat, sessions: ProjectSessions
    ) -> None:
        first = chat.app(ScriptedLlm([Reply.text("First answer.")]))

        async with first.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(first))
            await chat.send(pilot, "remember this")
            await chat.until(pilot, lambda: chat.messages(first) == 2)
            await pilot.press("ctrl+d")

        second = chat.app(ScriptedLlm([Reply.text("Second answer.")]), resume=True)

        async with second.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(second))

            assert "remember this" in chat.shown(second)

            await chat.send(pilot, "and this")
            await chat.until(pilot, lambda: chat.messages(second) == 4)

        assert len(sessions.all()) == 1

    async def test_undo_restores_the_last_turns_changes(self, chat: Chat, project: Path) -> None:
        app = chat.app(
            ScriptedLlm([Reply.tools(("editor", FIX)), Reply.text("done"), Reply.text("ok")])
        )

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "fix add")
            await chat.until(pilot, lambda: bool(app.query(ApprovalPrompt)))
            await pilot.press("y")
            await chat.until(pilot, lambda: chat.idle(app) and chat.messages(app) == 4)

            assert "a + b" in (project / "calc.py").read_text()

            await chat.send(pilot, "/undo")
            await chat.until(pilot, lambda: "restored 1 files" in chat.shown(app))

            assert "a - b" in (project / "calc.py").read_text()

            await chat.send(pilot, "why?")
            await chat.until(pilot, lambda: chat.messages(app) == 6)

            assert app.harness is not None
            sent = app.harness.session.messages[4]
            assert sent.role is Role.USER and "I undid the changes" in str(sent.content)


class TestCommandMenu:
    async def test_typing_a_slash_offers_matching_commands(self, chat: Chat) -> None:
        app = chat.app(ScriptedLlm([]))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await pilot.press("slash", "m", "o")
            menu = app.query_one(CommandMenu)

            assert [match.text for match in menu.matches] == ["/model ", "/mode "]

            await pilot.press("down", "tab")

            assert app.query_one(Composer).text == "/mode "
            assert next(match.text for match in menu.matches) == "/mode ask"

    async def test_enter_runs_a_command_without_arguments(self, chat: Chat) -> None:
        app = chat.app(ScriptedLlm([]))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await pilot.press("slash", "h", "e", "enter")
            await chat.until(pilot, lambda: "Commands" in chat.shown(app))

            assert app.query_one(Composer).text == ""


class TestArgumentCompletion:
    async def test_arguments_complete_from_their_choices(self, chat: Chat) -> None:
        app = chat.app(ScriptedLlm([]))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app) and bool(app._models))
            menu = app.query_one(CommandMenu)

            app.query_one(Composer).insert("/effort ")
            await pilot.pause()
            efforts = [match.label.plain for match in menu.matches]

            app.query_one(Composer).clear()
            app.query_one(Composer).insert("/model TWO")
            await pilot.pause()
            models = [match.text for match in menu.matches]

        assert efforts == ["low", "medium", "high  current", "xhigh", "max"]
        assert models == ["/model scripted/two"]


class TestCompact:
    async def test_compact_continues_in_a_new_session_with_the_summary(
        self, chat: Chat, sessions: ProjectSessions
    ) -> None:
        turns: list[Turn] = [
            Reply.text("Sure."),
            Reply.text("Goal: fix add in calc.py."),
            Reply.text("On it."),
        ]
        app = chat.app(ScriptedLlm(turns))

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "fix add")
            await chat.until(pilot, lambda: chat.idle(app) and chat.messages(app) == 2)
            first = app.harness.session.id if app.harness else ""
            await chat.send(pilot, "/compact")
            await chat.until(pilot, lambda: "compacted" in chat.shown(app))
            await chat.send(pilot, "keep going")
            await chat.until(pilot, lambda: chat.messages(app) == 2)

            assert app.harness is not None
            opening = str(app.harness.session.messages[0].content)

        assert app.harness is None or app.harness.session.id != first
        assert "Goal: fix add in calc.py." in opening and opening.endswith("keep going")
        assert len(sessions.all()) == 2


class TestActivityLine:
    async def test_it_sits_under_the_message_until_the_turn_ends(self, chat: Chat) -> None:
        llm = StallingLlm([])
        app = chat.app(llm)

        async with app.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(app))
            await chat.send(pilot, "think hard")
            await chat.until(pilot, llm.stalled.is_set)

            last = list(app.query("#transcript > *"))[-1]
            assert isinstance(last, ActivityLine) and "Thinking" in str(last.render())

            await pilot.press("escape")
            await chat.until(pilot, lambda: chat.idle(app))

            assert not app.query(ActivityLine)


class TestGlyphs:
    def test_wide_emoji_become_one_cell_symbols(self) -> None:
        assert Glyphs.narrow(
            "done \u2705 failed \u274c warn \u26a0\ufe0f hi \U0001f44b \u4e2d"
        ) == ("done \u2713 failed \u2717 warn \u26a0 hi  \u4e2d")


class TestResumePicker:
    async def test_resume_lists_earlier_sessions_and_switches_to_one(self, chat: Chat) -> None:
        first = chat.app(ScriptedLlm([Reply.text("Noted.")]))

        async with first.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(first))
            await chat.send(pilot, "the earlier one")
            await chat.until(pilot, lambda: chat.messages(first) == 2)
            await pilot.press("ctrl+d")

        second = chat.app(ScriptedLlm([]))

        async with second.run_test() as pilot:
            await chat.until(pilot, lambda: chat.idle(second))
            await chat.send(pilot, "/resume")
            await chat.until(pilot, lambda: bool(second.query(SessionPicker)))

            assert "the earlier one" in str(
                second.query_one(SessionPicker).get_option_at_index(0).prompt
            )

            await pilot.press("enter")
            await chat.until(pilot, lambda: chat.idle(second) and chat.messages(second) == 2)

            assert "the earlier one" in chat.shown(second)
