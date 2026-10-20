import json
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from fakes.clock import FakeClock
from fakes.llm import Reply, ScriptedLlm, Turn
from fakes.probe import FakeProbe
from juno.bootstrap import Adapters, ChatSetup, Harness
from juno.config import HarnessConfig, SandboxSettings
from juno.domain.events import ApprovalDecided, Event
from juno.domain.session import Session, State
from juno.domain.values import Role, RunPaths
from juno.entrypoints.cli import JunoCli
from juno.ports import LlmClient
from juno.projects import ProjectSessions
from juno.service.drivers import ScriptDriver
from juno.tools.approvals import RulesOnly
from juno.tools.permissions import Mode, Permissions

pytestmark = pytest.mark.anyio

FIX = {
    "action": "str_replace",
    "path": "calc.py",
    "old_str": "return a - b",
    "new_str": "return a + b",
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "calc"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "AGENTS.md").write_text("Run python -m calc to check.\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


@pytest.fixture
def sessions(project: Path, tmp_path: Path) -> ProjectSessions:
    return ProjectSessions(project, data_home=tmp_path / "data")


class ChatRun:
    def __init__(self, sessions: ProjectSessions, clock: FakeClock) -> None:
        self.sessions = sessions
        self.clock = clock
        self.events: list[Event] = []

    async def start(
        self,
        prompts: list[str],
        turns: list[Turn],
        *,
        yes: bool = True,
        mode: Mode = Mode.ASK,
    ) -> Harness:
        return await self._open(Harness.start_chat, self.sessions.new(), prompts, turns, yes, mode)

    async def resume(self, prompts: list[str], turns: list[Turn]) -> Harness:
        latest = self.sessions.latest()
        assert latest is not None
        return await self._open(Harness.resume_chat, latest, prompts, turns, True, Mode.ASK)

    async def _open(
        self,
        opener: Callable[..., Awaitable[Harness]],
        paths: RunPaths,
        prompts: list[str],
        turns: list[Turn],
        yes: bool,
        mode: Mode,
    ) -> Harness:
        setup = ChatSetup(
            driver=ScriptDriver(prompts),
            approver=RulesOnly(yes=yes),
            permissions=Permissions(mode=mode),
            instructions=self.sessions.instructions(),
        )
        harness = await opener(
            config=HarnessConfig(sandbox=SandboxSettings(kind="local")),
            chat=setup,
            paths=paths,
            adapters=Adapters(llm=ScriptedLlm(turns), clock=self.clock, probe=FakeProbe()),
        )
        harness.bus.subscribe(Event, self._record)
        return harness

    async def _record(self, event: Event) -> None:
        self.events.append(event)


@pytest.fixture
def chats(sessions: ProjectSessions, clock: FakeClock) -> ChatRun:
    return ChatRun(sessions, clock)


class TestChatSession:
    async def test_a_request_is_worked_through_and_the_session_waits(
        self, chats: ChatRun, project: Path
    ) -> None:
        turns: list[Turn] = [
            Reply.tools(("grep", {"pattern": "return a"})),
            Reply.tools(("editor", FIX)),
            Reply.tools(("bash", {"command": "python -c 'import calc; print(calc.add(2, 3))'"})),
            Reply.text("Fixed add; it now returns 5 for 2 and 3."),
        ]

        async with await chats.start(["fix add"], turns) as harness:
            await harness.run()
            session = harness.session
            system = harness.loop.journal.request([]).system

        assert (project / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
        assert session.state is State.AWAITING_CONTINUE and session.open_ended
        assert JunoCli.final_text(session) == "Fixed add; it now returns 5 for 2 and 3."
        assert "Run python -m calc to check." in system
        decided = [event for event in chats.events if isinstance(event, ApprovalDecided)]
        assert [(event.call.name, event.asked) for event in decided] == [
            ("grep", False),
            ("editor", True),
            ("bash", True),
        ]

    async def test_checkpoints_go_to_a_shadow_repo_not_the_project(
        self, chats: ChatRun, project: Path, sessions: ProjectSessions
    ) -> None:
        async with await chats.start(
            ["fix add"], [Reply.tools(("editor", FIX)), Reply.text("ok")]
        ) as harness:
            await harness.run()

        shadow = sessions.root / harness.session.id / "checkpoints.git"
        log = self.git("--git-dir", str(shadow), "log", "--oneline")
        own = self.git("-C", str(project), "log")

        assert "SessionStarted at message 0" in log.stdout
        assert own.returncode != 0 and "does not have any commits" in own.stderr

    async def test_denied_edits_do_not_touch_the_project(
        self, chats: ChatRun, project: Path
    ) -> None:
        async with await chats.start(
            ["fix add"], [Reply.tools(("editor", FIX)), Reply.text("I could not edit.")], yes=False
        ) as harness:
            await harness.run()
            results = self.tool_results(harness.session)

        assert results == ["not run: the user denied it"]
        assert "a - b" in (project / "calc.py").read_text()

    async def test_plan_mode_only_reads(self, chats: ChatRun, project: Path) -> None:
        turns: list[Turn] = [
            Reply.tools(("read", {"path": "calc.py"}), ("editor", FIX)),
            Reply.text("Here is the plan."),
        ]

        async with await chats.start(["plan the fix"], turns, mode=Mode.PLAN) as harness:
            await harness.run()
            results = self.tool_results(harness.session)

        assert results[0].startswith("     1\tdef add(a, b):")
        assert results[1] == "not run: plan mode is read-only"

    async def test_a_session_continues_with_its_history_todo_and_instructions(
        self, chats: ChatRun, project: Path
    ) -> None:
        todo = {"items": [{"text": "fix add", "status": "in_progress"}]}

        async with await chats.start(
            ["fix add"], [Reply.tools(("todo", todo)), Reply.text("Started.")]
        ) as first:
            await first.run()

        (project / "AGENTS.md").write_text("Changed instructions.\n")

        async with await chats.resume(["keep going"], [Reply.text("Done.")]) as second:
            await second.run()
            session = second.session
            todos = second.todo.render() if second.todo else ""
            system = second.loop.journal.request([]).system

        users = [
            m.content
            for m in session.messages
            if m.role is Role.USER and isinstance(m.content, str)
        ]
        assert users == ["fix add", "keep going"]
        assert todos == "[>] fix add"
        assert "Run python -m calc to check." in system

    @staticmethod
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False)

    @staticmethod
    def tool_results(session: Session) -> list[str]:
        return [
            str(block["content"])
            for message in session.messages
            if message.role is Role.USER and isinstance(message.content, list)
            for block in message.content
            if block.get("type") == "tool_result"
        ]


class TestProjectSessions:
    def test_sessions_live_outside_the_project_and_sort_by_age(
        self, sessions: ProjectSessions, project: Path, tmp_path: Path
    ) -> None:
        assert sessions.all() == [] and sessions.latest() is None

        first, second = sessions.new(), sessions.new()

        for paths in (first, second):
            paths.root.mkdir(parents=True)
            (paths.root / "run.db").write_text("")

        assert sessions.root.is_relative_to(tmp_path / "data" / "juno" / "projects")
        assert sessions.latest() == second and second.workspace == project
        assert sessions.find(first.root.name) == first

        with pytest.raises(LookupError, match="no session"):
            sessions.find("zzz")


class TestChatCli:
    @pytest.fixture
    def scripted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: FakeClock
    ) -> list[list[Turn]]:
        scripts: list[list[Turn]] = []

        def llm_for(config: HarnessConfig, session: Session) -> LlmClient:
            return ScriptedLlm(scripts.pop(0))

        monkeypatch.setattr(Harness, "llm_for", staticmethod(llm_for))
        monkeypatch.setattr("juno.bootstrap.SystemClock", lambda: clock)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        return scripts

    async def test_print_mode_answers_in_text(
        self, scripted: list[list[Turn]], project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scripted.append([Reply.tools(("editor", FIX)), Reply.text("Fixed it.")])

        code = await JunoCli().chat(prompt="fix add", project=project, yes=True)

        out = capsys.readouterr()
        assert (code, out.out) == (0, "Fixed it.\n")
        assert "juno: session " in out.err
        assert "a + b" in (project / "calc.py").read_text()

    async def test_print_mode_streams_json_events(
        self, scripted: list[list[Turn]], project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scripted.append([Reply.tools(("editor", FIX)), Reply.text("Could not.")])

        code = await JunoCli().chat(prompt="fix add", project=project, output="json")

        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        kinds = [line["type"] for line in lines]
        assert code == 0
        assert "SessionStarted" in kinds[:2] and kinds[-1] == "Result"
        assert "ApprovalDecided" in kinds and lines[-1]["text"] == "Could not."
        assert "a - b" in (project / "calc.py").read_text()

    async def test_continue_picks_up_the_latest_session(
        self, scripted: list[list[Turn]], project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        one: list[Turn] = [Reply.text("First.")]
        two: list[Turn] = [Reply.text("Second.")]
        scripted += [one, two]
        cli = JunoCli()

        await cli.chat(prompt="one", project=project)
        first = capsys.readouterr().err
        await cli.chat(prompt="two", project=project, resume_latest=True)
        second = capsys.readouterr()

        assert second.out == "Second.\n" and first == second.err

    async def test_without_print_mode_it_points_to_print_mode(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert await JunoCli().chat(project=project) == 2
        assert "use -p" in capsys.readouterr().err
