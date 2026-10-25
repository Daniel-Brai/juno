import sys
from pathlib import Path

import pytest

from fakes.clock import FakeClock
from fakes.llm import Reply, ScriptedLlm, Turn
from fakes.probe import FakeProbe
from juno.bootstrap import Adapters, ChatSetup, Harness
from juno.config import HarnessConfig, McpServer, SandboxSettings
from juno.domain.values import Role
from juno.projects import ProjectSessions
from juno.service.drivers import ScriptDriver
from juno.tools.approvals import RulesOnly
from juno.tools.permissions import Permissions

pytestmark = [pytest.mark.anyio, pytest.mark.slow]

SERVER = Path(__file__).parent.parent / "fakes" / "mcp_server.py"
DEMO = McpServer(command=sys.executable, args=(str(SERVER),))


class Chat:
    def __init__(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        self.sessions = ProjectSessions(project, data_home=tmp_path / "data")

    async def open(
        self, turns: list[Turn], servers: dict[str, McpServer], *, resume: bool = False
    ) -> Harness:
        setup = ChatSetup(
            driver=ScriptDriver(["go"]),
            approver=RulesOnly(yes=True),
            permissions=Permissions(),
            mcp=servers,
        )
        paths = self.sessions.latest() if resume else self.sessions.new()
        assert paths is not None
        opener = Harness.resume_chat if resume else Harness.start_chat
        return await opener(
            config=HarnessConfig(sandbox=SandboxSettings(kind="local")),
            chat=setup,
            paths=paths,
            adapters=Adapters(llm=ScriptedLlm(turns), clock=FakeClock(), probe=FakeProbe()),
        )


class TestMcpTools:
    async def test_a_server_tool_is_offered_and_called(self, tmp_path: Path) -> None:
        turns: list[Turn] = [Reply.tools(("mcp__demo__add", {"a": 2, "b": 3})), Reply.text("5")]

        async with await Chat(tmp_path).open(turns, {"demo": DEMO}) as harness:
            await harness.run()
            results = [
                block["content"]
                for message in harness.session.messages
                if message.role is Role.USER and isinstance(message.content, list)
                for block in message.content
            ]

        assert results == ["5"]

    async def test_a_broken_server_is_a_warning_not_a_failure(self, tmp_path: Path) -> None:
        broken = McpServer(command="juno-no-such-command")

        async with await Chat(tmp_path).open(
            [Reply.text("fine")], {"demo": DEMO, "broken": broken}
        ) as harness:
            await harness.run()

            assert [w for w in harness.warnings if "broken did not start" in w]
            assert "mcp__demo__add" in harness.registry.names

    async def test_a_session_resumes_after_its_servers_change(self, tmp_path: Path) -> None:
        chat = Chat(tmp_path)

        async with await chat.open([Reply.text("first")], {"demo": DEMO}) as first:
            await first.run()

        async with await chat.open([Reply.text("second")], {}, resume=True) as second:
            await second.run()

            assert "mcp__demo__add" not in second.registry.names
