from pathlib import Path

import pytest

from juno.domain.events import ApprovalDecided, Event
from juno.domain.values import Answer, ToolCall, ToolResult
from juno.service.context import RunContext
from juno.tools.approvals import Approvals, RulesOnly
from juno.tools.permissions import Action, Mode, Permissions, Rule

pytestmark = pytest.mark.anyio


class Calls:
    @staticmethod
    def bash(command: str) -> ToolCall:
        return ToolCall(id=f"call-{command}", name="bash", args={"command": command})

    @staticmethod
    def edit(path: str, action: str = "create") -> ToolCall:
        return ToolCall(id=f"call-{path}", name="editor", args={"action": action, "path": path})


class TestPermissions:
    def test_the_first_matching_rule_wins(self) -> None:
        permissions = Permissions(
            rules=[
                Rule(tool="bash", match=("git status*",), action=Action.ALLOW),
                Rule(tool="bash", match=("git *",), action=Action.DENY),
            ]
        )

        assert permissions.decide(Calls.bash("git status -s"), read_only=False)[0] is Action.ALLOW
        assert permissions.decide(Calls.bash("git commit"), read_only=False)[0] is Action.DENY
        assert permissions.decide(Calls.bash("ls"), read_only=False)[0] is Action.ASK

    @pytest.mark.parametrize(
        "command", ["sudo rm x", "cd /tmp && sudo ls", "rm -rf /", "git push --force origin"]
    )
    def test_destructive_commands_are_denied_even_in_yolo_mode(self, command: str) -> None:
        permissions = Permissions(mode=Mode.YOLO, rules=[Rule(tool="bash", action=Action.ALLOW)])

        assert permissions.decide(Calls.bash(command), read_only=False)[0] is Action.DENY

    @pytest.mark.parametrize(
        ("mode", "edit", "command"),
        [
            (Mode.ASK, Action.ASK, Action.ASK),
            (Mode.ACCEPT_EDITS, Action.ALLOW, Action.ASK),
            (Mode.PLAN, Action.DENY, Action.DENY),
            (Mode.YOLO, Action.ALLOW, Action.ALLOW),
        ],
    )
    def test_modes_set_the_default(self, mode: Mode, edit: Action, command: Action) -> None:
        permissions = Permissions(mode=mode)

        assert permissions.decide(Calls.edit("app.py"), read_only=False)[0] is edit
        assert permissions.decide(Calls.bash("make"), read_only=False)[0] is command

    def test_reads_are_allowed_in_every_mode(self) -> None:
        permissions = Permissions(mode=Mode.PLAN)

        assert permissions.decide(Calls.edit("app.py", "view"), read_only=False)[0] is Action.ALLOW
        assert permissions.decide(ToolCall(id="g", name="grep"), read_only=True)[0] is Action.ALLOW

    def test_path_rules_match_inside_folders(self) -> None:
        permissions = Permissions(
            rules=[Rule(tool="editor", paths=("./src/**",), action=Action.ALLOW)]
        )

        assert permissions.decide(Calls.edit("src/a/b.py"), read_only=False)[0] is Action.ALLOW
        assert permissions.decide(Calls.edit("./src/c.py"), read_only=False)[0] is Action.ALLOW
        assert permissions.decide(Calls.edit("tests/c.py"), read_only=False)[0] is Action.ASK

    def test_rules_can_match_tool_names_by_pattern(self) -> None:
        permissions = Permissions(rules=[Rule(tool="mcp__demo__*", action=Action.ALLOW)])
        call = ToolCall(id="1", name="mcp__demo__add", args={})
        other = ToolCall(id="2", name="mcp__github__push", args={})

        assert permissions.decide(call, read_only=False)[0] is Action.ALLOW
        assert permissions.decide(other, read_only=False)[0] is Action.ASK

    def test_project_rules_and_mode_come_before_user_ones(self, tmp_path: Path) -> None:
        project, user = tmp_path / "project", tmp_path / "user"
        (project / ".juno").mkdir(parents=True)
        user.mkdir()
        (project / ".juno" / "permissions.toml").write_text(
            'mode = "accept-edits"\n\n[[rule]]\ntool = "bash"\nmatch = ["make*"]\naction = "allow"\n'
        )
        (user / "permissions.toml").write_text(
            'mode = "plan"\n\n[[rule]]\ntool = "bash"\nmatch = ["make*", "ls*"]\naction = "deny"\n'
        )

        permissions = Permissions.load(project, user)

        assert permissions.mode is Mode.ACCEPT_EDITS
        assert permissions.decide(Calls.bash("make test"), read_only=False)[0] is Action.ALLOW
        assert permissions.decide(Calls.bash("ls"), read_only=False)[0] is Action.DENY

    def test_remembered_answers_are_saved_and_reloaded(self, tmp_path: Path) -> None:
        permissions = Permissions.load(tmp_path, tmp_path / "user")

        permissions.remember(Calls.bash('echo "hi" \\ there'))
        permissions.remember(Calls.edit("notes/today.md"))
        reloaded = Permissions.load(tmp_path, tmp_path / "user")

        assert reloaded.decide(Calls.bash('echo "hi" \\ there'), read_only=False)[0] is Action.ALLOW
        assert reloaded.decide(Calls.edit("notes/today.md"), read_only=False)[0] is Action.ALLOW
        assert reloaded.decide(Calls.bash("echo other"), read_only=False)[0] is Action.ASK


class Asker:
    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def approve(self, call: ToolCall, reason: str, /) -> Answer:
        self.asked.append(call.name)
        return self.answer


class TestApprovals:
    async def run(
        self, context: RunContext, permissions: Permissions, approver: Asker | RulesOnly
    ) -> tuple[ToolResult, list[Event], list[str]]:
        seen: list[Event] = []
        ran: list[str] = []

        async def record(event: Event) -> None:
            seen.append(event)

        async def proceed(call: ToolCall) -> ToolResult:
            ran.append(call.id)
            return ToolResult(call_id=call.id, content="done")

        context.bus.subscribe(Event, record)
        approvals = Approvals(permissions, approver, context, read_only=lambda _: False)
        result = await approvals(Calls.bash("make"), proceed)
        return result, seen, ran

    async def test_an_allowed_call_runs_and_is_recorded(self, context: RunContext) -> None:
        asker = Asker(Answer.ONCE)

        result, seen, ran = await self.run(context, Permissions(), asker)

        assert (result.content, ran, asker.asked) == ("done", ["call-make"], ["bash"])
        [decided] = seen
        assert isinstance(decided, ApprovalDecided) and decided.allowed and decided.asked

    async def test_a_denied_call_does_not_run(self, context: RunContext) -> None:
        result, seen, ran = await self.run(context, Permissions(), Asker(Answer.DENY))

        assert result.is_error and result.content == "not run: the user denied it"
        assert ran == []
        assert isinstance(seen[0], ApprovalDecided) and not seen[0].allowed

    async def test_always_adds_a_rule_so_the_next_call_is_not_asked(
        self, context: RunContext
    ) -> None:
        permissions = Permissions()
        asker = Asker(Answer.ALWAYS)

        await self.run(context, permissions, asker)
        await self.run(context, permissions, asker)

        assert asker.asked == ["bash"]

    @pytest.mark.parametrize(("yes", "ran"), [(False, []), (True, ["call-make"])])
    async def test_print_mode_answers_from_the_yes_flag(
        self, context: RunContext, yes: bool, ran: list[str]
    ) -> None:
        _, _, actual = await self.run(context, Permissions(), RulesOnly(yes=yes))

        assert actual == ran
