import json
import tomllib
from collections.abc import Mapping, Sequence
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from juno.domain.values import ToolCall


class Action(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Mode(StrEnum):
    ASK = "ask"
    ACCEPT_EDITS = "accept-edits"
    PLAN = "plan"
    YOLO = "yolo"


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str
    match: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    action: Action

    def applies(self, call: ToolCall) -> bool:
        if not fnmatchcase(call.name, self.tool):
            return False

        command = call.args.get("command")

        if self.match and not (
            isinstance(command, str)
            and any(fnmatchcase(command.strip(), pattern) for pattern in self.match)
        ):
            return False

        path = call.args.get("path")

        return not self.paths or (
            isinstance(path, str) and any(self._inside(path, pattern) for pattern in self.paths)
        )

    @staticmethod
    def _inside(path: str, pattern: str) -> bool:
        relative = PurePosixPath(path.removeprefix("./") or ".")
        return relative.full_match(pattern.removeprefix("./"))


class RuleFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Mode | None = None
    rule: tuple[Rule, ...] = ()


class Permissions:
    FILE: ClassVar[str] = "permissions.toml"
    BUILT_IN: ClassVar[tuple[Rule, ...]] = (
        Rule(
            tool="bash",
            match=(
                "sudo *",
                "* sudo *",
                "rm -rf /",
                "rm -rf /*",
                "rm -rf ~*",
                "* rm -rf /*",
                "git push --force*",
                "git push -f*",
                "mkfs*",
                "dd *of=/dev/*",
            ),
            action=Action.DENY,
        ),
    )
    READS: ClassVar[Mapping[str, frozenset[str]]] = {"editor": frozenset({"view"})}
    EDITS: ClassVar[frozenset[str]] = frozenset({"editor"})
    LOOKUPS: ClassVar[frozenset[str]] = frozenset({"web_fetch"})

    def __init__(
        self, *, mode: Mode = Mode.ASK, rules: Sequence[Rule] = (), store: Path | None = None
    ) -> None:
        self.mode = mode
        self._rules = list(rules)
        self._store = store

    @classmethod
    def load(cls, project: Path, user_dir: Path) -> Permissions:
        store = project / ".juno" / cls.FILE
        files = [RuleFile.model_validate(cls._read(path)) for path in (store, user_dir / cls.FILE)]
        mode = next((file.mode for file in files if file.mode is not None), Mode.ASK)
        return cls(mode=mode, rules=[rule for file in files for rule in file.rule], store=store)

    def decide(self, call: ToolCall, *, read_only: bool) -> tuple[Action, str]:
        for rule in self.BUILT_IN:
            if rule.applies(call):
                return Action.DENY, "juno never runs this command"

        for rule in self._rules:
            if rule.applies(call):
                return rule.action, f"a permission rule says {rule.action}"

        if read_only or call.args.get("action") in self.READS.get(call.name, ()):
            return Action.ALLOW, "it only reads"

        match self.mode:
            case Mode.YOLO:
                return Action.ALLOW, "yolo mode allows everything"
            case Mode.PLAN if call.name in self.LOOKUPS:
                return Action.ALLOW, "plan mode allows looking things up"
            case Mode.PLAN:
                return Action.DENY, "plan mode is read-only"
            case Mode.ACCEPT_EDITS if call.name in self.EDITS:
                return Action.ALLOW, "accept-edits mode allows file edits"
            case _:
                return Action.ASK, f"{call.name} changes things"

    def remember(self, call: ToolCall) -> Rule:
        command = call.args.get("command")
        path = call.args.get("path")
        rule = Rule(
            tool=call.name,
            match=(command.strip(),) if isinstance(command, str) else (),
            paths=(path,) if isinstance(path, str) and not isinstance(command, str) else (),
            action=Action.ALLOW,
        )
        self._rules.insert(0, rule)

        if self._store is not None:
            self._store.parent.mkdir(parents=True, exist_ok=True)

            with self._store.open("a") as handle:
                handle.write(self.toml(rule))

        return rule

    @staticmethod
    def toml(rule: Rule) -> str:
        lines = ["", "[[rule]]", f"tool = {json.dumps(rule.tool)}"]

        if rule.match:
            lines.append(f"match = {json.dumps(list(rule.match))}")

        if rule.paths:
            lines.append(f"paths = {json.dumps(list(rule.paths))}")

        lines.append(f"action = {json.dumps(rule.action.value)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _read(path: Path) -> dict[str, object]:
        return tomllib.loads(path.read_text()) if path.is_file() else {}
