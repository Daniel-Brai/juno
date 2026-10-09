from pathlib import Path

from fakes.probe import FakeProbe
from juno.adapters.sandboxes.local import LocalSandbox
from juno.domain.values import CommandResult, ResourceSnapshot


class Crash(BaseException):
    pass


class CrashingProbe(FakeProbe):
    def sample(self) -> ResourceSnapshot:
        raise Crash("the harness died while reading resources")


class CrashingSandbox(LocalSandbox):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace)

    async def exec(self, command: str, timeout_s: float) -> CommandResult:
        raise Crash("the harness died while a command ran")
