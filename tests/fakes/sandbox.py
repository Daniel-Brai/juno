from pathlib import Path

from juno.domain.values import CommandResult, Job


class StillAliveSandbox:
    def __init__(self, alive: set[str]) -> None:
        self._alive = alive

    async def exec(self, command: str, timeout_s: float) -> CommandResult:
        raise NotImplementedError

    async def spawn(self, command: str, log_path: Path, exit_path: Path) -> Job:
        raise NotImplementedError

    def poll(self, job: Job) -> int | None:
        return None

    def is_alive(self, job: Job) -> bool:
        return job.id in self._alive

    async def kill_group(self, pgid: int) -> None:
        raise NotImplementedError
