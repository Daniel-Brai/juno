import asyncio
from asyncio.subprocess import DEVNULL, PIPE, Process
from pathlib import Path
from typing import ClassVar

from juno.adapters.sandboxes.local import LocalSandbox
from juno.domain.values import Job


class PodmanError(Exception):
    pass


class PodmanSandbox(LocalSandbox):
    WORKSPACE: ClassVar[Path] = Path("/workspace")
    LOGS: ClassVar[Path] = Path("/juno/logs")
    DATA: ClassVar[Path] = Path("/data")
    PIDS_LIMIT: ClassVar[int] = 4_096
    START_WAIT_S: ClassVar[float] = 10.0
    EXEC_SCRIPT: ClassVar[str] = 'setsid bash -c "$1" & echo $! > "$2"; wait $!'
    JOB_SCRIPT: ClassVar[str] = 'echo $$ > "$2"; ( eval "$1" ) > "$4" 2>&1; echo $? > "$3"'
    KILL_BODY: ClassVar[str] = (
        '[ -n "$pg" ] || exit 0; kill -TERM -- "-$pg" 2>/dev/null || exit 0; '
        'for _ in $(seq 50); do kill -0 -- "-$pg" 2>/dev/null || exit 0; sleep 0.1; done; '
        'kill -KILL -- "-$pg" 2>/dev/null; exit 0'
    )
    KILL_GROUP: ClassVar[str] = f'pg="$1"; {KILL_BODY}'
    KILL_EXEC: ClassVar[str] = f'pg=$(cat "$1" 2>/dev/null); rm -f "$1"; {KILL_BODY}'

    def __init__(
        self,
        name: str,
        image: str,
        workspace: Path,
        logs: Path,
        data: Path | None = None,
        memory: str | None = None,
        gpus: tuple[int, ...] = (),
        network: bool = False,
        output_limit: int = 30_000,
    ) -> None:
        super().__init__(workspace, output_limit)
        self.name = name
        self._image = image
        self._logs = logs
        self._data = data
        self._memory = memory
        self._gpus = gpus
        self._network = network
        self._incarnation: float | None = None

    def run_argv(self) -> list[str]:
        argv = [
            "podman",
            "run",
            "--detach",
            "--init",
            "--name",
            self.name,
            "--userns=keep-id",
            "--workdir",
            str(self.WORKSPACE),
            "--pids-limit",
            str(self.PIDS_LIMIT),
            "--network",
            "slirp4netns" if self._network else "none",
            "--volume",
            f"{self._workspace}:{self.WORKSPACE}:Z",
            "--volume",
            f"{self._logs}:{self.LOGS}:Z",
        ]

        if self._data is not None:
            argv += ["--volume", f"{self._data}:{self.DATA}:ro,z"]

        if self._memory is not None:
            argv += ["--memory", self._memory]

        for gpu in self._gpus:
            argv += ["--device", f"nvidia.com/gpu={gpu}"]

        return [*argv, self._image, "sleep", "infinity"]

    async def start(self) -> None:
        if not await self._succeeds("container", "exists", self.name):
            await self._podman(*self.run_argv()[1:])
        elif await self._podman("inspect", "-f", "{{.State.Running}}", self.name) != "true":
            await self._podman("start", self.name)

        self._incarnation = float(
            await self._podman("inspect", "-f", "{{.State.StartedAt.UnixNano}}", self.name)
        )

    async def stop(self) -> None:
        await self._podman("rm", "--force", "--time", "5", self.name)
        self._incarnation = None

    def exec_argv(self, command: str, token: str) -> list[str]:
        return [
            "podman",
            "exec",
            "--workdir",
            str(self.WORKSPACE),
            self.name,
            "bash",
            "-c",
            self.EXEC_SCRIPT,
            "juno-exec",
            command,
            str(self.LOGS / f".exec-{token}.pid"),
        ]

    async def stop_exec(self, process: Process, token: str) -> None:
        await self._podman(
            "exec",
            self.name,
            "bash",
            "-c",
            self.KILL_EXEC,
            "juno-kill",
            str(self.LOGS / f".exec-{token}.pid"),
        )
        await super().stop_exec(process, token)

    async def spawn(self, command: str, log_path: Path, exit_path: Path) -> Job:
        pid_path = log_path.with_suffix(".pid")
        pid_path.unlink(missing_ok=True)
        await self._podman(
            "exec",
            "--detach",
            "--workdir",
            str(self.WORKSPACE),
            self.name,
            "setsid",
            "bash",
            "-c",
            self.JOB_SCRIPT,
            "juno-job",
            command,
            str(self.inside(pid_path)),
            str(self.inside(exit_path)),
            str(self.inside(log_path)),
        )
        pid = await self._read_pid(pid_path)
        return Job(
            command=command,
            pid=pid,
            pgid=pid,
            proc_start=self._incarnation,
            log_path=log_path,
            exit_path=exit_path,
        )

    def is_alive(self, job: Job) -> bool:
        return (
            self._incarnation is not None
            and job.proc_start == self._incarnation
            and not job.exit_path.exists()
        )

    def memory_bytes(self, _job: Job) -> int:
        return 0

    async def kill_group(self, pgid: int) -> None:
        await self._podman("exec", self.name, "bash", "-c", self.KILL_GROUP, "juno-kill", str(pgid))

    def inside(self, host_path: Path) -> Path:
        return self.LOGS / host_path.relative_to(self._logs)

    async def _read_pid(self, pid_path: Path) -> int:
        for _ in range(round(self.START_WAIT_S / self.POLL_S)):
            if (pid := self._pid_from(pid_path)) is not None:
                return pid

            await asyncio.sleep(self.POLL_S)

        raise PodmanError(f"job did not start within {self.START_WAIT_S}s")

    @staticmethod
    def _pid_from(path: Path) -> int | None:
        text = path.read_text().strip() if path.exists() else ""
        return int(text) if text else None

    async def _succeeds(self, *args: str) -> bool:
        process = await asyncio.create_subprocess_exec(
            "podman", *args, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL
        )
        return await process.wait() == 0

    async def _podman(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "podman", *args, stdin=DEVNULL, stdout=PIPE, stderr=PIPE
        )
        output, errors = await process.communicate()

        if process.returncode != 0:
            raise PodmanError(f"podman {args[0]} failed: {errors.decode().strip()}")

        return output.decode().strip()
