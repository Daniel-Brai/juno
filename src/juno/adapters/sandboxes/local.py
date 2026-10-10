import asyncio
import os
import signal
import time
from asyncio.subprocess import DEVNULL, PIPE, STDOUT, Process
from collections import deque
from pathlib import Path
from typing import ClassVar
from uuid import uuid7

import psutil

from juno.domain.values import CommandResult, Job


class OutputBuffer:
    def __init__(self, limit: int) -> None:
        self.dropped = 0
        self._half = limit // 2
        self._head = bytearray()
        self._tail: deque[bytes] = deque()
        self._tail_size = 0

    @property
    def truncated(self) -> bool:
        return self.dropped > 0

    def add(self, chunk: bytes) -> None:
        room = self._half - len(self._head)

        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]

        if chunk:
            self._tail.append(chunk)
            self._tail_size += len(chunk)
            self._trim_tail()

    def text(self) -> str:
        head = bytes(self._head).decode(errors="replace")
        tail = b"".join(self._tail).decode(errors="replace")

        if not self.dropped:
            return head + tail

        return f"{head}\n[... {self.dropped} bytes omitted ...]\n{tail}"

    def _trim_tail(self) -> None:
        while self._tail_size > self._half:
            excess = self._tail_size - self._half
            first = self._tail[0]

            if len(first) <= excess:
                self._tail.popleft()
                cut = len(first)
            else:
                self._tail[0] = first[excess:]
                cut = excess

            self._tail_size -= cut
            self.dropped += cut


class LocalSandbox:
    KILL_GRACE_S: ClassVar[float] = 5.0
    DRAIN_TIMEOUT_S: ClassVar[float] = 5.0
    POLL_S: ClassVar[float] = 0.1
    START_TOLERANCE_S: ClassVar[float] = 1.0
    READ_CHUNK: ClassVar[int] = 65_536
    JOB_WRAPPER: ClassVar[str] = '( eval "$1" ); echo $? > "$2"'

    def __init__(self, workspace: Path, output_limit: int = 30_000) -> None:
        self._workspace = workspace
        self._output_limit = output_limit
        self._jobs: dict[int, Process] = {}

    async def exec(self, command: str, timeout_s: float) -> CommandResult:
        started = time.monotonic()
        token = uuid7().hex
        process = await asyncio.create_subprocess_exec(
            *self.exec_argv(command, token),
            cwd=self._workspace,
            stdin=DEVNULL,
            stdout=PIPE,
            stderr=STDOUT,
            process_group=0,
        )
        buffer = OutputBuffer(self._output_limit)
        reader = asyncio.create_task(self._drain(process, buffer))
        timed_out = False

        try:
            async with asyncio.timeout(timeout_s):
                await process.wait()
        except TimeoutError:
            timed_out = True
        finally:
            await self.stop_exec(process, token)
            await process.wait()
            await self._finish_reading(reader)

        return CommandResult(
            exit_code=None if timed_out else process.returncode,
            output=buffer.text(),
            truncated=buffer.truncated,
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
        )

    async def start(self) -> None:
        self._workspace.mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        for process in list(self._jobs.values()):
            if process.returncode is None:
                await self.kill_group(process.pid)

    def exec_argv(self, command: str, _token: str) -> list[str]:
        return ["bash", "-c", command]

    async def stop_exec(self, process: Process, _token: str) -> None:
        await self.kill_group(process.pid)

    async def spawn(self, command: str, log_path: Path, exit_path: Path) -> Job:
        with log_path.open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                self.JOB_WRAPPER,
                "juno-job",
                command,
                str(exit_path),
                cwd=self._workspace,
                stdin=DEVNULL,
                stdout=log,
                stderr=STDOUT,
                process_group=0,
            )

        self._jobs[process.pid] = process
        return Job(
            command=command,
            pid=process.pid,
            pgid=process.pid,
            proc_start=self._start_time(process.pid),
            log_path=log_path,
            exit_path=exit_path,
        )

    def poll(self, job: Job) -> int | None:
        if job.exit_path.exists() and (text := job.exit_path.read_text().strip()):
            return int(text)

        process = self._jobs.get(job.pid)
        return process.returncode if process is not None else None

    def is_alive(self, job: Job) -> bool:
        if (process := self._jobs.get(job.pid)) is not None:
            return process.returncode is None

        start = self._start_time(job.pid)
        same_process = job.proc_start is None or (
            start is not None and abs(start - job.proc_start) < self.START_TOLERANCE_S
        )
        return start is not None and same_process

    def memory_bytes(self, job: Job, /) -> int:
        total = 0

        for process in psutil.process_iter(["pid", "memory_info"]):
            try:
                if os.getpgid(process.info["pid"]) == job.pgid:
                    total += process.info["memory_info"].rss
            except ProcessLookupError, PermissionError, TypeError:
                continue

        return total

    async def kill_group(self, pgid: int) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not self._signal(pgid, sig) or await self._group_exits(pgid, self.KILL_GRACE_S):
                return

    async def _group_exits(self, pgid: int, within_s: float) -> bool:
        for _ in range(max(1, round(within_s / self.POLL_S))):
            if not self._signal(pgid, 0):
                return True

            await asyncio.sleep(self.POLL_S)

        return not self._signal(pgid, 0)

    async def _drain(self, process: Process, buffer: OutputBuffer) -> None:
        if process.stdout is None:
            return

        while chunk := await process.stdout.read(self.READ_CHUNK):
            buffer.add(chunk)

    async def _finish_reading(self, reader: asyncio.Task[None]) -> None:
        try:
            async with asyncio.timeout(self.DRAIN_TIMEOUT_S):
                await reader
        except TimeoutError:
            reader.cancel()

    @staticmethod
    def _start_time(pid: int) -> float | None:
        try:
            process = psutil.Process(pid)
            return None if process.status() == psutil.STATUS_ZOMBIE else process.create_time()
        except psutil.NoSuchProcess:
            return None

    @staticmethod
    def _signal(pgid: int, sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return False

        return True
