import asyncio
from asyncio.subprocess import DEVNULL, PIPE
from pathlib import Path
from typing import ClassVar


class GitCheckpointer:
    IDENTITY: ClassVar[tuple[str, ...]] = (
        "-c",
        "user.name=juno",
        "-c",
        "user.email=juno@localhost",
        "-c",
        "commit.gpgsign=false",
    )

    def __init__(self, workspace: Path, max_file_bytes: int = 50 * 1024**2) -> None:
        self._workspace = workspace
        self._max_file_bytes = max_file_bytes

    async def checkpoint(self, message: str) -> str | None:
        if not (self._workspace / ".git").exists():
            await self._git("init", "-q")

        paths = [path for path in await self._changed_paths() if self._small_enough(path)]

        if not paths:
            return None

        await self._git(
            "add", "--pathspec-from-file=-", "--pathspec-file-nul", stdin="\0".join(paths)
        )
        await self._git("commit", "-q", "--no-verify", "-m", message)
        return (await self._git("rev-parse", "HEAD")).strip()

    async def _changed_paths(self) -> list[str]:
        output = await self._git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        entries = iter(output.split("\0"))
        paths: list[str] = []

        for entry in entries:
            if len(entry) < 4:
                continue

            paths.append(entry[3:])

            if "R" in entry[:2] or "C" in entry[:2]:
                next(entries, None)

        return paths

    def _small_enough(self, relative: str) -> bool:
        path = self._workspace / relative
        return not path.is_file() or path.stat().st_size <= self._max_file_bytes

    async def _git(self, *args: str, stdin: str | None = None) -> str:
        process = await asyncio.create_subprocess_exec(
            "git",
            *self.IDENTITY,
            *args,
            cwd=self._workspace,
            stdin=PIPE if stdin is not None else DEVNULL,
            stdout=PIPE,
            stderr=PIPE,
        )
        output, errors = await process.communicate(stdin.encode() if stdin is not None else None)

        if process.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {errors.decode().strip()}")

        return output.decode()
