import asyncio
import os
import shutil
from asyncio.subprocess import DEVNULL, PIPE
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from pathspec import GitIgnoreSpec

from juno.domain.commands import FindFiles, ReadFile, SearchText
from juno.tools.registry import Tool, ToolError
from juno.tools.workspace import Workspace


class SearchTools:
    MAX_FILES: ClassVar[int] = 500
    MAX_LINE: ClassVar[int] = 2_000
    GENERATED: ClassVar[tuple[str, ...]] = (
        ".git", ".hg", ".svn", ".jj",
        "_build", "deps", ".elixir_ls", ".lexical", "ebin", "_checkouts", "cover",
        "dist-newstyle", ".stack-work", ".cabal-sandbox",
        "target",
        "build", "cmake-build-*", "CMakeFiles", ".cache", ".ccls-cache", ".xmake", "bazel-*",
        "node_modules", "bower_components", "dist", ".next", ".nuxt", ".svelte-kit",
        ".turbo", ".parcel-cache", ".angular", ".expo", "coverage",
        ".gradle", ".idea", ".bsp", ".metals", ".bloop", ".kotlin",
        "vendor",
        ".build", "DerivedData", "Pods", ".swiftpm",
        ".dart_tool", ".pub-cache",
        "obj",
        "zig-cache", ".zig-cache", "zig-out",
        "_opam", "_esy",
        "elm-stuff",
        ".terraform", ".direnv", "result",
        ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
        ".tox", ".nox", ".ipynb_checkpoints", "*.egg-info",
    )  # fmt: skip

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._in_git: bool | None = None

    def tools(self) -> list[Tool[Any]]:
        return [
            Tool(
                name="glob",
                description="Find files by glob pattern, skipping files git ignores.",
                input_model=FindFiles,
                handler=self.glob,
                read_only=True,
            ),
            Tool(
                name="grep",
                description="Search file contents with a regular expression; shows path:line.",
                input_model=SearchText,
                handler=self.grep,
                read_only=True,
            ),
            Tool(
                name="read",
                description="Read a file with line numbers, optionally a range of lines.",
                input_model=ReadFile,
                handler=self.read,
                read_only=True,
            ),
        ]

    async def glob(self, args: FindFiles) -> str:
        base = self._workspace.resolve(args.path)
        pattern = args.pattern if "/" in args.pattern else f"**/{args.pattern}"
        files = self._matching(await self._files(base), pattern.removeprefix("./"))

        if not files:
            return "no files match"

        shown = sorted(files)[: self.MAX_FILES]
        more = len(files) - len(shown)
        prefix = self._workspace.relative(base)
        lines = [path if prefix == "." else f"{prefix}/{path}" for path in shown]
        return "\n".join(lines) + (f"\n... and {more} more" if more else "")

    async def grep(self, args: SearchText) -> str:
        target = self._workspace.resolve(args.path)
        argv = self._grep_argv(args, target, skip_generated=not await self._git_tracked())
        code, output, errors = await self._run(argv)

        if code == 1:
            return "no matches"

        if code != 0:
            raise ToolError(errors.strip() or f"search failed with exit code {code}")

        root = f"{self._workspace.root}/"
        return "\n".join(line.removeprefix(root) for line in output.splitlines())

    async def read(self, args: ReadFile) -> str:
        return await asyncio.to_thread(self._read, args)

    def _read(self, args: ReadFile) -> str:
        path = self._workspace.resolve(args.path)

        if path.is_dir():
            raise ToolError(f"{args.path} is a folder; use glob to list what is in it")

        if not path.is_file():
            raise ToolError(f"{args.path} does not exist; use glob to find the file first")

        data = path.read_bytes()

        if b"\0" in data[:8_192]:
            raise ToolError(f"{args.path} looks binary")

        lines = data.decode(errors="replace").splitlines()
        chosen = lines[args.offset - 1 : args.offset - 1 + args.limit]

        if not chosen:
            return f"{args.path} has {len(lines)} lines"

        body = "\n".join(
            f"{number:>6}\t{self._clip(line)}"
            for number, line in enumerate(chosen, start=args.offset)
        )
        end = args.offset + len(chosen) - 1
        return body if end >= len(lines) else f"{body}\n... {len(lines) - end} more lines"

    @staticmethod
    def _matching(paths: list[str], pattern: str) -> list[str]:
        return [path for path in paths if PurePosixPath(path).full_match(pattern)]

    def _clip(self, line: str) -> str:
        return line if len(line) <= self.MAX_LINE else f"{line[: self.MAX_LINE]}..."

    @classmethod
    def generated(cls, relative: str) -> bool:
        folders = PurePosixPath(relative).parts[:-1]
        return any(fnmatchcase(part, name) for part in folders for name in cls.GENERATED)

    def _grep_argv(self, args: SearchText, target: Path, *, skip_generated: bool) -> list[str]:
        skipped = self.GENERATED if skip_generated else (".git",)

        if shutil.which("rg"):
            argv = ["rg", "--line-number", "--no-heading", "--color", "never", "--hidden"]
            argv += ["--no-require-git", "--max-columns", "300", "--context", str(args.context)]
            argv += ["--ignore-case"] if args.ignore_case else []
            argv += ["--glob", args.glob] if args.glob else []

            for name in skipped:
                argv += ["--glob", f"!{name}/"]

            return [*argv, "--", args.pattern, str(target)]

        if shutil.which("git"):
            argv = ["git", "grep", "--no-index", "--exclude-standard", "-n", "-E"]
            argv += [f"-C{args.context}"] + (["-i"] if args.ignore_case else [])
            argv += ["-e", args.pattern, "--", str(target)]
            argv += [f":(glob){args.glob}"] if args.glob else []
            return argv + [f":(exclude,glob)**/{name}/**" for name in skipped]

        raise ToolError("grep needs ripgrep or git installed")

    async def _git_tracked(self) -> bool:
        if self._in_git is None:
            code, output, _ = await self._run(["git", "rev-parse", "--is-inside-work-tree"])
            self._in_git = code == 0 and output.strip() == "true"

        return self._in_git

    async def _files(self, base: Path) -> list[str]:
        if await self._git_tracked():
            code, output, _ = await self._run(
                ["git", "ls-files", "-co", "--exclude-standard", "-z", "--", "."], cwd=base
            )

            if code == 0:
                return [path for path in output.split("\0") if path]

        if shutil.which("rg"):
            argv = ["rg", "--files", "--hidden", "--no-require-git", "--null"]

            for name in self.GENERATED:
                argv += ["--glob", f"!{name}/"]

            code, output, _ = await self._run(argv, cwd=base)

            if code in {0, 1}:
                return [path for path in output.split("\0") if path and not self.generated(path)]

        return await asyncio.to_thread(self._walk, base)

    def _walk(self, base: Path) -> list[str]:
        root = self._workspace.root
        rules = IgnoreFiles(root)
        found: list[str] = []

        for parent in reversed(base.parents):
            if parent.is_relative_to(root):
                rules.enter(parent)

        for folder, subfolders, files in os.walk(base):
            here = Path(folder)
            rules.enter(here)
            subfolders[:] = [
                name
                for name in subfolders
                if not any(fnmatchcase(name, pattern) for pattern in self.GENERATED)
                and not rules.ignores(here / name, folder=True)
            ]
            found += [
                (here / name).relative_to(base).as_posix()
                for name in files
                if not rules.ignores(here / name, folder=False)
            ]

        return found

    async def _run(self, argv: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd or self._workspace.root, stdin=DEVNULL, stdout=PIPE, stderr=PIPE
        )
        output, errors = await process.communicate()
        return process.returncode or 0, output.decode(errors="replace"), errors.decode()


class IgnoreFiles:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._rules: dict[tuple[str, ...], GitIgnoreSpec] = {}

    def enter(self, folder: Path) -> None:
        file = folder / ".gitignore"

        if file.is_file():
            lines = file.read_text(errors="replace").splitlines()
            self._rules[folder.relative_to(self._root).parts] = GitIgnoreSpec.from_lines(lines)

    def ignores(self, path: Path, *, folder: bool) -> bool:
        parts = path.relative_to(self._root).parts
        ignored = False

        for base, spec in self._rules.items():
            if parts[: len(base)] != base:
                continue

            inside = "/".join(parts[len(base) :]) + ("/" if folder else "")
            verdict = spec.check_file(inside).include

            if verdict is not None:
                ignored = verdict

        return ignored
