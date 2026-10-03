import asyncio
from pathlib import Path

from juno.domain.commands import EditFile
from juno.tools.registry import Tool, ToolError
from juno.tools.workspace import Workspace


class EditorTools:
    def __init__(self, workspace: Workspace, view_limit: int = 2_000) -> None:
        self._workspace = workspace
        self._view_limit = view_limit

    def tools(self) -> list[Tool[EditFile]]:
        return [
            Tool(
                name="editor",
                description=(
                    "View, create and edit text files in the workspace. str_replace needs "
                    "old_str to match exactly once; view shows numbered lines."
                ),
                input_model=EditFile,
                handler=self.run,
            )
        ]

    async def run(self, args: EditFile) -> str:
        return await asyncio.to_thread(self.apply, args)

    def apply(self, args: EditFile) -> str:
        path = self._workspace.resolve(args.path)

        match args.action:
            case "view":
                return self._view(path, args.view_range)
            case "create":
                return self._create(path, self._require(args.file_text, "file_text"))
            case "str_replace":
                return self._replace(
                    path, self._require(args.old_str, "old_str"), args.new_str or ""
                )
            case "insert":
                line = self._require(args.insert_line, "insert_line")
                return self._insert(path, line, self._require(args.new_str, "new_str"))

    def _view(self, path: Path, view_range: tuple[int, int] | None) -> str:
        if path.is_dir():
            entries = sorted(path.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name))
            return "\n".join(
                f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries
            )

        lines = self._read(path).splitlines()
        first, last = view_range or (1, len(lines))

        if not 1 <= first <= max(last, 1) or last > max(len(lines), 1):
            raise ToolError(f"view_range must lie within 1-{len(lines)}")

        shown = lines[first - 1 : last][: self._view_limit]
        numbered = "\n".join(f"{number:>6}\t{line}" for number, line in enumerate(shown, first))
        hidden = (last - first + 1) - len(shown)
        return f"{numbered}\n[... {hidden} more lines ...]" if hidden > 0 else numbered

    def _create(self, path: Path, text: str) -> str:
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return f"{'overwrote' if existed else 'created'} {self._workspace.relative(path)}"

    def _replace(self, path: Path, old: str, new: str) -> str:
        text = self._read(path)
        count = text.count(old)

        if count != 1:
            raise ToolError(
                f"old_str occurs {count} times in {self._workspace.relative(path)}, it must occur once"
            )

        path.write_text(text.replace(old, new, 1))
        return f"edited {self._workspace.relative(path)}"

    def _insert(self, path: Path, line: int, new: str) -> str:
        lines = self._read(path).splitlines(keepends=True)

        if line > len(lines):
            raise ToolError(f"insert_line {line} is past the end ({len(lines)} lines)")

        lines.insert(line, new if new.endswith("\n") else f"{new}\n")
        path.write_text("".join(lines))
        return f"inserted after line {line} of {self._workspace.relative(path)}"

    def _read(self, path: Path) -> str:
        if not path.is_file():
            raise ToolError(f"{self._workspace.relative(path)} is not a file")

        return path.read_text()

    @staticmethod
    def _require[T](value: T | None, name: str) -> T:
        if value is None:
            raise ToolError(f"{name} is required for this action")

        return value
