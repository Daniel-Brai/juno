from pathlib import Path

from juno.tools.registry import ToolError


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, raw: str) -> Path:
        path = (self.root / raw).resolve()

        if not path.is_relative_to(self.root):
            raise ToolError(f"{raw} is outside the workspace")

        return path

    def relative(self, path: Path) -> str:
        resolved = path.resolve()
        return (
            str(resolved.relative_to(self.root))
            if resolved.is_relative_to(self.root)
            else str(path)
        )
