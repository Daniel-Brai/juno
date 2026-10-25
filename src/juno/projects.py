import hashlib
import os
import tomllib
from pathlib import Path
from typing import ClassVar
from uuid import uuid7

from juno.config import McpFile, McpServer
from juno.domain.values import RunPaths


class ProjectSessions:
    INSTRUCTIONS: ClassVar[tuple[str, ...]] = ("AGENTS.md", ".juno/instructions.md")
    DATABASE: ClassVar[str] = "run.db"
    MCP: ClassVar[str] = "mcp.toml"

    def __init__(self, project: Path, data_home: Path | None = None) -> None:
        self.project = project.resolve()
        home = data_home or Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
        digest = hashlib.sha256(str(self.project).encode()).hexdigest()[:12]
        self.root = home / "juno" / "projects" / f"{self.project.name}-{digest}"

    def new(self) -> RunPaths:
        return self._paths(uuid7().hex)

    def latest(self) -> RunPaths | None:
        sessions = self.all()
        return sessions[-1] if sessions else None

    def find(self, session_id: str) -> RunPaths:
        matches = [paths for paths in self.all() if paths.root.name.startswith(session_id)]

        if len(matches) != 1:
            found = "no session" if not matches else f"{len(matches)} sessions"
            raise LookupError(f"{found} in {self.project} start with {session_id!r}")

        return matches[0]

    def all(self) -> list[RunPaths]:
        if not self.root.is_dir():
            return []

        folders = sorted(
            folder
            for folder in self.root.iterdir()
            if folder.is_dir() and (folder / self.DATABASE).is_file()
        )
        return [self._paths(folder.name) for folder in folders]

    def mcp_servers(self, user_dir: Path) -> dict[str, McpServer]:
        servers: dict[str, McpServer] = {}

        for path in (user_dir / self.MCP, self.project / ".juno" / self.MCP):
            if path.is_file():
                servers |= McpFile.model_validate(tomllib.loads(path.read_text())).servers

        return servers

    def instructions(self) -> str | None:
        for name in self.INSTRUCTIONS:
            path = self.project / name

            if path.is_file():
                return path.read_text()

        return None

    def _paths(self, session_id: str) -> RunPaths:
        return RunPaths(root=self.root / session_id, project=self.project)
