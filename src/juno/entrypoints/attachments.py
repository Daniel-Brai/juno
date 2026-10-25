from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Expanded:
    text: str
    attached: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class Attachments:
    LIMIT: ClassVar[int] = 100_000
    TRAILING: ClassVar[str] = ",.;:!?)]}\"'"

    def __init__(self, project: Path) -> None:
        self._project = project.resolve()

    def expand(self, text: str) -> Expanded:
        attached: list[str] = []
        skipped: list[str] = []
        blocks: list[str] = []

        for word in text.split():
            if not word.startswith("@") or len(word) < 2:
                continue

            name = word[1:].rstrip(self.TRAILING)

            if name in attached:
                continue

            body, problem = self._read(name)

            if problem:
                skipped.append(f"{name}: {problem}")
            else:
                attached.append(name)
                blocks.append(f'<file path="{name}">\n{body}\n</file>')

        return Expanded("\n\n".join([text, *blocks]), attached, skipped)

    def _read(self, name: str) -> tuple[str, str | None]:
        path = (self._project / name).resolve()

        if not path.is_relative_to(self._project):
            return "", "outside the project"

        if not path.is_file():
            return "", "no such file"

        data = path.read_bytes()

        if len(data) > self.LIMIT:
            return "", f"larger than {self.LIMIT // 1000} KB"

        if b"\0" in data[:8_192]:
            return "", "binary"

        return data.decode(errors="replace").rstrip("\n"), None
