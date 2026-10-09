import tomllib
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from juno.domain.values import Direction, RunPaths, Task


class TaskFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    metric: str
    direction: Direction
    submission: str = "submission.csv"
    description: str = "description.md"
    data: str = "data"


class TaskFolder:
    MANIFEST: ClassVar[str] = "task.toml"
    SNAPSHOT: ClassVar[str] = "task.json"

    @classmethod
    def load(cls, folder: Path) -> Task:
        folder = folder.resolve()
        manifest = folder / cls.MANIFEST

        if not manifest.is_file():
            raise FileNotFoundError(f"{folder} has no {cls.MANIFEST}")

        spec = TaskFile.model_validate(tomllib.loads(manifest.read_text()))
        description = folder / spec.description
        data = folder / spec.data

        if not description.is_file():
            raise FileNotFoundError(f"{description} is missing")

        if not data.is_dir():
            raise FileNotFoundError(f"{data} is missing")

        return Task(
            id=spec.id,
            description=description.read_text(),
            data_dir=data,
            metric=spec.metric,
            direction=spec.direction,
            submission_file=spec.submission,
        )

    @classmethod
    def save(cls, task: Task, paths: RunPaths) -> None:
        paths.root.mkdir(parents=True, exist_ok=True)
        (paths.root / cls.SNAPSHOT).write_text(task.model_dump_json(indent=2))

    @classmethod
    def from_run(cls, paths: RunPaths) -> Task:
        snapshot = paths.root / cls.SNAPSHOT

        if not snapshot.is_file():
            raise FileNotFoundError(f"{paths.root} is not a juno run folder")

        return Task.model_validate_json(snapshot.read_text())
