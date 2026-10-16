import tomllib
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from juno.config import Seconds
from juno.tasks import TaskFolder


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskEntry(Entry):
    path: Path
    answers: Path

    def resolved(self, base: Path) -> TaskEntry:
        return TaskEntry(path=(base / self.path).resolve(), answers=(base / self.answers).resolve())


class Suite(Entry):
    tasks: tuple[TaskEntry, ...]


class Arm(Entry):
    flags: dict[str, bool] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class ExperimentSettings(Entry):
    name: str
    budget: Seconds
    max_cost: float | None = Field(default=None, gt=0)
    max_total_cost: float | None = Field(default=None, gt=0)
    seeds: int = Field(default=1, ge=1)
    parallel: int = Field(default=1, ge=1)
    gpus: tuple[int, ...] = ()
    gpus_per_run: int = Field(default=0, ge=0)
    retries: int = Field(default=1, ge=0)
    runs: Path = Path("runs")

    @model_validator(mode="after")
    def _enough_gpus(self) -> Self:
        if self.gpus_per_run > len(self.gpus):
            raise ValueError(f"gpus_per_run is {self.gpus_per_run} but only {len(self.gpus)} gpus")

        return self


class RunPlan(Entry):
    arm: str
    task_id: str
    task: TaskEntry
    seed: int
    folder: Path
    budget: float
    max_cost: float | None = None
    flags: dict[str, bool]
    settings: dict[str, Any]


class Experiment(Entry):
    SAFE_NAME: ClassVar[frozenset[str]] = frozenset("-_")

    experiment: ExperimentSettings
    tasks: tuple[TaskEntry, ...] = ()
    suite: Path | None = None
    arms: dict[str, Arm]

    @field_validator("arms")
    @classmethod
    def _named_safely(cls, arms: dict[str, Arm]) -> dict[str, Arm]:
        if not arms:
            raise ValueError("an experiment needs at least one arm")

        for name in arms:
            if not name or not all(char.isalnum() or char in cls.SAFE_NAME for char in name):
                raise ValueError(f"arm name {name!r} may only use letters, digits, - and _")

        return arms

    @classmethod
    def load(cls, manifest: Path) -> Experiment:
        manifest = manifest.resolve()
        base = manifest.parent
        raw = cls.model_validate(tomllib.loads(manifest.read_text()))
        tasks = [entry.resolved(base) for entry in raw.tasks]

        if raw.suite is not None:
            suite_path = (base / raw.suite).resolve()
            suite = Suite.model_validate(tomllib.loads(suite_path.read_text()))
            tasks += [entry.resolved(suite_path.parent) for entry in suite.tasks]

        if not tasks:
            raise ValueError(f"{manifest} lists no tasks; add [[tasks]] or a suite")

        settings = raw.experiment.model_copy(
            update={"runs": (base / raw.experiment.runs).resolve()}
        )
        return raw.model_copy(update={"experiment": settings, "tasks": tuple(tasks), "suite": None})

    @property
    def folder(self) -> Path:
        return self.experiment.runs / self.experiment.name

    @property
    def base_arm(self) -> str:
        return next(iter(self.arms))

    def plans(self) -> list[RunPlan]:
        task_ids = [TaskFolder.load(entry.path).id for entry in self.tasks]

        if len(set(task_ids)) < len(task_ids):
            raise ValueError(f"task ids repeat: {sorted(task_ids)}")

        return [
            RunPlan(
                arm=name,
                task_id=task_id,
                task=entry,
                seed=seed,
                folder=self.folder / name / task_id / f"seed-{seed}",
                budget=self.experiment.budget,
                max_cost=self.experiment.max_cost,
                flags=arm.flags,
                settings=arm.settings,
            )
            for name, arm in self.arms.items()
            for task_id, entry in zip(task_ids, self.tasks, strict=True)
            for seed in range(1, self.experiment.seeds + 1)
        ]
