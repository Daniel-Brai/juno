from typing import Literal

from pydantic import Field

from juno.domain.values import Value


class Command(Value):
    pass


class StartSession(Command):
    task_id: str
    budget_s: float = Field(gt=0)
    seed: int = 0


class ResumeSession(Command):
    session_id: str


class RunBash(Command):
    command: str = Field(min_length=1)
    timeout_s: float = Field(default=600, gt=0)


class EditFile(Command):
    action: Literal["view", "create", "str_replace", "insert"]
    path: str
    file_text: str | None = None
    old_str: str | None = None
    new_str: str | None = None
    insert_line: int | None = Field(default=None, ge=0)
    view_range: tuple[int, int] | None = None


class SubmitJob(Command):
    command: str = Field(min_length=1)
    name: str


class WaitJob(Command):
    job_id: str
    timeout_s: float = Field(gt=0)


class CancelJob(Command):
    job_id: str


class RegisterSubmission(Command):
    path: str
    val_score: float
    notes: str = ""


class QuerySystem(Command):
    pass
