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
    command: str = Field(min_length=1, description="Shell command to run in the workspace.")
    timeout_s: float = Field(
        default=600, gt=0, description="Seconds before the command and its children are killed."
    )


class EditFile(Command):
    action: Literal["view", "create", "str_replace", "insert"] = Field(
        description="view a file or folder, create or overwrite a file, replace one unique "
        "snippet, or insert text after a line."
    )
    path: str = Field(description="Path relative to the workspace.")
    file_text: str | None = Field(default=None, description="Full content for create.")
    old_str: str | None = Field(default=None, description="Exact text to replace; must occur once.")
    new_str: str | None = Field(default=None, description="Replacement or inserted text.")
    insert_line: int | None = Field(
        default=None, ge=0, description="Insert after this line; 0 inserts at the top."
    )
    view_range: tuple[int, int] | None = Field(
        default=None, description="First and last line to view, 1-based and inclusive."
    )


class SubmitJob(Command):
    command: str = Field(min_length=1, description="Shell command to run in the background.")
    name: str = Field(min_length=1, description="Short label for the job.")


class WaitJob(Command):
    job_id: str
    timeout_s: float = Field(gt=0, description="Stop waiting after this many seconds.")


class InspectJob(Command):
    job_id: str
    tail_lines: int = Field(default=40, ge=0, le=500, description="Log lines to return.")


class CancelJob(Command):
    job_id: str


class ListJobs(Command):
    pass


class RegisterSubmission(Command):
    path: str = Field(description="Submission file, relative to the workspace.")
    val_score: float = Field(description="Score on your own validation split.")
    notes: str = Field(default="", description="What produced this submission.")


class ListSubmissions(Command):
    pass


class QuerySystem(Command):
    pass
