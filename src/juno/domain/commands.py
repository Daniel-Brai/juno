from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator

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
        default=600,
        gt=0,
        description="Seconds to wait for the command before it moves to the background.",
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


class FindFiles(Command):
    pattern: str = Field(
        min_length=1, description="Glob such as **/*.py or src/*.ts; no slash means any folder."
    )
    path: str = Field(default=".", description="Folder to search, relative to the project.")


class SearchText(Command):
    pattern: str = Field(min_length=1, description="Regular expression to search for.")
    path: str = Field(default=".", description="File or folder to search.")
    glob: str | None = Field(default=None, description="Only search files matching this glob.")
    context: int = Field(default=0, ge=0, le=20, description="Lines to show around each match.")
    ignore_case: bool = False


class ReadFile(Command):
    path: str = Field(description="File to read, relative to the project.")
    offset: int = Field(default=1, ge=1, description="First line to read, 1-based.")
    limit: int = Field(default=2_000, ge=1, le=5_000, description="Most lines to read.")


class TodoItem(Value):
    text: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "done"] = "pending"


class UpdateTodos(Command):
    items: list[TodoItem] = Field(description="The whole list; it replaces the previous one.")


class RemoteArgs(Command):
    model_config = ConfigDict(frozen=True, extra="allow")

    @property
    def values(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class FetchUrl(Command):
    url: str = Field(description="An http or https address.")
    max_chars: int = Field(default=20_000, ge=500, le=100_000, description="Most text to return.")

    @field_validator("url")
    @classmethod
    def _web_address(cls, url: str) -> str:
        parts = urlsplit(url.strip())

        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("expected an http or https address")

        return url.strip()
