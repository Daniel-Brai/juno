import contextlib
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

from juno.domain.session import Session
from juno.domain.values import MessageRecord, RunPaths, Submission, Task, Usage
from juno.tasks import TaskFolder


class RunFolder:
    DATABASE: ClassVar[str] = "run.db"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = RunPaths(root=root)

    @property
    def database(self) -> Path:
        return self.root / self.DATABASE

    def session(self) -> Session | None:
        if not self.database.is_file():
            return None

        with self._connect() as db:
            row = db.execute(
                "SELECT data_json FROM sessions ORDER BY rowid DESC LIMIT 1"
            ).fetchone()

        return Session.model_validate_json(row[0]) if row else None

    def submissions(self, session_id: str) -> list[Submission]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT data_json FROM submissions WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            ).fetchall()

        return [Submission.model_validate_json(row[0]) for row in rows]

    def usage(self, session_id: str) -> Usage:
        with self._connect() as db:
            rows = db.execute(
                "SELECT data_json FROM messages WHERE session_id = ? AND role = 'assistant'",
                (session_id,),
            ).fetchall()

        records = (MessageRecord.model_validate_json(row[0]) for row in rows)
        return sum((record.usage for record in records if record.usage is not None), Usage())

    def opening(self, session_id: str) -> tuple[int, str]:
        with self._connect() as db:
            (count,) = db.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            row = db.execute(
                "SELECT data_json FROM messages WHERE session_id = ? AND role = 'user' "
                "ORDER BY idx LIMIT 1",
                (session_id,),
            ).fetchone()

        content = MessageRecord.model_validate_json(row[0]).message.content if row else ""
        return count, content if isinstance(content, str) else ""

    def task(self) -> Task:
        return TaskFolder.from_run(self.paths)

    def submission_file(self, submission: Submission) -> Path:
        return self.paths.submissions / submission.path.name

    @contextlib.contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)

        try:
            yield connection
        finally:
            connection.close()
