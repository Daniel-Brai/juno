import asyncio
from collections.abc import Iterable, Sequence
from contextvars import ContextVar, Token
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Self

import aiosqlite

from juno.domain.events import Event
from juno.domain.session import Session
from juno.domain.values import (
    Checkpoint,
    Job,
    MessageRecord,
    Submission,
    ToolCall,
    ToolCallRecord,
    ToolCallStatus,
    ToolResult,
)

type Row = Sequence[Any]


class SqliteRepo:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def _run(self, sql: str, params: Iterable[Any] = ()) -> None:
        await self._connection.execute(sql, tuple(params))

    async def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[Row]:
        async with self._connection.execute(sql, tuple(params)) as cursor:
            return list(await cursor.fetchall())

    async def _row(self, sql: str, params: Iterable[Any] = ()) -> Row | None:
        async with self._connection.execute(sql, tuple(params)) as cursor:
            return await cursor.fetchone()


class SqliteSessionRepo(SqliteRepo):
    async def save(self, session: Session) -> None:
        await self._run(
            "INSERT INTO sessions (id, task_id, data_json) VALUES (?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET data_json = excluded.data_json",
            (session.id, session.task_id, session.model_dump_json(exclude={"messages"})),
        )

    async def get(self, session_id: str) -> Session | None:
        row = await self._row("SELECT data_json FROM sessions WHERE id = ?", (session_id,))
        return Session.model_validate_json(row[0]) if row else None

    async def latest(self) -> Session | None:
        row = await self._row("SELECT data_json FROM sessions ORDER BY rowid DESC LIMIT 1")
        return Session.model_validate_json(row[0]) if row else None


class SqliteMessageRepo(SqliteRepo):
    async def append(self, session_id: str, record: MessageRecord) -> None:
        await self._run(
            "INSERT INTO messages (session_id, idx, role, request_hash, data_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                record.index,
                record.message.role,
                record.request_hash,
                record.model_dump_json(),
            ),
        )

    async def all(self, session_id: str) -> list[MessageRecord]:
        rows = await self._rows(
            "SELECT data_json FROM messages WHERE session_id = ? ORDER BY idx", (session_id,)
        )
        return [MessageRecord.model_validate_json(row[0]) for row in rows]


class SqliteToolCallRepo(SqliteRepo):
    async def record_intent(self, session_id: str, call: ToolCall) -> None:
        record = ToolCallRecord(call=call, status=ToolCallStatus.INTENT)
        await self._run(
            "INSERT INTO tool_calls (session_id, tool_use_id, name, status, data_json) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (session_id, tool_use_id) "
            "DO UPDATE SET status = excluded.status, data_json = excluded.data_json",
            (session_id, call.id, call.name, record.status, record.model_dump_json()),
        )

    async def record_result(
        self, session_id: str, result: ToolResult, status: ToolCallStatus = ToolCallStatus.DONE
    ) -> None:
        record = await self.find(session_id, result.call_id)

        if record is None:
            raise LookupError(f"no intent recorded for tool call {result.call_id}")

        finished = record.model_copy(update={"status": status, "result": result})
        await self._run(
            "UPDATE tool_calls SET status = ?, data_json = ? WHERE session_id = ? AND tool_use_id = ?",
            (status, finished.model_dump_json(), session_id, result.call_id),
        )

    async def find(self, session_id: str, call_id: str) -> ToolCallRecord | None:
        row = await self._row(
            "SELECT data_json FROM tool_calls WHERE session_id = ? AND tool_use_id = ?",
            (session_id, call_id),
        )
        return ToolCallRecord.model_validate_json(row[0]) if row else None


class SqliteJobRepo(SqliteRepo):
    async def save(self, session_id: str, job: Job) -> None:
        await self._run(
            "INSERT INTO jobs (session_id, id, state, data_json) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (session_id, id) DO UPDATE SET "
            "state = excluded.state, data_json = excluded.data_json",
            (session_id, job.id, job.state, job.model_dump_json()),
        )

    async def all(self, session_id: str) -> list[Job]:
        rows = await self._rows(
            "SELECT data_json FROM jobs WHERE session_id = ? ORDER BY rowid", (session_id,)
        )
        return [Job.model_validate_json(row[0]) for row in rows]


class SqliteSubmissionRepo(SqliteRepo):
    async def add(self, session_id: str, submission: Submission) -> None:
        await self._run(
            "INSERT INTO submissions (id, session_id, sha256, val_score, data_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                submission.id,
                session_id,
                submission.sha256,
                submission.val_score,
                submission.model_dump_json(),
            ),
        )

    async def all(self, session_id: str) -> list[Submission]:
        rows = await self._rows(
            "SELECT data_json FROM submissions WHERE session_id = ? ORDER BY rowid", (session_id,)
        )
        return [Submission.model_validate_json(row[0]) for row in rows]

    async def best(self, session_id: str) -> Submission | None:
        best: Submission | None = None

        for submission in await self.all(session_id):
            if best is None or submission.beats(best):
                best = submission

        return best


class SqliteCheckpointRepo(SqliteRepo):
    async def add(self, session_id: str, checkpoint: Checkpoint) -> None:
        await self._run(
            "INSERT INTO checkpoints (session_id, message_index, commit_sha, data_json) "
            "VALUES (?, ?, ?, ?)",
            (session_id, checkpoint.message_index, checkpoint.commit, checkpoint.model_dump_json()),
        )

    async def all(self, session_id: str) -> list[Checkpoint]:
        rows = await self._rows(
            "SELECT data_json FROM checkpoints WHERE session_id = ? ORDER BY rowid", (session_id,)
        )
        return [Checkpoint.model_validate_json(row[0]) for row in rows]


class SqliteEventRepo(SqliteRepo):
    async def add(self, event: Event) -> None:
        await self._run(
            "INSERT INTO events (session_id, at, type, data_json) VALUES (?, ?, ?, ?)",
            (event.session_id, event.at, type(event).__name__, event.model_dump_json()),
        )

    async def count(self, session_id: str) -> int:
        row = await self._row("SELECT count(*) FROM events WHERE session_id = ?", (session_id,))
        return int(row[0]) if row else 0

    async def types(self, session_id: str) -> list[str]:
        rows = await self._rows(
            "SELECT type FROM events WHERE session_id = ? ORDER BY id", (session_id,)
        )
        return [str(row[0]) for row in rows]


class SqliteTransaction:
    def __init__(self, store: SqliteStore) -> None:
        connection = store.connection
        self.sessions = SqliteSessionRepo(connection)
        self.messages = SqliteMessageRepo(connection)
        self.tool_calls = SqliteToolCallRepo(connection)
        self.jobs = SqliteJobRepo(connection)
        self.submissions = SqliteSubmissionRepo(connection)
        self.checkpoints = SqliteCheckpointRepo(connection)
        self.events = SqliteEventRepo(connection)
        self._store = store
        self._open = False
        self._token: Token[bool] | None = None

    async def __aenter__(self) -> Self:
        self._token = await self._store.claim()

        try:
            await self._store.connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            self._store.release(self._token)
            raise

        self._open = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._open:
                await self.rollback()
        finally:
            if self._token is not None:
                self._store.release(self._token)
                self._token = None

    async def commit(self) -> None:
        await self._store.connection.execute("COMMIT")
        self._open = False

    async def rollback(self) -> None:
        await self._store.connection.execute("ROLLBACK")
        self._open = False


class SqliteStore:
    VERSION: ClassVar[int] = 1
    PRAGMAS: ClassVar[tuple[str, ...]] = (
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = NORMAL",
        "PRAGMA busy_timeout = 5000",
        "PRAGMA foreign_keys = ON",
    )
    SCHEMA: ClassVar[str] = """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE TABLE messages (
            session_id TEXT NOT NULL REFERENCES sessions (id),
            idx INTEGER NOT NULL,
            role TEXT NOT NULL,
            request_hash TEXT,
            data_json TEXT NOT NULL,
            PRIMARY KEY (session_id, idx)
        );
        CREATE TABLE tool_calls (
            session_id TEXT NOT NULL REFERENCES sessions (id),
            tool_use_id TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (session_id, tool_use_id)
        );
        CREATE TABLE jobs (
            session_id TEXT NOT NULL REFERENCES sessions (id),
            id TEXT NOT NULL,
            state TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (session_id, id)
        );
        CREATE TABLE submissions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions (id),
            sha256 TEXT NOT NULL,
            val_score REAL NOT NULL,
            data_json TEXT NOT NULL,
            UNIQUE (session_id, sha256)
        );
        CREATE TABLE checkpoints (
            session_id TEXT NOT NULL REFERENCES sessions (id),
            message_index INTEGER NOT NULL,
            commit_sha TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            at REAL NOT NULL,
            type TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
    """

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection
        self._lock = asyncio.Lock()
        self._holding: ContextVar[bool] = ContextVar(f"juno-store-{id(self)}", default=False)

    @classmethod
    async def open(cls, path: Path) -> Self:
        connection = await aiosqlite.connect(path, isolation_level=None)

        for pragma in cls.PRAGMAS:
            await connection.execute(pragma)

        store = cls(connection)

        try:
            await store.migrate()
        except BaseException:
            await connection.close()
            raise

        return store

    async def close(self) -> None:
        await self.connection.close()

    def transaction(self) -> SqliteTransaction:
        return SqliteTransaction(self)

    async def claim(self) -> Token[bool]:
        if self._holding.get():
            raise RuntimeError("transactions cannot nest; finish the open one first")

        await self._lock.acquire()
        return self._holding.set(True)

    def release(self, token: Token[bool]) -> None:
        self._holding.reset(token)
        self._lock.release()

    async def migrate(self) -> None:
        async with self.connection.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()

        version = int(row[0]) if row else 0

        if version == 0:
            await self.connection.executescript(self.SCHEMA)
            await self.connection.execute(f"PRAGMA user_version = {self.VERSION}")
        elif version != self.VERSION:
            raise RuntimeError(f"run.db schema version {version} is not supported")
