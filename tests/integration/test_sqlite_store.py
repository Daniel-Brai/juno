import asyncio
import sqlite3
from pathlib import Path

import pytest

from juno.adapters.sqlite_store import SqliteStore
from juno.domain.events import TurnContinued
from juno.domain.session import Session, State
from juno.domain.values import (
    Checkpoint,
    Direction,
    Job,
    Message,
    MessageRecord,
    Role,
    Submission,
    ToolCall,
    ToolCallStatus,
    ToolResult,
    Usage,
)
from juno.ports import UnitOfWorkFactory

pytestmark = pytest.mark.anyio


class TestRepos:
    async def test_sessions_round_trip_and_update(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        session.state = State.RUNNING

        async with units() as uow:
            await uow.sessions.save(session)
            await uow.commit()

        async with units() as uow:
            stored = await uow.sessions.get(session.id)
            latest = await uow.sessions.latest()

        assert stored == session.model_copy(update={"messages": []})
        assert latest is not None and latest.id == session.id

    async def test_messages_keep_their_order_and_metadata(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        records = [
            MessageRecord(index=0, message=Message(role=Role.USER, content="task")),
            MessageRecord(
                index=1,
                message=Message(role=Role.ASSISTANT, content=[{"type": "text", "text": "hi"}]),
                request_hash="abc",
                usage=Usage(input_tokens=7),
            ),
        ]

        async with units() as uow:
            for record in reversed(records):
                await uow.messages.append(session.id, record)

            await uow.commit()

        async with units() as uow:
            assert await uow.messages.all(session.id) == records

    async def test_a_message_index_is_written_once(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        record = MessageRecord(index=0, message=Message(role=Role.USER, content="task"))

        async with units() as uow:
            await uow.messages.append(session.id, record)

            with pytest.raises(sqlite3.IntegrityError):
                await uow.messages.append(session.id, record)

    async def test_tool_calls_move_from_intent_to_result(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        call = ToolCall(id="toolu_1", name="bash", args={"command": "ls"})
        result = ToolResult(call_id="toolu_1", content="data/")

        async with units() as uow:
            await uow.tool_calls.record_intent(session.id, call)
            await uow.commit()

        async with units() as uow:
            intent = await uow.tool_calls.find(session.id, call.id)
            await uow.tool_calls.record_result(session.id, result)
            await uow.commit()

        async with units() as uow:
            done = await uow.tool_calls.find(session.id, call.id)

        assert intent is not None and intent.status is ToolCallStatus.INTENT
        assert done is not None and (done.status, done.result) == (ToolCallStatus.DONE, result)

    async def test_a_result_needs_an_intent(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        async with units() as uow:
            with pytest.raises(LookupError):
                await uow.tool_calls.record_result(session.id, ToolResult(call_id="x", content=""))

    async def test_jobs_are_upserted(self, units: UnitOfWorkFactory, session: Session) -> None:
        job = Job(
            id="job-1", command="sleep 1", pid=1, pgid=1, log_path=Path("a"), exit_path=Path("b")
        )

        async with units() as uow:
            await uow.jobs.save(session.id, job)
            await uow.jobs.save(session.id, job.model_copy(update={"exit_code": 0}))
            await uow.commit()

        async with units() as uow:
            assert [stored.exit_code for stored in await uow.jobs.all(session.id)] == [0]

    async def test_best_submission_follows_the_direction(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        async with units() as uow:
            for score, sha in [(0.7, "a"), (0.4, "b"), (0.9, "c")]:
                await uow.submissions.add(
                    session.id,
                    Submission(
                        path=Path(f"{sha}.csv"),
                        sha256=sha,
                        val_score=score,
                        direction=Direction.MINIMIZE,
                        created_at=0.0,
                    ),
                )

            await uow.commit()

        async with units() as uow:
            best = await uow.submissions.best(session.id)

        assert best is not None and best.sha256 == "b"

    async def test_checkpoints_and_events_are_appended(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        async with units() as uow:
            await uow.checkpoints.add(session.id, Checkpoint(commit="abc", message_index=3))
            await uow.events.add(TurnContinued(session_id=session.id, at=1.0, remaining_s=9))
            await uow.commit()

        async with units() as uow:
            assert await uow.checkpoints.all(session.id) == [
                Checkpoint(commit="abc", message_index=3)
            ]
            assert await uow.events.count(session.id) == 1


class TestUnitOfWork:
    async def test_uncommitted_work_is_rolled_back(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        async with units() as uow:
            await uow.checkpoints.add(session.id, Checkpoint(commit="abc", message_index=0))

        async with units() as uow:
            assert await uow.checkpoints.all(session.id) == []

    async def test_an_error_rolls_back_and_releases_the_store(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with units() as uow:
                await uow.checkpoints.add(session.id, Checkpoint(commit="abc", message_index=0))
                raise RuntimeError("boom")

        async with units() as uow:
            assert await uow.checkpoints.all(session.id) == []

    async def test_units_of_work_cannot_nest(self, units: UnitOfWorkFactory) -> None:
        async with units():
            with pytest.raises(RuntimeError, match="cannot nest"):
                async with units():
                    pass


class TestSchema:
    async def test_data_survives_reopening(self, tmp_path: Path, session: Session) -> None:
        path = tmp_path / "run.db"
        store = await SqliteStore.open(path)

        async with store.unit_of_work() as uow:
            await uow.sessions.save(session)
            await uow.commit()

        await store.close()
        reopened = await SqliteStore.open(path)

        async with reopened.unit_of_work() as uow:
            stored = await uow.sessions.get(session.id)

        await reopened.close()
        assert stored is not None and stored.id == session.id

    async def test_unknown_schema_versions_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "run.db"

        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA user_version = 99")

        with pytest.raises(RuntimeError, match="version 99"):
            await SqliteStore.open(path)


class TestConcurrency:
    async def test_units_of_work_from_separate_tasks_take_turns(
        self, units: UnitOfWorkFactory, session: Session
    ) -> None:
        order: list[str] = []

        async def write(name: str) -> None:
            async with units() as uow:
                order.append(f"{name}:in")
                await asyncio.sleep(0.01)
                await uow.checkpoints.add(session.id, Checkpoint(commit=name, message_index=0))
                await uow.commit()
                order.append(f"{name}:out")

        async with asyncio.TaskGroup() as group:
            group.create_task(write("a"))
            group.create_task(write("b"))

        assert order in (["a:in", "a:out", "b:in", "b:out"], ["b:in", "b:out", "a:in", "a:out"])

        async with units() as uow:
            assert len(await uow.checkpoints.all(session.id)) == 2
