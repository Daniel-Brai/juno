import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Any

from juno.domain.commands import ListSubmissions, RegisterSubmission
from juno.domain.events import SubmissionRegistered
from juno.domain.values import Direction, Submission
from juno.ports import Clock, UnitOfWorkFactory
from juno.service.bus import MessageBus
from juno.tools.registry import Tool, ToolError
from juno.tools.workspace import Workspace


class SubmissionTools:
    def __init__(
        self,
        units: UnitOfWorkFactory,
        workspace: Workspace,
        store_dir: Path,
        session_id: str,
        direction: Direction,
        clock: Clock,
        bus: MessageBus,
    ) -> None:
        self._units = units
        self._workspace = workspace
        self._store_dir = store_dir
        self._session_id = session_id
        self._direction = direction
        self._clock = clock
        self._bus = bus

    def tools(self) -> list[Tool[Any]]:
        better = "Lower" if self._direction is Direction.MINIMIZE else "Higher"

        return [
            Tool(
                name="submission_register",
                description=(
                    "Register a submission file with its score on your validation split. "
                    f"{better} scores are better. The best one is submitted when time runs out; "
                    "test scores are never shown."
                ),
                input_model=RegisterSubmission,
                handler=self.register,
                idempotent=True,
            ),
            Tool(
                name="submission_list",
                description="List registered submissions and their validation scores.",
                input_model=ListSubmissions,
                handler=self.list,
                read_only=True,
            ),
        ]

    async def register(self, args: RegisterSubmission) -> str:
        source = self._workspace.resolve(args.path)

        if not source.is_file():
            raise ToolError(f"{args.path} is not a file")

        digest = await asyncio.to_thread(self._sha256, source)
        target = self._store_dir / f"{digest}{source.suffix}"
        submission = Submission(
            path=target,
            sha256=digest,
            val_score=args.val_score,
            direction=self._direction,
            notes=args.notes,
            created_at=self._clock.now(),
        )

        async with self._units() as uow:
            submissions = await uow.submissions.all(self._session_id)
            existing = [item for item in submissions if item.sha256 == digest]

            if existing:
                return f"{args.path} is already registered with validation score {existing[0].val_score}"

            await asyncio.to_thread(shutil.copy2, source, target)
            await uow.submissions.add(self._session_id, submission)
            best = await uow.submissions.best(self._session_id)
            await uow.commit()

        await self._bus.publish(
            SubmissionRegistered(
                session_id=self._session_id, at=submission.created_at, submission=submission
            )
        )
        best_score = best.val_score if best is not None else args.val_score
        return (
            f"registered submission {len(submissions) + 1} with validation score "
            f"{args.val_score}; best so far {best_score}"
        )

    async def list(self, _: ListSubmissions) -> str:
        async with self._units() as uow:
            submissions = await uow.submissions.all(self._session_id)
            best = await uow.submissions.best(self._session_id)

        if not submissions:
            return "no submissions yet"

        return "\n".join(
            f"{number}. {item.val_score}{' (best)' if item == best else ''} {item.notes}".rstrip()
            for number, item in enumerate(submissions, 1)
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)

        return digest.hexdigest()
