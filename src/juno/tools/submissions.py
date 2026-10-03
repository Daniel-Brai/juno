import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Any

from juno.domain.commands import ListSubmissions, RegisterSubmission
from juno.domain.events import SubmissionRegistered
from juno.domain.values import Direction, Submission
from juno.ports import Clock, SubmissionRepo
from juno.service.bus import MessageBus
from juno.tools.registry import Tool, ToolError
from juno.tools.workspace import Workspace


class SubmissionTools:
    def __init__(
        self,
        repo: SubmissionRepo,
        workspace: Workspace,
        store_dir: Path,
        session_id: str,
        direction: Direction,
        clock: Clock,
        bus: MessageBus,
    ) -> None:
        self._repo = repo
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
                idempotent=True,
            ),
        ]

    async def register(self, args: RegisterSubmission) -> str:
        source = self._workspace.resolve(args.path)

        if not source.is_file():
            raise ToolError(f"{args.path} is not a file")

        digest = await asyncio.to_thread(self._sha256, source)
        existing = [
            item for item in await self._repo.all(self._session_id) if item.sha256 == digest
        ]

        if existing:
            return (
                f"{args.path} is already registered with validation score {existing[0].val_score}"
            )

        target = self._store_dir / f"{digest}{source.suffix}"
        await asyncio.to_thread(shutil.copy2, source, target)

        submission = Submission(
            path=target,
            sha256=digest,
            val_score=args.val_score,
            direction=self._direction,
            notes=args.notes,
            created_at=self._clock.now(),
        )
        await self._repo.add(self._session_id, submission)
        await self._bus.publish(
            SubmissionRegistered(
                session_id=self._session_id, at=submission.created_at, submission=submission
            )
        )

        count = len(await self._repo.all(self._session_id))
        best = await self._repo.best(self._session_id)
        best_score = best.val_score if best is not None else args.val_score
        return f"registered submission {count} with validation score {args.val_score}; best so far {best_score}"

    async def list(self, _: ListSubmissions) -> str:
        submissions = await self._repo.all(self._session_id)
        best = await self._repo.best(self._session_id)

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
