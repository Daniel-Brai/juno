from collections import defaultdict

from juno.domain.values import Submission


class InMemorySubmissionRepo:
    def __init__(self) -> None:
        self._by_session: defaultdict[str, list[Submission]] = defaultdict(list)

    async def add(self, session_id: str, submission: Submission) -> None:
        self._by_session[session_id].append(submission)

    async def all(self, session_id: str) -> list[Submission]:
        return list(self._by_session[session_id])

    async def best(self, session_id: str) -> Submission | None:
        best: Submission | None = None

        for submission in self._by_session[session_id]:
            if best is None or submission.beats(best):
                best = submission

        return best
