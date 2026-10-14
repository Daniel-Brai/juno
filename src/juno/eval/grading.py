import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from juno.domain.values import Direction, Submission
from juno.eval.runs import RunFolder


class GradingError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Table:
    columns: tuple[str, ...]
    rows: dict[str, dict[str, str]]

    @classmethod
    def read(cls, path: Path) -> Table:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)

            if not header or len(header) < 2:
                raise GradingError(f"{path.name} needs an id column and at least one value column")

            key, *columns = (name.strip() for name in header)
            rows: dict[str, dict[str, str]] = {}

            for line in reader:
                if not line:
                    continue

                if len(line) != len(header):
                    raise GradingError(
                        f"{path.name} has a row with {len(line)} of {len(header)} cells"
                    )

                row_id, *cells = (cell.strip() for cell in line)

                if row_id in rows:
                    raise GradingError(f"{path.name} repeats {key} {row_id!r}")

                rows[row_id] = dict(zip(columns, cells, strict=True))

        return cls(columns=tuple(columns), rows=rows)

    def matching(self, truth: Table) -> Table:
        missing = truth.rows.keys() - self.rows.keys()
        extra = self.rows.keys() - truth.rows.keys()

        if missing:
            raise GradingError(f"{len(missing)} ids are missing, e.g. {sorted(missing)[:3]}")

        if extra:
            raise GradingError(
                f"{len(extra)} ids are not in the test set, e.g. {sorted(extra)[:3]}"
            )

        return self

    def number(self, row_id: str, column: str) -> float:
        cell = self.rows[row_id].get(column)

        if cell is None:
            raise GradingError(f"column {column!r} is missing")

        try:
            value = float(cell)
        except ValueError:
            raise GradingError(f"{cell!r} for id {row_id} is not a number") from None

        if not math.isfinite(value):
            raise GradingError(f"{cell!r} for id {row_id} is not finite")

        return value

    def single_column(self) -> str:
        if len(self.columns) != 1:
            raise GradingError(f"expected one answer column, found {list(self.columns)}")

        return self.columns[0]


class Metric(Protocol):
    def score(self, truth: Table, predicted: Table) -> float: ...


class Accuracy:
    def score(self, truth: Table, predicted: Table) -> float:
        cells = [(row_id, column) for row_id in truth.rows for column in truth.columns]
        hits = sum(
            self._same(truth.rows[row_id][column], predicted.rows[row_id].get(column))
            for row_id, column in cells
        )
        return hits / len(cells)

    @staticmethod
    def _same(expected: str, actual: str | None) -> bool:
        if actual is None:
            raise GradingError("the submission is missing an answer column")

        try:
            return float(expected) == float(actual)
        except ValueError:
            return expected == actual


class MeanError:
    def __init__(self, *, squared: bool) -> None:
        self._squared = squared

    def score(self, truth: Table, predicted: Table) -> float:
        errors = [
            truth.number(row_id, column) - predicted.number(row_id, column)
            for row_id in truth.rows
            for column in truth.columns
        ]

        if self._squared:
            return math.sqrt(sum(error * error for error in errors) / len(errors))

        return sum(abs(error) for error in errors) / len(errors)


class LogLoss:
    EPSILON: ClassVar[float] = 1e-15

    def score(self, truth: Table, predicted: Table) -> float:
        column = truth.single_column()
        losses = [
            -math.log(self._probability(truth.rows[row_id][column], row_id, column, predicted))
            for row_id in truth.rows
        ]
        return sum(losses) / len(losses)

    def _probability(self, label: str, row_id: str, column: str, predicted: Table) -> float:
        if predicted.columns == (column,):
            positive = self._clip(predicted.number(row_id, column))

            if label not in {"0", "1"}:
                raise GradingError("one probability column needs 0/1 labels")

            return positive if label == "1" else 1 - positive

        if label not in predicted.columns:
            raise GradingError(f"no probability column for class {label!r}")

        probabilities = {
            name: self._clip(predicted.number(row_id, name)) for name in predicted.columns
        }
        return probabilities[label] / sum(probabilities.values())

    def _clip(self, value: float) -> float:
        return min(max(value, self.EPSILON), 1 - self.EPSILON)


class RocAuc:
    def score(self, truth: Table, predicted: Table) -> float:
        column = truth.single_column()
        labels = [truth.rows[row_id][column] for row_id in truth.rows]

        if set(labels) - {"0", "1"} or len(set(labels)) < 2:
            raise GradingError("AUC needs 0/1 labels with both classes present")

        scored = sorted(
            (predicted.number(row_id, column), label == "1")
            for row_id, label in zip(truth.rows, labels, strict=True)
        )
        rank_sum, positives = 0.0, sum(positive for _, positive in scored)
        start = 0

        while start < len(scored):
            end = start

            while end + 1 < len(scored) and scored[end + 1][0] == scored[start][0]:
                end += 1

            mean_rank = (start + end) / 2 + 1
            rank_sum += mean_rank * sum(positive for _, positive in scored[start : end + 1])
            start = end + 1

        negatives = len(scored) - positives
        return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


class Metrics:
    KNOWN: ClassVar[Mapping[str, Metric]] = {
        "accuracy": Accuracy(),
        "log_loss": LogLoss(),
        "logloss": LogLoss(),
        "rmse": MeanError(squared=True),
        "mae": MeanError(squared=False),
        "auc": RocAuc(),
        "roc_auc": RocAuc(),
    }

    @classmethod
    def get(cls, name: str) -> Metric:
        metric = cls.KNOWN.get(name.lower())

        if metric is None:
            raise GradingError(f"no grader for metric {name!r}; known: {', '.join(cls.KNOWN)}")

        return metric


class Medal(StrEnum):
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


class Leaderboard:
    def __init__(self, scores: Sequence[float], direction: Direction) -> None:
        if not scores:
            raise GradingError("a leaderboard needs at least one score")

        self._scores = list(scores)
        self._direction = direction

    @classmethod
    def read(cls, path: Path, direction: Direction) -> Leaderboard:
        table = Table.read(path)

        if "score" not in table.columns:
            raise GradingError(f"{path.name} needs a score column")

        return cls([table.number(row_id, "score") for row_id in table.rows], direction)

    def percentile(self, score: float) -> float:
        worse = sum(self._better(score, other) for other in self._scores)
        tied = sum(other == score for other in self._scores)
        return 100 * (worse + tied / 2) / len(self._scores)

    def medal(self, score: float) -> Medal | None:
        rank = 1 + sum(self._better(other, score) for other in self._scores)
        gold, silver, bronze = self.cutoffs(len(self._scores))

        if rank <= gold:
            return Medal.GOLD

        if rank <= silver:
            return Medal.SILVER

        return Medal.BRONZE if rank <= bronze else None

    @staticmethod
    def cutoffs(teams: int) -> tuple[int, int, int]:
        if teams < 100:
            return math.ceil(0.1 * teams), math.ceil(0.2 * teams), math.ceil(0.4 * teams)

        if teams < 250:
            return 10, math.ceil(0.2 * teams), math.ceil(0.4 * teams)

        gold = 10 + math.floor(0.002 * teams)

        if teams < 1_000:
            return gold, 50, 100

        return gold, math.ceil(0.05 * teams), math.ceil(0.1 * teams)

    def _better(self, score: float, other: float) -> bool:
        return score < other if self._direction is Direction.MINIMIZE else score > other


class Grade(BaseModel):
    model_config = ConfigDict(frozen=True)


class SubmissionGrade(Grade):
    submission_id: str
    sha256: str
    val_score: float
    test_score: float | None = None
    error: str | None = None
    percentile: float | None = None
    medal: Medal | None = None


class RunGrade(Grade):
    task_id: str
    metric: str
    direction: Direction
    leaderboard: bool
    submissions: tuple[SubmissionGrade, ...]
    selected: SubmissionGrade | None
    oracle: SubmissionGrade | None


class Answers:
    TRUTH: ClassVar[str] = "answers.csv"
    LEADERBOARD: ClassVar[str] = "leaderboard.csv"

    def __init__(self, folder: Path, direction: Direction) -> None:
        truth = folder / self.TRUTH

        if not truth.is_file():
            raise GradingError(f"{folder} has no {self.TRUTH}")

        board = folder / self.LEADERBOARD
        self.truth = Table.read(truth)
        self.leaderboard = Leaderboard.read(board, direction) if board.is_file() else None


class Grader:
    FILE: ClassVar[str] = "grades.json"

    def grade(self, run: RunFolder, answers_dir: Path) -> RunGrade:
        session = run.session()

        if session is None:
            raise GradingError(f"{run.root} holds no run")

        task = run.task()
        metric = Metrics.get(task.metric)
        answers = Answers(answers_dir, task.direction)
        submissions = run.submissions(session.id)
        grades = tuple(self._grade_one(run, item, metric, answers) for item in submissions)
        valid = [grade for grade in grades if grade.test_score is not None]

        return RunGrade(
            task_id=task.id,
            metric=task.metric,
            direction=task.direction,
            leaderboard=answers.leaderboard is not None,
            submissions=grades,
            selected=self._selected(submissions, grades),
            oracle=self.best(valid, task.direction),
        )

    def save(self, run: RunFolder, grade: RunGrade) -> Path:
        path = run.root / self.FILE
        path.write_text(grade.model_dump_json(indent=2))
        return path

    def load(self, run: RunFolder) -> RunGrade | None:
        path = run.root / self.FILE
        return RunGrade.model_validate_json(path.read_text()) if path.is_file() else None

    @staticmethod
    def best(grades: Sequence[SubmissionGrade], direction: Direction) -> SubmissionGrade | None:
        scored = [(grade.test_score, grade) for grade in grades if grade.test_score is not None]
        pick = min if direction is Direction.MINIMIZE else max
        chosen = pick(scored, key=lambda pair: pair[0], default=None)
        return chosen[1] if chosen else None

    @staticmethod
    def _selected(
        submissions: Sequence[Submission], grades: Sequence[SubmissionGrade]
    ) -> SubmissionGrade | None:
        chosen: Submission | None = None

        for submission in submissions:
            if chosen is None or submission.beats(chosen):
                chosen = submission

        return next(
            (grade for grade in grades if chosen and grade.submission_id == chosen.id), None
        )

    @staticmethod
    def _grade_one(
        run: RunFolder, submission: Submission, metric: Metric, answers: Answers
    ) -> SubmissionGrade:
        base = SubmissionGrade(
            submission_id=submission.id, sha256=submission.sha256, val_score=submission.val_score
        )
        path = run.submission_file(submission)

        try:
            if not path.is_file():
                raise GradingError(f"{path.name} is missing from the run folder")

            score = metric.score(answers.truth, Table.read(path).matching(answers.truth))
        except GradingError as error:
            return base.model_copy(update={"error": str(error)})
        except (UnicodeDecodeError, csv.Error) as error:
            return base.model_copy(update={"error": f"unreadable submission: {error}"})

        board = answers.leaderboard
        return base.model_copy(
            update={
                "test_score": score,
                "percentile": board.percentile(score) if board else None,
                "medal": board.medal(score) if board else None,
            }
        )
