import csv
import io
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from juno.config import Durations
from juno.domain.values import Direction, Usage
from juno.eval.experiment import Experiment
from juno.eval.grading import Grader, GradingError, Leaderboard, RunGrade, SubmissionGrade
from juno.eval.runs import RunFolder
from juno.eval.stats import Bootstrap, Estimate, Samples


class RunScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    task_id: str
    seed: int
    state: str
    metric: str
    leaderboard: bool
    selected_score: float | None
    oracle_score: float | None
    selected_percentile: float
    oracle_percentile: float
    selected_medal: bool | None
    oracle_medal: bool | None
    submissions: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float


class Scorecard:
    def __init__(self, experiment: Experiment, grader: Grader | None = None) -> None:
        self._experiment = experiment
        self._grader = grader or Grader()
        self.skipped: list[str] = []

    def collect(self) -> list[RunScore]:
        graded: list[tuple[str, int, str, RunGrade, Usage]] = []

        for plan in self._experiment.plans():
            folder = RunFolder(plan.folder)
            session = folder.session()
            label = f"{plan.arm}/{plan.task_id}/seed-{plan.seed}"

            if session is None or not session.terminal:
                self.skipped.append(f"{label} has not finished")
                continue

            try:
                grade = self._grader.grade(folder, plan.task.answers)
            except GradingError as error:
                self.skipped.append(f"{label} could not be graded: {error}")
                continue

            self._grader.save(folder, grade)
            graded.append((plan.arm, plan.seed, session.state, grade, folder.usage(session.id)))

        references = self._references([grade for *_, grade, _ in graded])
        return [
            self._score(arm, seed, state, grade, usage, references.get(grade.task_id))
            for arm, seed, state, grade, usage in graded
        ]

    @staticmethod
    def _references(grades: Sequence[RunGrade]) -> dict[str, Leaderboard]:
        pools: defaultdict[str, list[float]] = defaultdict(list)
        directions: dict[str, Direction] = {}

        for grade in grades:
            directions[grade.task_id] = grade.direction

            if not grade.leaderboard and grade.selected and grade.selected.test_score is not None:
                pools[grade.task_id].append(grade.selected.test_score)

        return {task: Leaderboard(pool, directions[task]) for task, pool in pools.items()}

    @staticmethod
    def _score(
        arm: str,
        seed: int,
        state: str,
        grade: RunGrade,
        usage: Usage,
        reference: Leaderboard | None,
    ) -> RunScore:

        def percentile(item: SubmissionGrade | None) -> float:
            if item is None or item.test_score is None:
                return 0.0

            if grade.leaderboard:
                return item.percentile or 0.0

            return reference.percentile(item.test_score) if reference else 0.0

        def medal(item: SubmissionGrade | None) -> bool | None:
            return (item is not None and item.medal is not None) if grade.leaderboard else None

        return RunScore(
            arm=arm,
            task_id=grade.task_id,
            seed=seed,
            state=state,
            metric=grade.metric,
            leaderboard=grade.leaderboard,
            selected_score=grade.selected.test_score if grade.selected else None,
            oracle_score=grade.oracle.test_score if grade.oracle else None,
            selected_percentile=percentile(grade.selected),
            oracle_percentile=percentile(grade.oracle),
            selected_medal=medal(grade.selected),
            oracle_medal=medal(grade.oracle),
            submissions=len(grade.submissions),
            input_tokens=usage.input_tokens + usage.cache_creation_input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cache_read_input_tokens,
            cost_usd=usage.cost_usd,
        )


class Markdown:
    @staticmethod
    def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(" --- " for _ in headers) + "|",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)


class EvalReport:
    MARKDOWN: ClassVar[str] = "report.md"
    CSV: ClassVar[str] = "report.csv"
    TIE: ClassVar[float] = 1e-9
    MIN_SEEDS: ClassVar[int] = 3

    def __init__(
        self,
        experiment: Experiment,
        scores: Sequence[RunScore],
        *,
        skipped: Sequence[str] = (),
        bootstrap: Bootstrap | None = None,
    ) -> None:
        self._experiment = experiment
        self._scores = list(scores)
        self._skipped = list(skipped)
        self._bootstrap = bootstrap or Bootstrap()
        self._arms = [arm for arm in experiment.arms if self._samples(arm, "selected_percentile")]

    def write(self) -> tuple[Path, Path]:
        folder = self._experiment.folder
        folder.mkdir(parents=True, exist_ok=True)
        markdown, table = folder / self.MARKDOWN, folder / self.CSV
        markdown.write_text(self.markdown())
        table.write_text(self.csv())
        return markdown, table

    def markdown(self) -> str:
        settings = self._experiment.experiment
        sections = [
            f"# {settings.name}",
            self._summary(),
        ]

        if self._arms:
            sections += ["## Arms", self._arm_table()]

        if len(self._arms) > 1:
            sections += [f"## Compared with {self._experiment.base_arm}", self._pair_table()]

        if self._scores:
            sections += ["## Tasks", self._task_table()]

        if self._skipped:
            sections += ["## Not in the report", "\n".join(f"- {line}" for line in self._skipped)]

        return "\n\n".join(sections) + "\n"

    def csv(self) -> str:
        buffer = io.StringIO()
        fields = list(RunScore.model_fields)
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()

        for score in self._scores:
            writer.writerow(score.model_dump(mode="json"))

        return buffer.getvalue()

    def _summary(self) -> str:
        settings = self._experiment.experiment
        tasks = sorted({score.task_id for score in self._scores})
        ranked = sorted({score.task_id for score in self._scores if score.leaderboard})
        relative = [task for task in tasks if task not in ranked]
        lines = [
            f"{len(self._scores)} finished runs over {len(tasks)} task{'s' if len(tasks) != 1 else ''}, "
            f"{settings.seeds} seeds per task and arm, with a {Durations.format(settings.budget)} "
            "budget each. Scores are percentiles, higher is better, and a run with no valid "
            "submission scores 0."
        ]

        if ranked:
            lines.append(f"Ranked against each task's leaderboard: {', '.join(ranked)}.")

        if relative:
            lines.append(
                "Ranked against this experiment's own self-selected scores, as these tasks "
                f"have no leaderboard: {', '.join(relative)}."
            )

        lines.append(
            f"Intervals are {self._bootstrap.confidence:.0%} bootstrap CIs from "
            f"{self._bootstrap.resamples} resamples of seeds within each task."
        )

        if len(self._arms) > 1 and self._fewest_seeds(self._arms) < self.MIN_SEEDS:
            lines.append(
                f"A verdict needs at least {self.MIN_SEEDS} finished seeds per task and arm: "
                "with fewer, resampling seeds gives intervals too narrow to trust, so arms "
                "below that get no verdict."
            )

        return " ".join(lines)

    def _arm_table(self) -> str:
        medals = any(score.leaderboard for score in self._scores)
        costs = any(score.cost_usd for score in self._scores)
        headers = ["Arm", "Runs", "Self-select", "Oracle", "Selection gap"]
        headers += ["Medal rate"] if medals else []
        headers += ["Tokens in", "Tokens out", "Cached"]
        headers += ["Cost"] if costs else []
        rows: list[list[str]] = []

        for arm in self._arms:
            runs = [score for score in self._scores if score.arm == arm]
            selected = self._bootstrap.mean(self._samples(arm, "selected_percentile"))
            oracle = self._bootstrap.mean(self._samples(arm, "oracle_percentile"))
            row = [
                arm,
                str(len(runs)),
                selected.format(),
                oracle.format(),
                f"{oracle.value - selected.value:.1f}",
            ]

            if medals:
                medal_samples = self._samples(arm, "selected_medal")
                row.append(self._bootstrap.mean(medal_samples).format() if medal_samples else "")

            row += [
                f"{sum(score.input_tokens for score in runs):,}",
                f"{sum(score.output_tokens for score in runs):,}",
                f"{sum(score.cached_tokens for score in runs):,}",
            ]

            if costs:
                row.append(f"${sum(score.cost_usd for score in runs):.4f}")

            rows.append(row)

        return Markdown.table(headers, rows)

    def _pair_table(self) -> str:
        base = self._experiment.base_arm
        base_samples = self._samples(base, "selected_percentile")
        rows: list[list[str]] = []

        for arm in self._arms:
            if arm == base:
                continue

            samples = self._samples(arm, "selected_percentile")
            delta = self._bootstrap.difference(samples, base_samples)
            won, lost, tied = self._tally(samples, base_samples)
            verdict = self._verdict(delta, self._fewest_seeds([arm, base]))
            rows.append([arm, delta.format(), str(won), str(lost), str(tied), verdict])

        headers = ["Arm", f"Self-select minus {base}", "Tasks won", "Lost", "Tied", "Verdict"]
        return Markdown.table(headers, rows)

    def _task_table(self) -> str:
        rows: list[list[str]] = []
        tasks = sorted({score.task_id for score in self._scores})

        for task in tasks:
            for arm in self._arms:
                runs = [s for s in self._scores if s.task_id == task and s.arm == arm]

                if not runs:
                    continue

                rows.append(
                    [
                        task,
                        runs[0].metric,
                        arm,
                        str(len(runs)),
                        self._mean_score([score.selected_score for score in runs]),
                        self._mean_score([score.oracle_score for score in runs]),
                        f"{Bootstrap.average([s.selected_percentile for s in runs]):.1f}",
                    ]
                )

        headers = [
            "Task",
            "Metric",
            "Arm",
            "Runs",
            "Self-select score",
            "Oracle score",
            "Percentile",
        ]
        return Markdown.table(headers, rows)

    def _samples(self, arm: str, field: str) -> Samples:
        samples: defaultdict[str, list[float]] = defaultdict(list)

        for score in self._scores:
            value = getattr(score, field)

            if score.arm == arm and value is not None:
                samples[score.task_id].append(100.0 * value if isinstance(value, bool) else value)

        return samples

    def _tally(self, samples: Samples, base: Samples) -> tuple[int, int, int]:
        won = lost = tied = 0

        for task in sorted(samples.keys() & base.keys()):
            gap = Bootstrap.average(samples[task]) - Bootstrap.average(base[task])

            if abs(gap) <= self.TIE:
                tied += 1
            elif gap > 0:
                won += 1
            else:
                lost += 1

        return won, lost, tied

    def _fewest_seeds(self, arms: Sequence[str]) -> int:
        counts = [
            len(values)
            for arm in arms
            for values in self._samples(arm, "selected_percentile").values()
        ]
        return min(counts, default=0)

    def _verdict(self, delta: Estimate, seeds: int) -> str:
        if seeds < self.MIN_SEEDS:
            return f"too few seeds ({seeds} per task)"

        if not delta.excludes_zero:
            return "no clear difference"

        return "better" if delta.value > 0 else "worse"

    @staticmethod
    def _mean_score(values: Sequence[float | None]) -> str:
        valid = [value for value in values if value is not None]
        missing = len(values) - len(valid)
        text = f"{Bootstrap.average(valid):.4g}" if valid else "no valid submission"
        return f"{text} ({missing} without a valid submission)" if valid and missing else text


class EvalStatus:
    @staticmethod
    def table(experiment: Experiment, now: float) -> str:
        rows: list[list[str]] = []

        for plan in experiment.plans():
            folder = RunFolder(plan.folder)
            session = folder.session()

            if session is None:
                rows.append([plan.arm, plan.task_id, str(plan.seed), "not started", "", "", ""])
                continue

            submissions = folder.submissions(session.id)
            best = None

            for submission in submissions:
                if best is None or submission.beats(best):
                    best = submission

            rows.append(
                [
                    plan.arm,
                    plan.task_id,
                    str(plan.seed),
                    session.state,
                    Durations.format(session.elapsed(now)),
                    f"{best.val_score:.4g} of {len(submissions)}" if best else "none registered",
                    f"${folder.usage(session.id).cost_usd:.4f}",
                ]
            )

        headers = ["Arm", "Task", "Seed", "State", "Elapsed", "Best validation", "Cost"]
        return Markdown.table(headers, rows)
