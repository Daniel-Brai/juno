import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
from cyclopts import App, Parameter

from juno.bootstrap import Adapters, Harness, RunReport
from juno.config import DottedKeys, Durations, HarnessConfig
from juno.domain.session import State
from juno.domain.values import RunPaths
from juno.eval.experiment import Experiment
from juno.eval.grading import Grader, GradingError, RunGrade
from juno.eval.report import EvalReport, EvalStatus, Scorecard
from juno.eval.runner import EvalRunner, JunoProcess
from juno.eval.runs import RunFolder
from juno.tasks import TaskFolder

type Provider = Literal["anthropic", "openrouter"]
type Effort = Literal["low", "medium", "high", "xhigh", "max"]
type SandboxKind = Literal["podman", "local"]
type LogFormat = Literal["json", "console"]


class JunoCli:
    DESCRIPTION = (
        "Your headless agent harness for long-running tasks and everyday coding. "
        "Give it a goal, walk away, and come back to finished work."
    )

    def __init__(self) -> None:
        self.app = App(name="juno", help=self.DESCRIPTION)
        self.app.command(self.run)
        self.app.command(self.resume)
        self.app.command(self.status)
        self.app.command(self.stop)
        self.app.command(self.grade)
        self.evaluate = App(
            name="eval", help="Run experiments, follow them and report the results."
        )
        self.evaluate.command(self.eval_run, name="run")
        self.evaluate.command(self.eval_status, name="status")
        self.evaluate.command(self.eval_report, name="report")
        self.app.command(self.evaluate)

    async def run(
        self,
        task: Path,
        /,
        *,
        runs: Path = Path("runs"),
        name: str | None = None,
        provider: Provider | None = None,
        model: str | None = None,
        effort: Effort | None = None,
        budget: str | None = None,
        max_cost: float | None = None,
        sandbox: SandboxKind | None = None,
        flag: list[str] | None = None,
        setting: Annotated[list[str] | None, Parameter(name="--set", negative=())] = None,
        log_format: LogFormat = "json",
    ) -> int:
        """Start a run on the task folder TASK.

        Parameters
        ----------
        task
            Folder holding task.toml, description.md and data/.
        runs
            Folder where run folders are created.
        name
            Run folder name; defaults to the task id and the start time.
        provider
            Model provider.
        model
            Model name, e.g. claude-opus-5-5 or an OpenRouter model id.
        effort
            Reasoning effort.
        budget
            Time budget such as 30m, 6h or 1h30m.
        max_cost
            Spending cap in US dollars. The run wraps up and submits its best validated result
            once model calls reach it; the last call can go slightly over.
        sandbox
            Where commands run.
        flag
            Feature flag as key=value, e.g. nudge.time=true. Repeatable.
        setting
            Any setting as key=value, e.g. model.effort=low or sandbox.gpus=[0]. Values are
            read as JSON when they parse, otherwise as text. Repeatable.
        log_format
            Log line format on stderr.
        """

        self.configure_logging(log_format)
        config = self.config(
            provider,
            model,
            effort,
            budget,
            sandbox,
            flag or [],
            max_cost=max_cost,
            settings=setting or [],
        )
        loaded = TaskFolder.load(task)
        paths = RunPaths(root=runs / (name or f"{loaded.id}-{time.strftime('%Y%m%d-%H%M%S')}"))
        TaskFolder.save(loaded, paths)

        async with await Harness.start(
            config=config, task=loaded, paths=paths, adapters=Adapters()
        ) as harness:
            print(f"juno run {harness.session.id} in {paths.root}", file=sys.stderr)
            await self.drive(harness, paths)

        return self.finish(await Harness.report(paths))

    async def resume(self, run: Path, /, *, log_format: LogFormat = "json") -> int:
        """Continue the run in folder RUN with the settings it started with.

        Parameters
        ----------
        run
            Run folder created by juno run.
        log_format
            Log line format on stderr.
        """

        self.configure_logging(log_format)
        paths = RunPaths(root=run)
        report = await Harness.report(paths)
        config = HarnessConfig(**report.session.setup)

        async with await Harness.resume(
            config=config, task=TaskFolder.from_run(paths), paths=paths, adapters=Adapters()
        ) as harness:
            await self.drive(harness, paths)

        return self.finish(await Harness.report(paths))

    async def status(self, run: Path, /) -> int:
        """Show the state, budget, best submission and cost of the run in folder RUN.

        Parameters
        ----------
        run
            Run folder created by juno run.
        """

        report = await Harness.report(RunPaths(root=run))
        print(self.describe(report, time.time()))
        return 0

    async def stop(self, run: Path, /) -> int:
        """Remove the sandbox of the run in folder RUN and stop its background jobs.

        An unfinished run keeps its history and can still be continued with juno resume.

        Parameters
        ----------
        run
            Run folder created by juno run.
        """

        paths = RunPaths(root=run)
        session = await Harness.stop(
            task=TaskFolder.from_run(paths), paths=paths, adapters=Adapters()
        )

        if session.terminal:
            print(f"removed the sandbox of finished run {session.id}")
        else:
            print(f"stopped run {session.id}; juno resume {run} continues it")

        return 0

    def grade(self, run: Path, /, *, answers: Path) -> int:
        """Score every submission in run folder RUN against hidden answers, outside the sandbox.

        Parameters
        ----------
        run
            Run folder created by juno run.
        answers
            Folder holding answers.csv and, optionally, leaderboard.csv.
        """

        folder = RunFolder(run)

        try:
            grade = Grader().grade(folder, answers)
        except GradingError as error:
            print(f"juno grade: {error}", file=sys.stderr)
            return 2

        path = Grader().save(folder, grade)
        print(self.describe_grade(grade))
        print(f"saved:       {path}")
        return 0

    async def eval_run(self, manifest: Path, /) -> int:
        """Launch every arm, task and seed in experiment MANIFEST, then grade and report.

        Finished runs are skipped and unfinished ones resumed, so running it again picks up
        where it stopped.

        Parameters
        ----------
        manifest
            Experiment TOML file.
        """

        experiment = Experiment.load(manifest)
        runner = EvalRunner(experiment, JunoProcess())
        outcomes = await runner.run(manifest)
        print(self.write_report(experiment))

        if runner.stopped:
            print(
                f"stopped at ${runner.spent():.4f}, the experiment's max_total_cost of "
                f"${experiment.experiment.max_total_cost:.2f}; unfinished runs can be resumed "
                "with eval run after raising it",
                file=sys.stderr,
            )

        return 0 if all(outcome.state == State.DONE for outcome in outcomes) else 1

    def eval_status(self, manifest: Path, /) -> int:
        """Show the state of every run in experiment MANIFEST.

        Parameters
        ----------
        manifest
            Experiment TOML file.
        """

        print(EvalStatus.table(Experiment.load(manifest), time.time()))
        return 0

    def eval_report(self, manifest: Path, /) -> int:
        """Grade the finished runs in experiment MANIFEST and write report.md and report.csv.

        Parameters
        ----------
        manifest
            Experiment TOML file.
        """

        print(self.write_report(Experiment.load(manifest)))
        return 0

    @staticmethod
    async def drive(harness: Harness, paths: RunPaths) -> None:
        try:
            await harness.run()
        except Exception:
            print(
                f"juno: the run stopped on an error and its sandbox is still up; "
                f"juno resume {paths.root} continues it, juno stop {paths.root} removes it",
                file=sys.stderr,
            )
            raise

    @staticmethod
    def write_report(experiment: Experiment) -> str:
        scorecard = Scorecard(experiment)
        report = EvalReport(experiment, scorecard.collect(), skipped=scorecard.skipped)
        markdown, table = report.write()
        return f"{report.markdown()}\nwritten to {markdown} and {table}"

    @staticmethod
    def describe_grade(grade: RunGrade) -> str:
        valid = [item for item in grade.submissions if item.test_score is not None]
        lines = [
            f"task:        {grade.task_id} ({grade.metric}, {grade.direction})",
            f"submissions: {len(grade.submissions)}, {len(valid)} valid",
        ]

        for label, item in (("selected:", grade.selected), ("oracle:", grade.oracle)):
            if item is None:
                continue

            score = f"{item.test_score:.4g}" if item.test_score is not None else item.error
            line = f"{label:<12} test {score}, validation {item.val_score:.4g}"

            if item.percentile is not None:
                line += f", percentile {item.percentile:.1f}"

            if item.medal is not None:
                line += f", {item.medal} medal"

            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def config(
        provider: Provider | None,
        model: str | None,
        effort: Effort | None,
        budget: str | None,
        sandbox: SandboxKind | None,
        flags: list[str],
        *,
        max_cost: float | None = None,
        settings: Sequence[str] = (),
    ) -> HarnessConfig:
        overrides: dict[str, Any] = {}
        sections: dict[str, dict[str, Any]] = {
            "model": {"provider": provider, "name": model, "effort": effort},
            "budget": {"total": budget, "max_cost": max_cost},
            "sandbox": {"kind": sandbox},
        }

        for section, values in sections.items():
            if chosen := {key: value for key, value in values.items() if value is not None}:
                overrides[section] = chosen

        if flags:
            base = HarnessConfig().flags
            overrides["flags"] = base.with_overrides(JunoCli.parse_flags(flags)).model_dump()

        return HarnessConfig(**DottedKeys.merge(overrides, JunoCli.parse_settings(settings)))

    @staticmethod
    def parse_flags(flags: list[str]) -> dict[str, bool]:
        parsed: dict[str, bool] = {}

        for item in flags:
            key, separator, raw = item.partition("=")

            if not separator or raw.lower() not in {"true", "false"}:
                raise ValueError(f"--flag expects key=true or key=false, got {item!r}")

            parsed[key.strip()] = raw.lower() == "true"

        return parsed

    @staticmethod
    def parse_settings(settings: Sequence[str]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}

        for item in settings:
            key, separator, raw = item.partition("=")

            if not separator or not key.strip():
                raise ValueError(f"--set expects key=value, got {item!r}")

            try:
                parsed[key.strip()] = json.loads(raw)
            except json.JSONDecodeError:
                parsed[key.strip()] = raw

        return DottedKeys.nest(parsed)

    @staticmethod
    def describe(report: RunReport, now: float) -> str:
        session = report.session
        best = report.best
        running = [job for job in report.jobs if job.state == "running"]
        lines = [
            f"run:         {session.id} ({session.task_id})",
            f"state:       {session.state}" + (f" ({session.failure})" if session.failure else ""),
            f"elapsed:     {Durations.format(session.elapsed(now))}",
            f"remaining:   {Durations.format(session.remaining(now))}",
            f"best:        {best.val_score if best else 'none yet'}"
            + (f" ({best.notes})" if best and best.notes else ""),
            f"jobs:        {len(report.jobs)} ({len(running)} running)",
            f"checkpoints: {report.checkpoints}",
            f"tokens:      {report.usage.input_tokens} in, {report.usage.output_tokens} out, "
            f"{report.usage.cache_read_input_tokens} cached",
        ]

        if session.cost_limit is not None:
            lines.append(f"cost:        ${report.usage.cost_usd:.4f} of ${session.cost_limit:.2f}")
        elif report.usage.cost_usd:
            lines.append(f"cost:        ${report.usage.cost_usd:.4f}")

        return "\n".join(lines)

    @staticmethod
    def finish(report: RunReport) -> int:
        print(JunoCli.describe(report, time.time()))
        return 0 if report.session.state is State.DONE else 1

    @staticmethod
    def configure_logging(log_format: LogFormat) -> None:
        renderer: Any = (
            structlog.processors.JSONRenderer()
            if log_format == "json"
            else structlog.dev.ConsoleRenderer()
        )
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                renderer,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        )


app = JunoCli().app
