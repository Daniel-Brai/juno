import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

import structlog
from cyclopts import App

from juno.bootstrap import Adapters, Harness, RunReport
from juno.config import Durations, HarnessConfig
from juno.domain.session import State
from juno.domain.values import RunPaths
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
        sandbox: SandboxKind | None = None,
        flag: list[str] | None = None,
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
        sandbox
            Where commands run.
        flag
            Feature flag as key=value, e.g. nudge.time=true. Repeatable.
        log_format
            Log line format on stderr.
        """

        self.configure_logging(log_format)
        config = self.config(provider, model, effort, budget, sandbox, flag or [])
        loaded = TaskFolder.load(task)
        paths = RunPaths(root=runs / (name or f"{loaded.id}-{time.strftime('%Y%m%d-%H%M%S')}"))
        TaskFolder.save(loaded, paths)

        async with await Harness.start(
            config=config, task=loaded, paths=paths, adapters=Adapters()
        ) as harness:
            print(f"juno run {harness.session.id} in {paths.root}", file=sys.stderr)
            await harness.run()

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
            await harness.run()

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

    @staticmethod
    def config(
        provider: Provider | None,
        model: str | None,
        effort: Effort | None,
        budget: str | None,
        sandbox: SandboxKind | None,
        flags: list[str],
    ) -> HarnessConfig:

        overrides: dict[str, Any] = {}
        sections: dict[str, dict[str, Any]] = {
            "model": {"provider": provider, "name": model, "effort": effort},
            "budget": {"total": budget},
            "sandbox": {"kind": sandbox},
        }

        for section, values in sections.items():
            if chosen := {key: value for key, value in values.items() if value is not None}:
                overrides[section] = chosen

        if flags:
            base = HarnessConfig().flags
            overrides["flags"] = base.with_overrides(JunoCli.parse_flags(flags)).model_dump()

        return HarnessConfig(**overrides)

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
    def describe(report: RunReport, now: float) -> str:
        session = report.session
        best = report.best
        running = [job for job in report.jobs if job.state == "running"]
        lines = [
            f"run:         {session.id} ({session.task_id})",
            f"state:       {session.state}",
            f"elapsed:     {Durations.format(session.elapsed(now))}",
            f"remaining:   {Durations.format(session.remaining(now))}",
            f"best:        {best.val_score if best else 'none yet'}"
            + (f" ({best.notes})" if best and best.notes else ""),
            f"jobs:        {len(report.jobs)} ({len(running)} running)",
            f"checkpoints: {report.checkpoints}",
            f"tokens:      {report.usage.input_tokens} in, {report.usage.output_tokens} out, "
            f"{report.usage.cache_read_input_tokens} cached",
        ]

        if report.usage.cost_usd:
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
