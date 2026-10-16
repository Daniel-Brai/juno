import asyncio
import os
import sys
from pathlib import Path

import pytest

from fakes.experiment import MANIFEST, ExperimentFiles, ScriptedLauncher
from juno.domain.session import State
from juno.eval.experiment import Experiment, RunPlan
from juno.eval.report import EvalReport, EvalStatus, Scorecard
from juno.eval.runner import EvalRunner, GpuPool, JunoProcess, Launcher
from juno.eval.runs import RunFolder

pytestmark = pytest.mark.anyio


class TestAblation:
    async def test_an_ablation_is_run_graded_and_reported_with_intervals(
        self, tmp_path: Path
    ) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        experiment = Experiment.load(manifest)

        outcomes = await EvalRunner(experiment, ScriptedLauncher()).run(manifest)

        assert [outcome.state for outcome in outcomes] == [State.DONE] * 12
        assert all(outcome.grade is not None for outcome in outcomes)
        assert (experiment.folder / "experiment.toml").read_text() == MANIFEST

        scorecard = Scorecard(experiment)
        scores = scorecard.collect()
        report = EvalReport(experiment, scores, skipped=scorecard.skipped)
        markdown = report.markdown()

        first = scores[0]
        assert (first.arm, first.task_id, first.seed) == ("base", "alpha", 1)
        assert (first.selected_score, first.oracle_score) == pytest.approx((0.7, 0.8))
        assert (first.submissions, first.input_tokens, first.output_tokens) == (2, 600, 60)
        beta = next(score for score in scores if score.task_id == "beta")
        assert beta.leaderboard and beta.selected_medal is not None
        assert beta.selected_percentile == pytest.approx(100 * 12.5 / 20)

        assert "## Compared with base" in markdown
        assert "| 0 | 2 | 0 | worse |" in markdown
        assert "Ranked against each task's leaderboard: beta." in markdown
        assert "own self-selected scores, as these tasks have no leaderboard: alpha." in markdown
        assert "## Not in the report" not in markdown

        markdown_path, csv_path = report.write()
        assert markdown_path.read_text() == markdown
        assert len(csv_path.read_text().splitlines()) == 13

    async def test_two_seeds_get_no_verdict(self, tmp_path: Path) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(MANIFEST.replace("seeds = 3", "seeds = 2"))
        experiment = Experiment.load(manifest)
        await EvalRunner(experiment, ScriptedLauncher()).run()

        markdown = EvalReport(experiment, Scorecard(experiment).collect()).markdown()

        assert "| too few seeds (2 per task) |" in markdown
        assert "worse" not in markdown
        assert "A verdict needs at least 3 finished seeds per task and arm" in markdown

    async def test_finished_runs_are_skipped_and_crashed_ones_resumed(self, tmp_path: Path) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(MANIFEST.replace("seeds = 3", "seeds = 1"))
        experiment = Experiment.load(manifest)
        launcher = ScriptedLauncher(crash_first=True)

        outcomes = await EvalRunner(experiment, launcher).run()

        assert [(outcome.state, outcome.exit_codes) for outcome in outcomes] == [
            (State.DONE, (1, 0))
        ] * 4
        assert [kind for kind, *_ in launcher.calls].count("resume") == 4

        again = ScriptedLauncher()
        rerun = await EvalRunner(experiment, again).run()

        assert again.calls == []
        assert all(outcome.attempts == 0 and outcome.grade for outcome in rerun)

    async def test_unfinished_runs_are_listed_not_scored(self, tmp_path: Path) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(MANIFEST.replace("seeds = 3", "seeds = 1\nretries = 0"))
        experiment = Experiment.load(manifest)

        await EvalRunner(experiment, ScriptedLauncher(crash_first=True)).run()
        scorecard = Scorecard(experiment)

        assert scorecard.collect() == []
        assert scorecard.skipped[0] == "base/alpha/seed-1 has not finished"
        assert "| base | alpha | 1 | running |" in EvalStatus.table(experiment, 0.0)
        assert (
            "## Not in the report"
            in EvalReport(experiment, [], skipped=scorecard.skipped).markdown()
        )

    async def test_runs_wait_for_free_gpus(self, tmp_path: Path) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(
            MANIFEST.replace(
                "seeds = 3\nparallel = 2",
                "seeds = 2\nparallel = 4\ngpus = [0, 1]\ngpus_per_run = 1",
            )
        )
        launcher = ScriptedLauncher()

        await EvalRunner(Experiment.load(manifest), launcher).run()

        given = [gpus for *_, gpus in launcher.calls]
        assert len(given) == 8 and set(given) <= {(0,), (1,)}


class HangingLauncher:
    def __init__(self) -> None:
        self.cancelled = 0

    async def start(self, plan: RunPlan, gpus: tuple[int, ...]) -> int:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

        return 0

    async def resume(self, plan: RunPlan) -> int:
        return await self.start(plan, ())


class SpendingRunner(EvalRunner):
    def __init__(
        self, experiment: Experiment, launcher: Launcher, *, readings: list[float]
    ) -> None:
        super().__init__(experiment, launcher, spend_poll_s=0.01)
        self._readings = readings

    def spent(self) -> float:
        return self._readings.pop(0) if len(self._readings) > 1 else self._readings[0]


class TestCostCaps:
    async def test_runs_stop_launching_once_the_total_cap_is_spent(self, tmp_path: Path) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(MANIFEST.replace("parallel = 2", "parallel = 1\nmax_total_cost = 0.25"))
        experiment = Experiment.load(manifest)
        launcher = ScriptedLauncher(cost_per_call=0.02)

        runner = EvalRunner(experiment, launcher)
        outcomes = await runner.run()

        finished = [outcome for outcome in outcomes if outcome.state == State.DONE]
        assert runner.stopped and runner.spent() >= 0.25
        assert len(finished) == 3 and len(launcher.calls) == 3
        assert all(outcome.stopped and outcome.state is None for outcome in outcomes[3:])

    async def test_a_spent_cap_stops_runs_in_progress(self, tmp_path: Path) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(MANIFEST.replace("parallel = 2", "parallel = 2\nmax_total_cost = 1"))
        launcher = HangingLauncher()
        runner = SpendingRunner(Experiment.load(manifest), launcher, readings=[0.0, 0.0, 0.0, 1.5])

        outcomes = await runner.run()

        assert runner.stopped and launcher.cancelled == 2
        assert all(outcome.stopped for outcome in outcomes)

    async def test_an_experiment_already_over_its_cap_launches_nothing(
        self, tmp_path: Path
    ) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(MANIFEST.replace("parallel = 2", "max_total_cost = 1"))
        launcher = ScriptedLauncher()

        outcomes = await SpendingRunner(Experiment.load(manifest), launcher, readings=[2.0]).run()

        assert launcher.calls == [] and all(outcome.stopped for outcome in outcomes)

    def test_the_per_run_cap_is_passed_to_each_run(self, tmp_path: Path) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        manifest.write_text(MANIFEST.replace("seeds = 3", "seeds = 3\nmax_cost = 0.5"))
        plan = Experiment.load(manifest).plans()[0]

        assert JunoProcess(["juno"]).start_argv(plan, ())[-2:] == ["--set", "budget.max_cost=0.5"]


class TestGpuPool:
    async def test_a_taker_waits_until_gpus_come_back(self) -> None:
        pool = GpuPool([0, 1])
        first = await pool.take(2)
        waiting = asyncio.create_task(pool.take_exact([1]))
        await asyncio.sleep(0)

        assert first == (0, 1) and not waiting.done()

        await pool.give([1])
        assert await waiting == (1,)


class TestJunoProcess:
    def plan(self, tmp_path: Path) -> RunPlan:
        plans = Experiment.load(ExperimentFiles.experiment(tmp_path)).plans()
        return next(plan for plan in plans if plan.arm == "no_system")

    def test_start_passes_the_plan_as_options(self, tmp_path: Path) -> None:
        plan = self.plan(tmp_path).model_copy(update={"settings": {"model.effort": "low"}})

        argv = JunoProcess(["juno"]).start_argv(plan, (2,))

        assert argv == [
            "juno",
            "run",
            str(plan.task.path),
            "--runs",
            str(plan.folder.parent),
            "--name",
            "seed-1",
            "--budget",
            "3600.0s",
            "--flag",
            "tools.system=false",
            "--set",
            'model.effort="low"',
            "--set",
            "sandbox.gpus=[2]",
        ]
        assert JunoProcess(["juno"]).resume_argv(plan) == ["juno", "resume", str(plan.folder)]

    async def test_output_goes_to_the_run_log(self, tmp_path: Path) -> None:
        plan = self.plan(tmp_path)
        process = JunoProcess([sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(4)"])

        assert await process.resume(plan) == 4
        assert (plan.folder / JunoProcess.LOG).read_text() == "resume\n"

    async def test_cancelling_stops_the_process_group(self, tmp_path: Path) -> None:
        plan = self.plan(tmp_path)
        pid_file = tmp_path / "pid"
        script = (
            f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
        )
        waiting = asyncio.create_task(JunoProcess([sys.executable, "-c", script]).resume(plan))

        for _ in range(200):
            if pid_file.exists() and pid_file.read_text():
                break

            await asyncio.sleep(0.05)

        waiting.cancel()

        with pytest.raises(asyncio.CancelledError):
            await waiting

        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


class TestRunFolder:
    def test_an_empty_folder_holds_no_run(self, tmp_path: Path) -> None:
        assert RunFolder(tmp_path).session() is None
