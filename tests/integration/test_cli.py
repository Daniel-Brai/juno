from pathlib import Path

import pytest

from fakes.clock import FakeClock
from fakes.experiment import ExperimentFiles, ScriptedLauncher
from fakes.llm import Reply, ScriptedLlm, Turn
from juno.bootstrap import Harness
from juno.config import HarnessConfig
from juno.domain.session import Session
from juno.domain.values import Direction, ModelReply, ModelRequest, RunPaths
from juno.entrypoints.cli import JunoCli
from juno.eval.experiment import Experiment
from juno.eval.runner import EvalRunner
from juno.ports import LlmClient
from juno.tasks import TaskFolder

pytestmark = pytest.mark.anyio


class TaskFiles:
    @staticmethod
    def write(root: Path) -> Path:
        folder = root / "spooky"
        (folder / "data").mkdir(parents=True)
        (folder / "data" / "train.csv").write_text("id,text,author\n1,dark,EAP\n")
        (folder / "description.md").write_text("Predict each sentence's author.\n")
        (folder / "task.toml").write_text(
            'id = "spooky"\nmetric = "log_loss"\ndirection = "minimize"\n'
        )
        return folder


class TestTaskFolder:
    def test_loads_a_task_folder(self, tmp_path: Path) -> None:
        task = TaskFolder.load(TaskFiles.write(tmp_path))

        assert (task.id, task.metric, task.direction) == ("spooky", "log_loss", Direction.MINIMIZE)
        assert task.description == "Predict each sentence's author.\n"
        assert task.data_dir == (tmp_path / "spooky" / "data").resolve()

    def test_a_run_folder_keeps_a_copy_of_the_task(self, tmp_path: Path) -> None:
        task = TaskFolder.load(TaskFiles.write(tmp_path))
        paths = RunPaths(root=tmp_path / "run")

        TaskFolder.save(task, paths)

        assert TaskFolder.from_run(paths) == task

    @pytest.mark.parametrize("missing", ["task.toml", "description.md"])
    def test_missing_files_are_named(self, tmp_path: Path, missing: str) -> None:
        folder = TaskFiles.write(tmp_path)
        (folder / missing).unlink()

        with pytest.raises(FileNotFoundError, match=missing.split(".")[0]):
            TaskFolder.load(folder)


class TestCliOptions:
    def test_flags_parse_to_booleans(self) -> None:
        assert JunoCli.parse_flags(["nudge.time=true", "tools.jobs=False"]) == {
            "nudge.time": True,
            "tools.jobs": False,
        }

    @pytest.mark.parametrize("bad", ["nudge.time", "nudge.time=yes"])
    def test_bad_flags_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="key=true"):
            JunoCli.parse_flags([bad])

    def test_options_override_the_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        config = JunoCli.config("openrouter", "x/y", "low", "30m", "local", ["nudge.time=true"])

        assert (config.model.provider, config.model.name, config.model.effort) == (
            "openrouter",
            "x/y",
            "low",
        )
        assert config.budget.total == 1_800
        assert config.sandbox.kind == "local"
        assert config.flags.nudge.time and config.flags.tools.jobs

    def test_max_cost_sets_the_budget_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        config = JunoCli.config(None, None, None, None, None, [], max_cost=0.5)

        assert config.budget.max_cost == 0.5

    def test_a_cost_cap_needs_a_priced_anthropic_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = JunoCli.config(None, "claude-future-9", None, None, None, [], max_cost=0.5)
        session = Session(task_id="t", started_at=0, deadline=60)

        with pytest.raises(ValueError, match="needs a price for claude-future-9"):
            Harness.llm_for(config, session)

    def test_settings_parse_as_json_or_text(self) -> None:
        assert JunoCli.parse_settings(
            ["model.effort=low", "sandbox.gpus=[0, 1]", "budget.max_jobs=2", "model.name=x/y"]
        ) == {
            "model": {"effort": "low", "name": "x/y"},
            "sandbox": {"gpus": [0, 1]},
            "budget": {"max_jobs": 2},
        }

    @pytest.mark.parametrize("bad", ["model.effort", "=low"])
    def test_bad_settings_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="key=value"):
            JunoCli.parse_settings([bad])

    def test_settings_reach_any_section_and_win_over_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        config = JunoCli.config(
            None,
            None,
            "high",
            None,
            None,
            [],
            settings=["model.effort=low", "sandbox.gpus=[1]", "watchdog.interval=2s"],
        )

        assert config.model.effort == "low"
        assert config.sandbox.gpus == (1,)
        assert config.watchdog.interval == 2


class TestCliRun:
    async def test_run_then_status_on_a_scripted_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        clock = FakeClock(start=1_000.0)

        def finish(request: ModelRequest) -> ModelReply:
            clock.advance(10 * 3_600)
            return Reply.text("done")

        script: list[Turn] = [
            Reply.tools(
                ("editor", {"action": "create", "path": "s.csv", "file_text": "id,p\n1,1\n"})
            ),
            Reply.tools(("submission_register", {"path": "s.csv", "val_score": 0.4})),
            finish,
        ]

        def scripted(config: HarnessConfig, session: Session) -> LlmClient:
            return ScriptedLlm(script)

        monkeypatch.setattr(Harness, "llm_for", staticmethod(scripted))
        monkeypatch.setattr("juno.bootstrap.SystemClock", lambda: clock)
        cli = JunoCli()

        code = await cli.run(
            TaskFiles.write(tmp_path),
            runs=tmp_path / "runs",
            name="first",
            sandbox="local",
            budget="1h",
        )
        assert code == 0
        assert "best:        0.4" in capsys.readouterr().out

        await cli.status(tmp_path / "runs" / "first")
        assert "state:       done" in capsys.readouterr().out


class TestCliEval:
    async def test_report_status_and_grade_read_a_finished_experiment(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        experiment = Experiment.load(manifest)
        await EvalRunner(experiment, ScriptedLauncher()).run(manifest)
        cli = JunoCli()

        assert cli.eval_report(manifest) == 0
        assert "| no_system |" in capsys.readouterr().out
        assert (experiment.folder / "report.md").is_file()

        assert cli.eval_status(manifest) == 0
        assert "| base | alpha | 1 | done |" in capsys.readouterr().out

        run = experiment.folder / "base" / "alpha" / "seed-1"
        assert cli.grade(run, answers=tmp_path / "answers" / "alpha") == 0
        assert "selected:    test 0.7, validation 0.99" in capsys.readouterr().out

        assert cli.grade(run, answers=tmp_path) == 2
        assert "has no answers.csv" in capsys.readouterr().err

    async def test_eval_run_launches_every_run_then_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = ExperimentFiles.experiment(tmp_path)
        monkeypatch.setattr("juno.entrypoints.cli.JunoProcess", ScriptedLauncher)

        assert await JunoCli().eval_run(manifest) == 0
        assert "## Compared with base" in capsys.readouterr().out
