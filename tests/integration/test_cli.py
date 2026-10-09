from pathlib import Path

import pytest

from fakes.clock import FakeClock
from fakes.llm import Reply, ScriptedLlm, Turn
from juno.bootstrap import Harness
from juno.config import HarnessConfig
from juno.domain.session import Session
from juno.domain.values import Direction, ModelReply, ModelRequest, RunPaths
from juno.entrypoints.cli import JunoCli
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
