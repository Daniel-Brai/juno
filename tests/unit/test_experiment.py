from pathlib import Path

import pytest
from pydantic import ValidationError

from juno.eval.experiment import Experiment


class Files:
    @staticmethod
    def task(root: Path, task_id: str) -> Path:
        folder = root / "tasks" / task_id
        (folder / "data").mkdir(parents=True)
        (folder / "description.md").write_text("Predict y.\n")
        (folder / "task.toml").write_text(
            f'id = "{task_id}"\nmetric = "rmse"\ndirection = "minimize"\n'
        )
        return folder

    @staticmethod
    def manifest(root: Path, body: str) -> Path:
        path = root / "experiment.toml"
        path.write_text(body)
        return path


MANIFEST = """\
[experiment]
name = "system-tool"
budget = "30m"
seeds = 2

[[tasks]]
path = "tasks/alpha"
answers = "answers/alpha"

[arms.base]

[arms.no_system]
flags = { "tools.system" = false }
settings = { "model.effort" = "low" }
"""


class TestExperiment:
    def test_plans_every_arm_task_and_seed(self, tmp_path: Path) -> None:
        Files.task(tmp_path, "alpha")
        experiment = Experiment.load(Files.manifest(tmp_path, MANIFEST))

        plans = experiment.plans()

        assert [(plan.arm, plan.task_id, plan.seed) for plan in plans] == [
            ("base", "alpha", 1),
            ("base", "alpha", 2),
            ("no_system", "alpha", 1),
            ("no_system", "alpha", 2),
        ]
        assert plans[0].budget == 1_800
        assert plans[0].task.answers == tmp_path / "answers" / "alpha"
        assert (
            plans[3].folder == tmp_path / "runs" / "system-tool" / "no_system" / "alpha" / "seed-2"
        )
        assert plans[3].flags == {"tools.system": False}
        assert plans[3].settings == {"model.effort": "low"}
        assert experiment.base_arm == "base"

    def test_a_suite_adds_tasks_relative_to_itself(self, tmp_path: Path) -> None:
        Files.task(tmp_path, "alpha")
        Files.task(tmp_path, "beta")
        (tmp_path / "suites").mkdir()
        (tmp_path / "suites" / "lite.toml").write_text(
            '[[tasks]]\npath = "../tasks/beta"\nanswers = "../answers/beta"\n'
        )
        manifest = Files.manifest(tmp_path, f'suite = "suites/lite.toml"\n\n{MANIFEST}')

        experiment = Experiment.load(manifest)

        assert [entry.path for entry in experiment.tasks] == [
            tmp_path / "tasks" / "alpha",
            tmp_path / "tasks" / "beta",
        ]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (("[arms.no_system]", "[arms.'no system']"), "may only use letters"),
            (("seeds = 2", "seeds = 2\ngpus_per_run = 1"), "only 0 gpus"),
            (('budget = "30m"', 'budget = "soon"'), "invalid duration"),
        ],
    )
    def test_bad_manifests_are_refused(
        self, tmp_path: Path, change: tuple[str, str], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            Experiment.load(Files.manifest(tmp_path, MANIFEST.replace(*change)))

    def test_a_manifest_needs_tasks(self, tmp_path: Path) -> None:
        body = MANIFEST.replace('[[tasks]]\npath = "tasks/alpha"\nanswers = "answers/alpha"\n', "")

        with pytest.raises(ValueError, match="lists no tasks"):
            Experiment.load(Files.manifest(tmp_path, body))
