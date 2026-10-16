from collections.abc import Awaitable, Callable
from pathlib import Path

from fakes.clock import FakeClock
from fakes.llm import Reply, ScriptedLlm, ScriptExhausted, Turn
from fakes.probe import FakeProbe
from juno.bootstrap import Adapters, Harness
from juno.config import BudgetSettings, Flags, HarnessConfig, SandboxSettings
from juno.domain.values import ModelReply, ModelRequest, RunPaths, Usage
from juno.eval.experiment import RunPlan
from juno.tasks import TaskFolder

ROWS = 10
ACCURACY = {
    ("base", "alpha"): [0.8, 0.9, 1.0],
    ("base", "beta"): [0.7, 0.8, 0.9],
    ("no_system", "alpha"): [0.3, 0.4, 0.5],
    ("no_system", "beta"): [0.2, 0.3, 0.4],
}
MANIFEST = """\
[experiment]
name = "system-tool"
budget = "1h"
seeds = 3
parallel = 2

[[tasks]]
path = "tasks/alpha"
answers = "answers/alpha"

[[tasks]]
path = "tasks/beta"
answers = "answers/beta"

[arms.base]

[arms.no_system]
flags = { "tools.system" = false }
"""


class ExperimentFiles:
    @staticmethod
    def experiment(root: Path) -> Path:
        for task_id in ("alpha", "beta"):
            folder = root / "tasks" / task_id
            (folder / "data").mkdir(parents=True)
            (folder / "description.md").write_text("Predict label.\n")
            (folder / "task.toml").write_text(
                f'id = "{task_id}"\nmetric = "accuracy"\ndirection = "maximize"\n'
            )
            answers = root / "answers" / task_id
            answers.mkdir(parents=True)
            (answers / "answers.csv").write_text(
                "id,label\n" + "".join(f"{row},1\n" for row in range(ROWS))
            )

        board = "id,score\n" + "".join(f"{rank},{rank / 20:.2f}\n" for rank in range(20))
        (root / "answers" / "beta" / "leaderboard.csv").write_text(board)
        manifest = root / "experiment.toml"
        manifest.write_text(MANIFEST)
        return manifest

    @staticmethod
    def predictions(accuracy: float) -> str:
        right = round(accuracy * ROWS)
        return "id,label\n" + "".join(f"{row},{int(row < right)}\n" for row in range(ROWS))


class Finish:
    def __init__(self, clock: FakeClock, usage: Usage) -> None:
        self._clock = clock
        self._usage = usage

    def __call__(self, request: ModelRequest) -> ModelReply:
        self._clock.advance(7 * 24 * 3_600)
        return Reply.text("done", usage=self._usage)


class ScriptedLauncher:
    def __init__(self, *, crash_first: bool = False, cost_per_call: float = 0.0) -> None:
        self.calls: list[tuple[str, str, tuple[int, ...]]] = []
        self._crash_first = crash_first
        self._cost = cost_per_call

    async def start(self, plan: RunPlan, gpus: tuple[int, ...]) -> int:
        self.calls.append(("start", plan.folder.name, gpus))
        paths = RunPaths(root=plan.folder)
        TaskFolder.save(TaskFolder.load(plan.task.path), paths)
        clock = FakeClock()
        turns = self.turns(plan, clock)
        return await self._run(
            plan, Harness.start, turns[:2] if self._crash_first else turns, clock
        )

    async def resume(self, plan: RunPlan) -> int:
        self.calls.append(("resume", plan.folder.name, ()))
        clock = FakeClock()
        return await self._run(plan, Harness.resume, self.turns(plan, clock)[2:], clock)

    def turns(self, plan: RunPlan, clock: FakeClock) -> list[Turn]:
        accuracy = ACCURACY[plan.arm, plan.task_id][plan.seed - 1]
        usage = Usage(input_tokens=100, output_tokens=10, cost_usd=self._cost)
        turns: list[Turn] = [
            self._write("good.csv", ExperimentFiles.predictions(accuracy), usage),
            Reply.tools(
                ("submission_register", {"path": "good.csv", "val_score": accuracy}), usage=usage
            ),
            self._write("overfit.csv", ExperimentFiles.predictions(accuracy - 0.1), usage),
            Reply.tools(
                ("submission_register", {"path": "overfit.csv", "val_score": 0.99}), usage=usage
            ),
        ]

        if plan.flags.get("tools.system", True):
            turns.append(Reply.tools(("system", {}), usage=usage))

        return [*turns, Finish(clock, usage)]

    @staticmethod
    async def _run(
        plan: RunPlan,
        opener: Callable[..., Awaitable[Harness]],
        turns: list[Turn],
        clock: FakeClock,
    ) -> int:
        config = HarnessConfig(
            sandbox=SandboxSettings(kind="local"),
            flags=Flags().with_overrides(plan.flags),
            budget=BudgetSettings(total=plan.budget),
        )
        harness = await opener(
            config=config,
            task=TaskFolder.from_run(RunPaths(root=plan.folder)),
            paths=RunPaths(root=plan.folder),
            adapters=Adapters(llm=ScriptedLlm(turns), clock=clock, probe=FakeProbe()),
        )

        try:
            await harness.run()
        except ScriptExhausted:
            return 1
        finally:
            await harness.close()

        return 0

    @staticmethod
    def _write(path: str, text: str, usage: Usage) -> ModelReply:
        return Reply.tools(
            ("editor", {"action": "create", "path": path, "file_text": text}), usage=usage
        )
