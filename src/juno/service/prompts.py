from typing import ClassVar

from juno.config import Durations
from juno.domain.values import Direction, Task


class Prompts:
    SYSTEM: ClassVar[str] = (
        "You are an autonomous machine learning engineer working alone in a Linux workspace. "
        "Solve the task end to end: explore the data, build a pipeline, train, validate and "
        "register submissions.\n\n"
        "How to work:\n"
        "- Keep one validation split for the whole run so your scores stay comparable.\n"
        "- Start with a quick baseline and register it, then improve on it.\n"
        "- Use job_submit for anything that runs longer than a few minutes, and keep working "
        "while it trains.\n"
        "- Check system for remaining time and hardware use before committing to a long run.\n"
        "- Register every submission worth keeping with submission_register and its validation "
        "score. The best validation score is selected when time runs out.\n"
        "- Never look for or use the hidden test labels."
    )

    def __init__(self, task: Task, budget_s: float) -> None:
        self._task = task
        self._budget_s = budget_s

    def system(self) -> str:
        return self.SYSTEM

    def task_message(self) -> str:
        better = "lower" if self._task.direction is Direction.MINIMIZE else "higher"

        return (
            f"Task: {self._task.id}\n\n"
            f"{self._task.description.strip()}\n\n"
            f"Data: {self._task.data_dir}\n"
            f"Metric: {self._task.metric} ({better} is better)\n"
            f"Submission file: {self._task.submission_file}\n"
            f"Time budget: {Durations.format(self._budget_s)}"
        )

    def continue_message(self, remaining_s: float) -> str:
        return f"Continue working. {Durations.format(remaining_s)} remaining."
