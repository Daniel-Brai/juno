from typing import ClassVar, Protocol

from juno.config import Durations
from juno.domain.values import Direction, Task


class Prompts(Protocol):
    def system(self) -> str: ...

    def opening(self) -> str | None: ...


class TaskPrompts:
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

    def opening(self) -> str:
        better = "lower" if self._task.direction is Direction.MINIMIZE else "higher"

        return (
            f"Task: {self._task.id}\n\n"
            f"{self._task.description.strip()}\n\n"
            f"Data: {self._task.data_dir}\n"
            f"Metric: {self._task.metric} ({better} is better)\n"
            f"Submission file: {self._task.submission_file}\n"
            f"Time budget: {Durations.format(self._budget_s)}"
        )


class ChatPrompts:
    SYSTEM: ClassVar[str] = (
        "You are a software engineer working in the user's project through tools. Read the "
        "code before changing it, make the smallest change that does the job, and run the "
        "project's checks or tests when they exist.\n\n"
        "How to work:\n"
        "- Answer greetings, questions about yourself and anything the conversation already "
        "covers directly, without tools. Look at the project only when the request needs it, "
        "and then only at what it needs.\n"
        "- Use glob, grep and read to find your way around; use bash for anything else.\n"
        "- Edit files with the editor tool, never with shell redirection.\n"
        "- Keep a todo list for work with several steps, and update it as you go.\n"
        "- Some calls need the user's approval. If one is denied, take that as an answer and "
        "do not try the same thing another way.\n"
        "- When you are done, say what changed and what you could not check.\n\n"
        "How to write:\n"
        "- Write the way a thoughtful colleague talks: plain sentences, short paragraphs, "
        "no filler or hype.\n"
        "- Do not use em dashes or en dashes. Use a comma, a full stop or brackets instead.\n"
        "- Skip decorative formatting. Use a list, a table, bold text or a heading only when "
        "it makes something clearer, such as steps to follow or values to compare.\n"
        "- Put code, commands, paths and identifiers in backticks."
    )

    def __init__(self, project: str, instructions: str | None, *, saved: str | None = None) -> None:
        self._project = project
        self._instructions = instructions
        self._saved = saved

    def system(self) -> str:
        if self._saved is not None:
            return self._saved

        parts = [self.SYSTEM, f"The project is in {self._project}."]

        if self._instructions:
            parts.append(f"Project instructions:\n\n{self._instructions.strip()}")

        return "\n\n".join(parts)

    def opening(self) -> None:
        return None
