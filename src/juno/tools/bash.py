from juno.config import Durations
from juno.domain.commands import RunBash
from juno.domain.session import Session
from juno.ports import Clock, Sandbox
from juno.tools.registry import Tool, ToolError


class BashTools:
    def __init__(
        self, sandbox: Sandbox, clock: Clock, session: Session, max_timeout_s: float
    ) -> None:
        self._sandbox = sandbox
        self._clock = clock
        self._session = session
        self._max_timeout_s = max_timeout_s

    def tools(self) -> list[Tool[RunBash]]:
        return [
            Tool(
                name="bash",
                description=(
                    "Run a shell command in the workspace and get its exit code and output. "
                    f"Commands are killed after {Durations.format(self._max_timeout_s)}; "
                    "start anything longer with job_submit."
                ),
                input_model=RunBash,
                handler=self.run,
            )
        ]

    async def run(self, args: RunBash) -> str:
        remaining = self._session.remaining(self._clock.now())

        if remaining <= 0:
            raise ToolError("the time budget is spent")

        timeout = min(args.timeout_s, self._max_timeout_s, remaining)
        result = await self._sandbox.exec(args.command, timeout)

        if result.timed_out:
            status = f"timed out after {Durations.format(timeout)}, process group killed"
        else:
            status = f"exit code {result.exit_code}"

        return f"{status}, took {Durations.format(result.duration_s)}\n{result.output}"
