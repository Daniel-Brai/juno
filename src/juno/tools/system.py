from juno.config import Durations
from juno.domain.commands import QuerySystem
from juno.ports import ResourceProbe
from juno.service.context import RunContext
from juno.tools.registry import Tool

GIB = 1024**3


class SystemTools:
    def __init__(self, probe: ResourceProbe, context: RunContext) -> None:
        self._probe = probe
        self._clock = context.clock
        self._session = context.session

    def tools(self) -> list[Tool[QuerySystem]]:
        return [
            Tool(
                name="system",
                description="Show CPU, RAM and GPU use, plus elapsed and remaining time.",
                input_model=QuerySystem,
                handler=self.run,
                read_only=True,
            )
        ]

    async def run(self, _: QuerySystem) -> str:
        snapshot = self._probe.sample()
        now = self._clock.now()
        lines = [
            f"CPU: {snapshot.cpu_percent:.0f}%",
            f"RAM: {snapshot.ram_used_bytes / GIB:.1f} of {snapshot.ram_total_bytes / GIB:.1f} GiB "
            f"({snapshot.ram_used_fraction:.0%})",
        ]
        lines += [
            f"GPU {gpu.index}: {gpu.utilization_percent:.0f}% busy, "
            f"{gpu.vram_used_bytes / GIB:.1f} of {gpu.vram_total_bytes / GIB:.1f} GiB VRAM"
            for gpu in snapshot.gpus
        ] or ["GPU: none detected"]
        lines += [
            f"Elapsed: {Durations.format(self._session.elapsed(now))}",
            f"Remaining: {Durations.format(self._session.remaining(now))}",
        ]
        return "\n".join(lines)
