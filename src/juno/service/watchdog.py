import asyncio
from typing import Protocol

from juno.domain.commands import CancelJob
from juno.domain.events import ResourcePressure
from juno.domain.values import Job
from juno.ports import ResourceProbe
from juno.service.context import RunContext


class StoppableJobs(Protocol):
    async def largest_running(self) -> Job | None: ...

    async def cancel(self, args: CancelJob) -> str: ...


class MemoryWatchdog:
    def __init__(
        self,
        probe: ResourceProbe,
        jobs: StoppableJobs | None,
        context: RunContext,
        *,
        limit: float,
        interval_s: float,
    ) -> None:
        self._probe = probe
        self._jobs = jobs
        self._bus = context.bus
        self._clock = context.clock
        self._session_id = context.session.id
        self._limit = limit
        self._interval_s = interval_s

    async def watch(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            await self.check()

    async def check(self) -> Job | None:
        snapshot = self._probe.sample()

        if snapshot.ram_used_fraction < self._limit:
            return None

        victim = await self._jobs.largest_running() if self._jobs is not None else None

        if victim is not None and self._jobs is not None:
            await self._jobs.cancel(CancelJob(job_id=victim.id))
            reason = f"RAM at {snapshot.ram_used_fraction:.0%}; stopped {victim.id} ({victim.name})"
        else:
            reason = f"RAM at {snapshot.ram_used_fraction:.0%}; no job to stop"

        await self._bus.publish(
            ResourcePressure(
                session_id=self._session_id, at=self._clock.now(), snapshot=snapshot, reason=reason
            )
        )
        return victim
