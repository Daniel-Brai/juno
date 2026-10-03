import asyncio
import time

import psutil

from juno.domain.values import ResourceSnapshot


class SystemClock:
    def now(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class PsutilProbe:
    def sample(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()

        return ResourceSnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_used_bytes=memory.total - memory.available,
            ram_total_bytes=memory.total,
        )
