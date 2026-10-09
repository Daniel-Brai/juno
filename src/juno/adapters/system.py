import asyncio
import time

import psutil
import pynvml

from juno.domain.values import GpuSnapshot, ResourceSnapshot


class SystemClock:
    def now(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SystemProbe:
    def __init__(self) -> None:
        self.has_gpus = self._start_nvml()

    def sample(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()

        return ResourceSnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_used_bytes=memory.total - memory.available,
            ram_total_bytes=memory.total,
            gpus=self.gpus(),
        )

    def gpus(self) -> tuple[GpuSnapshot, ...]:
        if not self.has_gpus:
            return ()

        snapshots: list[GpuSnapshot] = []

        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            snapshots.append(
                GpuSnapshot(
                    index=index,
                    utilization_percent=float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
                    vram_used_bytes=int(memory.used),
                    vram_total_bytes=int(memory.total),
                )
            )

        return tuple(snapshots)

    @staticmethod
    def _start_nvml() -> bool:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return False

        return True
