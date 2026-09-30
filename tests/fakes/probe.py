from juno.domain.values import GpuSnapshot, ResourceSnapshot

GIB = 1024**3


class FakeProbe:
    def __init__(self, snapshot: ResourceSnapshot | None = None) -> None:
        self.samples = 0
        self.snapshot = snapshot or ResourceSnapshot(
            cpu_percent=10.0,
            ram_used_bytes=8 * GIB,
            ram_total_bytes=64 * GIB,
            gpus=(
                GpuSnapshot(
                    index=0,
                    utilization_percent=0.0,
                    vram_used_bytes=0,
                    vram_total_bytes=80 * GIB,
                ),
            ),
        )

    def sample(self) -> ResourceSnapshot:
        self.samples += 1
        return self.snapshot

    def near_oom(self, used_fraction: float = 0.97) -> None:
        total = self.snapshot.ram_total_bytes
        self.snapshot = self.snapshot.model_copy(
            update={"ram_used_bytes": int(total * used_fraction)}
        )
