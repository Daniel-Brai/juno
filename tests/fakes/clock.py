import asyncio


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.sleeps: list[float] = []
        self._now = start

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock cannot go backwards")

        self._now += seconds
