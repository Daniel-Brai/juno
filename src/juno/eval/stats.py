import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

type Samples = Mapping[str, Sequence[float]]


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def format(self, digits: int = 1) -> str:
        return f"{self.value:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


class Bootstrap:
    def __init__(self, *, resamples: int = 2_000, confidence: float = 0.95, seed: int = 0) -> None:
        self.resamples = resamples
        self.confidence = confidence
        self.seed = seed

    def mean(self, samples: Samples) -> Estimate:
        tasks = sorted(self._tasks(samples))
        rng = random.Random(self.seed)
        draws = [
            self.macro({task: self._resample(rng, samples[task]) for task in tasks})
            for _ in range(self.resamples)
        ]
        return self._estimate(self.macro({task: samples[task] for task in tasks}), draws)

    def difference(self, first: Samples, second: Samples) -> Estimate:
        tasks = sorted(self._tasks(first) & self._tasks(second))

        if not tasks:
            raise ValueError("the two arms share no task with results")

        rng = random.Random(self.seed)
        draws = [
            sum(
                self.average(self._resample(rng, first[task]))
                - self.average(self._resample(rng, second[task]))
                for task in tasks
            )
            / len(tasks)
            for _ in range(self.resamples)
        ]
        point = sum(self.average(first[task]) - self.average(second[task]) for task in tasks)
        return self._estimate(point / len(tasks), draws)

    @classmethod
    def macro(cls, samples: Samples) -> float:
        return sum(cls.average(values) for values in samples.values()) / len(samples)

    @staticmethod
    def average(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _tasks(samples: Samples) -> set[str]:
        tasks = {task for task, values in samples.items() if values}

        if not tasks:
            raise ValueError("no task has results")

        return tasks

    @staticmethod
    def _resample(rng: random.Random, values: Sequence[float]) -> list[float]:
        return rng.choices(values, k=len(values))

    def _estimate(self, point: float, draws: list[float]) -> Estimate:
        draws.sort()
        tail = (1 - self.confidence) / 2
        return Estimate(
            value=point,
            low=draws[math.floor(tail * (len(draws) - 1))],
            high=draws[math.ceil((1 - tail) * (len(draws) - 1))],
        )
