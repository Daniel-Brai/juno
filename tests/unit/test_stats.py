import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from juno.eval.stats import Bootstrap, Estimate, Samples

scores = st.floats(min_value=0, max_value=100, allow_nan=False)
samples = st.dictionaries(
    st.sampled_from(["a", "b", "c", "d"]),
    st.lists(scores, min_size=1, max_size=5),
    min_size=1,
)
QUICK = Bootstrap(resamples=200)


class TestBootstrap:
    def test_macro_average_weighs_tasks_equally(self) -> None:
        assert Bootstrap.macro({"a": [0, 100, 100, 100], "b": [0]}) == 37.5

    def test_one_seed_per_task_leaves_no_spread(self) -> None:
        assert QUICK.mean({"a": [40], "b": [60]}) == Estimate(50, 50, 50)

    def test_a_clear_gap_excludes_zero(self) -> None:
        better = {"a": [80, 82, 81], "b": [70, 72, 71]}
        worse = {"a": [40, 42, 41], "b": [30, 33, 31]}

        delta = Bootstrap().difference(better, worse)

        assert delta.value == pytest.approx(40, abs=0.5)
        assert delta.excludes_zero and delta.low > 35

    def test_arms_must_share_a_task(self) -> None:
        with pytest.raises(ValueError, match="share no task"):
            QUICK.difference({"a": [1]}, {"b": [1]})

    def test_estimates_format_with_their_interval(self) -> None:
        assert Estimate(12.34, 8.1, 16.06).format() == "12.3 [8.1, 16.1]"

    @settings(max_examples=50, deadline=None)
    @given(samples)
    def test_the_interval_holds_the_point_estimate(self, data: Samples) -> None:
        estimate = QUICK.mean(data)

        assert estimate.low - 1e-9 <= estimate.value <= estimate.high + 1e-9

    @settings(max_examples=50, deadline=None)
    @given(samples, st.floats(min_value=-50, max_value=50))
    def test_shifting_every_score_shifts_the_interval(self, data: Samples, shift: float) -> None:
        moved = {task: [value + shift for value in values] for task, values in data.items()}

        before, after = QUICK.mean(data), QUICK.mean(moved)

        assert after.value == pytest.approx(before.value + shift, abs=1e-6)
        assert after.low == pytest.approx(before.low + shift, abs=1e-6)
        assert after.high == pytest.approx(before.high + shift, abs=1e-6)

    @settings(max_examples=50, deadline=None)
    @given(samples)
    def test_identical_arms_differ_by_an_interval_around_zero(self, data: Samples) -> None:
        delta = QUICK.difference(data, data)

        assert delta.value == 0
        assert delta.low <= 0 <= delta.high

    @settings(max_examples=50, deadline=None)
    @given(samples, samples)
    def test_swapping_arms_negates_the_point_estimate(
        self, first: Samples, second: Samples
    ) -> None:
        if not first.keys() & second.keys():
            return

        assert QUICK.difference(first, second).value == pytest.approx(
            -QUICK.difference(second, first).value
        )

    @given(samples)
    def test_a_fixed_seed_reproduces_the_interval(self, data: Samples) -> None:
        assert Bootstrap(resamples=100, seed=3).mean(data) == Bootstrap(resamples=100, seed=3).mean(
            data
        )
