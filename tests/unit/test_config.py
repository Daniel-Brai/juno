import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from juno.config import BudgetSettings, Durations, Flags, HarnessConfig, ModelSettings


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)

    for name in [name for name in os.environ if name.startswith("JUNO_")]:
        monkeypatch.delenv(name)

    return tmp_path


class TestDefaults:
    def test_use_medium_effort_on_opus(self) -> None:
        config = HarnessConfig()

        assert config.model == ModelSettings(name="anthropic/claude-opus-5.5", effort="medium")
        assert config.budget.total == 86_400
        assert (
            config.flags.tools.system and config.flags.tools.jobs and config.flags.tools.submission
        )
        assert not (config.flags.nudge.time or config.flags.nudge.resource)
        assert not (config.flags.delegation or config.flags.parallel or config.flags.compaction)

    def test_are_frozen(self) -> None:
        config = HarnessConfig()

        with pytest.raises(ValidationError):
            config.__setattr__("budget", config.budget)


class TestDurations:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("90s", 90),
            ("10m", 600),
            ("1h30m", 5_400),
            ("1.5h", 5_400),
            ("24h", 86_400),
            ("2d", 172_800),
        ],
    )
    def test_parse_human_durations(self, text: str, seconds: float) -> None:
        assert BudgetSettings.model_validate({"total": text}).total == seconds

    @pytest.mark.parametrize("text", ["", "soon", "10", "10x", "h", "1h soon", "-5m", "1.2.3h"])
    def test_reject_anything_else(self, text: str) -> None:
        with pytest.raises(ValidationError):
            BudgetSettings.model_validate({"total": text})

    def test_parse_is_usable_on_its_own(self) -> None:
        assert Durations.parse("6h") == 21_600

        with pytest.raises(ValueError, match="invalid duration"):
            Durations.parse("soon")

    def test_plain_seconds_pass_straight_through(self) -> None:
        assert BudgetSettings.model_validate({"total": 600}).total == 600

    @pytest.mark.parametrize("value", [None, [], {"h": 1}, 0, -5])
    def test_non_durations_fail_validation_cleanly(self, value: object) -> None:
        with pytest.raises(ValidationError):
            BudgetSettings.model_validate({"total": value})

    @pytest.mark.parametrize(
        ("seconds", "text"),
        [(0, "0s"), (-3, "0s"), (45, "45s"), (90, "1m 30s"), (5_400, "1h 30m"), (90_061, "1d 1h")],
    )
    def test_format_shows_the_two_largest_units(self, seconds: float, text: str) -> None:
        assert Durations.format(seconds) == text

    @pytest.mark.parametrize("text", ["45s", "1m30s", "1h30m", "2d"])
    def test_format_round_trips_through_parse(self, text: str) -> None:
        assert Durations.parse(Durations.format(Durations.parse(text))) == Durations.parse(text)


class TestSources:
    def test_toml_file_is_read_with_dotted_flags(self, isolated: Path) -> None:
        (isolated / "juno.toml").write_text(
            '[budget]\ntotal = "6h"\n\n[flags]\n"nudge.time" = true\n'
        )

        config = HarnessConfig()

        assert config.budget.total == 21_600
        assert config.flags.nudge.time

    def test_environment_beats_the_toml_file(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (isolated / "juno.toml").write_text('[model]\neffort = "low"\n')
        monkeypatch.setenv("JUNO_MODEL__EFFORT", "xhigh")

        assert HarnessConfig().model.effort == "xhigh"

    def test_arguments_beat_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JUNO_MODEL__EFFORT", "xhigh")

        assert HarnessConfig(model=ModelSettings(effort="max")).model.effort == "max"


class TestFlags:
    def test_overrides_return_a_new_set(self) -> None:
        flags = Flags()

        changed = flags.with_overrides({"tools.jobs": False, "nudge.time": True})

        assert not changed.tools.jobs and changed.nudge.time
        assert flags.tools.jobs and not flags.nudge.time

    def test_overrides_keep_sibling_flags(self) -> None:
        changed = Flags().with_overrides({"tools.jobs": False})

        assert changed.tools.system and changed.tools.submission

    @pytest.mark.parametrize("key", ["nudge.timing", "tool.jobs", "unknown"])
    def test_unknown_flags_are_rejected(self, key: str) -> None:
        with pytest.raises(ValidationError):
            Flags().with_overrides({key: True})
