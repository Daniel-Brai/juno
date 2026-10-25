from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class Durations:
    UNITS: ClassVar[Mapping[str, int]] = {"s": 1, "m": 60, "h": 3600, "d": 86400}

    @classmethod
    def parse(cls, text: str) -> float:
        total, amount, parts = 0.0, "", 0

        for char in text.lower().replace(" ", ""):
            if char.isdigit() or char == ".":
                amount += char
            elif char in cls.UNITS and amount:
                total += float(amount) * cls.UNITS[char]
                amount, parts = "", parts + 1
            else:
                raise cls._invalid(text)

        if amount or not parts:
            raise cls._invalid(text)

        return total

    @classmethod
    def coerce(cls, value: object) -> object:
        return cls.parse(value) if isinstance(value, str) else value

    @classmethod
    def format(cls, seconds: float) -> str:
        remaining = max(0, round(seconds))
        parts: list[str] = []

        for unit, size in sorted(cls.UNITS.items(), key=lambda item: item[1], reverse=True):
            amount, remaining = divmod(remaining, size)

            if amount:
                parts.append(f"{amount}{unit}")

        return " ".join(parts[:2]) or "0s"

    @staticmethod
    def _invalid(text: str) -> ValueError:
        return ValueError(f"invalid duration {text!r}, expected e.g. '90s', '10m', '1h30m'")


type Seconds = Annotated[float, BeforeValidator(Durations.coerce), Field(gt=0)]


class DottedKeys:
    @classmethod
    def nest(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        nested: dict[str, Any] = {}

        for key, value in data.items():
            *parents, leaf = key.split(".")
            branch = {leaf: cls.nest(value) if isinstance(value, Mapping) else value}

            for parent in reversed(parents):
                branch = {parent: branch}

            nested = cls.merge(nested, branch)

        return nested

    @classmethod
    def merge(cls, base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(base)

        for key, value in extra.items():
            current = merged.get(key)

            if isinstance(value, Mapping) and isinstance(current, Mapping):
                merged[key] = cls.merge(current, value)
            else:
                merged[key] = value

        return merged


class Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolFlags(Section):
    system: bool = True
    jobs: bool = True
    submission: bool = True


class NudgeFlags(Section):
    time: bool = False
    resource: bool = False


class PromptFlags(Section):
    strict_validation: bool = False


class Flags(Section):
    tools: ToolFlags = Field(default_factory=ToolFlags)
    nudge: NudgeFlags = Field(default_factory=NudgeFlags)
    prompt: PromptFlags = Field(default_factory=PromptFlags)
    delegation: bool = False
    parallel: bool = False
    broadcast: bool = False
    skills: bool = False
    compaction: bool = False
    network: bool = False
    mcp_servers: bool = False

    @model_validator(mode="before")
    @classmethod
    def _expand_dotted_keys(cls, data: Any) -> Any:
        return DottedKeys.nest(data) if isinstance(data, Mapping) else data

    def with_overrides(self, overrides: Mapping[str, Any]) -> Flags:
        return Flags.model_validate(DottedKeys.merge(self.model_dump(), DottedKeys.nest(overrides)))


class ModelNames:
    ANTHROPIC: ClassVar[str] = "anthropic/"

    @classmethod
    def openrouter(cls, name: str) -> str:
        if "/" in name or not name.startswith("claude-"):
            return name

        parts = name.split("-")
        version: list[str] = []

        while parts and parts[-1].isdigit() and len(version) < 2:
            version.insert(0, parts.pop())

        return f"{cls.ANTHROPIC}{'-'.join([*parts, '.'.join(version)]) if version else name}"

    @classmethod
    def anthropic(cls, name: str) -> str:
        return name.removeprefix(cls.ANTHROPIC).partition(":")[0].replace(".", "-")

    @classmethod
    def is_anthropic(cls, name: str) -> bool:
        return cls.openrouter(name).startswith(cls.ANTHROPIC)


class ModelSettings(Section):
    provider: Literal["anthropic", "openrouter"] = "anthropic"
    name: str = "anthropic/claude-opus-5.5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = Field(default=64_000, ge=1_024)


class McpServer(Section):
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None

    @model_validator(mode="after")
    def _one_way_in(self) -> McpServer:
        if (self.command is None) == (self.url is None):
            raise ValueError("an MCP server needs either a command or a url")

        return self


class McpFile(Section):
    servers: dict[str, McpServer] = Field(default_factory=dict)


class BudgetSettings(Section):
    total: Seconds = 86_400
    max_cost: float | None = Field(default=None, gt=0)
    bash_timeout: Seconds = 600
    max_jobs: int = Field(default=4, ge=1)


class SandboxSettings(Section):
    kind: Literal["podman", "local"] = "podman"
    image: str = "docker.io/library/python:3.13-slim"
    gpus: tuple[int, ...] = ()
    memory: str | None = None
    network: bool = False


class WatchdogSettings(Section):
    memory_fraction: float = Field(default=0.95, gt=0, le=1)
    interval: Seconds = 5


class HarnessConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JUNO_",
        env_nested_delimiter="__",
        toml_file="juno.toml",
        frozen=True,
        extra="forbid",
    )

    model: ModelSettings = Field(default_factory=ModelSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    watchdog: WatchdogSettings = Field(default_factory=WatchdogSettings)
    flags: Flags = Field(default_factory=Flags)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, TomlConfigSettingsSource(settings_cls)
