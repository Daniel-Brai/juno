import pytest

from fakes.clock import FakeClock
from fakes.probe import FakeProbe
from juno.domain.session import Session
from juno.domain.values import Message, ModelRequest, Role, ToolSpec

GATED_MARKERS = {"live_api": "live", "gpu": "gpu"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="run tests that call the real model API")
    parser.addoption("--gpu", action="store_true", help="run tests that need an NVIDIA GPU")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker, option in GATED_MARKERS.items():
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"needs --{option}"))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(start=1_000.0)


@pytest.fixture
def probe() -> FakeProbe:
    return FakeProbe()


@pytest.fixture
def session(clock: FakeClock) -> Session:
    return Session(task_id="spooky", started_at=clock.now(), deadline=clock.now() + 3_600)


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        system="You are an ML engineer.",
        tools=[ToolSpec(name="bash", description="Run a command", input_schema={"type": "object"})],
        messages=[Message(role=Role.USER, content="Solve the task.")],
    )
