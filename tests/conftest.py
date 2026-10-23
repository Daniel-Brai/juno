from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import structlog

from fakes.clock import FakeClock
from fakes.probe import FakeProbe
from fakes.run import RunFactory
from juno.adapters.stores.sqlite import SqliteStore
from juno.domain.session import Session
from juno.domain.values import Direction, Message, ModelRequest, Role, RunPaths, Task, ToolSpec
from juno.ports import TransactionFactory
from juno.service.bus import MessageBus
from juno.service.context import RunContext
from juno.tools.workspace import Workspace

GATED_MARKERS = {"live_api": "live", "gpu": "gpu"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="run tests that call the real model API")
    parser.addoption("--gpu", action="store_true", help="run tests that need an NVIDIA GPU")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker, option in GATED_MARKERS.items():
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"needs --{option}"))


@pytest.fixture(autouse=True)
def fresh_logging() -> Iterator[None]:
    yield
    structlog.reset_defaults()


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
def paths(tmp_path: Path) -> RunPaths:
    run = RunPaths(root=tmp_path / "run")

    for path in run.all():
        path.mkdir(parents=True)

    return run


@pytest.fixture
async def store(paths: RunPaths) -> AsyncIterator[SqliteStore]:
    store = await SqliteStore.open(paths.root / "run.db")
    yield store
    await store.close()


@pytest.fixture
async def transactions(store: SqliteStore, session: Session) -> TransactionFactory:
    async with store.transaction() as tx:
        await tx.sessions.save(session)
        await tx.commit()

    return store.transaction


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def context(
    session: Session, clock: FakeClock, transactions: TransactionFactory, bus: MessageBus
) -> RunContext:
    return RunContext(session=session, clock=clock, transactions=transactions, bus=bus)


@pytest.fixture
def workspace(paths: RunPaths) -> Workspace:
    return Workspace(paths.workspace)


@pytest.fixture
def task(paths: RunPaths) -> Task:
    data = paths.root / "data"
    data.mkdir()
    (data / "train.csv").write_text("id,text,author\n1,It was a dark night,EAP\n")

    return Task(
        id="spooky",
        description="Predict the author of each sentence.",
        data_dir=data,
        metric="log_loss",
        direction=Direction.MINIMIZE,
    )


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        system="You are an ML engineer.",
        tools=[ToolSpec(name="bash", description="Run a command", input_schema={"type": "object"})],
        messages=[Message(role=Role.USER, content="Solve the task.")],
    )


@pytest.fixture
async def runs(
    task: Task, paths: RunPaths, clock: FakeClock, probe: FakeProbe
) -> AsyncIterator[RunFactory]:
    factory = RunFactory(task, paths, clock, probe)
    yield factory
    await factory.close_all()
