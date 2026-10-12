import asyncio
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from uuid import uuid7

import pytest

from juno.adapters.sandboxes.podman import PodmanSandbox
from juno.config import SandboxSettings
from juno.domain.values import RunPaths

IMAGE = SandboxSettings().image
LIST_SLEEPS = (
    "for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < $f; echo; done | grep '^sleep 60' || true"
)


class Podman:
    @staticmethod
    def available() -> bool:
        if shutil.which("podman") is None:
            return False

        return subprocess.run(["podman", "image", "exists", IMAGE], check=False).returncode == 0


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.slow,
    pytest.mark.skipif(not Podman.available(), reason=f"needs podman and {IMAGE}"),
]


@pytest.fixture
async def sandbox(paths: RunPaths) -> AsyncIterator[PodmanSandbox]:
    data = paths.root / "data"
    data.mkdir()
    (data / "train.csv").write_text("id,label\n1,0\n")
    box = PodmanSandbox(
        SandboxSettings(image=IMAGE, memory="512m"),
        name=f"juno-test-{uuid7().hex[-12:]}",
        workspace=paths.workspace,
        logs=paths.logs,
        data=data,
    )
    await box.start()
    yield box
    await box.stop()


class TestPodmanExec:
    async def test_runs_inside_the_container_workspace(
        self, sandbox: PodmanSandbox, paths: RunPaths
    ) -> None:
        result = await sandbox.exec("pwd; ls /data; echo made > out.txt", timeout_s=30)

        assert result.exit_code == 0
        assert result.output.splitlines() == ["/workspace", "train.csv"]
        assert (paths.workspace / "out.txt").read_text() == "made\n"

    async def test_data_is_read_only_and_there_is_no_network(self, sandbox: PodmanSandbox) -> None:
        write = await sandbox.exec("touch /data/new", timeout_s=30)
        network = await sandbox.exec(
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\"",
            timeout_s=30,
        )

        assert write.exit_code != 0
        assert network.exit_code != 0

    async def test_a_timeout_kills_the_processes_in_the_container(
        self, sandbox: PodmanSandbox
    ) -> None:
        started = time.monotonic()
        result = await sandbox.exec("sleep 60 & sleep 60", timeout_s=1)
        leftovers = await sandbox.exec(LIST_SLEEPS, timeout_s=30)

        assert result.timed_out
        assert time.monotonic() - started < 20
        assert leftovers.output.strip() == ""


class TestPodmanJobs:
    async def test_a_job_reports_its_exit_code_and_log(
        self, sandbox: PodmanSandbox, paths: RunPaths
    ) -> None:
        job = await sandbox.spawn(
            "echo training; exit 3", paths.logs / "j.log", paths.logs / "j.exit"
        )

        code: int | None = None

        for _ in range(100):
            if (code := sandbox.poll(job)) is not None:
                break

            await asyncio.sleep(0.1)

        assert code == 3
        assert job.log_path.read_text() == "training\n"
        assert not sandbox.is_alive(job)

    async def test_kill_group_stops_a_running_job(
        self, sandbox: PodmanSandbox, paths: RunPaths
    ) -> None:
        job = await sandbox.spawn("sleep 60", paths.logs / "j.log", paths.logs / "j.exit")
        assert sandbox.is_alive(job)

        await sandbox.kill_group(job.pgid)
        leftovers = await sandbox.exec(LIST_SLEEPS, timeout_s=30)

        assert leftovers.output.strip() == ""

    async def test_a_restarted_container_marks_old_jobs_lost(
        self, sandbox: PodmanSandbox, paths: RunPaths
    ) -> None:
        job = await sandbox.spawn("sleep 60", paths.logs / "j.log", paths.logs / "j.exit")
        await asyncio.to_thread(
            subprocess.run, ["podman", "restart", "--time", "1", sandbox.name], check=True
        )
        await asyncio.sleep(1.1)

        await sandbox.start()

        assert not sandbox.is_alive(job)
