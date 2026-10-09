import asyncio
import os
import time

import pytest

from juno.adapters.sandboxes.local import LocalSandbox
from juno.domain.values import RunPaths

pytestmark = pytest.mark.anyio


@pytest.fixture
def sandbox(paths: RunPaths) -> LocalSandbox:
    return LocalSandbox(paths.workspace, output_limit=200)


class TestExec:
    async def test_runs_in_the_workspace(self, sandbox: LocalSandbox, paths: RunPaths) -> None:
        result = await sandbox.exec("pwd && echo hi >&2", timeout_s=10)

        assert result.exit_code == 0
        assert result.output.splitlines() == [str(paths.workspace), "hi"]

    async def test_reports_failing_exit_codes(self, sandbox: LocalSandbox) -> None:
        result = await sandbox.exec("exit 3", timeout_s=10)

        assert (result.exit_code, result.timed_out) == (3, False)

    async def test_timeout_kills_the_whole_process_group(self, sandbox: LocalSandbox) -> None:
        started = time.monotonic()

        result = await sandbox.exec("sleep 30 & echo $! > child.pid; sleep 30", timeout_s=0.5)

        assert result.timed_out and result.exit_code is None
        assert time.monotonic() - started < 10

    async def test_background_leftovers_are_cleaned_up(
        self, sandbox: LocalSandbox, paths: RunPaths
    ) -> None:
        await sandbox.exec("sleep 30 & echo $! > child.pid", timeout_s=10)
        child = int((paths.workspace / "child.pid").read_text())

        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)

    async def test_large_output_is_truncated(self, sandbox: LocalSandbox) -> None:
        result = await sandbox.exec("seq 1 10000", timeout_s=10)

        assert result.truncated
        assert result.output.startswith("1\n2\n")
        assert result.output.rstrip().endswith("10000")


class TestJobs:
    async def test_spawned_job_reports_its_exit_code(
        self, sandbox: LocalSandbox, paths: RunPaths
    ) -> None:
        job = await sandbox.spawn(
            "echo training; exit 4", paths.logs / "j.log", paths.logs / "j.exit"
        )

        for _ in range(100):
            if (code := sandbox.poll(job)) is not None:
                break

            await asyncio.sleep(0.05)

        assert code == 4
        assert job.log_path.read_text() == "training\n"

    async def test_kill_group_stops_a_running_job(
        self, sandbox: LocalSandbox, paths: RunPaths
    ) -> None:
        job = await sandbox.spawn("sleep 30", paths.logs / "j.log", paths.logs / "j.exit")

        await sandbox.kill_group(job.pgid)

        with pytest.raises(ProcessLookupError):
            os.killpg(job.pgid, 0)
