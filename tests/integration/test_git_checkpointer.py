import subprocess
from pathlib import Path

import pytest

from juno.adapters.git import GitCheckpointer

pytestmark = pytest.mark.anyio


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=True
        ).stdout

    def tracked(self) -> set[str]:
        return set(self.git("ls-files").split())

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    root = tmp_path / "workspace"
    root.mkdir()
    return Repo(root)


class TestGitCheckpointer:
    async def test_first_checkpoint_creates_the_repo(self, repo: Repo) -> None:
        (repo.root / "train.py").write_text("print('hi')\n")

        commit = await GitCheckpointer(repo.root).checkpoint("first")

        assert commit == repo.head()
        assert repo.tracked() == {"train.py"}

    async def test_nothing_changed_means_no_commit(self, repo: Repo) -> None:
        checkpointer = GitCheckpointer(repo.root)
        (repo.root / "train.py").write_text("x = 1\n")
        await checkpointer.checkpoint("first")

        assert await checkpointer.checkpoint("again") is None

    async def test_large_files_stay_out_of_history(self, repo: Repo) -> None:
        (repo.root / "model.bin").write_bytes(b"0" * 2_048)
        (repo.root / "train.py").write_text("x = 1\n")

        await GitCheckpointer(repo.root, max_file_bytes=1_024).checkpoint("first")

        assert repo.tracked() == {"train.py"}

    async def test_deletions_and_renames_are_recorded(self, repo: Repo) -> None:
        checkpointer = GitCheckpointer(repo.root)
        (repo.root / "a.py").write_text("a = 1\n")
        (repo.root / "b.py").write_text("b = 1\n")
        await checkpointer.checkpoint("first")

        (repo.root / "a.py").unlink()
        (repo.root / "b.py").rename(repo.root / "c.py")
        await checkpointer.checkpoint("second")

        assert repo.tracked() == {"c.py"}
