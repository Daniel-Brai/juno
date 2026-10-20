import subprocess
from pathlib import Path

import pytest

from juno.domain.commands import FindFiles, ReadFile, SearchText, TodoItem, UpdateTodos
from juno.domain.values import Message, Role
from juno.tools.registry import ToolError
from juno.tools.search import SearchTools
from juno.tools.todo import TodoTools
from juno.tools.workspace import Workspace

pytestmark = pytest.mark.anyio


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src" / "app").mkdir(parents=True)
    (root / "build").mkdir()
    (root / ".gitignore").write_text("build/\n")
    (root / "src" / "app" / "main.py").write_text("def main():\n    return greet('juno')\n")
    (root / "src" / "app" / "greet.py").write_text("def greet(name):\n    return f'hi {name}'\n")
    (root / "build" / "main.py").write_text("generated = True\n")
    (root / "README.md").write_text("".join(f"line {n}\n" for n in range(1, 11)))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


@pytest.fixture
def search(project: Path) -> SearchTools:
    return SearchTools(Workspace(project))


class TestGlob:
    async def test_finds_files_at_any_depth_and_skips_ignored_ones(
        self, search: SearchTools
    ) -> None:
        assert await search.glob(FindFiles(pattern="*.py")) == "src/app/greet.py\nsrc/app/main.py"

    async def test_searches_inside_a_folder(self, search: SearchTools) -> None:
        assert await search.glob(FindFiles(pattern="main.py", path="src")) == "src/app/main.py"
        assert await search.glob(FindFiles(pattern="*.rs")) == "no files match"

    @pytest.mark.parametrize("ripgrep", [True, False], ids=["ripgrep", "walk"])
    async def test_outside_git_generated_folders_of_every_ecosystem_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ripgrep: bool
    ) -> None:
        root = tmp_path / "polyglot"
        sources = ["lib/app.ex", "src/Main.hs", "src/main.rs", "src/main.cpp", "web/app.ts"]
        generated = [
            "_build/dev/app.beam",
            "deps/jason/mix.exs",
            "dist-newstyle/cache/plan.json",
            ".stack-work/install/x.hi",
            "target/debug/app.d",
            "cmake-build-debug/CMakeCache.txt",
            "web/node_modules/left-pad/index.ts",
            "web/.next/server/page.js",
            "app/.gradle/build.lock",
            "pkg/demo.egg-info/PKG-INFO",
            "secret/key.txt",
        ]

        for relative in sources + generated:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text("x\n")

        (root / ".gitignore").write_text("secret/\n")

        if not ripgrep:
            monkeypatch.setattr("juno.tools.search.shutil.which", lambda name: None)

        found = await SearchTools(Workspace(root)).glob(FindFiles(pattern="*"))

        ignored_by_file = [] if ripgrep else ["secret/key.txt"]
        assert sorted(found.splitlines()) == sorted([*sources, ".gitignore", *ignored_by_file])

    async def test_inside_git_tracked_vendored_code_is_still_found(
        self, project: Path, search: SearchTools
    ) -> None:
        (project / "vendor" / "lib").mkdir(parents=True)
        (project / "vendor" / "lib" / "dep.go").write_text("package lib\n")

        assert await search.glob(FindFiles(pattern="*.go")) == "vendor/lib/dep.go"

    @pytest.mark.parametrize(
        ("path", "generated"),
        [
            ("deps/phoenix/lib/a.ex", True),
            ("target/release/build/x.rs", True),
            ("cmake-build-release/a.o", True),
            ("src/foo.egg-info/x", True),
            ("src/target.rs", False),
            ("docs/build.md", False),
            ("lib/deps.ex", False),
        ],
    )
    def test_only_folders_count_as_generated(self, path: str, generated: bool) -> None:
        assert SearchTools.generated(path) is generated


class TestGrep:
    async def test_shows_project_relative_matches(self, search: SearchTools) -> None:
        found = await search.grep(SearchText(pattern="greet\\(", path="src"))

        assert sorted(found.splitlines()) == [
            "src/app/greet.py:1:def greet(name):",
            "src/app/main.py:2:    return greet('juno')",
        ]

    async def test_outside_git_generated_folders_are_not_searched(self, tmp_path: Path) -> None:
        root = tmp_path / "rust"
        (root / "src").mkdir(parents=True)
        (root / "target" / "debug").mkdir(parents=True)
        (root / "src" / "main.rs").write_text("fn needle() {}\n")
        (root / "target" / "debug" / "main.d").write_text("needle\n")

        found = await SearchTools(Workspace(root)).grep(SearchText(pattern="needle"))

        assert found == "src/main.rs:1:fn needle() {}"

    async def test_reports_no_matches_and_bad_patterns(self, search: SearchTools) -> None:
        assert await search.grep(SearchText(pattern="nowhere")) == "no matches"

        with pytest.raises(ToolError):
            await search.grep(SearchText(pattern="(unclosed"))


class TestRead:
    async def test_numbers_lines_and_says_what_is_left(self, search: SearchTools) -> None:
        text = await search.read(ReadFile(path="README.md", offset=3, limit=2))

        assert text == "     3\tline 3\n     4\tline 4\n... 6 more lines"

    async def test_refuses_folders_binaries_and_paths_outside(
        self, search: SearchTools, project: Path
    ) -> None:
        (project / "blob.bin").write_bytes(b"\0\1\2")

        for path, message in [("src", "not a file"), ("blob.bin", "binary"), ("../x", "outside")]:
            with pytest.raises(ToolError, match=message):
                await search.read(ReadFile(path=path))


class TestTodo:
    async def test_renders_the_whole_list(self) -> None:
        todos = TodoTools()

        text = await todos.update(
            UpdateTodos(
                items=[
                    TodoItem(text="read the code", status="done"),
                    TodoItem(text="fix the bug", status="in_progress"),
                    TodoItem(text="run the tests"),
                ]
            )
        )

        assert text == "[x] read the code\n[>] fix the bug\n[ ] run the tests"

    def test_the_last_list_in_the_history_comes_back(self) -> None:
        call = {"type": "tool_use", "id": "t", "name": "todo"}
        messages = [
            Message(role=Role.ASSISTANT, content=[{**call, "input": {"items": [{"text": "a"}]}}]),
            Message(role=Role.ASSISTANT, content=[{**call, "input": {"items": [{"text": "b"}]}}]),
            Message(role=Role.ASSISTANT, content=[{**call, "input": {"items": "broken"}}]),
        ]

        assert [item.text for item in TodoTools.from_history(messages).items] == ["b"]
