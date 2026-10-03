import pytest

from juno.domain.commands import EditFile
from juno.tools.editor import EditorTools
from juno.tools.registry import ToolError
from juno.tools.workspace import Workspace


@pytest.fixture
def editor(workspace: Workspace) -> EditorTools:
    (workspace.root / "train.py").write_text("import pandas\nprint('train')\n")
    return EditorTools(workspace, view_limit=3)


class TestEditor:
    def test_view_numbers_lines(self, editor: EditorTools) -> None:
        output = editor.apply(EditFile(action="view", path="train.py"))

        assert output == "     1\timport pandas\n     2\tprint('train')"

    def test_view_range_and_limit(self, editor: EditorTools, workspace: Workspace) -> None:
        (workspace.root / "long.py").write_text("\n".join(f"line {n}" for n in range(1, 11)))

        assert editor.apply(EditFile(action="view", path="long.py", view_range=(2, 2))) == (
            "     2\tline 2"
        )
        assert editor.apply(EditFile(action="view", path="long.py")).endswith(
            "[... 7 more lines ...]"
        )

    def test_view_lists_folders_first(self, editor: EditorTools, workspace: Workspace) -> None:
        (workspace.root / "src").mkdir()

        assert editor.apply(EditFile(action="view", path=".")) == "src/\ntrain.py"

    def test_create_writes_nested_files(self, editor: EditorTools, workspace: Workspace) -> None:
        output = editor.apply(EditFile(action="create", path="src/model.py", file_text="x = 1\n"))

        assert output == "created src/model.py"
        assert (workspace.root / "src/model.py").read_text() == "x = 1\n"

    def test_str_replace_edits_one_unique_match(
        self, editor: EditorTools, workspace: Workspace
    ) -> None:
        editor.apply(
            EditFile(action="str_replace", path="train.py", old_str="'train'", new_str="'fit'")
        )

        assert (workspace.root / "train.py").read_text() == "import pandas\nprint('fit')\n"

    @pytest.mark.parametrize("old", ["missing", "a"], ids=["absent", "repeated"])
    def test_str_replace_refuses_ambiguous_matches(self, editor: EditorTools, old: str) -> None:
        with pytest.raises(ToolError, match="must occur once"):
            editor.apply(EditFile(action="str_replace", path="train.py", old_str=old, new_str="b"))

    @pytest.mark.parametrize(
        ("line", "expected"), [(0, "# top\nimport pandas\n"), (1, "import pandas\n# top\n")]
    )
    def test_insert_after_a_line(
        self, editor: EditorTools, workspace: Workspace, line: int, expected: str
    ) -> None:
        editor.apply(EditFile(action="insert", path="train.py", insert_line=line, new_str="# top"))

        assert (workspace.root / "train.py").read_text().startswith(expected)

    def test_insert_past_the_end_is_refused(self, editor: EditorTools) -> None:
        with pytest.raises(ToolError, match="past the end"):
            editor.apply(EditFile(action="insert", path="train.py", insert_line=9, new_str="x"))

    def test_paths_outside_the_workspace_are_refused(self, editor: EditorTools) -> None:
        with pytest.raises(ToolError, match="outside the workspace"):
            editor.apply(EditFile(action="view", path="../../etc/passwd"))

    def test_missing_fields_are_named(self, editor: EditorTools) -> None:
        with pytest.raises(ToolError, match="file_text is required"):
            editor.apply(EditFile(action="create", path="new.py"))
