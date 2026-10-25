from pathlib import Path

from juno.entrypoints.attachments import Attachments


class TestAttachments:
    def test_files_named_with_at_are_appended_once(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hi')\n")

        expanded = Attachments(tmp_path).expand("why does @src/app.py fail? see @src/app.py.")

        assert expanded.attached == ["src/app.py"]
        assert expanded.text.endswith("<file path=\"src/app.py\">\nprint('hi')\n</file>")
        assert expanded.text.count("<file") == 1

    def test_unreadable_names_are_skipped_with_a_reason(self, tmp_path: Path) -> None:
        (tmp_path / "blob.bin").write_bytes(b"\0\1")
        (tmp_path / "big.txt").write_text("x" * (Attachments.LIMIT + 1))

        expanded = Attachments(tmp_path).expand(
            "mail me@example.com about @missing.py @../secret @blob.bin @big.txt"
        )

        assert expanded.attached == []
        assert expanded.skipped == [
            "missing.py: no such file",
            "../secret: outside the project",
            "blob.bin: binary",
            "big.txt: larger than 100 KB",
        ]
        assert (
            expanded.text == "mail me@example.com about @missing.py @../secret @blob.bin @big.txt"
        )
