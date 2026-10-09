from hypothesis import given
from hypothesis import strategies as st

from juno.adapters.sandboxes.local import OutputBuffer


class TestOutputBuffer:
    def test_small_output_is_kept_whole(self) -> None:
        buffer = OutputBuffer(limit=20)
        buffer.add(b"hello ")
        buffer.add(b"world")

        assert (buffer.text(), buffer.truncated) == ("hello world", False)

    def test_large_output_keeps_head_and_tail(self) -> None:
        buffer = OutputBuffer(limit=8)
        buffer.add(b"abcd")
        buffer.add(b"-" * 100)
        buffer.add(b"wxyz")

        assert buffer.text() == "abcd\n[... 100 bytes omitted ...]\nwxyz"

    @given(st.lists(st.binary(max_size=200), max_size=30), st.integers(min_value=2, max_value=100))
    def test_never_keeps_more_than_the_limit(self, chunks: list[bytes], limit: int) -> None:
        buffer = OutputBuffer(limit=limit)

        for chunk in chunks:
            buffer.add(chunk)

        total = sum(len(chunk) for chunk in chunks)
        kept = total - buffer.dropped

        assert kept <= limit
        assert buffer.truncated == (total > 2 * (limit // 2))
