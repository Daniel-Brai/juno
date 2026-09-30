from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fakes.llm import Reply
from juno.domain.values import (
    Direction,
    ResourceSnapshot,
    Submission,
    ToolCall,
    ToolResult,
    Usage,
)

tokens = st.integers(min_value=0, max_value=10**9)
usages = st.builds(
    Usage,
    input_tokens=tokens,
    output_tokens=tokens,
    cache_creation_input_tokens=tokens,
    cache_read_input_tokens=tokens,
)
json_scalars = st.none() | st.booleans() | st.integers() | st.text()


class TestUsage:
    @given(usages, usages)
    def test_addition_is_commutative(self, left: Usage, right: Usage) -> None:
        assert left + right == right + left

    @given(usages)
    def test_empty_usage_is_the_identity(self, usage: Usage) -> None:
        assert usage + Usage() == usage

    @given(usages)
    def test_survives_a_json_round_trip(self, usage: Usage) -> None:
        assert Usage.model_validate_json(usage.model_dump_json()) == usage


class TestToolCalls:
    def test_reply_exposes_its_tool_calls(self) -> None:
        reply = Reply.tools(("bash", {"command": "ls"}))

        [call] = reply.tool_calls()

        assert (call.name, call.args) == ("bash", {"command": "ls"})
        assert call.id == reply.content[0]["id"]

    def test_text_reply_has_no_tool_calls(self) -> None:
        assert Reply.text("done").tool_calls() == []

    @given(st.text(min_size=1), st.text(min_size=1), st.dictionaries(st.text(), json_scalars))
    def test_tool_call_survives_a_json_round_trip(
        self, call_id: str, name: str, args: dict[str, object]
    ) -> None:
        call = ToolCall(id=call_id, name=name, args=args)

        assert ToolCall.model_validate_json(call.model_dump_json()) == call

    def test_result_renders_as_a_tool_result_block(self) -> None:
        result = ToolResult(call_id="toolu_1", content="boom", is_error=True)

        assert result.to_block() == {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "boom",
            "is_error": True,
        }


class TestSubmission:
    @pytest.mark.parametrize(
        ("direction", "better", "worse"),
        [(Direction.MINIMIZE, 0.2, 0.5), (Direction.MAXIMIZE, 0.9, 0.7)],
        ids=["minimize", "maximize"],
    )
    def test_beats_follows_the_metric_direction(
        self, direction: Direction, better: float, worse: float
    ) -> None:
        def make(score: float) -> Submission:
            return Submission(
                path=Path("submission.csv"),
                sha256="0" * 64,
                val_score=score,
                direction=direction,
                created_at=0.0,
            )

        assert make(better).beats(make(worse))
        assert not make(worse).beats(make(better))


class TestResourceSnapshot:
    def test_ram_fraction_handles_unknown_totals(self) -> None:
        snapshot = ResourceSnapshot(cpu_percent=0.0, ram_used_bytes=0, ram_total_bytes=0)

        assert snapshot.ram_used_fraction == 0.0
