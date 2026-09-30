import pytest

from fakes.llm import Reply
from juno.domain.session import InvalidHistory, InvalidTransition, Session, State, Trigger
from juno.domain.values import Role, ToolResult, Usage

TABLE = [
    pytest.param(state, trigger, expected, id=f"{state}-{trigger}")
    for (state, trigger), expected in Session.TRANSITIONS.items()
]
LIVE_STATES = [state for state in State if state not in Session.TERMINAL]


class TestLifecycle:
    @pytest.mark.parametrize(("state", "trigger", "expected"), TABLE)
    def test_follows_the_transition_table(
        self, session: Session, state: State, trigger: Trigger, expected: State
    ) -> None:
        session.state = state

        assert session.fire(trigger) is expected
        assert session.state is expected

    def test_rejects_a_trigger_the_state_does_not_accept(self, session: Session) -> None:
        with pytest.raises(InvalidTransition):
            session.fire(Trigger.CONTINUE)

        assert session.state is State.STARTING

    @pytest.mark.parametrize("state", LIVE_STATES)
    def test_fatal_fails_any_live_state(self, session: Session, state: State) -> None:
        session.state = state

        assert session.fire(Trigger.FATAL) is State.FAILED

    @pytest.mark.parametrize("state", sorted(Session.TERMINAL))
    @pytest.mark.parametrize("trigger", list(Trigger))
    def test_terminal_states_accept_nothing(
        self, session: Session, state: State, trigger: Trigger
    ) -> None:
        session.state = state

        with pytest.raises(InvalidTransition):
            session.fire(trigger)

    def test_remaining_time_never_goes_negative(self, session: Session) -> None:
        assert session.remaining(session.deadline - 60) == 60
        assert session.remaining(session.deadline + 60) == 0


class TestHistory:
    def test_history_starts_with_the_user(self, session: Session) -> None:
        with pytest.raises(InvalidHistory):
            session.append_assistant(Reply.text("hello"))

    def test_roles_alternate(self, session: Session) -> None:
        session.append_user("Solve the task.")

        with pytest.raises(InvalidHistory):
            session.append_user("Again.")

    def test_assistant_turns_add_up_usage(self, session: Session) -> None:
        session.append_user("Solve the task.")
        session.append_assistant(
            Reply.tools(("bash", {"command": "ls"}), usage=Usage(input_tokens=10))
        )
        session.append_tool_results(
            [ToolResult(call_id=session.pending_tool_calls()[0].id, content="ok")]
        )
        session.append_assistant(Reply.text("done", usage=Usage(input_tokens=5, output_tokens=2)))

        assert session.usage == Usage(input_tokens=15, output_tokens=2)

    def test_pending_tool_calls_come_from_the_last_assistant_turn(self, session: Session) -> None:
        session.append_user("Solve the task.")
        session.append_assistant(Reply.tools(("bash", {"command": "ls"}), ("system", {})))

        assert [call.name for call in session.pending_tool_calls()] == ["bash", "system"]

    def test_tool_results_close_the_pending_calls(self, session: Session) -> None:
        session.append_user("Solve the task.")
        session.append_assistant(Reply.tools(("bash", {"command": "ls"})))
        call = session.pending_tool_calls()[0]

        message = session.append_tool_results([ToolResult(call_id=call.id, content="data/")])

        assert message.role is Role.USER
        assert message.content == [
            {"type": "tool_result", "tool_use_id": call.id, "content": "data/", "is_error": False}
        ]
        assert session.pending_tool_calls() == []

    @pytest.mark.parametrize("call_ids", [[], ["toolu_other"]], ids=["missing", "unknown"])
    def test_tool_results_must_match_the_calls(self, session: Session, call_ids: list[str]) -> None:
        session.append_user("Solve the task.")
        session.append_assistant(Reply.tools(("bash", {"command": "ls"})))

        with pytest.raises(InvalidHistory):
            session.append_tool_results([ToolResult(call_id=i, content="x") for i in call_ids])

    def test_tool_results_need_pending_calls(self, session: Session) -> None:
        session.append_user("Solve the task.")
        session.append_assistant(Reply.text("done"))

        with pytest.raises(InvalidHistory):
            session.append_tool_results([ToolResult(call_id="toolu_x", content="x")])
