import pytest

from fakes.llm import Reply
from juno.domain.session import (
    InvalidHistory,
    InvalidTransition,
    Limit,
    Session,
    State,
    Trigger,
)
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


class TestBudget:
    def test_elapsed_leaves_out_paused_time(self, session: Session) -> None:
        session.paused_s = 600

        assert session.elapsed(session.started_at + 1_800) == 1_200

    def test_the_first_limit_reached_ends_the_budget(self, session: Session) -> None:
        capped = session.model_copy(update={"cost_limit": 0.5})

        assert capped.limit_reached(session.started_at) is None
        capped.usage = Usage(cost_usd=0.5)
        assert capped.limit_reached(session.started_at) is Limit.COST
        assert capped.limit_reached(session.deadline) is Limit.TIME
        assert session.limit_reached(session.started_at) is None

    def test_a_finished_session_stops_the_clock_at_its_last_save(self, session: Session) -> None:
        session.fire(Trigger.START)
        session.fire(Trigger.FATAL)
        session.touch(session.started_at + 900)

        assert session.elapsed(session.started_at + 86_400) == 900

    def test_downtime_moves_the_deadline(self, session: Session) -> None:
        deadline = session.deadline
        session.touch(session.started_at + 100)

        downtime = session.pause_for_downtime(session.started_at + 400)

        assert downtime == 300
        assert (session.deadline, session.paused_s) == (deadline + 300, 300)
        assert session.last_active_at == session.started_at + 400

    @pytest.mark.parametrize("state", [State.DONE, State.FAILED])
    def test_finished_sessions_are_not_paused(self, session: Session, state: State) -> None:
        session.touch(session.started_at)
        session.state = state

        assert session.pause_for_downtime(session.started_at + 900) == 0
        assert session.paused_s == 0

    def test_a_session_never_saved_has_no_downtime(self, session: Session) -> None:
        assert session.pause_for_downtime(session.started_at + 900) == 0


class TestSetup:
    def test_changes_are_reported_as_dotted_keys(self, session: Session) -> None:
        session.setup = {
            "model": {"name": "a", "effort": "high"},
            "flags": {"tools": {"jobs": True}},
        }

        changes = session.setup_changes(
            {"model": {"name": "b", "effort": "high"}, "flags": {"tools": {"jobs": True, "new": 1}}}
        )

        assert changes == ["flags.tools.new", "model.name"]

    def test_the_same_setup_has_no_changes(self, session: Session) -> None:
        session.setup = {"model": {"name": "a"}}

        assert session.setup_changes({"model": {"name": "a"}}) == []


class TestBusyDowntime:
    def test_only_idle_downtime_is_paused(self, session: Session) -> None:
        deadline = session.deadline
        session.touch(session.started_at)

        paused = session.pause_for_downtime(session.started_at + 300, busy_s=100)

        assert paused == 200
        assert (session.deadline, session.paused_s) == (deadline + 200, 200)

    def test_downtime_spent_by_jobs_is_not_paused(self, session: Session) -> None:
        session.touch(session.started_at)

        assert session.pause_for_downtime(session.started_at + 300, busy_s=900) == 0
        assert session.paused_s == 0
