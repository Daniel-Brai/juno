from juno.domain.session import Limit, State
from juno.domain.values import (
    Checkpoint,
    Job,
    ResourceSnapshot,
    Submission,
    ToolCall,
    ToolResult,
    Usage,
    Value,
)


class Event(Value):
    session_id: str
    at: float


class SessionStarted(Event):
    task_id: str
    deadline: float | None


class ModelCalled(Event):
    message_index: int
    usage: Usage
    duration_s: float


class ToolStarted(Event):
    call: ToolCall


class ApprovalDecided(Event):
    call: ToolCall
    allowed: bool
    asked: bool
    reason: str


class ToolFinished(Event):
    result: ToolResult


class JobFinished(Event):
    job: Job


class SubmissionRegistered(Event):
    submission: Submission


class CheckpointTaken(Event):
    checkpoint: Checkpoint


class TurnContinued(Event):
    remaining_s: float | None


class Compacted(Event):
    message_index: int


class CacheMissed(Event):
    message_index: int
    misses: int


class ResourcePressure(Event):
    snapshot: ResourceSnapshot
    reason: str


class SessionResumed(Event):
    downtime_s: float
    jobs_busy_s: float
    paused_s: float
    deadline: float | None


class BudgetSpent(Event):
    limit: Limit
    cost_usd: float


class SessionEnded(Event):
    state: State
    failure: str | None = None
    usage: Usage
    handler_failures: int = 0
