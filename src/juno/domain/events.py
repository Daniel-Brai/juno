from juno.domain.session import State
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
    deadline: float


class ModelCalled(Event):
    message_index: int
    usage: Usage
    duration_s: float


class ToolStarted(Event):
    call: ToolCall


class ToolFinished(Event):
    result: ToolResult


class JobFinished(Event):
    job: Job


class SubmissionRegistered(Event):
    submission: Submission


class CheckpointTaken(Event):
    checkpoint: Checkpoint


class TurnContinued(Event):
    remaining_s: float


class Compacted(Event):
    message_index: int


class ResourcePressure(Event):
    snapshot: ResourceSnapshot
    reason: str


class SessionResumed(Event):
    downtime_s: float
    jobs_busy_s: float
    paused_s: float
    deadline: float


class SessionEnded(Event):
    state: State
    usage: Usage
    handler_failures: int = 0
