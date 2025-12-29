"""
SSE Event Types for Deep Research Streaming

Defines event types and data structures for Server-Sent Events
used in the streaming deep research endpoint.
"""

# Standard library
import json
from enum import Enum
from typing import Any, Dict, List, Optional

# Third-party
from pydantic import BaseModel


class SSEEventType(str, Enum):
    """SSE event types for deep research streaming."""
    
    # Planning phase
    PLAN_CONFIRMED = "plan_confirmed"
    
    # Task execution
    TASK_START = "task_start"
    TASK_DONE = "task_done"
    
    # Drill-down phase
    DRILLDOWN_START = "drilldown_start"
    DRILLDOWN_TASK_START = "drilldown_task_start"
    DRILLDOWN_TASK_DONE = "drilldown_task_done"
    
    # Synthesis phase
    SYNTHESIS_START = "synthesis_start"
    
    # Completion
    COMPLETE = "complete"
    ERROR = "error"


class PlanConfirmedData(BaseModel):
    """Data for plan_confirmed event."""
    task_count: int
    enabled_count: int


class TaskStartData(BaseModel):
    """Data for task_start event."""
    id: int
    question: str
    task_type: str
    iteration: int = 0


class TaskDoneData(BaseModel):
    """Data for task_done event."""
    id: int
    question: str
    answer: str
    sources: List[str]
    iteration: int = 0


class DrilldownStartData(BaseModel):
    """Data for drilldown_start event."""
    iteration: int
    new_task_count: int


class SynthesisStartData(BaseModel):
    """Data for synthesis_start event."""
    total_tasks: int


class ErrorData(BaseModel):
    """Data for error event."""
    message: str
    task_id: Optional[int] = None


def format_sse_event(event_type: SSEEventType, data: Any) -> Dict[str, str]:
    """
    Formats an SSE event for streaming.
    
    Args:
        event_type: The type of SSE event.
        data: The event data (Pydantic model or dict).
        
    Returns:
        Dict with 'event' and 'data' keys for SSE.
    """
    if hasattr(data, "model_dump"):
        json_data = json.dumps(data.model_dump(), ensure_ascii=False)
    elif isinstance(data, dict):
        json_data = json.dumps(data, ensure_ascii=False)
    else:
        json_data = json.dumps({"value": data}, ensure_ascii=False)
    
    return {
        "event": event_type.value,
        "data": json_data,
    }
