from enum import Enum
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


class Phase(str, Enum):
    IDLE = "idle"
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    DECISION = "decision"
    FINALIZE = "finalize"


@dataclass
class AgentState:
    tool_calls: Optional[List[Dict]] = None
    full_reply: str = ""
    is_complete: bool = False
    last_reflection_decision: Dict = field(default_factory=dict)
    executed_functions: List[Dict] = field(default_factory=list)
    iteration_history: List[Dict] = field(default_factory=list)
    phase: Phase = Phase.IDLE
    reflection_count: int = 0
    planned_tasks: List[str] = field(default_factory=list)
    current_task_index: int = 0
    last_progress_snapshot: Optional[Dict[str, Any]] = None
    last_reflection_iteration: int = 0
    last_replan_iteration: int = 0 

    @property
    def current_task(self) -> Optional[str]:
        if 0 <= self.current_task_index < len(self.planned_tasks):
            return self.planned_tasks[self.current_task_index]
        return None
