from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import uuid

from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


@dataclass
class Task:
    """
    Generic, reusable Task for NeuralCore.
    Supports flat lists (for easy AgentState compatibility) + hierarchical subtasks.
    Completely abstract — no domain-specific logic.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    parent_task_id: Optional[str] = None

    status: str = "pending"  # pending | in_progress | completed | failed | skipped
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[Any] = None
    expected_outcome: str = ""

    suggested_tool: str = ""
    used_tool: Optional[str] = None  # ← FIXED: now Optional

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    subtasks: List["Task"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _dependency_set: Set[str] = field(init=False, default_factory=set)

    def __post_init__(self):
        self._dependency_set = set(self.dependencies)

    def start(self, agent: Optional[Any] = None) -> None:
        """Mark task as in progress."""
        if self.status not in ("pending", "failed"):
            logger.warning(
                f"Task {self.task_id[:8]} cannot start from status '{self.status}'"
            )
            return

        self.status = "in_progress"
        self.start_time = datetime.now()
        if agent is not None:
            self.assigned_agent = agent
        logger.info(f"[TASK START] {self.task_id[:8]} | {self.description}")

    def complete(self, result: Any = None, error: Optional[str] = None) -> None:
        """Mark task completed or failed."""
        self.end_time = datetime.now()
        if error:
            self.status = "failed"
            self.error = error
            logger.error(f"[TASK FAILED] {self.task_id[:8]} | {error}")
        else:
            self.status = "completed"
            self.result = result
            logger.info(
                f"[TASK COMPLETE] {self.task_id[:8]} | outcome met: {bool(self.expected_outcome)}"
            )

    def is_ready(self, completed_ids: Set[str]) -> bool:
        """Check if all dependencies are completed."""
        if not self.dependencies:
            return True
        return self._dependency_set.issubset(completed_ids)

    def add_subtask(self, subtask: "Task") -> None:
        """Hierarchical support."""
        subtask.parent_task_id = self.task_id
        self.subtasks.append(subtask)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable snapshot."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "parent_task_id": self.parent_task_id,
            "status": self.status,
            "dependencies": self.dependencies,
            "assigned_agent": getattr(
                self.assigned_agent, "agent_id", str(self.assigned_agent)
            )
            if self.assigned_agent
            else None,
            "expected_outcome": self.expected_outcome,
            "suggested_tool": self.suggested_tool,
            "used_tool": self.used_tool,  # ← now safely accepts None
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": str(self.result)[:1000] if self.result is not None else None,
            "error": self.error,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        duration = (
            f" ({(self.end_time - self.start_time).total_seconds():.1f}s)"
            if self.start_time and self.end_time
            else ""
        )
        tool_info = f" | suggested={self.suggested_tool}" if self.suggested_tool else ""
        if self.used_tool:  # ← works perfectly with None or ""
            tool_info += f" | used={self.used_tool}"
        return f"[{self.status.upper()}] {self.task_id[:8]}… | {self.description[:80]}{tool_info}{duration}"
