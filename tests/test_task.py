"""Unit tests for neuralcore.tasks.task -- Task and TaskStatus."""

import sys
from pathlib import Path
from neuralcore.tasks.task import Task, TaskStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestTaskStatus:
    def test_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.SKIPPED == "skipped"


class TestTask:
    def test_default_creation(self):
        t = Task(description="Test task")
        assert t.status == TaskStatus.PENDING
        assert t.description == "Test task"
        assert t.task_id  # UUID generated
        assert t.dependencies == []
        assert t.subtasks == []

    def test_start(self):
        t = Task(description="Start test")
        t.start()
        assert t.status == TaskStatus.IN_PROGRESS
        assert t.start_time is not None

    def test_complete_success(self):
        t = Task(description="Complete test")
        t.start()
        t.complete(result="done")
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "done"
        assert t.end_time is not None
        assert t.error is None

    def test_complete_with_error(self):
        t = Task(description="Fail test")
        t.start()
        t.complete(error="Something went wrong")
        assert t.status == TaskStatus.FAILED
        assert t.error == "Something went wrong"

    def test_is_ready_no_dependencies(self):
        t = Task(description="No deps")
        assert t.is_ready(set()) is True

    def test_is_ready_with_dependencies(self):
        t1 = Task(description="Parent")
        t2 = Task(description="Child", dependencies=[t1.task_id])
        assert t2.is_ready(set()) is False
        assert t2.is_ready({t1.task_id}) is True

    def test_add_subtask(self):
        parent = Task(description="Parent")
        child = Task(description="Child")
        parent.add_subtask(child)
        assert len(parent.subtasks) == 1
        assert child.parent_task_id == parent.task_id

    def test_to_dict(self):
        t = Task(description="Dict test", expected_outcome="success")
        d = t.to_dict()
        assert d["description"] == "Dict test"
        assert d["expected_outcome"] == "success"
        assert d["status"] == "pending"
        assert "task_id" in d

    def test_summary(self):
        t = Task(description="Summary test", suggested_tool="echo_tool")
        s = t.summary()
        assert "Summary test" in s
        assert "echo_tool" in s

    def test_start_from_invalid_status(self):
        t = Task(description="Test")
        t.start()
        t.complete(result="ok")
        # Should not change from completed
        t.start()
        assert t.status == TaskStatus.COMPLETED
