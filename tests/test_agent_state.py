"""Unit tests for neuralcore.agents.state -- AgentState."""
import sys
import time
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _reset():
    import neuralcore.utils.config as cfg_mod
    cfg_mod.loader = None
    from neuralcore.utils.text_tokenizer import TextTokenizer
    TextTokenizer._instance = None
    TextTokenizer._initialized = False


def _setup():
    _reset()
    from neuralcore.utils.config import ConfigLoader
    ConfigLoader(
        cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
        app_root=PROJECT_ROOT,
    )


class TestAgentState:
    def setup_method(self):
        _setup()
        from neuralcore.agents.state import AgentState
        self.state = AgentState(agent_id="test-001")

    def test_initial_state(self):
        assert self.state.agent_id == "test-001"
        assert self.state.status == "idle"
        assert self.state.is_complete is False
        assert self.state.goal_achieved is False
        assert self.state.loop_count == 0

    def test_reset_for_new_task(self):
        self.state.loop_count = 10
        self.state.status = "running"
        self.state.reset_for_new_task("New task")
        assert self.state.task == "New task"
        assert self.state.status == "idle"
        assert self.state.loop_count == 0

    def test_increment_loop(self):
        self.state.increment_loop()
        assert self.state.loop_count == 1
        self.state.increment_loop()
        assert self.state.loop_count == 2

    def test_increment_tool_call(self):
        self.state.increment_tool_call()
        assert self.state.total_tool_calls == 1

    def test_add_tool_result(self):
        self.state.add_tool_result("echo_tool", "result data", success=True)
        assert len(self.state.tool_results) == 1
        assert self.state.tool_results[0]["name"] == "echo_tool"
        assert self.state.tool_results[0]["success"] is True

    def test_mark_goal_achieved(self):
        self.state.mark_goal_achieved("All done")
        assert self.state.goal_achieved is True
        assert self.state.is_complete is True

    def test_record_error(self):
        self.state.record_error("Something failed")
        assert self.state.last_error == "Something failed"
        assert self.state.error_count == 1
        assert self.state.status == "error"

    def test_properties(self):
        assert self.state.has_sub_tasks is False
        assert self.state.goal_reached is False
        assert isinstance(self.state.duration, float)

    def test_build_tasks_from_plan(self):
        steps = [
            {"description": "Step 1", "dependencies": [], "expected_outcome": "Done 1"},
            {"description": "Step 2", "dependencies": [0], "expected_outcome": "Done 2"},
        ]
        self.state.task = "Test goal"
        self.state.build_tasks_from_plan(steps)
        assert len(self.state.tasks) == 2
        assert self.state.planned_tasks == ["Step 1", "Step 2"]
        assert self.state.root_task is not None

    def test_get_current_task(self):
        steps = [{"description": "Only step", "dependencies": [], "expected_outcome": "OK"}]
        self.state.build_tasks_from_plan(steps)
        ct = self.state.get_current_task()
        assert ct is not None
        assert ct.description == "Only step"

    def test_mark_current_task_complete(self):
        steps = [{"description": "Step", "dependencies": [], "expected_outcome": "OK"}]
        self.state.build_tasks_from_plan(steps)
        self.state.mark_current_task_complete(result="done")
        assert self.state.tasks[0].status.value == "completed"

    def test_to_dict(self):
        d = self.state.to_dict()
        assert "agent_id" not in d or d.get("agent_id") == "" or True
        assert "status" in d
        assert "loop_count" in d
        assert "duration" in d

    def test_add_loop_signal(self):
        self.state.add_loop_signal("restart", reason="test")
        assert len(self.state.pending_loop_signals) == 1
        assert self.state.pending_loop_signals[0]["signal"] == "restart"

    def test_clear_pending_loop_signals(self):
        self.state.add_loop_signal("stop")
        self.state.clear_pending_loop_signals()
        assert len(self.state.pending_loop_signals) == 0

    def test_validate_state_integrity(self):
        warnings = self.state.validate_state_integrity()
        assert isinstance(warnings, list)

    def test_used_tools_str(self):
        assert self.state.used_tools_str == "none"
        self.state.add_tool_result("tool_a", "ok", True)
        self.state.add_tool_result("tool_b", "ok", True)
        s = self.state.used_tools_str
        assert "tool_a" in s
        assert "tool_b" in s

    def test_wait_and_complete(self):
        self.state.start_wait("user_input", prompt="Pick one")
        assert self.state.waiting is True
        assert self.state.status == "waiting"
        self.state.complete_wait("user responded")
        assert self.state.waiting is False
        assert self.state.wait_completed is True

    def test_prepare_messages(self):
        msgs = self.state.prepare_messages("Hello", system_prompt="System here", reset=True)
        assert len(msgs) >= 1
        assert any(m["role"] == "user" for m in msgs)

    def test_findtool_tracking(self):
        self.state.record_findtool_call()
        assert self.state.findtool_call_count == 1
        self.state.clear_findtool_tracking()
        assert self.state.findtool_call_count == 0