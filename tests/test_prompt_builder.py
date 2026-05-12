"""Unit tests for neuralcore.utils.prompt_builder -- PromptBuilder."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _setup():
    import neuralcore.utils.config as cfg_mod

    cfg_mod.loader = None
    from neuralcore.utils.text_tokenizer import TextTokenizer

    TextTokenizer._instance = None
    TextTokenizer._initialized = False
    from neuralcore.utils.config import ConfigLoader

    ConfigLoader(
        cli_path=str(PROJECT_ROOT / "data" / "test_config.yaml"),
        app_root=PROJECT_ROOT,
    )


class TestPromptBuilder:
    def setup_method(self):
        _setup()
        from neuralcore.utils.prompt_builder import PromptBuilder

        self.pb = PromptBuilder

    def test_final_answer_marker(self):
        assert self.pb.FINAL_ANSWER_MARKER == "[FINAL_ANSWER_COMPLETE]"

    def test_classify_intent(self):
        result = self.pb.classify_intent("Hello, how are you?")
        assert "CASUAL" in result
        assert "TASK" in result
        assert "exactly one word" in result.lower()

    def test_is_multi_step_task(self):
        result = self.pb.is_multi_step_task("list files in this dir")
        assert "SIMPLE" in result
        assert "COMPLEX" in result

    def test_task_decomposition(self):
        result = self.pb.task_decomposition("build a test suite")
        assert "steps" in result
        assert "JSON" in result

    def test_casual_chat_system_prompt(self):
        result = self.pb.casual_chat_system_prompt()
        assert "friendly" in result.lower()

    def test_default_agent_system_prompt(self):
        result = self.pb.default_agent_system_prompt()
        assert "helpful" in result.lower()

    def test_objective_reminder(self):
        body = "Current goal: Test"
        result = self.pb.objective_reminder(body)
        assert "OBJECTIVE REMINDER" in result
        assert "Test" in result

    def test_final_synthesis(self):
        result = self.pb.final_synthesis("original query here")
        assert "original query here" in result

    def test_inject_final_answer_instruction(self):
        result = self.pb.inject_final_answer_instruction("Base prompt")
        assert self.pb.FINAL_ANSWER_MARKER in result
        assert "Base prompt" in result

    def test_step_validation_prompt(self):
        from neuralcore.tasks.task import Task

        task = Task(description="Read file", expected_outcome="File contents returned")
        result = self.pb.step_validation_prompt(task, "file data here", 3, 0)
        assert "Read file" in result
        assert "YES" in result or "NO" in result

    def test_agentic_action_system_prefix(self):
        result = self.pb.agentic_action_system_prefix()
        assert "ACTION-ORIENTED" in result

    def test_loaded_tools_summary_empty(self):
        result = self.pb.loaded_tools_summary([])
        assert "No tools" in result

    def test_loaded_tools_summary_with_tools(self):
        result = self.pb.loaded_tools_summary(["echo_tool", "read_file"])
        assert "echo_tool" in result
        assert "read_file" in result

    def test_contamination_forbidden_phrases(self):
        phrases = self.pb.contamination_forbidden_phrases()
        assert isinstance(phrases, list)
        assert len(phrases) > 0

    def test_sub_task_execution(self):
        result = self.pb.sub_task_execution(
            original_query="build tests",
            task_desc="Write unit test",
            current_index=0,
            total_tasks=3,
            completed_context="None",
            used_tools_str="none",
            remaining_context="Steps 2, 3",
            marker="[DONE]",
            loop_count=1,
            expected_outcome="Test file created",
        )
        assert "Write unit test" in result
        assert "[DONE]" in result
