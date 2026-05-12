"""Unit tests for neuralcore.workflows.registry -- Workflow."""

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


class TestWorkflowRegistry:
    def setup_method(self):
        _setup()
        from neuralcore.workflows.registry import Workflow

        self.wf = Workflow()

    def test_step_registration(self):
        @self.wf.step("test_wf", name="step_one", description="First step")
        def step_one(iteration, state):
            pass

        assert "step_one" in self.wf.handlers
        assert "test_wf" in self.wf.workflows
        assert "step_one" in self.wf.workflows["test_wf"]["steps"]

    def test_condition_registration(self):
        @self.wf.condition("is_done", description="Check if done")
        def is_done(state):
            return True

        assert "is_done" in self.wf.conditions

    def test_evaluate_condition(self):
        @self.wf.condition("always_true")
        def always_true(state):
            return True

        result = self.wf.evaluate_condition("always_true", None)
        assert result is True

    def test_evaluate_missing_condition(self):
        result = self.wf.evaluate_condition("nonexistent", None)
        assert result is False

    def test_loop_registration(self):
        @self.wf.loop("test_loop", max_iterations=5, break_condition="is_done")
        async def test_loop(agent, state):
            yield ("done", {})

        assert "test_loop" in self.wf.loops
        meta = self.wf.loops["test_loop"]
        assert meta["max_iterations"] == 5
        assert meta["break_condition"] == "is_done"

    def test_loop_unlimited(self):
        @self.wf.loop("unlimited_loop", max_iterations=None)
        async def unlimited_loop(agent, state):
            yield ("done", {})

        assert self.wf.loops["unlimited_loop"]["max_iterations"] is None

    def test_list_workflows(self):
        @self.wf.step("wf_a", name="s1")
        def s1():
            pass

        @self.wf.step("wf_b", name="s2")
        def s2():
            pass

        result = self.wf.list_workflows()
        names = [w["name"] for w in result]
        assert "wf_a" in names
        assert "wf_b" in names

    def test_list_steps(self):
        @self.wf.step("wf_x", name="step_x", description="X step")
        def step_x():
            pass

        result = self.wf.list_steps()
        names = [s["name"] for s in result]
        assert "step_x" in names

    def test_list_loops(self):
        @self.wf.loop("loop_y", max_iterations=10)
        async def loop_y(agent, state):
            yield ("done", {})

        result = self.wf.list_loops()
        names = [l["name"] for l in result]
        assert "loop_y" in names

    def test_list_conditions(self):
        @self.wf.condition("cond_z")
        def cond_z():
            return False

        result = self.wf.list_conditions()
        assert "cond_z" in result

    def test_list_all(self):
        @self.wf.step("all_wf", name="all_step")
        def all_step():
            pass

        @self.wf.loop("all_loop", max_iterations=1)
        async def all_loop(agent, state):
            yield ("done", {})

        result = self.wf.list_all()
        assert "workflows" in result
        assert "steps" in result
        assert "loops" in result
        assert "totals" in result

    def test_search(self):
        @self.wf.step(
            "search_wf", name="file_reader", description="reads files from disk"
        )
        def file_reader():
            pass

        results = self.wf.search("file read")
        assert len(results) > 0
        assert results[0]["name"] == "file_reader"

    def test_search_empty_query(self):
        assert self.wf.search("") == []

    def test_get_step_metadata(self):
        @self.wf.step("meta_wf", name="meta_step", description="Meta test")
        def meta_step():
            pass

        meta = self.wf.get_step_metadata("meta_step")
        assert meta is not None
        assert meta["description"] == "Meta test"

    def test_normalize_list(self):
        assert self.wf._normalize_list(None) == []
        assert self.wf._normalize_list("single") == ["single"]
        assert self.wf._normalize_list(["a", "b"]) == ["a", "b"]
        assert self.wf._normalize_list([""]) == []
