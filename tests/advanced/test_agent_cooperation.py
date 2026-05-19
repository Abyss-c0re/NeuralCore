"""
Advanced Integration Tests for Multi-Agent Cooperation in NeuralCore.

This test suite validates the inter-agent cooperation system:
1. Task completion events and async waiting
2. Agent-to-agent task delegation (request_agent / await_task_completion)
3. Context draining from sub-agents (drain_agent_context / inject_drained_context)
4. Parallel task dispatch with dependency resolution (dispatch_parallel)
5. Full tool output extraction across agent boundaries
6. PromptBuilder cooperative context helpers

Uses the same MockLLMServer pattern as the existing advanced tests.
All agent communication flows through the framework's real message queue.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from neuralcore.utils.mock_llm_server import MockLLMServer

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).parent
NEURALCORE_ROOT = TESTS_DIR.parent.parent
TOOLS_DIR = TESTS_DIR / "tools"
CONFIG_PATH = TESTS_DIR / "test_config.yaml"
TOKENIZER_PATH = NEURALCORE_ROOT / "data" / "tokenizer" / "tokenizer.json"

sys.path.insert(0, str(NEURALCORE_ROOT / "src"))
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))


def _reset_singletons():
    import neuralcore.utils.config as config_mod
    import neuralcore.clients.factory as cfactory_mod
    import neuralcore.actions.registry as reg_mod

    config_mod.loader = None
    cfactory_mod._factory = None
    reg_mod.registry.sets.clear()
    reg_mod.registry.all_actions.clear()
    reg_mod.registry._index.clear()

    # Force re-import of tool modules so @tool decorators re-fire
    for mod_name in list(sys.modules.keys()):
        if "context_tools" in mod_name:
            del sys.modules[mod_name]


def _build_config_dict(mock_base_url: str) -> Dict[str, Any]:
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    cfg: Dict[str, Any] = raw_cfg if isinstance(raw_cfg, dict) else {}

    clients = cfg.setdefault("clients", {})
    main = clients.setdefault("main", {})
    main["base_url"] = mock_base_url
    main["tokenizer"] = str(TOKENIZER_PATH)

    tools_section = cfg.setdefault("tools", {})
    context_test_tools = tools_section.setdefault("ContextTestTools", {})
    context_test_tools["folder"] = str(TOOLS_DIR)

    return cfg


def _create_agent(factory, loader, agent_id: str, name: str):
    """Create an agent with overridden id and name."""
    base_config = loader.get_agent_config("test_agent")
    override = dict(base_config)
    override["id"] = agent_id
    override["name"] = name
    return factory.create_agent(
        agent_id=agent_id,
        config=override,
        app_root=NEURALCORE_ROOT,
    )


# =========================================================================
# Phase 1: Task completion events (unit-level, no agents)
# =========================================================================
@pytest.mark.asyncio
async def test_task_completion_event():
    """Task.get_completion_event() + Task.complete() correctly signals waiters."""
    from neuralcore.tasks.task import Task, TaskStatus

    task = Task(description="Test completion event", expected_outcome="Event is set")
    event = task.get_completion_event()
    assert not event.is_set()

    # Complete the task in a background coroutine
    async def _complete_after_delay():
        await asyncio.sleep(0.05)
        task.complete(result="done")

    asyncio.create_task(_complete_after_delay())

    # Wait for completion
    await asyncio.wait_for(event.wait(), timeout=2.0)
    assert event.is_set()
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "done"


# =========================================================================
# Phase 2: Task on_complete callbacks
# =========================================================================
@pytest.mark.asyncio
async def test_task_on_complete_callback():
    """on_complete callbacks fire when the task completes."""
    from neuralcore.tasks.task import Task

    callback_received = []

    task = Task(description="Callback test")
    task.on_complete(lambda t: callback_received.append(t.task_id))

    task.complete(result="ok")
    assert len(callback_received) == 1
    assert callback_received[0] == task.task_id


# =========================================================================
# Phase 3: Task completion event with error
# =========================================================================
@pytest.mark.asyncio
async def test_task_completion_event_on_failure():
    """Completion event is set even on task failure."""
    from neuralcore.tasks.task import Task, TaskStatus

    task = Task(description="Fail test")
    event = task.get_completion_event()
    task.complete(error="Something broke")

    assert event.is_set()
    assert task.status == TaskStatus.FAILED
    assert task.error == "Something broke"


# =========================================================================
# Phase 4: PromptBuilder cooperative helpers
# =========================================================================
def test_delegated_task_prompt():
    """PromptBuilder.delegated_task_prompt produces valid delegation prompt."""
    _reset_singletons()

    from neuralcore.utils.config import ConfigLoader

    ConfigLoader(
        cli_path=str(NEURALCORE_ROOT / "data" / "test_config.yaml"),
        app_root=NEURALCORE_ROOT,
    )
    from neuralcore.utils.prompt_builder import PromptBuilder

    prompt = PromptBuilder.delegated_task_prompt(
        task_description="Analyze the weather data for Berlin",
        expected_outcome="Weather analysis report produced",
        requesting_agent_id="agent_alpha",
    )
    assert "Analyze the weather data" in prompt
    assert "agent_alpha" in prompt
    assert "EXPECTED OUTCOME" in prompt
    assert PromptBuilder.FINAL_ANSWER_MARKER in prompt


def test_cooperation_context_section():
    """PromptBuilder.cooperation_context_section formats delegation status."""
    _reset_singletons()

    from neuralcore.utils.config import ConfigLoader

    ConfigLoader(
        cli_path=str(NEURALCORE_ROOT / "data" / "test_config.yaml"),
        app_root=NEURALCORE_ROOT,
    )
    from neuralcore.utils.prompt_builder import PromptBuilder

    pending = [
        {"task_id": "abc12345", "description": "Fetch data", "agent_name": "AgentB"}
    ]
    completed = [
        {
            "task_id": "def67890",
            "description": "Process data",
            "status": "completed",
            "result": "Data processed successfully",
        }
    ]
    section = PromptBuilder.cooperation_context_section(pending, completed)
    assert "AGENT COOPERATION STATUS" in section
    assert "PENDING" in section
    assert "COMPLETED" in section
    assert "AgentB" in section
    assert "Data processed" in section


def test_delegated_result_summary():
    """PromptBuilder.delegated_result_summary formats result block."""
    _reset_singletons()

    from neuralcore.utils.config import ConfigLoader

    ConfigLoader(
        cli_path=str(NEURALCORE_ROOT / "data" / "test_config.yaml"),
        app_root=NEURALCORE_ROOT,
    )
    from neuralcore.utils.prompt_builder import PromptBuilder

    summary = PromptBuilder.delegated_result_summary(
        task_description="Analyze weather",
        agent_name="WeatherAgent",
        result_text="Temperature is 22C in Berlin",
        tool_results_count=3,
    )
    assert "DELEGATED TASK RESULT" in summary
    assert "WeatherAgent" in summary
    assert "22C" in summary
    assert "3" in summary


# =========================================================================
# Phase 5: Full multi-agent cooperation with mock LLM
# =========================================================================
@pytest.mark.asyncio
async def test_multi_agent_cooperation():
    """
    Full end-to-end multi-agent cooperation test:
    - Creates two agents (alpha and beta) connected to a mock LLM
    - Alpha delegates a task to Beta via request_agent
    - Beta executes the task (with tool calls via mock LLM)
    - Alpha awaits completion and drains Beta's context
    - Validates the full tool output is transferred
    """
    _reset_singletons()

    server = MockLLMServer(port=9933)
    await server.start()

    log_collector: List[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: log_collector.append(
        f"[{record.levelname}] {record.getMessage()}"
    )
    logging.getLogger("neuralcore").addHandler(handler)

    try:
        # ── SETUP ──
        cfg = _build_config_dict(server.base_url)
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as config_mod

        loader = ConfigLoader(cli_path=None, app_root=NEURALCORE_ROOT)
        loader.config = loader.parse_config(cfg)
        config_mod.loader = loader

        sys.path.insert(
            0, str(NEURALCORE_ROOT / "data" / "examples" / "workflows")
        )
        import importlib

        try:
            importlib.import_module("default_flow")
        except Exception:
            pass

        from neuralcore.clients.factory import ClientFactory
        import neuralcore.clients.factory as cfactory_mod

        cfactory_mod._factory = ClientFactory(loader)
        cfactory_mod._factory.build()
        loader.load_tool_sets(sets_to_load=["ContextTestTools"])

        from neuralcore.agents.factory import AgentFactory

        factory = AgentFactory(loader)

        # Create two agents
        agent_alpha = _create_agent(factory, loader, "agent_alpha", "Alpha Agent")
        agent_beta = _create_agent(factory, loader, "agent_beta", "Beta Agent")

        # Load tools on both agents
        agent_alpha.action_manager.load_toolsets(["ContextTestTools"])
        agent_beta.action_manager.load_toolsets(["ContextTestTools"])

        assert "get_weather_data" in agent_alpha.action_manager.loaded_tools
        assert "get_weather_data" in agent_beta.action_manager.loaded_tools

        # ── Phase 5a: Pre-populate Beta with tool results ──
        # Simulate Beta having done some work (tool executions)
        r1 = await agent_beta.action_manager.execute_direct(
            "get_weather_data", city="Berlin"
        )
        assert "Berlin" in r1
        r2 = await agent_beta.action_manager.execute_direct("get_system_metrics")
        assert "CPU" in r2
        r3 = await agent_beta.action_manager.execute_direct(
            "get_project_status", project_name="Phoenix"
        )
        assert "Phoenix" in r3

        # Verify Beta's context manager has the data
        beta_stm = agent_beta.context_manager.short_term_mem
        beta_tool_items = [
            i for i in beta_stm.values() if i.source_type == "tool_outcome"
        ]
        assert len(beta_tool_items) >= 3, (
            f"Expected >= 3 tool outcomes in Beta's STM, got {len(beta_tool_items)}"
        )

        # ── Phase 5b: Alpha drains Beta's context ──
        drained = agent_alpha.drain_agent_context(agent_beta)

        assert drained["agent_id"] == "agent_beta"
        assert drained["agent_name"] == "Beta Agent"
        assert len(drained["tool_results"]) >= 3
        assert len(drained["context_items"]) >= 3
        assert len(drained["tool_call_history"]) >= 3
        # context_summary may be empty if no chat messages were added;
        # the critical data is in tool_results and context_items
        assert isinstance(drained["context_summary"], str)

        # Check that tool results contain the actual data
        all_results_str = " ".join(
            str(tr.get("result", "")) for tr in drained["tool_results"]
        )
        assert "Berlin" in all_results_str
        assert "CPU" in all_results_str
        assert "Phoenix" in all_results_str

        # ── Phase 5c: Alpha injects drained context ──
        await agent_alpha.inject_drained_context(drained)

        # Verify Alpha now has Beta's tool results
        alpha_tool_results = agent_alpha.state.tool_results
        source_agents = [tr.get("source_agent") for tr in alpha_tool_results]
        assert "Beta Agent" in source_agents, (
            f"Expected 'Beta Agent' in source_agents, got {source_agents}"
        )

        # ── Phase 5d: Task delegation with completion event ──
        from neuralcore.tasks.task import Task, TaskStatus

        delegated_task = Task(
            description="Analyze weather patterns for Tokyo",
            expected_outcome="Weather analysis for Tokyo completed",
        )

        # Enqueue LLM response for the delegated task execution
        server.enqueue_response(
            {
                "tool_calls": [
                    {"name": "get_weather_data", "arguments": {"city": "Tokyo"}}
                ]
            }
        )
        # Second response for the finish
        server.enqueue_response({"content": "Weather analysis complete for Tokyo."})

        # Alpha requests Beta to execute the task
        returned_task = await agent_alpha.request_agent(
            target_agent=agent_beta,
            task=delegated_task,
            drain_context=True,
        )

        assert returned_task is delegated_task
        assert delegated_task.requesting_agent_id == "agent_alpha"
        assert delegated_task.assigned_agent == "agent_beta"
        assert "agent_beta" in agent_alpha.state.active_sub_agents
        assert delegated_task.task_id in agent_alpha.state.sub_task_ids

        # Execute the delegated task on Beta (simulating Beta processing it)
        await agent_beta.task_manager.execute_delegated(delegated_task)

        # Verify the task completed
        assert delegated_task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        assert delegated_task.result_payload is not None
        assert "tool_results" in delegated_task.result_payload
        assert "context_summary" in delegated_task.result_payload

        # Verify the completion event was signaled
        event = delegated_task.get_completion_event()
        assert event.is_set()

        # Alpha can now await (already done, should return immediately)
        result = await agent_alpha.await_task_completion(
            delegated_task, timeout=5.0
        )
        assert result["status"] in ("completed", "failed")
        assert result["task_id"] == delegated_task.task_id
        assert result["result_payload"] is not None

        # ── Phase 5e: Await timeout test ──
        timeout_task = Task(
            description="This task will never complete",
            expected_outcome="Should timeout",
        )
        timeout_task.get_completion_event()  # create event but never set it

        timeout_result = await agent_alpha.await_task_completion(
            timeout_task, timeout=0.1
        )
        assert timeout_result["status"] == "timeout"
        assert timeout_result["error"] is not None

        # ── Phase 5f: Parallel dispatch with dependencies ──
        task_a = Task(
            description="Gather weather data",
            expected_outcome="Weather data collected",
        )
        task_b = Task(
            description="Gather system metrics",
            expected_outcome="System metrics collected",
        )
        task_c = Task(
            description="Synthesize reports",
            expected_outcome="Final report generated",
            dependencies=[task_a.task_id, task_b.task_id],
        )
        # Re-init dependency set since we set dependencies after creation
        task_c._dependency_set = {task_a.task_id, task_b.task_id}

        # Enqueue mock LLM responses for all three task executions
        for _ in range(6):
            server.enqueue_response(
                {"content": "Task execution result from mock LLM."}
            )

        dispatch_events = []
        async for event, payload in agent_alpha.task_manager.dispatch_parallel(
            tasks=[task_a, task_b, task_c],
            agents=[agent_alpha, agent_beta],
            timeout=30.0,
        ):
            dispatch_events.append((event, payload))

        event_types = [e[0] for e in dispatch_events]
        assert "dispatch_started" in event_types
        assert "task_dispatched" in event_types
        assert "dispatch_finished" in event_types

        # Verify the dispatch respected dependencies
        dispatched_events = [e for e in dispatch_events if e[0] == "task_dispatched"]
        dispatched_task_ids = [e[1]["task_id"] for e in dispatched_events]

        # task_c depends on task_a and task_b, so it should be dispatched last
        assert dispatched_task_ids.index(task_c.task_id) > dispatched_task_ids.index(
            task_a.task_id
        )
        assert dispatched_task_ids.index(task_c.task_id) > dispatched_task_ids.index(
            task_b.task_id
        )

        # ── Summary ──
        cooperation_logs = [
            l for l in log_collector if "COOPERATION" in l or "DELEGATED" in l
        ]
        assert len(cooperation_logs) >= 3, (
            f"Expected >= 3 cooperation log entries, got {len(cooperation_logs)}"
        )

    finally:
        await server.stop()
        logging.getLogger("neuralcore").removeHandler(handler)


# =========================================================================
# Phase 6: handle_delegated_task via control message
# =========================================================================
@pytest.mark.asyncio
async def test_handle_delegated_task_control():
    """Agent.handle_delegated_task processes a delegated_task control message."""
    _reset_singletons()

    server = MockLLMServer(port=9934)
    await server.start()

    try:
        cfg = _build_config_dict(server.base_url)
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as config_mod

        loader = ConfigLoader(cli_path=None, app_root=NEURALCORE_ROOT)
        loader.config = loader.parse_config(cfg)
        config_mod.loader = loader

        sys.path.insert(
            0, str(NEURALCORE_ROOT / "data" / "examples" / "workflows")
        )
        import importlib

        try:
            importlib.import_module("default_flow")
        except Exception:
            pass

        from neuralcore.clients.factory import ClientFactory
        import neuralcore.clients.factory as cfactory_mod

        cfactory_mod._factory = ClientFactory(loader)
        cfactory_mod._factory.build()
        loader.load_tool_sets(sets_to_load=["ContextTestTools"])

        from neuralcore.agents.factory import AgentFactory

        factory = AgentFactory(loader)
        agent = _create_agent(factory, loader, "handler_agent", "Handler Agent")
        agent.action_manager.load_toolsets(["ContextTestTools"])

        from neuralcore.tasks.task import Task, TaskStatus

        task = Task(
            description="Run system diagnostics",
            expected_outcome="Diagnostics report generated",
        )
        task.get_completion_event()
        task.requesting_agent_id = "external_agent"

        # Enqueue mock responses
        server.enqueue_response({"content": "System diagnostics complete."})
        server.enqueue_response({"content": "All checks passed."})

        control_msg = {
            "event": "delegated_task",
            "task_id": task.task_id,
            "description": task.description,
            "expected_outcome": task.expected_outcome,
            "requesting_agent_id": "external_agent",
        }

        await agent.handle_delegated_task(control_msg, task=task)

        assert task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        assert task.result_payload is not None
        assert task.get_completion_event().is_set()

    finally:
        await server.stop()
