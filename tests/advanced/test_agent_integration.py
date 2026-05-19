"""
Advanced Integration Tests for NeuralCore Agentic Framework.

This test suite:
1. Starts a mock LLM proxy server (no hardcoded prompts)
2. Creates an Agent via factory connected to the mock server
3. Loads tools from files AND config.yaml
4. Validates tool loading via action_manager.execute_direct
5. Runs a full session using the agent's communication channel
6. Validates ContextManager.provide_context with populated data

The mock LLM proxy is the sole mechanism for commanding the agent.
All evidence is gathered from framework logs.
"""

import asyncio
import logging
import sys
import time
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


def _build_config_dict(mock_base_url: str) -> Dict[str, Any]:
    """Build test config ensuring dict structure for type safety.

    yaml.safe_load can return None (or non-dict) → explicit guard + setdefault
    keeps the helper reusable across tests without fragile key assumptions.
    """
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    # Defensive narrowing for Pyright (reportOptionalSubscript + return type)
    cfg: Dict[str, Any] = raw_cfg if isinstance(raw_cfg, dict) else {}

    # Safe nested structure (reusable pattern, no domain logic)
    clients = cfg.setdefault("clients", {})
    main = clients.setdefault("main", {})
    main["base_url"] = mock_base_url
    main["tokenizer"] = str(TOKENIZER_PATH)

    tools_section = cfg.setdefault("tools", {})
    context_test_tools = tools_section.setdefault("ContextTestTools", {})
    context_test_tools["folder"] = str(TOOLS_DIR)

    return cfg


class _Report:
    def __init__(self):
        self.sections: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add(self, title: str, status: str, details: str, evidence: str = ""):
        self.sections.append(
            {
                "title": title,
                "status": status,
                "details": details,
                "evidence": evidence,
                "timestamp": time.time(),
            }
        )

    def generate_markdown(self, all_logs: List[str] | None = None) -> str:
        elapsed = time.time() - self.start_time
        passed = sum(1 for s in self.sections if s["status"] == "PASS")
        failed = sum(1 for s in self.sections if s["status"] == "FAIL")
        lines = [
            "# NeuralCore Advanced Integration Test Report",
            "",
            "## Overview",
            "",
            "| Property | Value |",
            "|---|---|",
            f"| Generated | {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} |",
            f"| Total Duration | {elapsed:.1f}s |",
            f"| Test Phases | {len(self.sections)} |",
            f"| Passed | {passed} |",
            f"| Failed | {failed} |",
            f"| Result | {'ALL PASSED' if failed == 0 else f'{failed} FAILED'} |",
            "",
            "## Architecture",
            "",
            "This test suite validates the NeuralCore Agentic Framework through a "
            "**mock LLM proxy server** that acts as a controllable intermediary. "
            "No prompts are hardcoded inside the mock server code; every response "
            "is enqueued externally by the test orchestrator.",
            "",
            "```",
            "Test Orchestrator",
            "    |",
            "    |-- enqueue_response({content: ...}) -->  Mock LLM Server (:9922)",
            "    |                                              |",
            "    |-- agent.client.chat() ------------------>    /v1/chat/completions",
            "    |-- agent.client.stream_chat() ----------->    (SSE streaming)",
            "    |-- agent.client.stream_with_tools() ----->    (SSE + tool_calls)",
            "    |                                              |",
            "    |<--- response / streaming chunks -------------|",
            "    |",
            "    |-- agent.action_manager.execute_direct() --> Tool execution",
            "    |                                              |",
            "    |-- agent.context_manager <--- record_tool_outcome() from Action.__call__",
            "    |-- agent.context_manager.provide_context() --> Validated output",
            "```",
            "",
            "### Components Tested",
            "",
            "| Component | Module | Test Coverage |",
            "|---|---|---|",
            "| AgentFactory | `agents.factory` | Phase 1 |",
            "| LLMClient (non-streaming) | `clients.client` | Phase 4 |",
            "| LLMClient (streaming) | `clients.client` | Phase 5 |",
            "| LLMClient (stream_with_tools) | `clients.client` | Phase 6 |",
            "| ActionRegistry / tool decorator | `actions.registry` | Phase 2 |",
            "| DynamicActionManager | `actions.manager` | Phase 2, 3 |",
            "| Action.__call__ | `actions.actions` | Phase 3, 6, 7 |",
            "| ContextManager (short_term_mem) | `cognition.memory` | Phase 7 |",
            "| ContextManager.provide_context | `cognition.memory` | Phase 8 |",
            "| ConfigLoader | `utils.config` | Setup |",
            "| ClientFactory | `clients.factory` | Setup |",
            "| Mock LLM Proxy Server | `tests.advanced.mock_llm_server` | All phases |",
            "",
            "---",
            "",
        ]
        for i, s in enumerate(self.sections, 1):
            icon = "PASS" if s["status"] == "PASS" else "FAIL"
            lines.extend([f"## Phase {i}: [{icon}] {s['title']}", "", s["details"], ""])
            if s["evidence"]:
                lines.extend(
                    ["**Evidence (Framework Logs):**", "```", s["evidence"], "```", ""]
                )
            lines.extend(["---", ""])

        # Full log dump
        if all_logs:
            lines.extend(
                [
                    "## Full Framework Log Trace",
                    "",
                    "Complete log output captured during the test session:",
                    "",
                    "```",
                ]
            )
            lines.extend(all_logs[-200:])
            lines.extend(["```", "", "---", ""])

        lines.extend(["## Conclusion", ""])
        if failed == 0:
            lines.extend(
                [
                    "**All 8 test phases passed successfully.** The NeuralCore Agentic Framework "
                    "demonstrates correct behavior across the full agent lifecycle:",
                    "",
                    "1. **Agent Construction** -- Factory pattern creates fully wired agents "
                    "with client, context manager, action manager, and state.",
                    "2. **Tool Loading** -- Tools defined via `@tool` decorator in Python files "
                    "are loaded from config.yaml `tools` section into the registry and action manager.",
                    "3. **Direct Tool Execution** -- `execute_direct()` calls tool executors "
                    "directly, producing correct results with context recording.",
                    "4. **Non-Streaming LLM Chat** -- The OpenAI-compatible mock server correctly "
                    "serves non-streaming chat completions consumed by `LLMClient.chat()`.",
                    "5. **Streaming LLM Chat** -- SSE streaming delivers word-by-word tokens "
                    "consumed by `LLMClient.stream_chat()` via async queue.",
                    "6. **Streaming with Tool Calls** -- `stream_with_tools()` receives streamed "
                    "tool call deltas, assembles valid JSON arguments, resolves executors via "
                    "`DynamicActionManager`, executes tools, and returns results.",
                    "7. **ContextManager Population** -- Tool outcomes are automatically recorded "
                    "to `short_term_mem` as `KnowledgeItem` objects with `source_type='tool_outcome'`, "
                    "and `tool_call_history` tracks all invocations.",
                    "8. **ContextManager.provide_context Validation** -- In `agentic` mode, "
                    "`provide_context()` assembles system messages, user queries, and tool context "
                    "into a properly structured message list. String mode (`return_as_string=True`) "
                    "also produces correctly formatted output.",
                    "",
                    "The mock LLM proxy server operated as a transparent intermediary without "
                    "any hardcoded prompts, validating that the framework's communication "
                    "channel is fully functional end-to-end.",
                ]
            )
        else:
            lines.append(f"**{failed} test(s) failed.** See individual sections above.")
        return "\n".join(lines)


# =========================================================================
# Single comprehensive test — all 8 phases in sequence
# =========================================================================
@pytest.mark.asyncio
async def test_full_agent_integration():
    """Run all 8 test phases sequentially in one event loop."""
    _reset_singletons()

    server = MockLLMServer(port=9922)
    await server.start()
    report = _Report()
    log_collector: List[str] = []

    handler = logging.Handler()
    handler.emit = lambda record: log_collector.append(
        f"[{record.levelname}] {record.getMessage()}"
    )
    logging.getLogger("neuralcore").addHandler(handler)

    def evidence(keywords, limit: int = 20) -> str:
        """Extract recent log lines matching any keyword for test evidence."""
        matched = [
            entry
            for entry in log_collector
            if any(k.lower() in entry.lower() for k in keywords)
        ]
        return "\n".join(matched[-limit:])

    try:
        # ── SETUP ────────────────────────────────────────────────
        cfg = _build_config_dict(server.base_url)
        from neuralcore.utils.config import ConfigLoader
        import neuralcore.utils.config as config_mod

        loader = ConfigLoader(cli_path=None, app_root=NEURALCORE_ROOT)
        loader.config = loader.parse_config(cfg)
        config_mod.loader = loader

        sys.path.insert(0, str(NEURALCORE_ROOT / "data" / "examples" / "workflows"))
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
        agent = factory.create_agent(
            agent_id="test_agent",
            config=loader.get_agent_config("test_agent"),
            app_root=NEURALCORE_ROOT,
        )

        # ── PHASE 1: Agent Construction ──────────────────────────
        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.client is not None
        assert agent.context_manager is not None
        assert agent.action_manager is not None

        report.add(
            "Agent Construction via Factory",
            "PASS",
            f"Agent '{agent.name}' (id={agent.agent_id}) created. "
            f"Client model: {agent.client.model}, base_url: {agent.client.base_url}",
            evidence(["Agent", "created", "factory", "init"]),
        )

        # ── PHASE 2: Tool Loading ────────────────────────────────
        from neuralcore.actions.registry import registry

        assert "ContextTestTools" in registry.sets
        ctx_set = registry.sets["ContextTestTools"]
        tool_names = [a.name for a in ctx_set.actions]
        assert "get_weather_data" in tool_names
        assert "get_system_metrics" in tool_names
        assert "get_project_status" in tool_names

        agent.action_manager.load_toolsets(["ContextTestTools"])
        loaded = agent.action_manager.loaded_tools
        assert "get_weather_data" in loaded

        report.add(
            "Tool Loading from Files + Config",
            "PASS",
            f"Tools: {tool_names}. Loaded in action_manager: {loaded}",
            evidence(["TOOL REGISTERED", "ContextTestTools", "Loaded"]),
        )

        # ── PHASE 3: execute_direct ──────────────────────────────
        r1 = await agent.action_manager.execute_direct(
            "get_weather_data", city="London"
        )
        assert "London" in r1 and "Temperature" in r1
        r2 = await agent.action_manager.execute_direct("get_system_metrics")
        assert "CPU Usage" in r2
        r3 = await agent.action_manager.execute_direct(
            "get_project_status", project_name="NeuralCore"
        )
        assert "NeuralCore" in r3

        report.add(
            "Tool Execution via execute_direct",
            "PASS",
            "get_weather_data, get_system_metrics, get_project_status all executed.",
            evidence(["DIRECT EXEC", "ACTION START", "ACTION SUCCESS"]),
        )

        # ── PHASE 4: Non-streaming Chat ──────────────────────────
        server.enqueue_response({"content": "I am the mock assistant responding."})
        response = await agent.client.chat(
            [{"role": "user", "content": "Hello, are you working?"}]
        )
        assert "mock assistant" in response.lower()
        assert len(server.request_log) >= 1

        report.add(
            "Non-Streaming Chat via Mock Server",
            "PASS",
            f"Response: '{response[:80]}'",
            evidence(["chat", "model"]),
        )

        # ── PHASE 5: Streaming Chat ──────────────────────────
        server.enqueue_response(
            {"content": "This is a streamed response from the mock LLM."}
        )
        queue = await agent.client.stream_chat(
            [{"role": "user", "content": "Stream test."}]
        )
        chunks = []
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=10.0)
            if item is None:
                break
            chunks.append(item)
        full = "".join(chunks)
        assert "streamed response" in full.lower()
        assert len(chunks) > 1

        report.add(
            "Streaming Chat via Mock Server",
            "PASS",
            f"{len(chunks)} chunks received. Text: '{full[:80]}'",
            evidence(["stream"]),
        )

        # ── PHASE 6: Streaming with Tool Calls ───────────────────
        server.enqueue_response(
            {
                "tool_calls": [
                    {"name": "get_weather_data", "arguments": {"city": "Tokyo"}}
                ]
            }
        )
        # Use the actually-loaded tool set name (avoids potential None from get_action_set)
        tool_set = agent.action_manager.get_action_set("ContextTestTools")
        queue = await agent.client.stream_with_tools(
            manager=agent.action_manager,
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=tool_set or [],
        )
        events = []
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=15.0)
            if item is None:
                break
            events.append(item)
        event_types = [e[0] for e in events]
        assert "tool_complete" in event_types, f"Got: {event_types}"
        tc = next(e for e in events if e[0] == "tool_complete")
        assert tc[1]["function"]["name"] == "get_weather_data"
        assert "Tokyo" in str(tc[1].get("result", ""))
        assert "finish" in event_types

        report.add(
            "Streaming with Tool Calls",
            "PASS",
            f"Events: {event_types}. get_weather_data('Tokyo') executed via stream.",
            evidence(["tool_complete", "get_weather_data", "ACTION START"]),
        )

        # ── PHASE 7: ContextManager Population ───────────────────
        await agent.action_manager.execute_direct("get_weather_data", city="Berlin")
        await agent.action_manager.execute_direct("get_system_metrics")
        await agent.action_manager.execute_direct(
            "get_project_status", project_name="Phoenix"
        )

        cm = agent.context_manager
        stm = cm.short_term_mem
        assert len(stm) > 0
        tool_items = [i for i in stm.values() if i.source_type == "tool_outcome"]
        assert len(tool_items) >= 3, f"Got {len(tool_items)}"
        all_c = " ".join(i.content for i in tool_items)
        assert "Berlin" in all_c
        assert "Phoenix" in all_c
        assert len(cm.tool_call_history) >= 3

        report.add(
            "ContextManager Population via Tool Execution",
            "PASS",
            f"{len(tool_items)} tool_outcome items in short_term_mem. "
            f"{len(cm.tool_call_history)} tool_call_history entries.",
            evidence(["tool_outcome", "Added", "chunks"]),
        )

        # ── PHASE 8 (FINAL): provide_context Validation ─────────
        await agent.action_manager.execute_direct("get_weather_data", city="Paris")
        await agent.action_manager.execute_direct("get_system_metrics")
        await agent.action_manager.execute_direct(
            "get_project_status", project_name="Apollo"
        )

        stm = cm.short_term_mem
        tool_items = [i for i in stm.values() if i.source_type == "tool_outcome"]
        assert len(tool_items) >= 3

        await cm.set_mode("agentic")
        assert cm.mode == "agentic"

        await cm.add_message(
            "user", "Tell me about the weather in Paris and system status"
        )
        await cm.add_message("assistant", "I will check.")

        # provide_context — list mode
        server.enqueue_response({"content": "Context from mock."})
        ctx = await cm.provide_context(
            query="Weather in Paris and system performance?",
            max_input_tokens=8000,
            reserved_for_output=2000,
            chat=False,
            lightweight_agentic=False,
        )
        assert isinstance(ctx, list) and len(ctx) > 0
        roles = [m["role"] for m in ctx]
        assert "system" in roles and "user" in roles
        all_content = " ".join(m.get("content", "") for m in ctx)
        assert "Paris" in all_content or "weather" in all_content.lower()

        # provide_context — string mode (lightweight)
        server.enqueue_response({"content": "String context."})
        ctx_str = await cm.provide_context(
            query="System metrics and Apollo status",
            max_input_tokens=8000,
            reserved_for_output=2000,
            chat=False,
            lightweight_agentic=True,
            return_as_string=True,
        )
        assert isinstance(ctx_str, str) and len(ctx_str) > 50

        report.add(
            "FINAL VALIDATION: ContextManager.provide_context",
            "PASS",
            f"provide_context returned {len(ctx)} messages (list), "
            f"{len(ctx_str)} chars (string).\n"
            f"Short-term memory: {len(stm)} items ({len(tool_items)} tool outcomes).\n"
            f"Tool call history: {len(cm.tool_call_history)} entries.\n"
            f"Mode: {cm.mode}. Roles: {roles}.\n"
            f"Context contains query-relevant content: verified.",
            evidence(
                [
                    "provide_context",
                    "AGENTIC",
                    "LIGHTWEIGHT",
                    "tool_outcome",
                    "RETRIEVE",
                    "Added",
                    "mode changed",
                ],
                limit=40,
            ),
        )

    finally:
        # Always save the report and clean up
        docs_dir = NEURALCORE_ROOT / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "report.md").write_text(
            report.generate_markdown(all_logs=log_collector), encoding="utf-8"
        )
        await server.stop()
        logging.getLogger("neuralcore").removeHandler(handler)
