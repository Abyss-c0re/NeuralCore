# NeuralCore

## Overview

**NeuralCore** is an experimental adaptive agentic framework. The project is at a very early stage of development and is not ready for production or hype. There is no documentation at this time. For a demonstration of its capabilities, please see [NeuralVoid](https://github.com/Abyss-c0re/NeuralVoid).

### Licensing

This project is **dual-licensed**:

1.  **AGPLv3 (Open Source)**
    *   Free to use, modify, and distribute.
    *   Modifications must be shared under the same license.

2.  **Commercial License**
    *   Available for proprietary use without AGPL obligations.
    *   **Contact:** `info@abyss-core.com`

---

## Why NeuralCore?

Most agent frameworks mix framework code with business logic, becoming rigid, unsafe, and difficult to maintain as complexity grows.

**NeuralCore** was built differently — with a **strict architectural boundary**:

> **NeuralCore** = pure generic, reusable framework  
> **Your application (NeuralVoid)** = all tools, workflows, and business logic

This separation gives you clean architecture, maximum flexibility, and the ability to build extremely powerful agents without fighting the framework.

---

## What Makes NeuralCore Special

*   **Rich, expressive tool system** — Full-featured `@tool` decorator with tags, custom names, descriptions, and `require_confirmation`.
*   **Self-aware agents** — Tools can automatically receive the current `agent` instance, giving them direct access to memory, logs, context, and other agents.
*   **Dynamic & safe tool management** — `DynamicActionManager` with per-step loading/unloading, protected persistent tools (`FindTool`, `DeploySubAgent`, `GetContext`, `GetDeploymentStatus`), and runtime discovery.
*   **Composable sequences** — Multi-step flows with context propagation, human confirmation, and pausing/resuming.
*   **World-class ContextManager** — Advanced RAG (FastEmbed + TF-IDF), topic detection, automatic sub-agent noise pruning, investigation mode, `TaskContext`, findings/hypotheses tracking, and smart chunking.
*   **Powerful workflow engine** — Declarative workflows with `@workflow.set`, `@workflow.loop` (with `max_iterations` + `break_condition`), and `@workflow.condition`.

---

## Usage Examples

### 1. Rich Tool Definition

```python
@tool(
    "TerminalTools",
    tags=["filesystem", "search", "regex"],
    name="search_text",
    description="Search for text pattern inside files.",
)
async def search_text(pattern: str, file_path: str, recursive: bool = False) -> str:
    ...
```

### 2. Self-Aware Tools (Real Example)

Tools can be fully self-aware and even handle parent/sub-agent relationships:

```python
@tool("ContextManager", name="GetContext", description="Search your own memory")
async def provide_context(
    agent, query: str, *, agent_metadata: Optional[dict] = None
):
    """Tools can be fully self-aware and even handle parent/sub-agent relationships."""
    agent_metadata = agent_metadata or {}
    agent_metadata.setdefault("is_sub_agent", getattr(agent, "sub_agent", False))
    agent_metadata.setdefault("agent_id", getattr(agent, "agent_id", "unknown"))

    parent_agent = getattr(agent, "parent", None)
    if getattr(agent, "sub_agent", False) and parent_agent:
        return await parent_agent.context_manager.provide_context(query)
    return await agent.context_manager.provide_context(query)
```

### 3. Sequences

Create safe, multi-step actions with human oversight:

```python
from neuralcore.actions.sequence import ActionFromSequence

code_explorer = ActionFromSequence.from_sequence(
    name="explore_codebase",
    description="Safely explore a codebase with human oversight",
    steps=[list_directory, read_file, analyze_code],
    propagate=True,
    confirm_predicate=lambda result: "sensitive" in str(result).lower()
)
```

### 4. Advanced Workflows with Loops & Conditions

Leverage declarative workflows for complex agentic loops:

```python
from neuralcore.workflows.registry import workflow
from neuralcore.agents.state import AgentState

@workflow.condition("subtask_complete")
def subtask_complete(state: AgentState, args=None):
    ...

@workflow.loop("agentic_loop", max_iterations=50, break_condition="subtask_complete")
async def agentic_loop(agent, state: AgentState):
    ...

class AgentFlow:
    @workflow.set("orchestrator", name="plan_microtasks")
    async def _wf_plan_microtasks(self, iteration: int, state: AgentState):
        ...

    @workflow.set("orchestrator", name="launch_next_subtask")
    async def _wf_launch_next_subtask(self, iteration: int, state: AgentState):
        ...
```

---

## Configuration

Everything is orchestrated centrally through `config.yaml`:

```yaml
agents:
  agent_001:
    name: "Headless Shell Agent"
    tool_sets:
      - TerminalTools
      - FileEditingTools
      - CodingTools
      - WebTools
    workflow: orchestrator
```

---

## Use Cases

NeuralCore is ideal for building:

*   **Codebase Intelligence Agents** — Deep exploration, analysis, and safe modification of large repositories.
*   **Research & Synthesis Agents** — Multi-source intelligence with structured reasoning and long-term memory.
*   **Multi-Agent Orchestration Systems** — Main agents that intelligently plan, deploy, monitor, and coordinate multiple sub-agents.
*   **Self-Aware & Collaborative Agents** — Agents that can inspect their own state, logs, memory, and delegate tasks to others.
*   **Safe Automation Agents** — Terminal, file, web, and code operations with proper safety boundaries.