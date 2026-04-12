# NeuralCore

## Overview

**NeuralCore** is an experimental adaptive agentic framework focused on clean architectural separation and advanced context management. The project is in active early-stage development and is not yet intended for production use. For a practical demonstration and reference implementation, see [NeuralVoid](https://github.com/Abyss-c0re/NeuralVoid).

### Licensing

This project is **dual-licensed**:

1. **AGPLv3 (Open Source)**  
   Free to use, modify, and distribute. Modifications must be shared under the same license.

2. **Commercial License**  
   Available for proprietary use without AGPL obligations.  
   **Contact:** `info@abyss-core.com`

---

## Why NeuralCore?

Most agent frameworks tightly couple core logic with application-specific code, leading to brittle, hard-to-maintain systems as complexity grows.

**NeuralCore** enforces a strict architectural boundary:

> **NeuralCore** = pure generic, reusable framework layer  
> **Client applications (e.g. NeuralVoid)** = all domain-specific tools, workflows, and business logic

This separation delivers clean architecture, improved safety, and the flexibility required for sophisticated agentic systems.

---

## What Makes NeuralCore Special

- **Rich, expressive tool system** — Full-featured `@tool` decorator supporting tags, custom names, descriptions, and `require_confirmation`.
- **Self-aware agents** — Tools can automatically receive the current `agent` instance, granting direct access to memory, logs, context, and peer agents.
- **Dynamic & safe tool management** — `DynamicActionManager` with per-step loading/unloading, protected persistent tools, and runtime discovery.
- **Composable action sequences** — Multi-step flows with context propagation, human confirmation, and safe pausing/resuming.
- **Advanced ContextManager & RAG** — Hybrid retrieval (FastEmbed dense + TF-IDF sparse), topic detection, automatic sub-agent noise pruning, investigation mode, `TaskContext`, findings/hypotheses tracking, and configurable smart chunking.
- **Powerful declarative workflow engine** — Supports `@workflow.set`, `@workflow.loop` (with iteration limits and break conditions), and `@workflow.condition`.

---

## Experiments & Scientific Validation

### Neuroscience-Inspired Self-Audit of NeuralCore Memory System (2026-04-12)

**Experiment Overview**  
A NeuralCore-based agent performed a large-scale comparative RAG analysis between its own memory subsystem implementation (**~1500 lines of code** in `src/neuralcore/cognition/memory.py`) and **830 pages** of the canonical textbook *Neuroscience, Third Edition* (Purves et al., 2004).

Running on **CachyOS x86_64 (Linux kernel 6.19.11-1-cachyos)**, the agent mapped the module’s core RAG mechanisms — token-based chunking with overlap, hybrid sparse/dense embeddings, topic consolidation with drift detection, temporal history tracking, and off-topic pruning — onto established neuroscientific concepts such as hippocampal binding, systems consolidation, dual-coding theory, and signal-to-noise filtering in memory stabilization.

The resulting report also proposed concrete, neuroscience-inspired enhancements (Hebbian-style plasticity simulation, offline sleep-consolidation phase, retrieval practice mechanisms, and hierarchical topic organization) while preserving the strict abstraction boundaries of NeuralCore.

**Significance**  
- Demonstrates robust cross-domain RAG capability on large scientific literature combined with production code.
- Exhibits early meta-cognitive behavior: the framework analyzing and critiquing its own memory architecture through a biological lens.
- Validates clean separation of concerns — the entire analysis remained inside NeuralCore’s generic abstractions with no client logic leakage.
- Serves as a canonical reference for reflective, scientifically grounded agent workflows built using NeuralVoid patterns.

**Lab Setup**  
- **OS**: CachyOS x86_64 (Linux kernel 6.19.11-1-cachyos)  
- **Hardware**: AMD Ryzen 9 5900X (24 threads) @ 5.62 GHz, AMD Radeon RX 6800 (discrete), 62.71 GiB RAM  
- **LLM**: Qwen3.5-9B (Q4_K_M, 128k context) via llama.cpp  
- **Embeddings**: fastembed (`BAAI/bge-small-en-v1.5`, local cache)  
- **Workflow**: NeuralVoid orchestrator + sub-agent ReAct loops

**Artifacts** (all inside this repository)  
- Agent Generated Report → [`docs/experiments/neuroscience-memory-audit-2026-04-12.md`](docs/experiments/neuroscience-memory-audit-2026-04-12.md)  
- Raw Experiment Log → [`logs/experiments/neuroscience-memory-audit-20260412.log`](docs/experiments/neuroscience-memory-audit-20260412.log)  
- Source under audit → [`src/neuralcore/cognition/memory.py`](src/neuralcore/cognition/memory.py)

*Reproducible using NeuralVoid reference patterns. Similar audit workflows can be designed and visualized in NeuralLabs.*

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
    @workflow.step("orchestrator", name="plan_microtasks")
    async def _wf_plan_microtasks(self, iteration: int, state: AgentState):
        ...

    @workflow.step("orchestrator", name="launch_next_subtask")
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