# NeuralCore

**NeuralCore** is a **state-based, event-driven, action-oriented** adaptive agentic framework.

It is an **event-driven state machine** with adjustable workflows, loops, and dynamically loaded tools — designed from the ground up for clean architectural separation and long-term maintainability.

> **Core Principle (Strictly Enforced)**: NeuralCore is the **generic, reusable foundation**. It contains zero client-specific business logic, tools, or workflows. All domain-specific implementation lives in separate client applications (see [NeuralVoid](https://github.com/Abyss-c0re/NeuralVoid) for the official reference pattern).

**NeuralVoid** is not "the app" — it is the **living documentation and reference implementation** showing exactly how to build on NeuralCore without ever modifying the core itself. Real client projects should follow the same pattern.

## What Makes NeuralCore Special

- **State Based, Event Driven, Action Oriented** — Every interaction is an event that updates state and triggers actions.
- **Everything is a Tool (Action)** — Including the agent’s own internal methods (search context, deploy sub-agents, describe images, etc.). The agent only ever sees `FindTool`.
- **Dynamic Tool Registry** — Tools are registered via `@tool` decorator on the client side. The agent discovers and loads the most relevant ones on-demand using fuzzy + keyword matching.
- **Fully Config-Driven Workflows** — Loops, conditions (including complex state-based ones), reusable steps, go_to jumps, step overrides, and even human-in-the-loop waits are all defined or overridden in `config.yaml`. No code changes needed for most behaviors.
- **Sub-Agent Inheritance** — Child agents automatically inherit workflows, steps, and tool configurations from parents.
- **Self-Improving Cognition** — Built-in `KnowledgeConsolidator` with real Learning-To-Rank (LambdaMART) and automatic concept distillation.

Most agent frameworks hard-code behavior. NeuralCore makes **almost everything adjustable at runtime via configuration** while keeping the core pristine.

## Key Features

### Hybrid Retrieval-Augmented Generation (RAG)
- Dense retrieval via **FastEmbed** (local-first, with full offline path support, cache directory handling, and model path expansion)
- Sparse retrieval via **TF-IDF** (lazy async rebuild, robust error handling, n-gram support)
- **Reciprocal Rank Fusion (RRF)** for combining signals
- Intelligent **file/parameter mention boosting** for tool outcomes — dramatically improves relevance when agents mention specific files or arguments
- Tokenizer-aware chunking (768 tokens / 128 overlap recommended) with fallback for very large tool outputs
- Contamination guards and smart deduplication to keep the knowledge base clean

### ContextManager — Neuroscience-Inspired Episodic Memory
The `ContextManager` (in `neuralcore/cognition/memory.py`) implements a **neuroscience-inspired episodic memory system** that mirrors hippocampal indexing and neocortical integration:

- **Topic-Based Clustering & Off-Topic Detection**: Automatic episodic segmentation (like hippocampal place cells) and context/state transition detection when the agent drifts from its current "mental state".
- **Investigation State & TaskContext**: Goal-directed memory tracking (`goal`, `subtasks`, `findings`, `hypotheses`) with per-subtask containers that automatically feed consolidated experiences back into long-term storage — exactly like prefrontal-hippocampal coupling.
- **Multi-Mode Retrieval**: `chat`, rich `agentic`, and `lightweight_agentic` modes with token-aware pruning and smart sub-agent noise removal.
- **Embedding Pipeline + Caching**: Hybrid sparse (TF-IDF ≈ neocortical semantic indexing) + dense (FastEmbed) retrieval with prefixing ("query" vs "passage") and async execution.

`provide_context()` is the single most important method — it assembles high-fidelity, biologically-plausible prompts while respecting token budgets.

### KnowledgeConsolidator — Neuroscience-Inspired Consolidation Engine
Located in `neuralcore/cognition/consolidator.py`, this is the **learning layer** that turns static RAG into a living, evolving memory system — directly inspired by multi-stage memory consolidation (hippocampal indexing → neocortical integration):

- **Multi-Cue Feature Extraction**: Combines semantic similarity, keyword overlap, recency, source type, and investigation alignment — mimicking biological multi-cue binding.
- **LambdaMART Reranker (LightGBM)**: Performs synaptic-weight-style optimization by learning non-linear feature interactions to prioritize "important" memories. Trained on real interaction data collected during every `rerank()` call.
- **Recency Score + Investigation Align**: Introduces temporal decay (mimicking hippocampal replay / sleep consolidation) and goal-directed relevance gating (prefrontal-hippocampal coupling).
- **Concept Distillation**: Periodically extracts higher-level abstract concepts ("strategies", "patterns", "anti-patterns") and stores them as `extracted_concept` items — exactly like systems consolidation.
- **Smart Triggers & Concept Graph**: Only activates on meaningful knowledge growth; maintains relationships between distilled concepts.

Overall alignment with neuroscience principles: **8.5/10** (strong multi-stage retrieval and goal-directed filtering; future work includes dynamic plasticity models and explicit temporal binding).

This is the meta-cognitive engine that lets agents continuously improve how they remember.

### Dynamic Tool System — FindTool is the Only Tool the Agent Sees
- **Everything is a Tool**: Internal agent methods (search memory, deploy sub-agents, describe images, post messages, etc.) are exposed as tools via the same `@tool` decorator.
- **Central Registry**: All tools are auto-registered on the client side into a central registry using `@tool`.
- **FindTool Only**: The agent is **never** given the full tool list. It only has access to `FindTool`, which performs fuzzy + keyword search across the registry and dynamically loads the most relevant tools for the current step/context.
- **Per-Step Configuration**: `DynamicActionManager` can load/unload entire toolsets, apply overrides, or hide tools per workflow step — all configurable in YAML.
- **Self-Aware Tools**: Tools can receive the full `agent` instance and even handle parent/sub-agent relationships.

### WorkflowEngine — Decorator-First, Config-Driven State Machine

**Critical architecture**: All workflows, steps, loops, and conditions **must first be registered** using decorators (`@workflow.step`, `@workflow.loop`, `@workflow.condition`) on the **client side**. Only after registration can they be reused, composed, and adjusted via `config.yaml`.

- **Client-Side Registration**: In your client app (e.g. NeuralVoid), you decorate Python functions/methods. These are automatically picked up into the central registry.
- **YAML Composition**: The `workflows:` section in `config.yaml` then defines **which** registered steps/loops to use, their execution order, conditions, overrides (client, temperature, toolset, etc.), and `go_to` / `insert_steps` logic.
- **Rich, State-Based Conditions**: Use simple strings (`"goal_achieved"`, `"error_rate_high"`, `"no_progress_last_n"`, sub-task conditions) or complex `and`/`or`/`not` logic evaluated directly against `AgentState`. Custom conditions are registered via decorator and callable from YAML.
- **Advanced Runtime Control** (all configurable):
  - `go_to` (by name/index + data)
  - `insert_steps` (dynamic injection)
  - Per-step `overrides`, `retries`, `timeout`, `if` conditions
  - Built-in `wait` step (`time` / `human` / `subtask` / `agent` / `condition`)
- **Sub-Agent Inheritance & Live Reload**: Child agents inherit parent workflows/steps; `reload_workflow_config()` enables hot-reloading.
- **Example from real config**:
  ```yaml
  workflows:
    orchestrator:
      steps:
        - name: plan_microtasks
          overrides: { client: reasoning, temperature: 0.35 }
        - name: launch_next_subtask
        - name: wait_for_subtask
  ```

This design makes **registered components reusable across workflows** while giving you full runtime control through configuration — the best of both worlds.

### Additional Capabilities
- Multi-client LLM support (main, reasoning, embeddings) with per-step overrides
- Full async/await throughout (embedding calls, tool execution, index rebuilds)
- Robust error handling, logging, and stats tracking (`context_stats`)
- Experimental self-reflection: Agents can analyze their own memory implementation and execution logs against neuroscience literature (or any provided text). The original large-scale audit (comparing ~1675 LOC of memory code vs. 830 pages of *Neuroscience, Third Edition*) was user-initiated with a specific prompt and textbook. See `/docs/experiments/`.

## Quick Start — Use NeuralVoid as Your Guide

**Do not start from scratch.** Clone the reference implementation:

```bash
git clone https://github.com/Abyss-c0re/NeuralVoid.git
cd NeuralVoid
```

Study how NeuralVoid:
- Organizes tools into separate modules (`TerminalTools`, `FileEditingTools`, etc.)
- Uses `config.yaml` to wire agents, clients, tool sets, and workflows
- Implements the `orchestrator` workflow that decomposes goals into subtasks and spawns specialized sub-agents
- Calls `context_manager.provide_context(..., lightweight_agentic=...)` inside loops
- Records tool outcomes and lets the `KnowledgeConsolidator` run periodically

Then create **your own** client application in exactly the same style. NeuralCore stays untouched.

### Installing NeuralCore via uv

To add **NeuralCore** as a dependency in your client project using the `uv` package manager (recommended for fast, reproducible installs):

```bash
uv add neuralcore@git+https://github.com/Abyss-c0re/NeuralCore.git
```

This will install the latest version directly from the repository. For a specific commit or branch, append `#branch-or-commit` to the URL.

## Example: Tool Definition (NeuralVoid Pattern)

```python
from neuralcore.tools.base import tool
from typing import Optional

@tool(
    "ResearchTools",
    tags=["web", "analysis", "synthesis"],
    name="deep_research",
    description="Perform multi-step research on a topic and consolidate findings.",
    require_confirmation=False
)
async def deep_research(agent, query: str, depth: int = 3) -> str:
    # Self-aware: access full context and state
    ctx = agent.context_manager
    await ctx.add_external_content("research_query", query, metadata={"depth": depth})
    
    # ... perform steps, spawn sub-agents if needed ...
    
    return consolidated_findings
```

## Configuration Example (from NeuralVoid)

```yaml
agents:
  head_agent:
    name: "Autonomous Research Agent"
    tool_sets:
      - TerminalTools
      - ResearchTools
      - CodingTools
    workflow: research_orchestrator
    max_tokens: 28000
    clients:
      main: { model: "Qwen3.5-9B", tokenizer: "...", temperature: 0.7 }
      reasoning: { model: "...", temperature: 0.2 }

workflows:
  research_orchestrator:
    type: loop
    max_iterations: 30
    break_condition: "all_subtasks_complete"
```

## Advanced Topics

### Lightweight Mode for Long-Running Agents
```python
messages = await ctx.provide_context(
    query=current_goal,
    lightweight_agentic=True,
    state=agent.state,
    max_input_tokens=8000
)
```

Returns a compact prompt containing only: objective reminder, last 10 compact tool results, current subtask, loaded tools summary, and explicit tool expectations.

### Knowledge Consolidation
```python
consolidator = KnowledgeConsolidator(agent)
await consolidator.extract_and_consolidate(
    trace=agent.context_manager.tool_call_history[-30:],
    task_goal=agent.state.task
)
```

The consolidator automatically decides when to distill new concepts and whether to retrain the reranker.

### Sub-Agent Spawning (see NeuralVoid `start_complex_deployment`)
Restricted tool sets + forwarded context + isolated `AgentState` = safe parallel exploration.

## Project Structure

```
src/neuralcore/
├── agents/           # Base Agent + state management
├── cognition/        # ContextManager + KnowledgeConsolidator
├── tools/            # @tool decorator + DynamicActionManager
├── workflows/        # Declarative workflow engine
├── clients/          # LLM / embedding client factory
├── utils/            # Tokenizer, prompt builder, search helpers, logger
└── ...
```

## License

**Dual License**

- **AGPLv3** — Free for open-source projects (any modifications must be made available under the same license)
- **Commercial License** — Available for proprietary use without AGPL obligations. Contact: info@abyss-core.com

## Status & Roadmap

**Active early-stage development** The core architecture and key mechanisms (Hybrid RAG, ContextManager, KnowledgeConsolidator) are stabilizing. Not yet recommended for production workloads without thorough evaluation.

Feedback, issue reports, and commercial partnership inquiries are welcome.

---

*Built with the conviction that the best agent frameworks separate what is universal from what is specific.*