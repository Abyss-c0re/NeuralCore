# NeuralCore

A high-performance Python framework for orchestrating AI workflows with **OpenAI-compatible LLMs**, dynamic tool execution, and self-reflective agent patterns.

---

## ✨ Core Capabilities

### 📡 Streaming LLM Interface
- **Async/Sync support** – Non-blocking `stream_chat()` and synchronous `chat_sync()` methods
- **Incremental chunk delivery** – Real-time text streaming via `asyncio.Queue`
- **Vision capabilities** – Base64-encoded image analysis (e.g., for debugging or OCR)

### 🛠️ Dynamic Tool Execution
- **Action Registry System** – Runtime tool discovery and registration
- **Streaming tool calls** – Per-tool delta events: `tool_start`, `tool_delta`, `tool_complete`
- **Duplicate prevention** – Signature tracking avoids infinite loops
- **Confirmation handling** – Graceful user prompts for sensitive operations

### 🧠 Self-Reflective Agent Patterns
- **Stuck Detection** – Auto-triggers "Review & Reflect" when tool calls are missing or loop detected
- **Final Summary Generation** – Markdown report after task completion: iterations, tools used, conversation flow
- **Hidden Completion Markers** – System-level signals for clean task termination

### 🧩 Multi-Agent Architecture
- **AgentFactory** – Factory-based agent creation and lifecycle management
- **Context Manager Integration** – Persistent context with token-aware chunking
- **Cognitive Memory Layer** – Stateful session tracking across turns

---

## 📦 Quick Start

### Installation

```bash
# Using pip (editable mode)
pip install -e .

# Or using uv
uv pip install -e .
```

### Basic Usage

#### Async Streaming Chat

```python
from neuralcore.core.client import LLMClient

client = LLMClient(
    base_url="https://api.example.com/v1",
    model="gpt-4-turbo",
    api_key="your-api-key"
)

async def main():
    messages = [{"role": "user", "content": "Hello!"}]
    
    # Non-streaming (simplest)
    response = await client.chat(messages)
    print(response)  # Full response as string
    
    # Streaming (for real-time UI updates)
    stream = await client.stream_chat(messages)
    async for kind, chunk in stream:
        if kind == "content":
            print(chunk, end="")
```

#### Tool-Based Agent Workflow

```python
from neuralcore.agents import AgentRunner
from neuralcore.actions.registry import ActionRegistry

# Register tools (e.g., file operations, LLM calls, etc.)
registry = ActionRegistry()
registry.register("file_read", FileReadAction())
registry.register("web_search", WebSearchAction())

agent_runner = AgentRunner(
    client=client,
    registry=registry,
    max_iterations=25,
)

async def task_flow():
    async for event, payload in agent_runner.run(
        "Summarize this file: /path/to/report.pdf",
        tools=registry,
        system_prompt="You are a helpful assistant with file access."
    ):
        # Stream events to UI
        yield event, payload

# Run the workflow
async for event, payload in task_flow():
    print(f"[{event}] {payload}")
```

#### Synchronous (Blocking) Mode

```python
# For simpler scripts or REPL usage
result = client.ask_sync("What is 2+2?")
print(result)  # "4"
```

---

## 🏗️ Architecture Overview

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `LLMClient` | `src/neuralcore/core/client.py` | Main LLM interface (streaming, chat, tools, embeddings) |
| `AgentRunner` | `src/neuralcore/agents/agent_core.py` | Self-reflective agent execution with final summaries |
| `ActionRegistry` | `src/neuralcore/actions/registry.py` | Tool discovery and schema management |
| `DynamicActionManager` | `src/neuralcore/actions/manager.py` | Runtime action loading/unloading |
| `ContextManager` | (implicit) | Token-aware context with chunking |

### Module Structure

```
NeuralCore/
├── src/neuralcore/
│   ├── core/              # LLM client, prompt builder, client factory
│   │   ├── __init__.py    # Exports: CoreAgent, NeuralProcessor, CognitionEngine
│   │   └── client.py      # Main streaming/chat/tool interface
│   │
│   ├── actions/           # Dynamic tool registry and lifecycle
│   │   ├── actions.py     # Action base class and types
│   │   ├── manager.py     # DynamicActionManager (load/unload tools)
│   │   └── registry.py    # ActionRegistry (schema + executor mapping)
│   │
│   ├── agents/            # Self-reflective agent execution
│   │   └── agent_core.py  # AgentRunner with stuck detection & summaries
│   │
│   ├── utils/             # Shared utilities
│   │   ├── logger.py      # Structured logging (async + sync)
│   │   ├── config.py      # Configuration management
│   │   ├── text_tokenizer.py  # Tokenization for chunking
│   │   ├── llm_tools.py   # LLM-specific helpers
│   │   ├── tool_browser.py   # Tool discovery/registration
│   │   └── exceptions_handler.py  # Error handling (e.g., ConfirmationRequired)
│   │
│   └── cognition/         # Cognitive memory and state
│       ├── __init__.py    # Exports: CognitiveLayer, MemoryManager
│       └── memory.py      # Persistent context state
│
├── pyproject.toml        # Project metadata + dependencies
├── .gitignore            # Git ignore patterns
└── README.md             # This file
```

---

## 🔄 Execution Flow (Agent-Based)

```
1. User Prompt → Added to ContextManager
   ↓
2. LLM Stream Chat (with tools) → Returns Queue of events
   ↓
3. Event Processing:
   ├── "content"      → Text chunks streamed to UI
   ├── "tool_delta"   → Tool argument deltas
   ├── "tool_complete"→ Full tool results
   └── "finish"       → Final state (success/error/cancelled)
   ↓
4. No Tool Calls Detected? → Trigger Self-Reflection (if stuck)
   ↓
5. Task Complete Signal? → Generate Markdown Summary Report
```

---

## 🧪 Testing

```bash
# Run tests with uv
uv run pytest

# Or install dev dependencies first
pip install -e ".[dev]"
pytest
```

---

## 📚 Dependencies (Runtime)

From `pyproject.toml`:

- `numpy` – Numerical operations & embedding vector math
- `openai` – LLM API client (`AsyncOpenAI`, `OpenAI`)
- `aiofiles` – Async file I/O utilities
- `rapidfuzz` – Fuzzy string matching
- `tokenizers` – Model tokenization support

### Dev Dependencies

- `pytest`, `pytest-asyncio` – Testing framework
- `black` – Code formatting

---

## 🔧 LLMClient API Reference

### Async Methods (Non-blocking)

| Method | Purpose | Returns |
|--------|---------|---------|
| `ask()` | Single-turn Q&A (streaming mode) | `asyncio.Queue` or `str` |
| `chat()` | Multi-turn conversation (non-streaming) | `str` |
| `stream_chat()` | Incremental chunk streaming | `asyncio.Queue` |
| `call_tools()` | Async tool invocation | `List[ChatCompletionMessageToolCall]` or `None` |
| `describe_image()` | Vision/image analysis | `str` |
| `fetch_embedding()` | Vector embedding generation | `np.ndarray` or `None` |

### Sync Methods (Blocking)

| Method | Purpose | Returns |
|--------|---------|---------|
| `ask_sync()` | Single-turn Q&A (sync mode) | `str` |
| `chat_sync()` | Multi-turn conversation (sync) | `str` |
| `call_tools_sync()` | Sync tool invocation | `List[ChatCompletionMessageToolCall]` or `None` |
| `describe_image_sync()` | Vision/image analysis (sync) | `str` |

---

## 🧠 AgentRunner Key Features

- **Self-Reflection Loop** – Detects when stuck, asks LLM to "Review & Reflect"
- **Final Summary Report** – Markdown output after task completion:
  ```markdown
  # 🏁 Agent Execution Report
  **Original Task:** ...
  **Status:** SUCCESS/ERROR/CANCELLED
  **Iterations:** X
  **Tool Calls:** Y
  ## 🛠️ Tool Usage
  | Tool | Args | Result |
  | --- | --- | --- |
  | web_search | {"q": "..." } | "Found 3 results..." |
  ```
- **Hidden Completion Marker** – `[FINAL_ANSWER_COMPLETE]` signal for clean CLI exit

---

## 📝 License

MIT

---

**NeuralCore v0.1.0** — Built for production-ready AI orchestration.
