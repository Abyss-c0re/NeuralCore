"""Test the mock LLM server works correctly with the openai library."""

import json
import pytest
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
)  # Explicit typing for OpenAI client compatibility


@pytest.mark.asyncio(loop_scope="session")
async def test_non_streaming_chat(mock_server):
    """Test basic non-streaming chat completion."""
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    resp = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "Hello world"}],
        stream=False,
    )
    assert resp.choices[0].message.content is not None
    assert len(resp.choices[0].message.content) > 0
    assert resp.choices[0].finish_reason == "stop"


@pytest.mark.asyncio(loop_scope="session")
async def test_streaming_chat(mock_server):
    """Test streaming chat completion (SSE)."""
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    stream = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "Tell me something"}],
        stream=True,
    )
    chunks = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    full_text = "".join(chunks)
    assert len(full_text) > 0


@pytest.mark.asyncio(loop_scope="session")
async def test_streaming_tool_calls(mock_server):
    """Test streaming with tool calls."""
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    # Explicit type annotation satisfies OpenAI client's ChatCompletionToolUnionParam
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "echo_tool",
                "description": "Echo the input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"}
                    },
                    "required": ["message"],
                },
            },
        }
    ]

    stream = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "Please read this file for me"}],
        tools=tools,
        stream=True,
    )

    tool_call_name = ""
    tool_call_args = ""
    finish_reason = None

    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        if choice.delta.tool_calls:
            for tc in choice.delta.tool_calls:
                # Guard against Optional[ChatCompletionMessageToolCallFunction] in delta chunks
                if tc.function is not None:
                    if tc.function.name:
                        tool_call_name = tc.function.name
                    if tc.function.arguments:
                        tool_call_args += tc.function.arguments

    assert tool_call_name == "echo_tool"
    assert len(tool_call_args) > 0
    assert finish_reason == "tool_calls"


@pytest.mark.asyncio(loop_scope="session")
async def test_non_streaming_tool_calls(mock_server):
    """Test non-streaming tool call response."""
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    # Explicit type annotation satisfies OpenAI client's ChatCompletionToolUnionParam
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        }
    ]

    resp = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "Write hello to file.txt"}],
        tools=tools,
        stream=False,
    )

    assert resp.choices[0].message.tool_calls is not None
    tc = resp.choices[0].message.tool_calls[0]
    # Type narrowing for ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall union
    assert tc.type == "function"
    assert tc.function.name == "write_file"
    assert resp.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio(loop_scope="session")
async def test_intent_classification(mock_server):
    """Test the engine classifies intents correctly."""
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    # CASUAL intent
    resp = await client.chat.completions.create(
        model="mock-model",
        messages=[
            {"role": "system", "content": "Classify as CASUAL or TASK"},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    assert content is not None
    assert "CASUAL" in content

    # TASK intent
    resp = await client.chat.completions.create(
        model="mock-model",
        messages=[
            {"role": "system", "content": "Classify as CASUAL or TASK"},
            {"role": "user", "content": "Read the config file and analyze it"},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    assert content is not None
    assert "TASK" in content


@pytest.mark.asyncio(loop_scope="session")
async def test_task_decomposition(mock_server):
    """Test the engine returns valid JSON for task decomposition."""
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    resp = await client.chat.completions.create(
        model="mock-model",
        messages=[
            {"role": "system", "content": "You are a task decomposition expert."},
            {
                "role": "user",
                "content": "Break this request into actionable steps: build a test suite",
            },
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    assert content is not None
    plan = json.loads(content)
    assert "steps" in plan
    assert len(plan["steps"]) > 0


@pytest.mark.asyncio(loop_scope="session")
async def test_embeddings(mock_server):
    """Test the embeddings endpoint."""
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    resp = await client.embeddings.create(
        model="mock-embed",
        input="Test embedding text",
    )
    assert len(resp.data) == 1
    assert len(resp.data[0].embedding) == 384


@pytest.mark.asyncio(loop_scope="session")
async def test_request_logging(mock_server):
    """Test that the server logs requests."""
    initial_count = len(mock_server.request_log)
    client = AsyncOpenAI(
        base_url=mock_server.base_url,
        api_key="test-key",
        timeout=30.0,
    )
    await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "log test"}],
        stream=False,
    )
    assert len(mock_server.request_log) > initial_count
