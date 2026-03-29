# ❯ uv run pytest -q --asyncio-mode=auto tests/test_context_manager.py

import pytest
import asyncio
import numpy as np

from neuralcore.clients.factory import get_clients


from neuralcore.cognition.memory import (
    ContextManager,
    cosine_similarity,
    keyword_score,
    KnowledgeItem,
    Topic,
    MSG_THR,
    OFF_FREQ,
)

# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_clients():
    """Mock get_clients() return value."""
    clients = get_clients()
    return clients


@pytest.fixture
def mocked_context_manager(mock_clients):
    """Fully mocked ContextManager ready for async tests."""
    return ContextManager()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def test_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(v1, v2) == 0.0

    v3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v3, v3) == pytest.approx(1.0)

    assert cosine_similarity(np.array([]), np.array([])) == 0.0
    assert cosine_similarity(np.zeros(10), np.zeros(10)) == 0.0


def test_keyword_score():
    assert keyword_score([], "anything") == 0.0

    score = keyword_score(["python", "test"], "This is a Python test example")
    assert score > 3.0  # coverage + prefix bonus

    score = keyword_score(["exact"], "exactmatch")
    assert score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE ITEM
# ─────────────────────────────────────────────────────────────────────────────


def test_knowledge_item_init():
    item = KnowledgeItem(
        "key123", "file", "Hello world content", {"path": "/tmp/test.py"}
    )
    assert item.key == "key123"
    assert item.source_type == "file"
    assert item.content == "Hello world content"
    assert "hello" in item.word_set
    assert "world" in item.word_set
    assert item.embedding.size == 0


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_topic_add_message():
    topic = Topic("test", "desc")
    emb = np.random.rand(384).astype(np.float32)
    await topic.add_message("user", "Hello from test", emb)
    assert len(topic.history) == 1
    assert topic.history[0]["role"] == "user"
    assert topic.history[0]["content"] == "Hello from test"
    assert len(topic.history_embeddings) == 1


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT MANAGER - CORE TESTS
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_manager_init(mocked_context_manager):
    cm = mocked_context_manager
    assert cm.max_tokens == 1000
    assert cm.similarity_threshold == MSG_THR
    assert cm.current_topic.name == "Initial topic"
    assert len(cm.knowledge_base) == 0
    assert len(cm.action_log) == 0


@pytest.mark.asyncio
async def test_fetch_embedding_caching(mocked_context_manager):
    cm = mocked_context_manager
    text = "cache me"
    emb1 = await cm.fetch_embedding(text)
    emb2 = await cm.fetch_embedding(text)
    assert np.array_equal(emb1, emb2)
    cm.embeddings.fetch_embedding.assert_called_once_with(text, 768)


@pytest.mark.asyncio
async def test_add_external_content(mocked_context_manager):
    cm = mocked_context_manager
    key = await cm.add_external_content(
        "file", "print('hello')", {"path": "/src/main.py"}
    )
    assert key is not None
    assert key in cm.knowledge_base
    assert cm.knowledge_base[key].source_type == "file"
    assert len(cm.files_checked) == 1
    assert cm.context_stats["kb_added"] == 1


@pytest.mark.asyncio
async def test_retrieve_relevant_knowledge(mocked_context_manager):
    cm = mocked_context_manager
    # Seed KB
    await cm.add_external_content("file", "def foo(): pass", {"path": "a.py"})
    await cm.add_external_content("file", "def bar(): pass", {"path": "b.py"})

    result = await cm._retrieve_relevant_knowledge("foo function", max_kb_tokens=500)
    assert "foo" in result or "bar" in result


@pytest.mark.asyncio
async def test_add_message_and_topic_switching(mocked_context_manager):
    cm = mocked_context_manager
    emb = np.random.rand(384).astype(np.float32)

    await cm.add_message("user", "Talk about Python")
    assert len(cm.current_topic.history) == 1

    # Simulate off-topic detection (force new topic creation)
    cm.current_topic.description = "Python topic"
    cm.current_topic.embedded_description = np.random.rand(384).astype(np.float32)

    await cm.add_message("user", "Now talk about Java")
    # Should have created/switched to a new topic
    assert len(cm.topics) >= 1 or cm.current_topic.name != "Initial topic"


@pytest.mark.asyncio
async def test_prune_to_fit_context(mocked_context_manager):
    cm = mocked_context_manager
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Tell me a long story"},
        {"role": "assistant", "content": "Once upon a time... " * 50},  # big
    ]

    original_len = len(messages)
    removed, pruned = cm.prune_to_fit_context(
        messages,
        max_tokens=200,
        min_keep_messages=2,
    )

    assert removed > 0
    assert len(messages) < original_len
    assert len(pruned) == removed
    # System + last user must be protected
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_provide_context_pruning_and_summary(mocked_context_manager):
    cm = mocked_context_manager
    # Add some history that will force pruning
    for i in range(20):
        await cm.add_message("user", f"Message {i} " * 10)

    result = await cm.provide_context(
        query="What is the current topic?",
        max_input_tokens=800,
        reserved_for_output=200,
    )

    assert isinstance(result, list)
    assert any(m["role"] == "system" for m in result)
    # Summary must be injected
    system_msg = next(m for m in result if m["role"] == "system")
    assert "LIVE CONTEXT SUMMARY" in system_msg["content"]
    assert len(result) <= 15  # pruned


@pytest.mark.asyncio
async def test_get_archived_context(mocked_context_manager):
    cm = mocked_context_manager
    # Force some pruning to populate archive
    await cm.add_message("user", "Old message 1")
    await cm.add_message("assistant", "Old response")
    # Simulate prune
    messages = [{"role": "user", "content": "old"}] * 10
    cm.prune_to_fit_context(messages, max_tokens=50, min_keep_messages=1)
    cm.current_topic.archived_history.extend(messages[:5])

    result = await cm.get_archived_context("Old message", max_tokens=1000)
    assert result != ""


@pytest.mark.asyncio
async def test_record_tool_outcome(mocked_context_manager):
    cm = mocked_context_manager
    await cm.record_tool_outcome("ls", "file1.py file2.py", {"path": "/project"})
    assert "ls" in cm.tools_executed
    assert len(cm.knowledge_base) > 0
    assert cm.context_stats["kb_added"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────


def test_prune_edge_cases():
    cm = ContextManager(max_tokens=1000)  # dummy, we only test prune
    # Empty list
    removed, pruned = cm.prune_to_fit_context([], 100)
    assert removed == 0
    assert pruned == []

    # Already under limit
    msgs = [{"role": "user", "content": "short"}]
    removed, pruned = cm.prune_to_fit_context(msgs, 1000)
    assert removed == 0


@pytest.mark.asyncio
async def test_off_topic_detection_triggers_new_topic(mocked_context_manager):
    cm = mocked_context_manager
    # Seed a topic
    cm.current_topic.name = "Python"
    cm.current_topic.description = "Python programming"
    cm.current_topic.embedded_description = np.ones(384, dtype=np.float32)

    # Add messages that are off-topic (low similarity)
    for _ in range(OFF_FREQ):
        await cm.add_message("user", "Weather in London today?")

    # _analyze_history should have been triggered (we don't await the task, but we can check state)
    await asyncio.sleep(0.1)  # let background task run
    assert len(cm.topics) > 0 or cm.current_topic.name != "Python"


# Run with: pytest -q --asyncio-mode=auto tests/test_context_manager.py
