"""Unit tests for neuralcore.cognition.items -- KnowledgeItem and Topic."""
import sys
import asyncio
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuralcore.cognition.items import KnowledgeItem, Topic


class TestKnowledgeItem:
    def test_creation(self):
        item = KnowledgeItem(
            key="test_key",
            source_type="tool_outcome",
            content="This is test content for the knowledge item.",
        )
        assert item.key == "test_key"
        assert item.source_type == "tool_outcome"
        assert "test" in item.word_set
        assert "content" in item.word_set

    def test_metadata(self):
        item = KnowledgeItem(
            key="k1",
            source_type="message",
            content="Hello world",
            metadata={"source": "test"},
        )
        assert item.metadata["source"] == "test"

    def test_default_embedding(self):
        item = KnowledgeItem(key="k", source_type="x", content="y")
        assert isinstance(item.embedding, np.ndarray)
        assert len(item.embedding) == 0

    def test_word_set_extraction(self):
        item = KnowledgeItem(
            key="k",
            source_type="t",
            content="Python is a great programming language",
        )
        assert "python" in item.word_set
        assert "programming" in item.word_set
        assert "language" in item.word_set


class TestTopic:
    def test_creation(self):
        topic = Topic(name="Test Topic", description="A test topic")
        assert topic.name == "Test Topic"
        assert topic.description == "A test topic"
        assert len(topic.history) == 0

    @pytest.mark.asyncio
    async def test_add_message(self):
        topic = Topic(name="Chat")
        emb = np.random.randn(10).astype(np.float32)
        await topic.add_message("user", "Hello", emb, 5)
        assert len(topic.history) == 1
        assert topic.history[0]["role"] == "user"
        assert topic.history[0]["content"] == "Hello"
        assert len(topic.history_embeddings) == 1
        assert topic.history_tokens[0] == 5

    @pytest.mark.asyncio
    async def test_multiple_messages(self):
        topic = Topic(name="Multi")
        emb = np.zeros(10)
        await topic.add_message("user", "Q1", emb, 3)
        await topic.add_message("assistant", "A1", emb, 5)
        await topic.add_message("user", "Q2", emb, 4)
        assert len(topic.history) == 3