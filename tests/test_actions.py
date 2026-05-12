"""Unit tests for neuralcore.actions -- Action, ActionSet, ActionRegistry."""
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuralcore.actions.actions import Action, ActionSet


class TestAction:
    def test_creation(self):
        async def my_tool(message: str) -> str:
            return f"got: {message}"

        a = Action(
            name="test_tool",
            description="A test tool",
            parameters={"message": {"type": "string", "description": "input"}},
            executor=my_tool,
            required=["message"],
            tags=["test"],
        )
        assert a.name == "test_tool"
        assert a.description == "A test tool"
        assert a.type == "tool"
        assert a.usage_count == 0
        assert "test" in a.tags

    def test_schema_generation(self):
        async def my_tool(x: int, y: int) -> str:
            return str(x + y)

        a = Action(
            name="add",
            description="Add numbers",
            parameters={
                "x": {"type": "integer", "description": "first number"},
                "y": {"type": "integer", "description": "second number"},
            },
            executor=my_tool,
            required=["x", "y"],
        )
        schema = a._raw_schema
        assert schema["name"] == "add"
        assert "x" in schema["parameters"]["properties"]
        assert "y" in schema["parameters"]["properties"]
        assert "required" in schema["parameters"]

    def test_agent_excluded_from_schema(self):
        async def tool_with_agent(agent, file_path: str) -> str:
            return "ok"

        a = Action(
            name="agent_tool",
            description="Needs agent",
            parameters={
                "agent": {"type": "object"},
                "file_path": {"type": "string"},
            },
            executor=tool_with_agent,
            required=["file_path"],
        )
        assert "agent" not in a._raw_schema["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execution(self):
        async def my_tool(message: str) -> str:
            return f"echo: {message}"

        a = Action(
            name="echo",
            description="Echo tool",
            parameters={"message": {"type": "string"}},
            executor=my_tool,
            required=["message"],
        )
        result = await a(message="hello")
        assert result == "echo: hello"
        assert a.usage_count == 1

    def test_invalid_action_type_raises(self):
        with pytest.raises(ValueError, match="action_type must be"):
            Action(
                name="bad",
                description="bad",
                parameters={},
                executor=lambda: None,
                action_type="invalid",
            )


class TestActionSet:
    def test_creation_and_add(self):
        aset = ActionSet(name="TestSet")
        async def tool1(x: str) -> str:
            return x

        a = Action(
            name="tool1",
            description="Tool 1",
            parameters={"x": {"type": "string"}},
            executor=tool1,
        )
        aset.add(a)
        assert len(aset.actions) == 1
        assert aset.get_executor("tool1") is not None

    def test_get_llm_tools(self):
        aset = ActionSet(name="TestSet")
        async def tool1(x: str) -> str:
            return x

        a = Action(
            name="tool1",
            description="Tool 1",
            parameters={"x": {"type": "string"}},
            executor=tool1,
        )
        aset.add(a)
        llm_tools = aset.get_llm_tools()
        assert len(llm_tools) == 1
        assert llm_tools[0]["type"] == "function"
        assert llm_tools[0]["function"]["name"] == "tool1"


class TestActionRegistry:
    def _make_registry(self):
        from neuralcore.actions.registry import ActionRegistry
        reg = ActionRegistry()
        aset = ActionSet(name="MySet")
        async def tool1(msg: str) -> str:
            return msg

        a = Action(
            name="greet",
            description="Greet someone hello world",
            parameters={"msg": {"type": "string"}},
            executor=tool1,
            tags=["greeting", "hello"],
        )
        aset.add(a)
        reg.register_set("MySet", aset)
        return reg

    def test_register_and_search(self):
        reg = self._make_registry()
        results = reg.search("hello greeting", limit=5)
        assert len(results) > 0
        assert results[0][0].name == "greet"

    def test_list_all_tools(self):
        reg = self._make_registry()
        tools = reg.list_all_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "greet"

    def test_register_duplicate_set_raises(self):
        reg = self._make_registry()
        with pytest.raises(ValueError, match="already exists"):
            reg.register_set("MySet", ActionSet(name="MySet"))

    def test_search_empty_query(self):
        reg = self._make_registry()
        assert reg.search("") == []

    def test_list_all_with_schema(self):
        reg = self._make_registry()
        tools = reg.list_all_tools(include_schema=True)
        assert "parameters" in tools[0]

    def test_list_all_as_llm_format(self):
        reg = self._make_registry()
        tools = reg.list_all_tools(as_llm_format=True)
        assert len(tools) == 1
        # In LLM format, tools have type: "function"
        t = tools[0]
        assert "type" in t or "name" in t