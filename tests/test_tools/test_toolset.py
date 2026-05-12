"""Test toolset for NeuralCore unit tests."""

from neuralcore.actions.registry import tool


@tool(
    "TestTools",
    tags=["test", "echo"],
    name="echo_tool",
    description="Echoes input back for testing.",
)
async def echo_tool(message: str) -> str:
    return f"Echo: {message}"


@tool(
    "TestTools",
    tags=["test", "math"],
    name="add_numbers",
    description="Adds two numbers together.",
)
async def add_numbers(a: int, b: int) -> str:
    return f"Result: {a + b}"


@tool(
    "TestTools",
    tags=["test", "read"],
    name="mock_read_file",
    description="Mock file reader for tests.",
)
async def mock_read_file(file_path: str) -> str:
    return f"Contents of {file_path}: [mock file content for testing]"
