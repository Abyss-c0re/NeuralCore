"""
NeuralCore - Modular AI Agent Framework

Core components for building autonomous agents with tool use,
multi-agent coordination, knowledge management, and workflows.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neuralcore")
except PackageNotFoundError:
    __version__ = "0.1.0"

# =============================================================================
# Public API - Core Components
# =============================================================================

# Agents
from .agents.core import Agent
from .agents.factory import AgentFactory
from .agents.state import AgentState

# Clients
from .clients.factory import get_clients, get_client_factory
from .clients.client import LLMClient

# Configuration & Logging
from .utils.config import get_loader, ConfigLoader
from .utils.logger import Logger
from .utils.prompt_builder import PromptBuilder

# Workflows
from .workflows.engine import WorkflowEngine
from .workflows.registry import workflow

# Actions / Tools
from .actions.registry import registry, tool, sequenced
from .actions.manager import (
    ActionRegistry,
    DynamicActionManager,
    ToolBrowser,
)

# Bridge (for external control / WebSocket)
from .bridge.websocket import WebSocketBridge

__all__ = [
    "__version__",
    # Agents
    "Agent",
    "AgentFactory",
    "AgentState",
    # Clients
    "get_clients",
    "get_client_factory",
    "LLMClient",
    # Config & Logging
    "get_loader",
    "ConfigLoader",
    "Logger",
    "PromptBuilder",
    # Workflows
    "WorkflowEngine",
    "workflow",
    # Actions
    "registry",
    "tool",
    "sequenced",
    "ActionRegistry",
    "DynamicActionManager",
    "ToolBrowser",
    # Bridge
    "WebSocketBridge",
]