"""
NeuralCore Factories
Pure, abstract builders. No YAML parsing here — that stays in ConfigLoader.
Used by NeuralHub for deployment and NeuralLabs for live editing.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from neuralcore.clients.factory import get_clients
from neuralcore.agents.core import Agent
from neuralcore.actions.registry import registry
from neuralcore.actions.manager import DynamicActionManager, ToolBrowser


class AgentFactory:
    """
    Clean factory responsible ONLY for building Agent instances from already-parsed config dicts.
    Keeps NeuralCore abstract and reusable.
    """

    def __init__(self, loader):
        self.loader = loader   # reference back only for tool loading & app_root

    def create_agent(
        self,
        agent_id: str,
        config: Dict[str, Any],
        app_root: Path,
        sub_agent: bool = False,
        config_override: Optional[Dict] = None,
    ) -> Agent:
        """
        Build a fully configured Agent from a parsed config dict.
        This is the clean entry point for external management.
        """
        if not isinstance(config, dict) or not config:
            raise ValueError(f"Agent config for '{agent_id}' is empty or not found")

        # Use override if provided (for sub-agents or live edits)
        final_config = dict(config_override) if config_override is not None else dict(config)

        # Client resolution (still needed here — it's part of agent construction)
        clients = get_clients()
        client_name = final_config.get("client", "main")
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
        client = clients[client_name]

        # Instantiate the Agent with minimal constructor
        agent = Agent(
            agent_id=final_config.get("id", agent_id),
            loader=self.loader,
            app_root=app_root,
            # Pass the already-parsed config directly — no more parsing inside Agent
            config=final_config,          # ← NEW: we will update Agent.__init__ to accept this
            sub_agent=sub_agent,
        )

        # Wire the resolved client
        agent.client = client

        # Apply remaining scalar fields from config
        agent.name = final_config.get("name", f"Agent-{agent_id}")
        agent.description = final_config.get("description", "")
        agent.max_iterations = final_config.get("max_iterations", 25)
        agent.max_reflections = final_config.get("max_reflections", 4)
        agent.temperature = final_config.get("temperature", 0.3)
        agent.max_tokens = final_config.get("max_tokens", 12048)
        agent.system_prompt = final_config.get(
            "system_prompt", getattr(client, "system_prompt", "")
        )

        # Tool loading (still delegated to loader for now)
        tools_to_load = final_config.get("tool_sets", [])
        if tools_to_load:
            self.loader.load_tool_sets(sets_to_load=tools_to_load)

        print(f"[INFO] Agent '{agent.name}' created via AgentFactory")
        return agent