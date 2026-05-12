from pathlib import Path
from typing import Any, Dict, Optional

from neuralcore.clients.factory import get_clients
from neuralcore.agents.core import Agent


class AgentFactory:
    """
    The single source of truth for building Agent instances.
    Resolves all config into concrete values before constructing the Agent.
    """

    def __init__(self, loader):
        self.loader = loader

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
        All config resolution happens here — Agent receives ready-to-use values.
        """
        if not isinstance(config, dict) or not config:
            raise ValueError(f"Agent config for '{agent_id}' is empty or not found")

        final_config = (
            dict(config_override) if config_override is not None else dict(config)
        )

        # Resolve client
        clients = get_clients()
        client_name = final_config.get("client", "main")
        if client_name not in clients:
            raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
        client = clients[client_name]

        # Resolve config sections for infrastructure
        app_config = self.loader.get_app_config() if self.loader else {}
        embeddings_config = (
            self.loader.config.get("embeddings", {}) if self.loader else {}
        )

        # Tool loading
        tools_to_load = final_config.get("tool_sets", [])
        if tools_to_load and self.loader:
            self.loader.load_tool_sets(sets_to_load=tools_to_load)

        # Construct Agent with fully resolved values
        agent = Agent(
            agent_id=final_config.get("id", agent_id),
            name=final_config.get("name", f"Agent-{agent_id}"),
            description=final_config.get("description", ""),
            client=client,
            app_root=app_root,
            system_prompt=final_config.get(
                "system_prompt", getattr(client, "system_prompt", "")
            ),
            max_iterations=final_config.get("max_iterations", 25),
            temperature=final_config.get("temperature", 0.3),
            max_tokens=final_config.get("max_tokens", 12048),
            config=final_config,
            loader=self.loader,
            sub_agent=sub_agent,
            app_config=app_config,
            embeddings_config=embeddings_config,
        )

        print(f"[INFO] Agent '{agent.name}' created via AgentFactory")
        return agent
