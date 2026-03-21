from pathlib import Path
from neuralcore.agents.core import Agent

from neuralcore.utils.config import get_loader

from neuralcore.agents.core import get_clients


def load_agent_from_config(agent_id: str, app_root: Path, loader=None) -> Agent:
    """
    Instantiate an Agent fully from config:
      - Loads client
      - Sets max_iterations, max_tokens, etc.
      - Loads only the tool sets defined in agent's config
    """
    if loader is None:
        loader = get_loader()

    agent_cfg = loader.get_agent_config(agent_id)
    if not agent_cfg:
        raise ValueError(f"No agent config found for '{agent_id}'")

    clients = get_clients()
    client_name = agent_cfg.get("client")
    if client_name not in clients:
        raise ValueError(f"Client '{client_name}' not found for agent '{agent_id}'")
    client = clients[client_name]

    # Instantiate Agent
    agent = Agent(
        agent_id=agent_cfg.get("id", agent_id), loader=loader, app_root=app_root
    )

    # Override client and meta info
    agent.client = client
    agent.name = agent_cfg.get("name", f"Agent-{agent_id}")
    agent.description = agent_cfg.get("description", "")
    agent.max_iterations = agent_cfg.get("max_iterations", 25)
    agent.max_reflections = agent_cfg.get("max_reflections", 4)
    agent.temperature = agent_cfg.get("temperature", 0.3)
    agent.max_tokens = agent_cfg.get("max_tokens", 12048)

    return agent
