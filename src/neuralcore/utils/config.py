import os
import yaml
from pathlib import Path


class ConfigLoader:
    """
    Loads config from YAML + ENV hybrid with secret resolution.
    Provides defaults if nothing is specified.
    """

    DEFAULT_API_KEY = "not-needed"
    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(self, cli_path: str | None = None):
        self.config_path = self._resolve_config_path(cli_path)
        self.config = self._load_yaml(self.config_path)

    def _resolve_config_path(self, cli_path: str | None) -> Path | None:
        """Determine the config file path from CLI, ENV, or default locations."""
        if cli_path:
            return Path(cli_path)

        if env_path := os.getenv("APP_CONFIG"):
            return Path(env_path)

        local_path = Path(self.DEFAULT_CONFIG_FILE)
        if local_path.exists():
            return local_path

        # Corrected default global path
        global_path = Path.home() / ".neuralcore" / self.DEFAULT_CONFIG_FILE
        if global_path.exists():
            return global_path

        # If no config found, return None
        return None

    def _load_yaml(self, path: Path | None) -> dict:
        """Load YAML config file, or return empty dict if not found."""
        if path is None or not path.exists():
            return {}
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"YAML config must be a dict at top level, got {type(data)}")
            return data

    def get_client_config(self, client_name: str) -> dict:
        """Return config dict for a specific client."""
        clients = self.config.get("clients", {})
        return clients.get(client_name, {})

    def resolve_secret(self, client_name: str) -> str:
        """
        Resolve API key for the given client.
        Priority: ENV > YAML > default ("not-needed")
        """
        cfg = self.get_client_config(client_name)

        # Check for ENV override first
        if env_key := cfg.get("api_key_env"):
            if val := os.getenv(env_key):
                return val

        # Then YAML
        if val := cfg.get("api_key"):
            return val

        # Fallback
        return self.DEFAULT_API_KEY

    def get_system_prompt(self) -> str:
        """Return system prompt from config or default."""
        return self.config.get(
            "system_prompt",
            "You are terminal chat assistant, use provided tools if needed",
        )

    def get_agent_config(self, agent_name: str) -> dict:
        """Return config for an agent (like max_iterations)"""
        agents = self.config.get("agents", {})
        return agents.get(agent_name, {})

    def get_app_config(self) -> dict:
        """Return runtime config for the single interactive app"""
        return self.config.get("app", {})

    def get_logging_config(self) -> dict:
        """
        Return logging-related configuration with defaults.
        Includes:
          - logging_enabled
          - log_level
          - log_to_file
          - log_to_ui
          - log_file
        """
        app_cfg = self.get_app_config()
        log_dir_default = Path.home() / ".neuralcore"

        # Ensure the default directory exists
        log_dir_default.mkdir(parents=True, exist_ok=True)

        log_file = app_cfg.get("log_file", log_dir_default / "neuralcore.log")

        # Expand ~ if user provided path like '~/logs/app.log'
        if isinstance(log_file, str):
            log_file = Path(os.path.expanduser(log_file))

        return {
            "logging_enabled": app_cfg.get("logging_enabled", True),
            "log_level": app_cfg.get("log_level", "info"),
            "log_to_file": app_cfg.get("log_to_file", True),
            "log_to_ui": app_cfg.get("log_to_ui", False),
            "log_file": log_file,
        }
