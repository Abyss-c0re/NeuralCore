import asyncio
import logging
from pathlib import Path
from neuralcore.utils.config import ConfigLoader
from collections import deque


class Logger:
    _logger = None
    _renderer = None  # optional TUI renderer
    _config = None  # caching the logging config
    _memory_handler = None  # in-memory log storage

    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    @classmethod
    def _load_config(cls) -> dict:
        """Lazy-load logging config from ConfigLoader."""
        if cls._config is None:
            cls._config = ConfigLoader().get_logging_config()
        return cls._config

    @classmethod
    def get_logger(
        cls,
        name: str = "neuralcore",
        renderer=None,
        memory_level="debug",
        memory_size=1000,
    ) -> logging.Logger:
        """
        Returns a singleton logger instance.

        Args:
            name: logger name
            renderer: optional TUI renderer
            memory_level: max log level to store in memory
            memory_size: max number of entries to keep in memory
        """
        if renderer is not None:
            cls._renderer = renderer

        config = cls._load_config()
        LOG = config.get("logging_enabled", True)
        LOG_LEVEL = config.get("log_level", "info")
        LOG_TO_FILE = config.get("log_to_file", True)
        LOG_TO_UI = config.get("log_to_ui", False)
        LOG_FILE = Path(
            config.get("log_file", Path.home() / ".neuralcore" / "neuralcore.log")
        )

        if cls._logger is None:
            cls._logger = logging.getLogger(name)
            cls._logger.setLevel(cls.LEVELS.get(LOG_LEVEL.lower(), logging.INFO))
            cls._logger.propagate = False

            if not cls._logger.handlers and LOG:
                if LOG_TO_FILE:
                    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(LOG_FILE)
                    file_formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                    file_handler.setFormatter(file_formatter)
                    cls._logger.addHandler(file_handler)

                if LOG_TO_UI:
                    cls._logger.addHandler(FancyPrintHandler(renderer=cls._renderer))

            # Setup memory handler
            cls._memory_handler = MemoryHandler(
                level=cls.LEVELS.get(memory_level.lower(), logging.DEBUG),
                max_entries=memory_size,
            )
            cls._logger.addHandler(cls._memory_handler)

        return cls._logger

    @classmethod
    def get_log_data(cls, level="info", max_entries=50):
        """
        Retrieve last `max_entries` log records at specified level or higher.

        Args:
            level: log level to retrieve ("debug", "info", etc.)
            max_entries: number of recent entries to return

        Returns:
            List of log strings
        """
        if not cls._memory_handler:
            return []

        lvl_no = cls.LEVELS.get(level.lower(), logging.INFO)
        return cls._memory_handler.get_logs(level=lvl_no, max_entries=max_entries)


class MemoryHandler(logging.Handler):
    """
    Stores log records in memory with filtering by level.
    """

    def __init__(self, level=logging.DEBUG, max_entries=1000):
        super().__init__(level=level)
        self.max_entries = max_entries
        self._logs = deque(maxlen=max_entries)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= self.level:
            msg = self.format(record)
            self._logs.append((record.levelno, msg))

    def get_logs(self, level=logging.INFO, max_entries=50):
        # Filter logs at exact level (not above)
        filtered = [msg for lvl, msg in self._logs if lvl == level]
        return filtered[-max_entries:]  # last `max_entries`


class FancyPrintHandler(logging.Handler):
    """Logging handler that prints to a TUI renderer or fallback to print."""

    def __init__(self, renderer=None):
        super().__init__()
        self.renderer = renderer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            formatted_msg = f"[{record.levelname}] {msg}"
            colored_msg = self._apply_color(formatted_msg, record.levelno)

            if self.renderer:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    asyncio.create_task(
                        self.renderer.stream_message(colored_msg, role="system")
                    )
                else:
                    asyncio.run(
                        self.renderer.stream_message(colored_msg, role="system")
                    )
            else:
                print(colored_msg)

        except Exception:
            self.handleError(record)

    def _apply_color(self, msg: str, level: int) -> str:
        color_map = {
            logging.DEBUG: "blue",
            logging.INFO: "green",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
            logging.CRITICAL: "purple",
        }
        color = color_map.get(level, "white")
        return f"[{color}]{msg}[/]"
