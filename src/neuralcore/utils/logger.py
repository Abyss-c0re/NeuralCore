import asyncio
import logging

# Logging configuration
LOG = True
LOG_LEVEL = "info"  # Possible values: debug, info, warning, error, critical
LOG_TO_FILE = True
LOG_TO_UI = False

class Logger:
    _logger = None
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    _logging_enabled = LOG
    _use_fancy_print = LOG_TO_UI
    _use_file_handler = LOG_TO_FILE

    @classmethod
    def get_logger(
        cls,
        name: str = "neuralcore",
        level: str = LOG_LEVEL,
        log_file: str = "neuralcore.log",
        renderer=None  # Optional renderer argument
    ) -> logging.Logger:
        """
        Returns a logger instance with a selectable log level, logging to file and/or TUI.
        """
        if cls._logger is None:
            cls._logger = logging.getLogger(name)
            log_level = cls.LEVELS.get(level.lower(), logging.INFO)
            cls._logger.setLevel(log_level)

            # Add file handler
            if cls._use_file_handler and cls._logging_enabled:
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                cls._logger.addHandler(file_handler)

            # Add FancyPrintHandler if logging is enabled and configured
            if cls._use_fancy_print and cls._logging_enabled:
                cls._logger.addHandler(FancyPrintHandler(renderer=renderer))  # Pass renderer here

        return cls._logger


class FancyPrintHandler(logging.Handler):
    """
    Logging handler that prints to the TUI via the unified Rendering class,
    streaming messages with Markdown support. Falls back to normal print if no renderer is provided.
    """

    def __init__(self, renderer=None):
        super().__init__()
        # If no renderer is provided, use normal print.
        self.renderer = renderer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Format message and wrap with color Markdown
            msg = self.format(record)
            formatted_msg = f"[{record.levelname}] {msg}"
            colored_msg = self._apply_color(formatted_msg, record.levelno)

            if self.renderer:
                # If renderer is provided, use it
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Schedule streaming without blocking
                    asyncio.create_task(self.renderer.stream_message(colored_msg, role="system"))
                else:
                    # Fallback: run a temporary loop
                    asyncio.run(self.renderer.stream_message(colored_msg, role="system"))
            else:
                # If no renderer, fallback to normal print
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
        # Markdown-style coloring
        return f"[{color}]{msg}[/]"