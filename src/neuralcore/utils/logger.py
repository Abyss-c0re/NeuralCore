import asyncio
import logging

# Logging configuration
LOG = True
LOG_LEVEL = "info"
LOG_TO_FILE = True
LOG_TO_UI = False


class Logger:
    _logger = None
    _renderer = None  # <-- store renderer globally

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
        renderer=None,
    ) -> logging.Logger:
        """
        Returns a singleton logger instance.
        If a renderer is passed once, it will be stored and reused.
        """

        # Store renderer if provided
        if renderer is not None:
            cls._renderer = renderer

        if cls._logger is None:
            cls._logger = logging.getLogger(name)

            log_level = cls.LEVELS.get(level.lower(), logging.INFO)
            cls._logger.setLevel(log_level)

            # Prevent propagation to root logger
            cls._logger.propagate = False

            # Avoid duplicate handlers
            if not cls._logger.handlers:

                if cls._use_file_handler and cls._logging_enabled:
                    file_handler = logging.FileHandler(log_file)
                    file_formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                    file_handler.setFormatter(file_formatter)
                    cls._logger.addHandler(file_handler)

                if cls._use_fancy_print and cls._logging_enabled:
                    cls._logger.addHandler(
                        FancyPrintHandler(renderer=cls._renderer)
                    )

        return cls._logger


class FancyPrintHandler(logging.Handler):
    """
    Logging handler that prints to the TUI via the renderer.
    Falls back to normal print if renderer is unavailable.
    """

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