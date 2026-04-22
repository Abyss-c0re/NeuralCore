import asyncio
import logging
import queue
from pathlib import Path
from collections import deque

import aiofiles
from neuralcore.utils.config import ConfigLoader


class Logger:
    _logger = None
    _renderer = None
    _config = None
    _memory_handler = None

    # Async file logging state
    _file_queue: queue.Queue | None = None
    _file_writer_task: asyncio.Task | None = None
    _log_file_path: Path | None = None

    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    @classmethod
    def _load_config(cls) -> dict:
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
                    cls._log_file_path = LOG_FILE
                    cls._file_queue = queue.Queue(maxsize=5000)
                    cls._logger.addHandler(AsyncQueueHandler(cls._file_queue))
                    cls._start_file_writer()  # ← robust start

                if LOG_TO_UI:
                    cls._logger.addHandler(FancyPrintHandler(renderer=cls._renderer))

            cls._memory_handler = MemoryHandler(
                level=cls.LEVELS.get(memory_level.lower(), logging.DEBUG),
                max_entries=memory_size,
            )
            cls._logger.addHandler(cls._memory_handler)

        return cls._logger

    @classmethod
    def _start_file_writer(cls):
        if cls._file_writer_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                cls._file_writer_task = asyncio.create_task(cls._file_writer())
        except RuntimeError:
            pass

    @classmethod
    async def _file_writer(cls):
        if not cls._log_file_path or not cls._file_queue:
            return

        path = cls._log_file_path
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(path, mode="a", encoding="utf-8") as f:
                while True:
                    try:
                        line = await asyncio.to_thread(
                            cls._file_queue.get, block=True, timeout=0.5
                        )
                        if line is None:  # shutdown signal
                            await f.flush()
                            break
                        await f.write(line + "\n")
                        await f.flush()  # ← THIS was missing!
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    except Exception as exc:
                        print(f"[Logger] File writer error: {exc}")
                        await asyncio.sleep(1)
        except Exception as exc:
            print(f"[Logger] Failed to open log file {path}: {exc}")

    @classmethod
    async def shutdown(cls):
        """Clean shutdown: stops the file writer gracefully."""
        if cls._file_queue:
            cls._file_queue.put_nowait(None)  # sentinel
        if cls._file_writer_task:
            try:
                await asyncio.wait_for(cls._file_writer_task, timeout=2.0)
            except asyncio.TimeoutError:
                cls._file_writer_task.cancel()
        cls._file_writer_task = None

    @classmethod
    def get_log_data(cls, level="info", max_entries=50):
        if not cls._memory_handler:
            return []
        lvl_no = cls.LEVELS.get(level.lower(), logging.INFO)
        return cls._memory_handler.get_logs(level=lvl_no, max_entries=max_entries)


# ===================== Handlers (unchanged) =====================


class MemoryHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG, max_entries=1000):
        super().__init__(level=level)
        self.max_entries = max_entries
        self._logs = deque(maxlen=max_entries)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= self.level:
            msg = self.format(record)
            self._logs.append((record.levelno, msg))

    def get_logs(self, level=logging.INFO, max_entries=50):
        filtered = [msg for lvl, msg in self._logs if lvl == level]
        return filtered[-max_entries:]


class AsyncQueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.queue = q
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.queue.put_nowait(msg)
            Logger._start_file_writer()
        except queue.Full:
            pass
        except Exception:
            self.handleError(record)


class FancyPrintHandler(logging.Handler):
    def __init__(self, renderer=None):
        super().__init__()
        self.renderer = renderer
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

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
                    print(colored_msg)
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
