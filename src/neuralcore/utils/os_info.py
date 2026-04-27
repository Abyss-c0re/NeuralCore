import re
import platform
from neuralcore.utils.logger import Logger


logger = Logger.get_logger()


def get_os_info() -> str:
    """Detect OS and return a nice string (sync - fast enough)."""
    try:
        if platform.system() == "Linux":
            try:
                with open("/etc/os-release", "r", encoding="utf-8") as f:
                    content = f.read()

                pretty_match = re.search(r'^PRETTY_NAME="?(.+?)"?$', content, re.MULTILINE)
                if pretty_match:
                    return pretty_match.group(1).strip()

                id_match = re.search(r'^ID="?(.+?)"?$', content, re.MULTILINE)
                version_match = re.search(r'^VERSION_ID="?(.+?)"?$', content, re.MULTILINE)

                distro_id = id_match.group(1).strip() if id_match else "unknown"
                version = f" {version_match.group(1).strip()}" if version_match else ""
                return f"{distro_id.capitalize()}{version} Linux"

            except Exception as e:
                logger.debug(f"/etc/os-release detection failed: {e}")

        # Generic fallback
        system = platform.system()
        release = platform.release()
        machine = platform.machine()
        node = platform.node()

        if system == "Darwin":
            return f"macOS {release} ({machine})"
        elif system == "Windows":
            return f"Windows {release} ({machine})"
        else:
            return f"{system} {release} ({machine}) on {node}"

    except Exception as e:
        logger.warning(f"OS detection failed completely: {e}")
        return f"Unknown OS ({platform.system()} {platform.release()})"