import os
import subprocess

def exec_ls(path: str = ".") -> str:
    """List files and print to TUI."""
    result = subprocess.run(["ls", "-la", path], capture_output=True, text=True)
    output = result.stdout.strip()
    return output


def exec_cat(file_path: str) -> str:
    """Cat file and print to TUI."""
    result = subprocess.run(["cat", file_path], capture_output=True, text=True)
    output = result.stdout.strip()
    return output


def exec_mv(source: str, destination: str) -> str:
    """Move file and print confirmation."""
    subprocess.run(["mv", source, destination], check=True)
    output = f"Moved '{source}' → '{destination}'"
    return output


def exec_cp(source: str, destination: str) -> str:
    """Copy file and print confirmation."""
    subprocess.run(["cp", source, destination], check=True)
    output = f"Copied '{source}' → '{destination}'"
    return output


def exec_mkdir(path: str) -> str:
    """Make directory and print confirmation."""
    subprocess.run(["mkdir", "-p", path], check=True)
    output = f"Directory created: '{path}'"

    return output


def exec_delete_file(file_path: str) -> str:
    """Delete a file (requires confirmation)."""
    if not os.path.isfile(file_path):
        return f"File not found: '{file_path}'"
    os.remove(file_path)
    return f"Deleted file '{file_path}'"


def exec_delete_dir(dir_path: str) -> str:
    """Delete a directory recursively (requires confirmation)."""
    import shutil

    if not os.path.isdir(dir_path):
        return f"Directory not found: '{dir_path}'"
    shutil.rmtree(dir_path)
    return f"Deleted directory '{dir_path}'"


def exec_find(path: str = ".", name: str = "") -> str:
    """Find files and print results."""
    cmd = ["find", path]
    if name:
        cmd += ["-name", name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    return output


def exec_pwd() -> str:
    """Print working directory."""
    return os.getcwd()


def exec_cd(path: str) -> str:
    """Change the current working directory of the Python process."""
    try:
        os.chdir(path)
        return f"Changed directory to: {os.getcwd()}"
    except FileNotFoundError:
        return f"cd: no such file or directory: '{path}'"
    except NotADirectoryError:
        return f"cd: not a directory: '{path}'"
    except PermissionError:
        return f"cd: permission denied: '{path}'"
    except Exception as e:
        return f"cd error: {str(e)}"


def exec_touch(file_path: str) -> str:
    """Create empty file or update timestamp."""
    subprocess.run(["touch", file_path], check=True)
    return f"Touched '{file_path}'"


def exec_head(file_path: str, lines: int = 10) -> str:
    """Show first N lines of a file."""
    result = subprocess.run(
        ["head", "-n", str(lines), file_path], capture_output=True, text=True
    )
    return result.stdout.rstrip()


def exec_tail(file_path: str, lines: int = 10) -> str:
    """Show last N lines of a file."""
    result = subprocess.run(
        ["tail", "-n", str(lines), file_path], capture_output=True, text=True
    )
    return result.stdout.rstrip()


def exec_wc(
    file_path: str, lines: bool = True, words: bool = True, chars: bool = False
) -> str:
    """Count lines/words/characters."""
    flags = []
    if lines:
        flags.append("-l")
    if words:
        flags.append("-w")
    if chars:
        flags.append("-c")

    if not flags:
        flags = ["-lwc"]  # default behavior like wc

    result = subprocess.run(["wc", *flags, file_path], capture_output=True, text=True)
    return result.stdout.strip()


def exec_grep(
    pattern: str, file_path: str, recursive: bool = False, case_sensitive: bool = True
) -> str:
    """Search for pattern in file or directory."""
    cmd = ["grep"]
    if not case_sensitive:
        cmd.append("-i")
    if recursive:
        cmd.append("-r")

    cmd += [pattern, file_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 1:
        return "(no matches)"
    elif result.returncode == 0:
        return result.stdout.rstrip()
    else:
        return f"grep error: {result.stderr.strip() or '(exit code ' + str(result.returncode) + ')'}"


def exec_tree(path: str = ".", max_depth: int = 3) -> str:
    """Display directory tree (requires tree command)."""
    try:
        result = subprocess.run(
            ["tree", "-L", str(max_depth), path], capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.rstrip()
        else:
            return result.stderr.strip() or "tree command failed"
    except FileNotFoundError:
        return "tree command not available on this system"


def exec_file(path: str) -> str:
    """Determine file type."""
    result = subprocess.run(["file", path], capture_output=True, text=True)
    return result.stdout.strip()


def exec_stat(path: str) -> str:
    """Show file status."""
    result = subprocess.run(["stat", path], capture_output=True, text=True)
    return result.stdout.rstrip()


def exec_realpath(path: str) -> str:
    """Resolve absolute path."""
    result = subprocess.run(["realpath", path], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"realpath failed: {result.stderr.strip()}"


def exec_which(command: str) -> str:
    """Locate command in PATH."""
    result = subprocess.run(["which", command], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"{command} not found in PATH"


def exec_awk(file_path: str, script: str) -> str:
    """Process file with awk (non-interactive)."""
    import subprocess

    cmd = ["awk", "-f", "-", file_path]
    result = subprocess.run(cmd, input=script, capture_output=True, text=True)

    return result.stdout or result.stderr
