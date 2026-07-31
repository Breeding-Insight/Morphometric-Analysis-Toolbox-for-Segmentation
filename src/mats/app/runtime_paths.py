"""Runtime-aware paths and copyable shell commands for the Streamlit app.

The app can only inspect the filesystem of the process that runs Streamlit.
For a local ``mats app`` launch, that is the user's machine; for a hosted
deployment, it is the host machine. Keeping this logic here makes every page
describe the same runtime instead of assuming the terminal's current directory.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def display_path(path: str | Path) -> str:
    """Return an absolute, user-specific path suitable for UI text."""
    return str(Path(path).expanduser().resolve(strict=False))


def current_python() -> Path:
    """Return the interpreter that is running this Streamlit process."""
    return Path(sys.executable)


def demo_output_directory() -> Path:
    """Return the per-user directory used by Help's sample commands."""
    return Path.home() / "mats_demo"


def source_install_root() -> Path | None:
    """Return the source root for an editable install, if this is one."""
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def shell_command(*parts: str | Path) -> str:
    """Quote a command for the host platform without depending on its CWD."""
    values = [str(part) for part in parts]
    if os.name == "nt":
        return subprocess.list2cmdline(values)
    return shlex.join(values)


def python_command(*arguments: str | Path) -> str:
    """Build a command using the exact interpreter that runs MATS."""
    return shell_command(current_python(), *arguments)


def mats_command(*arguments: str | Path) -> str:
    """Build a MATS CLI command in the current Python environment."""
    return python_command("-m", "mats.cli", *arguments)


def shell_language() -> str:
    """Return Streamlit's closest syntax highlighter for the host shell."""
    return "powershell" if os.name == "nt" else "bash"
