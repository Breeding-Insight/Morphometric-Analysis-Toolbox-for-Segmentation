"""Cross-platform native folder selection for the locally hosted MATS app."""

import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys


class FolderPickerError(RuntimeError):
    """Raised when no graphical folder picker can be opened."""


_MACOS_SCRIPT = """
on run argv
    set promptText to item 1 of argv
    set initialPath to item 2 of argv
    try
        set selectedFolder to choose folder with prompt promptText ¬
            default location (POSIX file initialPath as alias)
        return POSIX path of selectedFolder
    on error number -128
        return ""
    end try
end run
"""


_TK_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
try:
    selected = filedialog.askdirectory(
        title=sys.argv[1],
        initialdir=sys.argv[2],
        mustexist=True,
        parent=root,
    )
    if selected:
        print(selected, end="")
finally:
    root.destroy()
"""


def _existing_start_directory(initial_directory):
    """Return the nearest existing directory for a dialog's initial location."""
    candidate = Path(initial_directory or Path.home()).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = Path.home()

    if candidate.is_file():
        candidate = candidate.parent
    while not candidate.is_dir() and candidate != candidate.parent:
        candidate = candidate.parent
    return str(candidate if candidate.is_dir() else Path.home())


def _run_dialog(command, *, cancelled_return_codes=()):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise FolderPickerError(str(exc)) from exc

    stderr = result.stderr.strip()
    if result.returncode in cancelled_return_codes and not stderr:
        return None
    if result.returncode != 0:
        detail = stderr or f"dialog exited with status {result.returncode}"
        raise FolderPickerError(detail)

    selected = result.stdout.rstrip("\r\n")
    return selected or None


def _pick_with_macos(title, initial_directory):
    executable = shutil.which("osascript")
    if not executable:
        raise FolderPickerError("macOS folder selection requires osascript")
    return _run_dialog(
        [executable, "-e", _MACOS_SCRIPT, "--", title, initial_directory]
    )


def _powershell_script(title, initial_directory):
    """Build a PowerShell dialog script without interpolating untrusted text."""

    def encoded(value):
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    return f"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$titleBytes = [Convert]::FromBase64String('{encoded(title)}')
$pathBytes = [Convert]::FromBase64String('{encoded(initial_directory)}')
$dialogTitle = [System.Text.Encoding]::UTF8.GetString($titleBytes)
$initialPath = [System.Text.Encoding]::UTF8.GetString($pathBytes)
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = $dialogTitle
$dialog.SelectedPath = $initialPath
$dialog.ShowNewFolderButton = $true
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::Write($dialog.SelectedPath)
}}
"""


def _pick_with_windows(title, initial_directory):
    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not executable:
        raise FolderPickerError("Windows folder selection requires PowerShell")
    return _run_dialog(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-STA",
            "-Command",
            _powershell_script(title, initial_directory),
        ]
    )


def _pick_with_linux(title, initial_directory):
    zenity = shutil.which("zenity")
    if zenity:
        initial_path = os.path.join(initial_directory, "")
        return _run_dialog(
            [
                zenity,
                "--file-selection",
                "--directory",
                f"--title={title}",
                f"--filename={initial_path}",
            ],
            cancelled_return_codes=(1,),
        )

    kdialog = shutil.which("kdialog")
    if kdialog:
        return _run_dialog(
            [
                kdialog,
                "--title",
                title,
                "--getexistingdirectory",
                initial_directory,
            ],
            cancelled_return_codes=(1,),
        )

    raise FolderPickerError("neither Zenity nor KDialog is installed")


def _pick_with_tk(title, initial_directory):
    return _run_dialog(
        [sys.executable, "-c", _TK_SCRIPT, title, initial_directory]
    )


def choose_folder(title, initial_directory):
    """Open a native folder dialog on the Streamlit server's local desktop.

    Returns the selected path, or ``None`` when the user cancels. If the
    platform's preferred dialog is unavailable, a Tk dialog is attempted.
    """
    initial_directory = _existing_start_directory(initial_directory)
    if sys.platform == "darwin":
        preferred_picker = _pick_with_macos
    elif sys.platform == "win32":
        preferred_picker = _pick_with_windows
    elif sys.platform.startswith("linux"):
        preferred_picker = _pick_with_linux
    else:
        preferred_picker = None

    errors = []
    if preferred_picker is not None:
        try:
            return preferred_picker(title, initial_directory)
        except FolderPickerError as exc:
            errors.append(str(exc))

    try:
        return _pick_with_tk(title, initial_directory)
    except FolderPickerError as exc:
        errors.append(str(exc))

    detail = "; ".join(error for error in errors if error)
    message = "No graphical folder picker could be opened"
    if detail:
        message += f": {detail}"
    raise FolderPickerError(message)
