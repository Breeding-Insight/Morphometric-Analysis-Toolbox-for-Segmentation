import subprocess

import pytest

from mats.app import folder_picker


def _completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_macos_picker_returns_selected_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(folder_picker.sys, "platform", "darwin")
    monkeypatch.setattr(
        folder_picker.shutil,
        "which",
        lambda name: "/usr/bin/osascript" if name == "osascript" else None,
    )

    def fake_run(command, **kwargs):
        assert command[:2] == ["/usr/bin/osascript", "-e"]
        assert command[-2:] == ["Choose photos", str(tmp_path)]
        return _completed(command, stdout="/Users/researcher/leaf photos/\n")

    monkeypatch.setattr(folder_picker.subprocess, "run", fake_run)

    selected = folder_picker.choose_folder("Choose photos", tmp_path)

    assert selected == "/Users/researcher/leaf photos/"


def test_windows_picker_uses_powershell_without_interpolating_paths(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(folder_picker.sys, "platform", "win32")
    monkeypatch.setattr(
        folder_picker.shutil,
        "which",
        lambda name: "powershell.exe" if name == "powershell.exe" else None,
    )

    def fake_run(command, **kwargs):
        assert command[0] == "powershell.exe"
        assert "-STA" in command
        script = command[-1]
        assert "leaf ' photos" not in script
        return _completed(command, stdout=r"C:\Data\Leaf Photos")

    monkeypatch.setattr(folder_picker.subprocess, "run", fake_run)

    selected = folder_picker.choose_folder("Choose photos", tmp_path / "leaf ' photos")

    assert selected == r"C:\Data\Leaf Photos"


def test_linux_picker_prefers_zenity(monkeypatch, tmp_path):
    monkeypatch.setattr(folder_picker.sys, "platform", "linux")
    monkeypatch.setattr(
        folder_picker.shutil,
        "which",
        lambda name: "/usr/bin/zenity" if name == "zenity" else None,
    )

    def fake_run(command, **kwargs):
        assert command[0] == "/usr/bin/zenity"
        assert "--directory" in command
        return _completed(command, stdout="/data/images\n")

    monkeypatch.setattr(folder_picker.subprocess, "run", fake_run)

    assert folder_picker.choose_folder("Choose photos", tmp_path) == "/data/images"


def test_cancel_returns_none_without_opening_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(folder_picker.sys, "platform", "linux")
    monkeypatch.setattr(
        folder_picker.shutil,
        "which",
        lambda name: "/usr/bin/zenity" if name == "zenity" else None,
    )
    monkeypatch.setattr(
        folder_picker.subprocess,
        "run",
        lambda command, **kwargs: _completed(command, returncode=1),
    )

    assert folder_picker.choose_folder("Choose photos", tmp_path) is None


def test_picker_failure_has_manual_entry_friendly_error(monkeypatch, tmp_path):
    monkeypatch.setattr(folder_picker.sys, "platform", "linux")
    monkeypatch.setattr(folder_picker.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        folder_picker.subprocess,
        "run",
        lambda command, **kwargs: _completed(
            command, returncode=1, stderr="no display available"
        ),
    )

    with pytest.raises(
        folder_picker.FolderPickerError,
        match="No graphical folder picker could be opened",
    ):
        folder_picker.choose_folder("Choose photos", tmp_path)
