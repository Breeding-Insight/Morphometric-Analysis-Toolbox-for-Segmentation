"""Runtime-path helpers keep app guidance specific to the active installation."""

import re
from pathlib import Path

from mats.app import runtime_paths


def test_display_path_expands_to_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert runtime_paths.display_path("sample output") == str(tmp_path / "sample output")


def test_python_command_quotes_the_active_interpreter_and_paths_with_spaces(monkeypatch, tmp_path):
    interpreter = tmp_path / "MATS Python" / "python"
    monkeypatch.setattr(runtime_paths.sys, "executable", str(interpreter))

    command = runtime_paths.python_command("-m", "pip", "install", "mats-morpho[qr]")

    assert str(interpreter) in command
    assert "-m pip install" in command
    assert "mats-morpho[qr]" in command
    if runtime_paths.os.name != "nt":
        assert "'" in command


def test_source_install_root_requires_a_project_manifest(monkeypatch, tmp_path):
    root = tmp_path / "source checkout"
    module_path = root / "src" / "mats" / "app" / "runtime_paths.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    monkeypatch.setattr(runtime_paths, "__file__", str(module_path))

    assert runtime_paths.source_install_root() is None

    (root / "pyproject.toml").write_text("[project]\nname = 'mats-morpho'\n")

    assert runtime_paths.source_install_root() == root


def test_source_install_root_does_not_depend_on_the_terminal_directory(monkeypatch, tmp_path):
    expected_root = Path(runtime_paths.__file__).resolve().parents[3]
    monkeypatch.chdir(tmp_path)

    assert runtime_paths.source_install_root() == expected_root


def test_app_source_does_not_embed_a_personal_user_profile_path():
    app_root = Path(__file__).resolve().parents[1] / "src" / "mats" / "app"
    personal_path = re.compile(r"(?:/Users/[^/\\\"']+|/home/[^/\\\"']+|[A-Za-z]:\\\\Users\\\\)")
    matches = [
        f"{path}: {match.group(0)}"
        for path in app_root.rglob("*.py")
        for match in personal_path.finditer(path.read_text())
    ]

    assert not matches
