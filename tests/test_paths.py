"""Checkpoint path resolution precedence -- no torch required."""

import importlib
from pathlib import Path


def _reload_paths(monkeypatch, **env):
    """Reload mats.paths with a controlled environment and return the module."""
    for key in ("MATS_WEIGHTS_DIR", "XDG_CACHE_HOME",
                "RF_DETR_MARKER_CHECKPOINT", "BIREFNET_CHECKPOINT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import mats.paths as paths
    return importlib.reload(paths)


def test_env_var_override_wins(monkeypatch, tmp_path):
    ckpt = tmp_path / "custom.pth"
    ckpt.write_bytes(b"x")
    paths = _reload_paths(monkeypatch, RF_DETR_MARKER_CHECKPOINT=str(ckpt))
    assert paths.RF_DETR_MARKER_CHECKPOINT == ckpt


def test_mats_weights_dir_used(monkeypatch, tmp_path):
    paths = _reload_paths(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    assert paths.WEIGHTS_DIR == tmp_path
    assert paths.RF_DETR_MARKER_CHECKPOINT == tmp_path / paths.RF_DETR_MARKER_FILENAME
    assert paths.BIREFNET_CHECKPOINT == tmp_path / paths.BIREFNET_FILENAME


def test_default_cache_dir(monkeypatch, tmp_path):
    paths = _reload_paths(monkeypatch, XDG_CACHE_HOME=str(tmp_path))
    assert paths.WEIGHTS_DIR == tmp_path / "mats" / "weights"


def test_falls_back_to_first_candidate_when_missing(monkeypatch, tmp_path):
    # Nothing on disk -> returns the primary candidate path for a clear error message.
    paths = _reload_paths(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    assert isinstance(paths.RF_DETR_MARKER_CHECKPOINT, Path)
    assert not paths.RF_DETR_MARKER_CHECKPOINT.is_file()
    assert paths.RF_DETR_MARKER_CHECKPOINT.parent == tmp_path


LFS_POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 1\n"


def test_lfs_pointer_helper(monkeypatch, tmp_path):
    paths = _reload_paths(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    stub = tmp_path / "stub.pth"
    stub.write_bytes(LFS_POINTER)
    assert paths.looks_like_lfs_pointer(stub) is True
    assert paths.looks_like_lfs_pointer(tmp_path / "missing.pth") is False


def test_resolver_skips_lfs_pointer(monkeypatch, tmp_path):
    # An un-smudged LFS stub at the primary candidate must not be treated as the
    # weight; resolution falls through (here, back to the stub path as last resort).
    weights_dir = tmp_path / "wd"
    weights_dir.mkdir()
    (weights_dir / "rf_detr_marker.pth").write_bytes(LFS_POINTER)
    paths = _reload_paths(monkeypatch, MATS_WEIGHTS_DIR=str(weights_dir))
    resolved = paths.RF_DETR_MARKER_CHECKPOINT
    # It resolved to *a* path, but that path is a pointer stub, so callers using
    # looks_like_lfs_pointer treat it as "not really present".
    assert paths.looks_like_lfs_pointer(resolved) is True
