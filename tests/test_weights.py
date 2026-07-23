"""Weights resolution / fetch logic -- no network, no torch."""

import importlib

import pytest

LFS_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:deadbeef\nsize 2647028978\n"
)


def _fresh_weights(monkeypatch, **env):
    """Reload mats.paths + mats.weights under a controlled environment."""
    for key in ("MATS_WEIGHTS_DIR", "XDG_CACHE_HOME", "MATS_NO_AUTO_FETCH",
                "RF_DETR_MARKER_CHECKPOINT", "BIREFNET_CHECKPOINT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import mats.paths as paths
    importlib.reload(paths)
    import mats.weights as weights
    return importlib.reload(weights)


def test_lfs_pointer_detected(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    stub = tmp_path / "birefnet_leaf.pth"
    stub.write_bytes(LFS_POINTER)
    assert weights.looks_like_lfs_pointer(stub) is True
    # ...and a real (large enough / non-magic) file is not a pointer
    real = tmp_path / "real.pth"
    real.write_bytes(b"\x80\x02" + b"x" * 4096)
    assert weights.looks_like_lfs_pointer(real) is False


def test_pointer_is_not_counted_present(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    (tmp_path / "rf_detr_marker.pth").write_bytes(LFS_POINTER)
    assert weights._is_present(tmp_path / "rf_detr_marker.pth") is False


def test_fetch_without_repo_prints_manual(monkeypatch, tmp_path, capsys):
    weights = _fresh_weights(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    assert weights._HF_REPO_ID is None
    code = weights.fetch()
    assert code == 1
    assert "not configured" in capsys.readouterr().err


def test_ensure_weight_returns_present_file(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    real = tmp_path / "rf_detr_marker.pth"
    real.write_bytes(b"\x80\x02" + b"x" * 4096)
    assert weights.ensure_weight("rf-detr") == real


def test_ensure_weight_honors_no_auto_fetch(monkeypatch, tmp_path):
    weights = _fresh_weights(
        monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path), MATS_NO_AUTO_FETCH="1"
    )
    with pytest.raises(FileNotFoundError, match="auto-fetch is disabled"):
        weights.ensure_weight("birefnet")


def test_ensure_weight_lfs_stub_message(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, MATS_WEIGHTS_DIR=str(tmp_path))
    (tmp_path / "birefnet_leaf.pth").write_bytes(LFS_POINTER)
    with pytest.raises(FileNotFoundError, match="git lfs"):
        weights.ensure_weight("birefnet")
