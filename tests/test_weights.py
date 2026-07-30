"""Weights resolution / fetch logic -- no network, no torch."""

import hashlib
import importlib
from pathlib import Path

import pytest

LFS_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:deadbeef\nsize 2647028978\n"
)


def _fresh_weights(monkeypatch, tmp_path, **env):
    """Reload mats.paths + mats.weights under a controlled environment.

    Isolates path resolution from real files outside tmp_path: the repo
    checkout ships a real weights/rf_detr_marker.pth (Git LFS) and a dev
    machine may also have a real weights/birefnet_leaf.pth on disk. Neither
    should leak into tests that want to simulate "nothing present" -- only
    files the test itself writes under tmp_path should be visible.

    Also isolates the two download channels from this machine's real state:
    _REPO_ROOT defaults to tmp_path (no .git there, so the Git LFS channel
    reads as "not a checkout" unless a test creates tmp_path/.git itself),
    and _HF_REPO_ID is forced to None (so a release build's configured
    _DEFAULT_HF_REPO_ID can't make the Hugging Face channel look available in
    a test that didn't ask for it).
    """
    for key in ("MATS_WEIGHTS_DIR", "XDG_CACHE_HOME", "MATS_NO_AUTO_FETCH",
                "RF_DETR_MARKER_CHECKPOINT", "BIREFNET_CHECKPOINT",
                "MATS_WEIGHTS_HF_REPO", "MATS_WEIGHTS_HF_REVISION"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    real_is_file = Path.is_file
    tmp_str = str(tmp_path)
    monkeypatch.setattr(
        Path, "is_file",
        lambda self: real_is_file(self) if str(self).startswith(tmp_str) else False,
    )
    import mats.paths as paths
    importlib.reload(paths)
    import mats.weights as weights
    importlib.reload(weights)
    monkeypatch.setattr(weights, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(weights, "_HF_REPO_ID", None)
    return weights


class _FakeProcess:
    """Stand-in for subprocess.Popen: poll()/returncode are available immediately."""

    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_lfs_pointer_detected(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    stub = tmp_path / "birefnet_leaf.pth"
    stub.write_bytes(LFS_POINTER)
    assert weights.looks_like_lfs_pointer(stub) is True
    # ...and a real (large enough / non-magic) file is not a pointer
    real = tmp_path / "real.pth"
    real.write_bytes(b"\x80\x02" + b"x" * 4096)
    assert weights.looks_like_lfs_pointer(real) is False


def test_pointer_is_not_counted_present(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    (tmp_path / "rf_detr_marker.pth").write_bytes(LFS_POINTER)
    assert weights._is_present(tmp_path / "rf_detr_marker.pth") is False


def test_fetch_without_any_source_prints_manual(monkeypatch, tmp_path, capsys):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    assert weights._HF_REPO_ID is None
    code = weights.fetch()
    assert code == 1
    assert "No automatic download source" in capsys.readouterr().err


def test_ensure_weight_returns_present_file(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    real = tmp_path / "rf_detr_marker.pth"
    real.write_bytes(b"\x80\x02" + b"x" * 4096)
    assert weights.ensure_weight("rf-detr") == real


def test_ensure_weight_honors_no_auto_fetch(monkeypatch, tmp_path):
    weights = _fresh_weights(
        monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path), MATS_NO_AUTO_FETCH="1"
    )
    with pytest.raises(FileNotFoundError, match="auto-fetch is disabled"):
        weights.ensure_weight("birefnet")


def test_ensure_weight_pulls_checkout_pointer_via_lfs(monkeypatch, tmp_path):
    """A pointer stub in the checkout (the post-.lfsconfig fresh-clone state)
    should trigger an attempted `git lfs pull`, not a dead-end error telling
    the user to run a bare `git lfs pull` (which fetchexclude would ignore).
    """
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path / "cache"))
    checkout = tmp_path / "checkout"
    (checkout / "weights").mkdir(parents=True)
    (checkout / ".git").mkdir()
    (checkout / "weights" / "birefnet_leaf.pth").write_bytes(LFS_POINTER)
    monkeypatch.setattr(weights, "_REPO_ROOT", checkout)

    calls = []

    def fake_download(name, progress_callback=None):
        calls.append(name)
        return False  # simulate the pull failing (e.g. no network in this test)

    monkeypatch.setattr(weights, "_download_from_lfs", fake_download)

    with pytest.raises(FileNotFoundError, match="Could not obtain"):
        weights.ensure_weight("birefnet")
    assert "birefnet" in calls


def test_doctor_ok_when_only_rf_detr_present(monkeypatch, tmp_path, capsys):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    (tmp_path / "rf_detr_marker.pth").write_bytes(b"\x80\x02" + b"x" * 4096)
    assert weights.doctor() == 0
    out = capsys.readouterr().out
    assert "MISSING" not in out
    assert "not fetched (optional" in out


def test_doctor_fails_when_rf_detr_missing(monkeypatch, tmp_path, capsys):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    assert weights.doctor() == 1
    assert "MISSING" in capsys.readouterr().out


def test_available_sources_both_unavailable_by_default(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    hf, lfs = weights.available_sources("birefnet")
    assert hf.available is False
    assert "configured" in hf.reason
    assert lfs.available is False
    assert "checkout" in lfs.reason.lower()


def test_available_sources_lfs_available_in_checkout(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path / "cache"))
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(weights, "_REPO_ROOT", checkout)
    monkeypatch.setattr(weights, "_git_lfs_installed", lambda: True)
    hf, lfs = weights.available_sources("birefnet")
    assert lfs.available is True


def test_available_sources_hf_available_when_configured(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    monkeypatch.setattr(weights, "_HF_REPO_ID", "org/mats-weights")
    monkeypatch.setattr(weights, "_hf_available", lambda: True)
    hf, lfs = weights.available_sources("birefnet")
    assert hf.available is True


def test_status_missing_for_excluded_checkout_pointer(monkeypatch, tmp_path):
    """A BiRefNet pointer stub at the checkout location (what a fresh clone
    produces under .lfsconfig's fetchexclude) is "missing", not "invalid" --
    that's the expected, by-design state, not an error.
    """
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path / "cache"))
    checkout = tmp_path / "checkout"
    (checkout / "weights").mkdir(parents=True)
    (checkout / ".git").mkdir()
    (checkout / "weights" / "birefnet_leaf.pth").write_bytes(LFS_POINTER)
    monkeypatch.setattr(weights, "_REPO_ROOT", checkout)

    status = weights.get_weight_status("birefnet")
    assert status.state == "missing"
    assert "excluded" in status.detail.lower()
    assert {s.id for s in status.sources} == {"hf", "lfs"}


def test_download_from_lfs_success(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path / "cache"))
    checkout = tmp_path / "checkout"
    (checkout / "weights").mkdir(parents=True)
    (checkout / ".git").mkdir()
    monkeypatch.setattr(weights, "_REPO_ROOT", checkout)
    monkeypatch.setattr(weights, "free_bytes", lambda path: 10 ** 12)

    real_bytes = b"\x80\x02" + b"x" * 4094
    digest = hashlib.sha256(real_bytes).hexdigest()
    monkeypatch.setitem(weights._MANIFEST["birefnet"], "size_bytes", len(real_bytes))
    monkeypatch.setitem(weights._MANIFEST["birefnet"], "sha256", digest)

    def fake_popen(*args, **kwargs):
        (checkout / "weights" / "birefnet_leaf.pth").write_bytes(real_bytes)
        return _FakeProcess(0)

    monkeypatch.setattr(weights.subprocess, "Popen", fake_popen)

    assert weights._download_from_lfs("birefnet") is True
    assert (checkout / "weights" / "birefnet_leaf.pth").read_bytes() == real_bytes


def test_download_from_lfs_failure(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path / "cache"))
    checkout = tmp_path / "checkout"
    (checkout / "weights").mkdir(parents=True)
    (checkout / ".git").mkdir()
    monkeypatch.setattr(weights, "_REPO_ROOT", checkout)
    monkeypatch.setattr(weights, "free_bytes", lambda path: 10 ** 12)
    monkeypatch.setattr(weights.subprocess, "Popen", lambda *a, **k: _FakeProcess(1))

    assert weights._download_from_lfs("birefnet") is False


def test_install_weight_hf_source_without_repo_raises(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    with pytest.raises(weights.WeightDownloadError, match="Hugging Face"):
        weights.install_weight("birefnet", source="hf")


def test_install_weight_lfs_source_outside_checkout_raises(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    with pytest.raises(weights.WeightDownloadError, match="checkout"):
        weights.install_weight("birefnet", source="lfs")


def test_install_weight_already_ready_short_circuits(monkeypatch, tmp_path):
    weights = _fresh_weights(monkeypatch, tmp_path, MATS_WEIGHTS_DIR=str(tmp_path))
    real = tmp_path / "birefnet_leaf.pth"
    real.write_bytes(b"\x80\x02" + b"x" * 4096)
    monkeypatch.setitem(weights._MANIFEST["birefnet"], "size_bytes", real.stat().st_size)
    status = weights.install_weight("birefnet", source="hf")
    assert status.state == "ready"
