"""Worker/device-policy behavior without loading inference models."""

import pytest

from mats import core


def _result_for(path):
    return path, {
        "sample_id": path,
        "status": "ok",
        "result_row": core.measurement_na_row(path, "test"),
    }, None


def test_parallel_cpu_policy_allows_model_backed_workers(monkeypatch, tmp_path):
    seen_devices = []

    def fake_process(*args):
        seen_devices.append(args[-1])
        return _result_for(args[0])

    monkeypatch.setattr(core, "_process_batch_image", fake_process)
    result = core.run_leaf_morpho_batch(
        ["one.jpg", "two.jpg"],
        str(tmp_path),
        str(tmp_path / "results.csv"),
        workers=2,
        execution_device="cpu",
    )

    assert result["workers"] == 2
    assert result["execution_device"] == "cpu"
    assert len(seen_devices) == 2
    assert set(seen_devices) == {"cpu"}


def test_auto_device_policy_keeps_model_backed_runs_serialized(monkeypatch, tmp_path):
    seen_devices = []

    def fake_process(*args):
        seen_devices.append(args[-1])
        return _result_for(args[0])

    monkeypatch.setattr(core, "_process_batch_image", fake_process)
    result = core.run_leaf_morpho_batch(
        ["one.jpg", "two.jpg"],
        str(tmp_path),
        str(tmp_path / "results.csv"),
        workers=2,
        execution_device="auto",
    )

    assert result["workers"] == 1
    assert result["execution_device"] == "auto"
    assert seen_devices == ["auto", "auto"]


def test_cpu_override_wins_over_accelerator_environment(monkeypatch):
    monkeypatch.setenv("RF_DETR_DEVICE", "cuda")
    monkeypatch.setenv("BIREFNET_DEVICE", "cuda")

    assert core.resolve_rfdetr_device("cpu") == "cpu"
    assert core.resolve_birefnet_device("cpu").type == "cpu"


def test_default_worker_count_uses_available_cpu_capacity(monkeypatch):
    monkeypatch.setattr(core, "available_cpu_workers", lambda: 8)

    workers, reason = core.default_worker_count(
        ["one_target_box.jpg", "two_target_box.jpg"],
        "masks",
        "threshold",
    )

    assert workers == 2
    assert reason == "thresholding precomputed target boxes on CPU"


def test_worker_safety_check_rejects_high_risk_count_without_acknowledgement(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "available_cpu_workers", lambda: 8)

    with pytest.raises(ValueError, match="break-glass acknowledgement"):
        core.run_leaf_morpho_batch(
            ["one.jpg"],
            str(tmp_path),
            str(tmp_path / "results.csv"),
            workers=7,
            execution_device="cpu",
            worker_safety_check=True,
        )


def test_worker_safety_check_allows_acknowledged_high_risk_count(monkeypatch, tmp_path):
    seen_devices = []
    monkeypatch.setattr(core, "available_cpu_workers", lambda: 8)

    def fake_process(*args):
        seen_devices.append(args[-1])
        return _result_for(args[0])

    monkeypatch.setattr(core, "_process_batch_image", fake_process)
    result = core.run_leaf_morpho_batch(
        ["one.jpg", "two.jpg"],
        str(tmp_path),
        str(tmp_path / "results.csv"),
        workers=7,
        execution_device="cpu",
        worker_safety_check=True,
        break_glass_acknowledged=True,
    )

    assert result["workers"] == 7
    assert set(seen_devices) == {"cpu"}
