"""Worker/device-policy behavior without loading inference models."""

import concurrent.futures
import csv
import threading
import time
from types import SimpleNamespace

import pytest

# mats.core imports torch/numpy/cv2/rfdetr at module level even though this
# file never runs inference -- skip cleanly on the dependency-light CI matrix
# (`pip install --no-deps -e .`), same idiom as the streamlit-gated app tests.
pytest.importorskip("numpy")
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
        seen_devices.append(args[-2])
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
        seen_devices.append(args[-2])
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


def test_hybrid_otsu_policy_keeps_gpu_rfdetr_and_cpu_fanout(monkeypatch, tmp_path):
    seen_devices = []

    def fake_process(*args):
        seen_devices.append(args[-2])
        return _result_for(args[0])

    monkeypatch.setattr(core, "_process_batch_image", fake_process)
    result = core.run_leaf_morpho_batch(
        ["one.jpg", "two.jpg"],
        str(tmp_path),
        str(tmp_path / "results.csv"),
        workers=2,
        execution_device="hybrid",
        mask_method="threshold",
    )

    assert result["workers"] == 2
    assert result["execution_device"] == "hybrid"
    assert set(seen_devices) == {"hybrid"}


def test_hybrid_policy_rejects_birefnet(tmp_path):
    with pytest.raises(ValueError, match="Otsu thresholding only"):
        core.run_leaf_morpho_batch(
            ["one.jpg"],
            str(tmp_path),
            str(tmp_path / "results.csv"),
            workers=2,
            execution_device="hybrid",
            mask_method="birefnet",
        )


def test_batch_writes_selected_results_unit(monkeypatch, tmp_path):
    def fake_process(*args):
        sample_id = args[0]
        row = core.measurement_row(sample_id, 100, 10, 20, 5.0, 10.0)
        return sample_id, {"sample_id": sample_id, "status": "ok", "result_row": row}, None

    monkeypatch.setattr(core, "_process_batch_image", fake_process)
    results_path = tmp_path / "results.csv"
    core.run_leaf_morpho_batch(
        ["leaf.jpg"],
        str(tmp_path),
        str(results_path),
        results_unit="mm",
        compact_csv=False,
        workers=1,
        execution_device="cpu",
    )

    with results_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["leaf_area_mm2"] == "200.0"
    assert row["width_mm"] == "20.0"
    assert row["length_mm"] == "20.0"
    assert row["px_per_mm_width"] == "0.5"


def test_gpu_marker_inference_uses_one_shared_lane(monkeypatch):
    active = 0
    peak_active = 0
    active_lock = threading.Lock()

    class FakeModel:
        def predict(self, _image, threshold):
            nonlocal active, peak_active
            assert threshold == core.RF_DETR_MARKER_CONFIDENCE
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.02)
            with active_lock:
                active -= 1
            return SimpleNamespace(xyxy=[])

    monkeypatch.setattr(core, "resolve_rfdetr_device", lambda _override=None: "cuda")
    monkeypatch.setattr(core, "get_marker_model", lambda _override=None: FakeModel())
    image = core.np.zeros((10, 10, 3), dtype=core.np.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(core.detect_marker_geometry, [image, image]))

    assert peak_active == 1


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
        seen_devices.append(args[-2])
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


def test_worker_safety_check_requires_birefnet_parallel_acknowledgement(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "available_cpu_workers", lambda: 8)

    with pytest.raises(ValueError, match="CPU-parallel BiRefNet"):
        core.run_leaf_morpho_batch(
            ["one.jpg", "two.jpg"],
            str(tmp_path),
            str(tmp_path / "results.csv"),
            workers=2,
            execution_device="cpu",
            mask_method="birefnet",
            worker_safety_check=True,
        )


def test_worker_safety_check_allows_acknowledged_birefnet_parallelism(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "available_cpu_workers", lambda: 8)
    monkeypatch.setattr(core, "_process_batch_image", lambda *args: _result_for(args[0]))

    result = core.run_leaf_morpho_batch(
        ["one.jpg", "two.jpg"],
        str(tmp_path),
        str(tmp_path / "results.csv"),
        workers=2,
        execution_device="cpu",
        mask_method="birefnet",
        worker_safety_check=True,
        birefnet_parallel_acknowledged=True,
    )

    assert result["workers"] == 2
