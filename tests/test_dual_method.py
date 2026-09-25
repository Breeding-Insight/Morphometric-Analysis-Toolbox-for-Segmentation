"""Measure with Otsu and BiRefNet in one run, without model downloads."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("rfdetr")
from mats import core


def _input(tmp_path, name="leaf"):
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[20:80, 20:80] = 0
    image[43:57, 43:57] = 255
    image[5:10, 5:10] = 0
    path = tmp_path / f"{name}_target_box.png"
    assert cv2.imwrite(str(path), image)
    return path


def _fake_birefnet(calls, leaf=True):
    def predict(image, device_override=None):
        calls.append(image.shape)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if leaf:
            mask[30:70, 30:70] = 255
        return mask
    return predict


def _run(sources, output, mask_method, **kwargs):
    kwargs.setdefault("workers", 1)
    return core.run_leaf_morpho_batch(
        [str(source) for source in sources], str(output), str(output / "results.csv"),
        template_dimensions=(10, 10, "cm"), mask_method=mask_method,
        compact_csv=False, write_failures=True, **kwargs,
    )


def test_resolve_mask_methods_accepts_both_and_sequences():
    assert core.resolve_mask_methods("threshold") == ("threshold",)
    assert core.resolve_mask_methods("both") == ("threshold", "birefnet")
    assert core.resolve_mask_methods(["birefnet", "threshold", "birefnet"]) == (
        "birefnet", "threshold"
    )
    for bad in ((), "nonsense", ["threshold", "nonsense"]):
        with pytest.raises(ValueError):
            core.resolve_mask_methods(bad)
    assert core.method_suffixed_path("/out/results.csv", "birefnet") == "/out/results_birefnet.csv"


def test_each_method_matches_its_single_method_run(tmp_path, monkeypatch):
    source = _input(tmp_path)
    calls = []
    monkeypatch.setattr(core, "predict_birefnet_mask", _fake_birefnet(calls))
    exports = {"pre_cleanup_methods": ("threshold", "birefnet"), "overlay": True,
               "cutout": True, "axes": True}
    threshold = tmp_path / "threshold"
    birefnet = tmp_path / "birefnet"
    both = tmp_path / "both"
    _run([source], threshold, "threshold", export_options=exports)
    _run([source], birefnet, "birefnet", export_options=exports)
    calls.clear()
    run = _run([source], both, "both", export_options=exports)

    assert calls == [(100, 100, 3)]
    assert run["methods"] == ("threshold", "birefnet")
    assert (run["succeeded"], run["failed"]) == (1, 0)
    assert "results_path" not in run and "result_rows" not in run
    for method, single in (("threshold", threshold), ("birefnet", birefnet)):
        assert (both / f"results_{method}.csv").read_bytes() == (single / "results.csv").read_bytes()
        assert run["by_method"][method]["results_path"] == str(both / f"results_{method}.csv")
        for suffix in ("_mask.png", "_overlay.jpg", "_cutout.jpg", "_measurement_axes.jpg"):
            stem, ext = suffix.rsplit(".", 1)
            assert (both / f"leaf{stem}_{method}.{ext}").read_bytes() == (
                single / f"leaf{suffix}"
            ).read_bytes()
        assert (both / f"leaf_mask_precleanup_{method}.png").exists()
    assert not (both / "leaf_mask.png").exists()
    assert {item["method"] for item in run["artifacts"] if item["kind"] == "results_csv"} == {
        "threshold", "birefnet"
    }


def test_a_method_failure_only_fails_that_methods_outputs(tmp_path, monkeypatch):
    source = _input(tmp_path)
    monkeypatch.setattr(core, "predict_birefnet_mask", _fake_birefnet([], leaf=False))
    single = tmp_path / "single"
    both = tmp_path / "both"
    _run([source], single, "birefnet")
    run = _run([source], both, "both")

    assert (run["succeeded"], run["failed"]) == (0, 1)
    assert run["by_method"]["threshold"]["succeeded"] == 1
    assert run["by_method"]["birefnet"]["failed"] == 1
    assert "LEAF_MASK: leaf not detected" in (both / "results_birefnet.csv").read_text()
    assert (both / "leaf_morpho_failures_birefnet.csv").read_bytes() == (
        single / "leaf_morpho_failures.csv"
    ).read_bytes()
    assert not (both / "leaf_morpho_failures_threshold.csv").exists()


def test_raw_measurement_source_applies_to_each_method(tmp_path, monkeypatch):
    import json

    source = _input(tmp_path)
    monkeypatch.setattr(core, "predict_birefnet_mask", _fake_birefnet([]))
    run = _run([source], tmp_path / "raw", "both", measurement_source="pre-cleanup")
    assert run["succeeded"] == 1
    for method in ("threshold", "birefnet"):
        assert run["by_method"][method]["result_rows"][0]["leaf_area_cm2"] != "NA"
        metadata = json.loads((tmp_path / "raw" / f"results_{method}.csv.meta.json").read_text())
        assert metadata["measurement_source"] == "pre-cleanup"
        assert metadata["mask_method"] == method


def test_failure_before_segmentation_is_recorded_for_every_method(tmp_path):
    unreadable = tmp_path / "broken.jpg"
    unreadable.write_bytes(b"not an image")
    run = _run([unreadable], tmp_path / "out", "both")

    assert (run["succeeded"], run["failed"]) == (0, 1)
    for method in ("threshold", "birefnet"):
        assert "READ_IMAGE" in (tmp_path / "out" / f"results_{method}.csv").read_text()
        assert (tmp_path / "out" / f"leaf_morpho_failures_{method}.csv").exists()


def test_parallel_workers_record_every_image_for_every_method(tmp_path, monkeypatch):
    sources = [_input(tmp_path, "leaf_a"), _input(tmp_path, "leaf_b")]
    monkeypatch.setattr(core, "predict_birefnet_mask", _fake_birefnet([]))
    run = _run(sources, tmp_path / "out", "both", workers=2, execution_device="cpu")

    assert (run["succeeded"], run["processed"]) == (2, 2)
    for method in ("threshold", "birefnet"):
        rows = run["by_method"][method]["result_rows"]
        assert sorted(row["sample_id"] for row in rows) == ["leaf_a", "leaf_b"]


def test_target_box_mode_keeps_a_single_csv(tmp_path):
    source = _input(tmp_path)
    run = _run([source], tmp_path / "out", "both", output_mode="target-boxes")

    assert run["methods"] == ("threshold",)
    assert run["results_path"] == str(tmp_path / "out" / "results.csv")
    assert (tmp_path / "out" / "results.csv").exists()


def test_both_methods_refuse_hybrid_execution(tmp_path):
    with pytest.raises(ValueError, match="Otsu thresholding only"):
        _run([_input(tmp_path)], tmp_path / "out", "both", workers=2, execution_device="hybrid")
