"""Exercise export behavior on a real Otsu target box without model downloads."""

import csv
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("rfdetr")
from mats import core


def _input(tmp_path):
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[20:80, 20:80] = 0
    image[43:57, 43:57] = 255
    image[5:10, 5:10] = 0
    path = tmp_path / "leaf_target_box.png"
    assert cv2.imwrite(str(path), image)
    return path


def _run(source, output, **options):
    return core.run_leaf_morpho_batch(
        [str(source)], str(output), str(output / "results.csv"),
        template_dimensions=(10, 10, "cm"), workers=1,
        compact_csv=False, export_options=options,
    )


def test_pre_cleanup_and_optional_files_preserve_measurements(tmp_path):
    source = _input(tmp_path)
    source_bytes = source.read_bytes()
    plain = tmp_path / "plain"
    exported = tmp_path / "exported"
    plain_run = _run(source, plain, target_boxes=False, cleaned_masks=False)
    export_run = _run(source, exported, target_boxes=False, cleaned_masks=True,
                      pre_cleanup_methods=("threshold",), overlay=True,
                      cutout=True, axes=True)

    assert (plain / "results.csv").read_bytes() == (exported / "results.csv").read_bytes()
    assert plain_run["result_rows"] == export_run["result_rows"]
    raw = cv2.imread(str(exported / "leaf_mask_precleanup_threshold.png"), 0)
    cleaned = cv2.imread(str(exported / "leaf_mask.png"), 0)
    assert raw[50, 50] == 0 and raw[7, 7] == 255
    assert cleaned[50, 50] == 255 and cleaned[7, 7] == 0
    assert not (exported / "leaf_target_box.jpg").exists()
    assert not (plain / "leaf_mask.png").exists()
    assert source.read_bytes() == source_bytes
    assert {item["kind"] for item in export_run["artifacts"]} == {
        "results_csv", "results_metadata", "mask", "pre_cleanup", "overlay", "cutout", "axes"
    }


def test_both_methods_export_once_and_do_not_change_otsu_measurement(tmp_path, monkeypatch):
    source = _input(tmp_path)
    baseline = _run(source, tmp_path / "baseline")
    calls = []

    def fake_birefnet(image, device_override=None):
        calls.append(image.shape)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[30:70, 30:70] = 255
        return mask

    monkeypatch.setattr(core, "predict_birefnet_mask", fake_birefnet)
    both = _run(source, tmp_path / "both",
                pre_cleanup_methods=("threshold", "birefnet"))
    assert calls == [(100, 100, 3)]
    assert baseline["result_rows"] == both["result_rows"]
    assert (tmp_path / "both" / "leaf_mask_precleanup_threshold.png").exists()
    assert (tmp_path / "both" / "leaf_mask_precleanup_birefnet.png").exists()


def test_birefnet_measurements_are_independent_of_threshold_export(tmp_path, monkeypatch):
    source = _input(tmp_path)
    calls = []

    def fake_birefnet(image, device_override=None):
        calls.append(image.shape)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[30:70, 30:70] = 255
        return mask

    monkeypatch.setattr(core, "predict_birefnet_mask", fake_birefnet)
    baseline = core.run_leaf_morpho_batch(
        [str(source)], str(tmp_path / "birefnet"), str(tmp_path / "birefnet" / "results.csv"),
        template_dimensions=(10, 10, "cm"), mask_method="birefnet", workers=1,
    )
    both = core.run_leaf_morpho_batch(
        [str(source)], str(tmp_path / "both"), str(tmp_path / "both" / "results.csv"),
        template_dimensions=(10, 10, "cm"), mask_method="birefnet", workers=1,
        export_options={"pre_cleanup_methods": ("threshold", "birefnet")},
    )
    assert calls == [(100, 100, 3), (100, 100, 3)]
    assert baseline["result_rows"] == both["result_rows"]


def test_custom_cutoff_drives_the_threshold_pre_cleanup_mask(tmp_path):
    # A pale (gray 160) leaf sits above the "high" preset but below a custom 177.
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[20:80, 20:80] = 160
    source = tmp_path / "pale_target_box.png"
    assert cv2.imwrite(str(source), image)

    def run(output, threshold_value):
        return core.run_leaf_morpho_batch(
            [str(source)], str(output), str(output / "results.csv"),
            template_dimensions=(10, 10, "cm"), threshold_value=threshold_value,
            workers=1, export_options={"pre_cleanup_methods": ("threshold",)},
        )

    custom = run(tmp_path / "custom", 177)
    raw = cv2.imread(str(tmp_path / "custom" / "pale_mask_precleanup_threshold.png"), 0)
    _, expected = cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 177, 255, cv2.THRESH_BINARY_INV
    )
    assert custom["succeeded"] == 1
    assert np.array_equal(raw, expected)
    assert run(tmp_path / "high", core.THRESHOLD_LEVELS["high"])["succeeded"] == 0


def test_secondary_method_failure_keeps_measurement(tmp_path, monkeypatch):
    source = _input(tmp_path)

    def fail_birefnet(image, device_override=None):
        raise RuntimeError("mock inference error")

    monkeypatch.setattr(core, "predict_birefnet_mask", fail_birefnet)
    run = _run(source, tmp_path / "out", pre_cleanup_methods=("birefnet",))
    assert run["succeeded"] == 1
    assert run["result_rows"][0]["leaf_area_cm2"] != "NA"
    assert any("mock inference error" in row["status"] for row in run["failure_rows"])


def test_manifest_excludes_files_left_by_prior_run(tmp_path):
    source = _input(tmp_path)
    output = tmp_path / "reused"
    _run(source, output, pre_cleanup_methods=("threshold",))
    later = _run(source, output, cleaned_masks=False)
    assert (output / "leaf_mask_precleanup_threshold.png").exists()
    assert {item["kind"] for item in later["artifacts"]} == {"results_csv", "results_metadata"}


def test_pre_cleanup_measurement_keeps_holes_and_nearby_specks(tmp_path):
    source = _input(tmp_path)
    cleaned = core.run_leaf_morpho_batch(
        [str(source)], str(tmp_path / "cleaned"), str(tmp_path / "cleaned" / "results.csv"),
        template_dimensions=(10, 10, "cm"), workers=1, compact_csv=False,
        export_options={"axes": True, "pre_cleanup_methods": ("threshold",)},
    )
    raw = core.run_leaf_morpho_batch(
        [str(source)], str(tmp_path / "raw"), str(tmp_path / "raw" / "results.csv"),
        template_dimensions=(10, 10, "cm"), workers=1, compact_csv=False,
        measurement_source="pre-cleanup",
        export_options={"axes": True, "pre_cleanup_methods": ("threshold",)},
    )
    clean_row, raw_row = cleaned["result_rows"][0], raw["result_rows"][0]
    assert raw_row["leaf_area_cm2"] < clean_row["leaf_area_cm2"]  # hole stays empty
    assert raw_row["width_cm"] > clean_row["width_cm"]  # detached speck extends width
    assert raw_row["length_cm"] > clean_row["length_cm"]
    assert json.loads((tmp_path / "raw" / "results.csv.meta.json").read_text()) == {
        "measurement_source": "pre-cleanup",
        "mask_method": "threshold",
        "results_unit": "cm",
        "csv_schema": "full",
        "clean_margin": 1.0,
        "stray_gap": 0.25,
    }
    assert "stray_gap" not in json.loads(
        (tmp_path / "cleaned" / "results.csv.meta.json").read_text()
    )
    assert raw["measurement_source"] == "pre-cleanup"
    assert raw["artifacts"][-1]["kind"] == "results_metadata"
    assert not np.array_equal(
        cv2.imread(str(tmp_path / "raw" / "leaf_measurement_axes.jpg")),
        cv2.imread(str(tmp_path / "cleaned" / "leaf_measurement_axes.jpg")),
    )


def _stray_input(tmp_path):
    """A leaf with a border strip, a far speck, and a speck just above it."""
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    image[60:140, 60:140] = 0          # leaf: 80 px, so the 0.25 gap limit is ~28 px
    image[0:4, 30:170] = 0             # printed edge along the top border
    image[10:16, 10:16] = 0            # far speck, ~64 px from the leaf
    image[45:50, 95:100] = 0           # near speck, ~10 px above the leaf
    path = tmp_path / "leaf_target_box.png"
    assert cv2.imwrite(str(path), image)
    return path


def _pre_cleanup_run(source, output, **options):
    return core.run_leaf_morpho_batch(
        [str(source)], str(output), str(output / "results.csv"),
        template_dimensions=(10, 10, "cm"), workers=1, compact_csv=False,
        measurement_source="pre-cleanup",
        export_options={"pre_cleanup_methods": ("threshold",), "axes": True,
                        "preview_dir": str(output / "previews")},
        **options,
    )


def _extent_px(row):
    return (round(row["width_cm"] * row["px_per_cm_width"]),
            round(row["length_cm"] * row["px_per_cm_height"]))


def test_pre_cleanup_measurement_drops_border_and_far_pieces(tmp_path):
    source = _stray_input(tmp_path)
    output = tmp_path / "out"
    run = _pre_cleanup_run(source, output)
    row = run["result_rows"][0]
    # Width is the leaf alone; length reaches up to the near speck (rows 45-139).
    assert _extent_px(row) == (80, 95)
    assert round(row["leaf_area_cm2"] * row["px_per_cm_width"] * row["px_per_cm_height"]) == (
        80 * 80 + 25
    )

    raw = cv2.imread(str(output / "leaf_mask_precleanup_threshold.png"), 0)
    assert raw[1, 100] == 255 and raw[12, 12] == 255      # the export stays raw
    measured = cv2.imread(next(
        item["path"] for item in run["preview_artifacts"] if item["kind"] == "preview_mask"
    ), 0)
    assert measured[1, 100] == 0 and measured[12, 12] == 0
    assert measured[47, 97] == 255 and measured[100, 100] == 255

    leaf_only = _pre_cleanup_run(source, tmp_path / "leaf_only", stray_gap=0)
    assert _extent_px(leaf_only["result_rows"][0]) == (80, 80)


def test_pre_cleanup_adjustment_measures_without_stray_pieces(tmp_path):
    pytest.importorskip("streamlit")
    from mats.app.output_adjustment import apply_threshold_adjustment

    source = _stray_input(tmp_path)
    output = tmp_path / "out"
    summary = _pre_cleanup_run(source, output, threshold_value=125, stray_gap=0.5)
    paths = {
        item["kind"]: item["path"]
        for item in (*summary["artifacts"], *summary["preview_artifacts"])
        if item["sample_id"] == "leaf"
    }
    run = {
        "by_method": {"threshold": {"results_path": summary["results_path"]}},
        "results_unit": "cm", "measurement_source": "pre-cleanup",
        "stray_gap": summary["stray_gap"], "output_path": str(output),
        "mask_methods": ("threshold",), "artifacts": list(summary["artifacts"]),
    }
    pair = {
        "sample_id": "leaf", "target_box": str(source), "mask": paths["preview_mask"],
        "raw_mask": paths["pre_cleanup"], "mask_source": "pre-cleanup",
    }
    before = summary["result_rows"][0]
    apply_threshold_adjustment(run, pair, 125)

    with open(summary["results_path"], newline="") as handle:
        adjusted = next(csv.DictReader(handle))
    assert float(adjusted["width_cm"]) == pytest.approx(before["width_cm"])
    assert float(adjusted["length_cm"]) == pytest.approx(before["length_cm"])
    assert pair["mask"] == paths["preview_mask"]
    assert pair["raw_mask"] == paths["pre_cleanup"]
    assert cv2.imread(pair["mask"], 0)[1, 100] == 0
    assert cv2.imread(pair["raw_mask"], 0)[1, 100] == 255
    metadata = json.loads(Path(f"{summary['results_path']}.meta.json").read_text())
    assert metadata["stray_gap"] == 0.5

    # The explorer's own settings drive the overwrite and are recorded with it.
    apply_threshold_adjustment(run, pair, 125, stray_gap=0, clean_margin=2)
    with open(summary["results_path"], newline="") as handle:
        leaf_only = next(csv.DictReader(handle))
    width = float(leaf_only["width_cm"]) * float(leaf_only["px_per_cm_width"])
    length = float(leaf_only["length_cm"]) * float(leaf_only["px_per_cm_height"])
    assert (round(width), round(length)) == (80, 80)       # the near speck is gone
    metadata = json.loads(Path(f"{summary['results_path']}.meta.json").read_text())
    assert metadata["threshold_adjustments"]["leaf"] == {
        "cutoff": 125, "fill_holes": True, "clean_margin": 2.0, "stray_gap": 0.0,
    }
    assert run["threshold_adjustments"]["leaf"]["stray_gap"] == 0.0


def _framed_input(tmp_path):
    """A small leaf inside the printed box outline, as a target box shows it.

    The outline lands on the box edge as four strips, with white corners where
    the marker boxes were whited out. Each strip outweighs the 20 x 20 leaf.
    """
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    image[:4, 30:-30] = image[-4:, 30:-30] = 0
    image[30:-30, :4] = image[30:-30, -4:] = 0
    image[190:210, 190:210] = 0
    path = tmp_path / "framed_target_box.png"
    assert cv2.imwrite(str(path), image)
    return path


def test_pre_cleanup_measurement_ignores_edge_lines_that_outweigh_the_leaf(tmp_path):
    source = _framed_input(tmp_path)
    output = tmp_path / "out"
    row = _pre_cleanup_run(source, output)["result_rows"][0]
    assert _extent_px(row) == (20, 20)
    raw = cv2.imread(str(output / "framed_mask_precleanup_threshold.png"), 0)
    assert raw[1, 200] == 255                          # the export keeps the lines
    metadata = json.loads((output / "results.csv.meta.json").read_text())
    assert metadata["clean_margin"] == 1.0

    # Without the margin, a strip is the largest piece and replaces the leaf.
    unguarded = _pre_cleanup_run(source, tmp_path / "no_margin", clean_margin=0)
    assert _extent_px(unguarded["result_rows"][0]) != (20, 20)


def test_remove_flashfill_adjustment_clears_the_edge_margin(tmp_path):
    pytest.importorskip("streamlit")
    from mats.app.output_adjustment import measure_threshold_adjustment

    source = _framed_input(tmp_path)
    output = tmp_path / "out"
    summary = core.run_leaf_morpho_batch(
        [str(source)], str(output), str(output / "results.csv"),
        template_dimensions=(10, 10, "cm"), workers=1, compact_csv=False,
        threshold_value=125,
    )
    run = {
        "by_method": {"threshold": {"results_path": summary["results_path"]}},
        "results_unit": "cm", "measurement_source": "cleaned",
        "output_path": str(output), "mask_methods": ("threshold",),
        "artifacts": list(summary["artifacts"]),
    }
    prepared = measure_threshold_adjustment(
        run, {"sample_id": "framed", "target_box": str(source)}, 125, remove_fill=True,
    )
    only_leaf = np.zeros((400, 400), dtype=np.uint8)
    only_leaf[190:210, 190:210] = 255
    assert np.array_equal(prepared["measurement_mask"], only_leaf)


def test_raw_preview_survives_removed_input_without_becoming_export(tmp_path):
    source = _input(tmp_path)
    output = tmp_path / "out"
    preview_dir = tmp_path / "private_previews"
    run = core.run_leaf_morpho_batch(
        [str(source)], str(output), str(output / "results.csv"),
        template_dimensions=(10, 10, "cm"), workers=1,
        measurement_source="pre-cleanup",
        export_options={
            "target_boxes": False, "cleaned_masks": False,
            "preview_dir": str(preview_dir),
        },
    )
    source.unlink()  # Uploaded files are discarded after the GUI run.
    previews = {item["kind"]: item for item in run["preview_artifacts"]}
    assert {"preview_target_box", "preview_mask", "preview_raw_mask"} == set(previews)
    assert cv2.imread(previews["preview_target_box"]["path"]) is not None
    measured = cv2.imread(previews["preview_mask"]["path"], cv2.IMREAD_GRAYSCALE)
    assert measured[50, 50] == 0 and measured[7, 7] == 255
    # The explorer re-cleans the true raw mask with each specimen's settings.
    raw = cv2.imread(previews["preview_raw_mask"]["path"], cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(raw, core.threshold_mask(cv2.imread(previews["preview_target_box"]["path"]), None))
    assert {item["kind"] for item in run["artifacts"]} == {
        "results_csv", "results_metadata",
    }


def test_cleaned_preview_uses_measurement_mask_when_export_is_disabled(tmp_path):
    source = _input(tmp_path)
    output = tmp_path / "out"
    run = core.run_leaf_morpho_batch(
        [str(source)], str(output), str(output / "results.csv"),
        template_dimensions=(10, 10, "cm"), workers=1,
        export_options={"cleaned_masks": False, "preview_dir": str(tmp_path / "previews")},
    )
    preview = next(
        item for item in run["preview_artifacts"] if item["kind"] == "preview_mask"
    )
    cleaned = cv2.imread(preview["path"], cv2.IMREAD_GRAYSCALE)
    assert cleaned[50, 50] == 255 and cleaned[7, 7] == 0
    raw_preview = next(
        item for item in run["preview_artifacts"] if item["kind"] == "preview_raw_mask"
    )
    raw = cv2.imread(raw_preview["path"], cv2.IMREAD_GRAYSCALE)
    assert raw[50, 50] == 0 and raw[7, 7] == 255
    assert not (output / "leaf_mask.png").exists()


def test_secondary_birefnet_export_requires_supported_execution(tmp_path):
    source = _input(tmp_path)
    with pytest.raises(ValueError, match="Otsu thresholding only"):
        core.run_leaf_morpho_batch(
            [str(source)], str(tmp_path / "out"), str(tmp_path / "out" / "results.csv"),
            workers=2, execution_device="hybrid",
            export_options={"pre_cleanup_methods": ("birefnet",)},
        )
