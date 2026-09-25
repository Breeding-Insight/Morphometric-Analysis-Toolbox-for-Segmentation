from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import zipfile

import pytest

# Every heavy import is gated: CI installs with `--no-deps`, so a bare
# module-level `import numpy` here is a collection error, not a skip.
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("streamlit")
cv2 = pytest.importorskip("cv2")
from streamlit.testing.v1 import AppTest
from mats.app.Home import (
    PENDING_WORKSPACE_TAB_KEY,
    THRESHOLD_CUSTOM_VALUE_KEY,
    THRESHOLD_LEVEL_KEY,
    WORKSPACE_TAB_KEY,
    _WORKBENCH_STYLES,
    _resolve_sheet_layout,
    _scale_axes_by_sample,
    _threshold_marks_html,
    build_cutout_image,
    build_overlay_image,
    collect_output_pairs,
    gather_output_files,
    files_from_manifest,
    pairs_from_manifest,
    generate_export_overlays,
    merge_viewer_pairs,
    normalize_measurements,
    select_export_files,
    summarize_measurements,
    write_output_zip,
    zip_download_name,
)
from mats.dimensions import parse_template_dimensions
from mats.mask_settings import CLEAN_MARGIN_DEFAULT, CLEAN_SIZE_DEFAULT, STRAY_GAP_DEFAULT


HOME_PAGE = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "Home.py"
DIAGNOSTICS_PAGE = HOME_PAGE.parent / "pages" / "0_Diagnostics.py"
CPU_OPTIONS_PAGE = HOME_PAGE.parent / "pages" / "3_CPU_Options.py"


def test_workspace_navigation_targets_current_streamlit_tab_markup():
    assert '[data-testid="stTab"]' in _WORKBENCH_STYLES
    assert '[role="tablist"]' in _WORKBENCH_STYLES
    assert 'data-baseweb="tab"' not in _WORKBENCH_STYLES
    assert 'font-size: 2.16rem' not in _WORKBENCH_STYLES


def test_launch_button_has_tactile_hover_and_press_styles():
    selector = (
        '.st-key-launch_analysis [data-testid="stButton"] '
        'button[data-testid="stBaseButton-primary"]'
    )
    assert f"{selector}:not(:disabled):hover" in _WORKBENCH_STYLES
    assert f"{selector}:not(:disabled):active" in _WORKBENCH_STYLES
    assert f"{selector}:disabled" in _WORKBENCH_STYLES
    assert "prefers-reduced-motion: reduce" in _WORKBENCH_STYLES


def _write_output_pair(output_dir, sample_id):
    target_box = output_dir / f"{sample_id}_target_box.jpg"
    mask = output_dir / f"{sample_id}_mask.png"
    target_box.write_bytes(b"target")
    mask.write_bytes(b"mask")
    return target_box, mask


def test_collect_output_pairs_ignores_outputs_from_earlier_sessions(tmp_path):
    _write_output_pair(tmp_path, "old")
    new_target, new_mask = _write_output_pair(tmp_path, "new")

    pairs = collect_output_pairs(tmp_path, ["new"])

    assert pairs == [{
        "sample_id": "new",
        "target_box": str(new_target),
        "mask": str(new_mask),
    }]


def test_manifest_selects_only_current_run_outputs(tmp_path):
    old_target, old_mask = _write_output_pair(tmp_path, "old")
    new_target, new_mask = _write_output_pair(tmp_path, "new")
    manifest = [
        {"path": str(new_target), "sample_id": "new", "kind": "target_box"},
        {"path": str(new_mask), "sample_id": "new", "kind": "mask"},
    ]
    assert files_from_manifest(manifest) == [new_target, new_mask]
    assert pairs_from_manifest(manifest) == [{
        "sample_id": "new", "target_box": str(new_target), "mask": str(new_mask)
    }]
    assert old_target not in files_from_manifest(manifest)
    assert old_mask not in files_from_manifest(manifest)


def test_manifest_pairs_each_method_with_its_own_mask():
    manifest = [
        {"path": "/out/leaf_target_box.jpg", "sample_id": "leaf", "kind": "target_box",
         "method": None},
        {"path": "/out/leaf_mask_threshold.png", "sample_id": "leaf", "kind": "mask",
         "method": "threshold"},
        {"path": "/out/leaf_mask_birefnet.png", "sample_id": "leaf", "kind": "mask",
         "method": "birefnet"},
    ]
    for method in ("threshold", "birefnet"):
        assert pairs_from_manifest(manifest, method=method) == [{
            "sample_id": "leaf",
            "target_box": "/out/leaf_target_box.jpg",
            "mask": f"/out/leaf_mask_{method}.png",
        }]


def test_manifest_uses_raw_mask_for_raw_measurements_and_keeps_methods_separate():
    exported = [
        {"path": "/out/leaf_target_box.jpg", "sample_id": "leaf", "kind": "target_box"},
        {"path": "/out/leaf_mask_threshold.png", "sample_id": "leaf", "kind": "mask",
         "method": "threshold"},
        {"path": "/out/leaf_mask_birefnet.png", "sample_id": "leaf", "kind": "mask",
         "method": "birefnet"},
    ]
    previews = [
        {"path": "/tmp/leaf_target_box.png", "sample_id": "leaf",
         "kind": "preview_target_box"},
        {"path": "/tmp/leaf_raw_threshold.png", "sample_id": "leaf",
         "kind": "preview_mask", "method": "threshold"},
        {"path": "/tmp/leaf_raw_birefnet.png", "sample_id": "leaf",
         "kind": "preview_mask", "method": "birefnet"},
    ]
    assert pairs_from_manifest(
        exported, method="threshold", measurement_source="pre-cleanup",
        preview_artifacts=previews,
    ) == [{
        "sample_id": "leaf", "target_box": "/tmp/leaf_target_box.png",
        "mask": "/tmp/leaf_raw_threshold.png", "mask_source": "pre-cleanup",
    }]
    assert pairs_from_manifest(
        exported, method="birefnet", measurement_source="pre-cleanup",
        preview_artifacts=previews,
    )[0]["mask"] == "/tmp/leaf_raw_birefnet.png"


def test_pre_cleanup_runs_show_the_measured_mask_not_the_raw_export():
    # The pre-cleanup export keeps stray pieces; the measured mask drops them.
    exported = [
        {"path": "/out/leaf_mask_precleanup_threshold.png", "sample_id": "leaf",
         "kind": "pre_cleanup", "method": "threshold"},
    ]
    previews = [
        {"path": "/tmp/leaf_measured.png", "sample_id": "leaf",
         "kind": "preview_mask", "method": "threshold"},
    ]
    assert pairs_from_manifest(
        exported, method="threshold", measurement_source="pre-cleanup",
        preview_artifacts=previews,
    ) == [{
        "sample_id": "leaf", "target_box": None,
        "raw_mask": "/out/leaf_mask_precleanup_threshold.png",
        "mask": "/tmp/leaf_measured.png", "mask_source": "pre-cleanup",
    }]


def test_manifest_uses_private_previews_when_image_exports_are_disabled():
    previews = [
        {"path": "/tmp/leaf_box.png", "sample_id": "leaf", "kind": "preview_target_box"},
        {"path": "/tmp/leaf_mask.png", "sample_id": "leaf", "kind": "preview_mask",
         "method": "threshold"},
    ]
    assert pairs_from_manifest(
        [], method="threshold", preview_artifacts=previews,
    ) == [{"sample_id": "leaf", "target_box": "/tmp/leaf_box.png",
           "mask": "/tmp/leaf_mask.png"}]


def test_manifest_attaches_only_the_selected_methods_raw_preview():
    previews = [
        {"path": f"/tmp/leaf_raw_{method}.png", "sample_id": "leaf",
         "kind": "preview_raw_mask", "method": method}
        for method in ("threshold", "birefnet")
    ]
    previews += [
        {"path": f"/tmp/leaf_clean_{method}.png", "sample_id": "leaf",
         "kind": "preview_mask", "method": method}
        for method in ("threshold", "birefnet")
    ]
    for method in ("threshold", "birefnet"):
        assert pairs_from_manifest(
            [], method=method, preview_artifacts=previews,
        ) == [{
            "sample_id": "leaf", "target_box": None,
            "raw_mask": f"/tmp/leaf_raw_{method}.png",
            "mask": f"/tmp/leaf_clean_{method}.png",
        }]


def test_merge_viewer_pairs_accumulates_current_session_without_duplicates():
    first = {
        "sample_id": "first",
        "target_box": "/outputs/first_target_box.jpg",
        "mask": "/outputs/first_mask.png",
    }
    updated = {
        "sample_id": "first",
        "target_box": "/outputs/recreated_first_target_box.jpg",
        "mask": "/outputs/first_mask.png",
    }
    second = {
        "sample_id": "second",
        "target_box": None,
        "mask": "/outputs/second_mask.png",
    }

    pairs = merge_viewer_pairs([first], [updated, second])

    assert pairs == [updated, second]


def _write_real_output_pair(output_dir, sample_id, size=10, fill=200):
    """Write a real, decodable target-box JPG + binary mask PNG for image-helper tests."""
    target_box = np.full((size, size, 3), fill, dtype=np.uint8)
    binary_mask = np.zeros((size, size), dtype=np.uint8)
    binary_mask[2:-2, 2:-2] = 255
    target_box_path = output_dir / f"{sample_id}_target_box.jpg"
    mask_path = output_dir / f"{sample_id}_mask.png"
    cv2.imwrite(str(target_box_path), target_box)
    cv2.imwrite(str(mask_path), binary_mask)
    return target_box_path, mask_path


def test_build_overlay_image_tints_only_the_masked_region():
    target_box = np.zeros((10, 10, 3), dtype=np.uint8)
    binary_mask = np.zeros((10, 10), dtype=np.uint8)
    binary_mask[2:8, 2:8] = 255

    overlay = build_overlay_image(target_box, binary_mask, color=(0, 255, 0), alpha=1.0)

    assert overlay.shape == target_box.shape
    assert tuple(overlay[0, 0]) == (0, 0, 0)
    assert tuple(overlay[5, 5]) == (0, 255, 0)


def test_build_cutout_image_blacks_out_everything_outside_the_mask():
    target_box = np.full((10, 10, 3), 200, dtype=np.uint8)
    binary_mask = np.zeros((10, 10), dtype=np.uint8)
    binary_mask[2:8, 2:8] = 255

    cutout = build_cutout_image(target_box, binary_mask)

    assert tuple(cutout[0, 0]) == (0, 0, 0)
    assert tuple(cutout[5, 5]) == (200, 200, 200)


def test_generate_export_overlays_writes_only_the_requested_kinds(tmp_path):
    _write_real_output_pair(tmp_path, "leaf_1")

    generate_export_overlays(tmp_path, include_overlay=True, include_cutout=False)

    assert (tmp_path / "leaf_1_overlay.jpg").is_file()
    assert not (tmp_path / "leaf_1_cutout.jpg").is_file()


def test_generate_export_overlays_skips_masks_without_a_target_box(tmp_path):
    (tmp_path / "orphan_mask.png").write_bytes(b"not a real png but presence is what matters")

    generate_export_overlays(tmp_path, include_overlay=True, include_cutout=True)

    assert not (tmp_path / "orphan_overlay.jpg").is_file()
    assert not (tmp_path / "orphan_cutout.jpg").is_file()


def test_generate_export_overlays_regenerates_stale_files(tmp_path):
    # JPEG re-encoding is lossy, so compare with tolerance rather than exact equality.
    _write_real_output_pair(tmp_path, "leaf_1", fill=200)
    generate_export_overlays(tmp_path, include_overlay=False, include_cutout=True)
    first_cutout = cv2.imread(str(tmp_path / "leaf_1_cutout.jpg"))
    assert abs(int(first_cutout[5, 5][0]) - 200) <= 5

    _write_real_output_pair(tmp_path, "leaf_1", fill=50)
    generate_export_overlays(tmp_path, include_overlay=False, include_cutout=True)
    second_cutout = cv2.imread(str(tmp_path / "leaf_1_cutout.jpg"))

    assert abs(int(second_cutout[5, 5][0]) - 50) <= 5


def test_gather_output_files_includes_overlay_and_cutout_only_when_requested(tmp_path):
    _write_real_output_pair(tmp_path, "leaf_1")
    generate_export_overlays(tmp_path, include_overlay=True, include_cutout=True)
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text("sample_id\nleaf_1\n")

    plain = gather_output_files(tmp_path, results_path)
    with_extras = gather_output_files(
        tmp_path, results_path, include_overlay=True, include_cutout=True
    )

    plain_names = {path.name for path in plain}
    extra_names = {path.name for path in with_extras}
    assert "leaf_1_overlay.jpg" not in plain_names
    assert "leaf_1_cutout.jpg" not in plain_names
    assert "leaf_1_overlay.jpg" in extra_names
    assert "leaf_1_cutout.jpg" in extra_names


def test_home_page_renders_analyze_view_without_worker_control():
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.run(timeout=30)

    assert not app.exception
    assert not app.number_input
    subheaders = [item.value for item in app.subheader]
    assert "Launch analysis" in subheaders
    assert "**WORKSPACE NAVIGATION**" in [item.value for item in app.markdown]
    assert "**5 · Preflight**" in [item.value for item in app.markdown]
    assert app.session_state["results_unit"] == "cm"


def test_diagnostics_moves_out_of_workspace_tabs():
    app = AppTest.from_file(str(HOME_PAGE))
    app.run(timeout=30)

    assert not app.exception
    assert "diagnostics_context" in app.session_state
    assert "Diagnostics" not in app.session_state[WORKSPACE_TAB_KEY]


def test_sidebar_diagnostics_offers_home_when_opened_without_context():
    app = AppTest.from_file(str(DIAGNOSTICS_PAGE)).run(timeout=30)

    assert not app.exception
    assert any(item.value == "Diagnostics" for item in app.title)


def test_sidebar_diagnostics_renders_latest_preflight():
    home = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    app = AppTest.from_file(str(DIAGNOSTICS_PAGE))
    app.session_state["diagnostics_context"] = home.session_state["diagnostics_context"]
    app.session_state["diagnostics_inputs"] = home.session_state["diagnostics_inputs"]
    with patch("streamlit.page_link"):
        app.run(timeout=30)

    assert not app.exception
    assert "Compute status" in [item.value for item in app.subheader]
    assert "**Preflight Overview**" in [item.value for item in app.markdown]


def test_pending_workspace_tab_is_applied_before_navigation():
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[PENDING_WORKSPACE_TAB_KEY] = "Analyze"
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state[WORKSPACE_TAB_KEY] == "Analyze"
    assert PENDING_WORKSPACE_TAB_KEY not in app.session_state
    assert any(item.value == "Results" for item in app.subheader)


def test_home_page_offers_input_and_output_folder_pickers():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert app.button(key="choose_input_folder").label == "Choose input folder"
    assert app.button(key="choose_output_folder").label == "Choose output folder"
    assert app.button(key="open_export").disabled
    assert app.text_input(key="input_directory").value == str(Path.home())
    assert app.text_input(key="output_directory").value == str(
        Path.home() / "mats_outputs"
    )


def test_sidebar_export_shortcut_opens_export_after_a_run(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,leaf_area_cm2,width_cm,length_cm\nleaf_1,12.5,2.5,7.0\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
    )
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state[WORKSPACE_TAB_KEY] == "Setup"
    assert not app.button(key="open_export").disabled

    app.button(key="open_export").click().run(timeout=30)

    assert not app.exception
    assert app.session_state[WORKSPACE_TAB_KEY] == "Export"
    assert any(item.value == "Export" for item in app.subheader)


def test_home_page_has_a_dedicated_launch_section():
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.run(timeout=30)

    assert not app.exception
    assert any(item.value == "Launch analysis" for item in app.subheader)
    launch_button = app.button(key="run_leaf_morphometrics")
    assert launch_button.label == "Run leaf morphometrics"
    assert launch_button.icon == ":material/rocket_launch:"
    assert launch_button.proto.type == "primary"
    assert launch_button.disabled
    assert any("preflight" in item.value for item in app.markdown)


def test_analyze_keeps_scale_and_segmentation_in_distinct_blocks():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert app.checkbox(key="segment_threshold").label == "Classic thresholding"
    assert app.checkbox(key="segment_threshold").value
    assert app.checkbox(key="segment_birefnet").label == "BiRefNet"
    assert not app.checkbox(key="segment_birefnet").value
    assert {"**1 · Scale**", "**2 · Segmentation**", "**3 · Measurement Output**"} <= {
        item.value for item in app.markdown
    }
    assert "**4 · Image output options**" in {item.value for item in app.markdown}
    assert app.checkbox(key="export_target_boxes").value
    assert app.checkbox(key="export_cleaned_masks").value
    assert not app.checkbox(key="measure_pre_cleanup").value
    assert app.checkbox(key="write_failures").value
    assert not app.checkbox(key="export_pre_cleanup").value
    assert not app.checkbox(key="export_overlay").value


def test_raw_measurement_checkbox_persists_in_setup():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    app.checkbox(key="measure_pre_cleanup").set_value(True).run(timeout=30)
    assert not app.exception
    assert app.checkbox(key="measure_pre_cleanup").value


def test_cleanup_settings_are_grayed_out_unless_measuring_pre_cleanup():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    margin, gap = app.number_input(key="clean_margin"), app.number_input(key="stray_gap")
    size = app.number_input(key="clean_size")
    assert (margin.value, gap.value, size.value) == (
        CLEAN_MARGIN_DEFAULT, STRAY_GAP_DEFAULT, CLEAN_SIZE_DEFAULT,
    )
    assert margin.disabled and gap.disabled and size.disabled

    app.checkbox(key="measure_pre_cleanup").set_value(True).run(timeout=30)
    assert not app.number_input(key="clean_margin").disabled
    assert not app.number_input(key="stray_gap").disabled
    assert not app.number_input(key="clean_size").disabled
    app.number_input(key="clean_margin").set_value(2.0).run(timeout=30)
    app.number_input(key="stray_gap").set_value(0.6).run(timeout=30)
    app.number_input(key="clean_size").set_value(3).run(timeout=30)
    assert not app.exception
    assert app.number_input(key="clean_margin").value == 2.0
    assert app.number_input(key="stray_gap").value == 0.6
    assert app.number_input(key="clean_size").value == 3
    assert app.session_state["diagnostics_context"]["clean_size"] == 3

    # Unchecked, the grayed-out clean size keeps its value but never reaches a run.
    app.checkbox(key="measure_pre_cleanup").set_value(False).run(timeout=30)
    assert not app.exception
    assert app.number_input(key="clean_size").disabled
    assert app.session_state["diagnostics_context"]["clean_size"] == 0


def _custom_threshold_sliders(app):
    return [slider for slider in app.slider if slider.key == THRESHOLD_CUSTOM_VALUE_KEY]


def test_threshold_level_offers_custom_after_auto_without_a_slider():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    level = app.selectbox(key=THRESHOLD_LEVEL_KEY)
    assert list(level.options) == ["auto", "custom", "low", "medium", "high"]
    assert level.value == "auto"
    assert _custom_threshold_sliders(app) == []


@pytest.mark.parametrize("with_birefnet", [False, True])
def test_custom_threshold_level_shows_slider_while_otsu_is_checked(with_birefnet):
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    app.checkbox(key="segment_birefnet").set_value(with_birefnet).run(timeout=30)
    app.selectbox(key=THRESHOLD_LEVEL_KEY).set_value("custom").run(timeout=30)

    assert not app.exception
    (slider,) = _custom_threshold_sliders(app)
    assert (slider.min, slider.max, slider.value) == (1, 255, 125)
    widget_keys = [getattr(node, "key", None) for node in app._tree]
    assert widget_keys.index("segment_threshold") < widget_keys.index(THRESHOLD_LEVEL_KEY)
    assert widget_keys.index(THRESHOLD_LEVEL_KEY) < widget_keys.index(THRESHOLD_CUSTOM_VALUE_KEY)
    assert widget_keys.index(THRESHOLD_CUSTOM_VALUE_KEY) < widget_keys.index("segment_birefnet")
    slider.set_value(177).run(timeout=30)
    assert app.session_state[THRESHOLD_CUSTOM_VALUE_KEY] == 177


def _threshold_level_boxes(app):
    return [box for box in app.selectbox if box.key == THRESHOLD_LEVEL_KEY]


def test_unchecking_otsu_hides_the_threshold_level_and_keeps_the_choice():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    app.selectbox(key=THRESHOLD_LEVEL_KEY).set_value("custom").run(timeout=30)
    _custom_threshold_sliders(app)[0].set_value(177).run(timeout=30)
    app.checkbox(key="segment_birefnet").check().run(timeout=30)
    app.checkbox(key="segment_threshold").uncheck().run(timeout=30)

    assert not app.exception
    assert _threshold_level_boxes(app) == []
    assert _custom_threshold_sliders(app) == []

    app.checkbox(key="segment_threshold").check().run(timeout=30)
    assert app.selectbox(key=THRESHOLD_LEVEL_KEY).value == "custom"
    assert _custom_threshold_sliders(app)[0].value == 177


def test_no_segmentation_method_blocks_preflight():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    app.checkbox(key="segment_threshold").uncheck().run(timeout=30)

    assert not app.exception
    assert "Choose at least one segmentation method." in [item.value for item in app.warning]
    assert _threshold_level_boxes(app) == []

    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.run(timeout=30)
    assert app.button(key="run_leaf_morphometrics").disabled
    assert any(
        item.value.startswith("Needs attention:") and "Segmentation method" in item.value
        for item in app.caption
    )


def test_threshold_marks_label_each_preset_at_its_track_position():
    html = _threshold_marks_html()

    for label, value, left in (("low", 100, "38.98%"), ("med", 125, "48.82%"),
                               ("high", 150, "58.66%")):
        assert f'left: {left}">{label}<br>{value}</span>' in html


def test_blocking_preflight_links_to_sidebar_diagnostics():
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state[WORKSPACE_TAB_KEY] == "Analyze"
    assert "pages/0_Diagnostics.py" in HOME_PAGE.read_text()


def test_measurement_normalization_supports_full_and_compact_schemas():
    full_schema = pd.DataFrame({
        "sample_id": ["full"],
        "leaf_area_cm2": [12.5],
        "width_cm": [2.5],
        "length_cm": [7.0],
        "scale_aspect_ratio": [1.01],
    })
    compact_schema = pd.DataFrame({
        "sample_id": ["compact"],
        "area_cm2": [8.0],
        "width_cm": [2.0],
        "length_cm": [4.0],
    })

    full = normalize_measurements(full_schema)
    compact = normalize_measurements(compact_schema)

    assert full.loc[0, "leaf_area"] == 12.5
    assert compact.loc[0, "leaf_area"] == 8.0
    assert summarize_measurements(full) == {
        "count": 1,
        "median_area": 12.5,
        "median_width": 2.5,
        "median_length": 7.0,
    }


def test_measurement_normalization_supports_inch_columns():
    frame = pd.DataFrame({
        "sample_id": ["inch"],
        "area_in2": [2.5],
        "width_in": [1.5],
        "length_in": [3.0],
    })

    measurements = normalize_measurements(frame, "in")

    assert measurements.loc[0, "leaf_area"] == 2.5
    assert measurements.loc[0, "width"] == 1.5
    assert measurements.loc[0, "length"] == 3.0


def _last_run(results_paths, succeeded, failed, total, **extra):
    """Build a session ``last_run`` for ``{method: results_path}``."""
    return {
        "succeeded": succeeded,
        "failed": failed,
        "total": total,
        "workers": 1,
        "worker_reason": "test",
        "execution_device": "cpu",
        "output_path": str(Path(next(iter(results_paths.values()))).parent),
        "mask_methods": tuple(results_paths),
        "by_method": {
            method: {
                "succeeded": succeeded,
                "failed": failed,
                "results_path": str(path),
                "failure_rows": [],
                "failure_overflow": 0,
            }
            for method, path in results_paths.items()
        },
        **extra,
    }


def test_results_tab_renders_measurement_dashboard(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,leaf_area_cm2,width_cm,length_cm,px_per_cm_width,"
        "px_per_cm_height,scale_aspect_ratio,source\n"
        "leaf_1,12.5,2.5,7.0,100,100,1.0,0\n"
        "leaf_2,8.0,2.0,4.0,100,101,0.99,0\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=2, failed=0, total=2,
    )
    app.run(timeout=30)

    assert not app.exception
    assert any(item.value == "Results" for item in app.subheader)
    assert [item.label for item in app.metric] == [
        "Successful measurements",
        "Median leaf area",
        "Median leaf width",
        "Median leaf length",
        "Median axis-scale ratio",
        "Within 2% of 1.0",
        "Observed range",
    ]
    assert {"**Measurement Table**", "**Selected Specimen**"} <= {
        item.value for item in app.markdown
    }
    section_titles = [item.value for item in app.markdown]
    table_index = section_titles.index("**Measurement Table**")
    assert section_titles.index("**Leaf Width and Length**") < table_index
    assert section_titles.index("**Leaf Area Distribution**") < table_index
    assert section_titles.index("**Scale Quality Control**") < table_index
    assert "**QR decoder trace**" not in {item.value for item in app.markdown}
    assert any("Select a row in the measurement table" in item.value for item in app.info)


def test_selected_threshold_sample_mounts_interactive_preview(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "sample_id,area_cm2,width_cm,length_cm\n"
        "leaf_1,4.0,2.0,2.0\nleaf_2,8.0,2.0,4.0\n"
    )
    target = tmp_path / "leaf_1_preview_target_box.png"
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image[5:35, 5:35] = 140
    image[10:30, 10:30] = 100
    assert cv2.imwrite(str(target), image)
    mask = tmp_path / "leaf_1_mask.png"
    original_mask = np.zeros((40, 40), dtype=np.uint8)
    original_mask[10:30, 10:30] = 255
    assert cv2.imwrite(str(mask), original_mask)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=2, failed=0, total=2,
        run_id="preview-test", threshold_value=125, measurement_source="cleaned",
        results_unit="cm", scale_axes_by_sample={"threshold": {"leaf_1": (10, 10)}},
        artifacts=[{
            "path": str(mask), "sample_id": "leaf_1",
            "kind": "mask", "method": "threshold",
        }],
    )
    app.session_state["threshold_preview_cutoffs"] = {"preview-test:leaf_1": 150}
    app.session_state["viewer_pairs"] = {
        "threshold": [{
            "sample_id": "leaf_1", "target_box": str(target), "mask": str(mask),
        }]
    }
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected):
        app.run(timeout=30)

    assert not app.exception
    assert "**Explore and adjust output**" in {item.value for item in app.markdown}
    assert app.button(key="reset_preview-test:leaf_1").label == "Reset to saved threshold"
    assert not app.button(key="use_preview-test:leaf_1").disabled
    assert not app.button(key="apply_adjustment_preview-test:leaf_1").disabled
    with patch("streamlit.dataframe", return_value=selected):
        app.button(key="use_preview-test:leaf_1").click().run(timeout=30)
    assert not app.exception
    assert app.session_state[THRESHOLD_LEVEL_KEY] == "custom"
    assert app.session_state[THRESHOLD_CUSTOM_VALUE_KEY] == 150

    with patch("streamlit.dataframe", return_value=selected):
        app.button(key="apply_adjustment_preview-test:leaf_1").click().run(timeout=30)
    assert not app.exception
    adjusted = pd.read_csv(results_path)
    assert adjusted.loc[0, ["area_cm2", "width_cm", "length_cm"]].tolist() == [9.0, 3.0, 3.0]
    assert adjusted.loc[1].to_dict() == {
        "sample_id": "leaf_2", "area_cm2": 8.0, "width_cm": 2.0, "length_cm": 4.0,
    }
    saved_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    assert saved_mask[7, 7] == 255
    assert app.session_state["last_run"]["threshold_adjustments"]["leaf_1"] == {
        "cutoff": 150, "fill_holes": True,
    }


def test_bulk_adjustment_updates_only_marked_specimens_and_preserves_each_scale(tmp_path):
    from mats.app.output_adjustment import apply_threshold_adjustments

    results = tmp_path / "results.csv"
    results.write_text(
        "sample_id,area_cm2,width_cm,length_cm\n"
        "leaf_1,1,1,1\nleaf_2,2,2,2\nleaf_3,3,3,3\n"
    )
    pairs = []
    artifacts = []
    for sample_id in ("leaf_1", "leaf_2", "leaf_3"):
        target = tmp_path / f"{sample_id}_target_box.png"
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[5:35, 5:35] = 100
        assert cv2.imwrite(str(target), image)
        mask = tmp_path / f"{sample_id}_mask.png"
        assert cv2.imwrite(str(mask), np.zeros((40, 40), dtype=np.uint8))
        pairs.append({"sample_id": sample_id, "target_box": str(target), "mask": str(mask)})
        artifacts.append({"path": str(mask), "sample_id": sample_id,
                          "kind": "mask", "method": "threshold"})
    run = _last_run(
        {"threshold": results}, succeeded=3, failed=0, total=3,
        results_unit="cm", measurement_source="cleaned", artifacts=artifacts,
        scale_axes_by_sample={"threshold": {
            "leaf_1": (10, 10), "leaf_2": (20, 20), "leaf_3": (10, 10),
        }},
    )

    assert apply_threshold_adjustments(run, pairs[:2], 125) == ["leaf_1", "leaf_2"]

    saved = pd.read_csv(results).set_index("sample_id")
    assert saved.loc["leaf_1", "area_cm2"] == 9.0
    assert saved.loc["leaf_2", "area_cm2"] == 2.25
    assert saved.loc["leaf_3", "area_cm2"] == 3.0
    assert cv2.imread(pairs[0]["mask"], 0)[10, 10] == 255
    assert cv2.imread(pairs[1]["mask"], 0)[10, 10] == 255
    assert not cv2.imread(pairs[2]["mask"], 0).any()


def test_adjust_tab_overwrites_marked_specimens_from_their_own_button(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "sample_id,area_cm2,width_cm,length_cm\n"
        "leaf_1,1,1,1\nleaf_2,2,2,2\n"
    )
    pairs = []
    for sample_id in ("leaf_1", "leaf_2"):
        target = tmp_path / f"{sample_id}_target_box.png"
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[5:35, 5:35] = 100
        assert cv2.imwrite(str(target), image)
        mask = tmp_path / f"{sample_id}_mask.png"
        assert cv2.imwrite(str(mask), np.zeros((40, 40), dtype=np.uint8))
        pairs.append({"sample_id": sample_id, "target_box": str(target), "mask": str(mask)})
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"threshold": results}, succeeded=2, failed=0, total=2,
        run_id="bulk-ui", results_unit="cm", measurement_source="cleaned",
        threshold_value=125,
        scale_axes_by_sample={"threshold": {"leaf_1": (10, 10), "leaf_2": (20, 20)}},
    )
    app.session_state["viewer_pairs"] = {"threshold": pairs}
    app.run(timeout=30)

    assert not app.exception
    assert not app.checkbox                      # no bulk switch any more
    assert app.button(key="apply_adjustment_bulk-ui:leaf_1").label == "Overwrite this specimen"
    marked_button = app.button(key="apply_marked_adjustment_bulk-ui:leaf_1")
    assert marked_button.label == "Overwrite all marked specimens (0)"
    assert marked_button.disabled

    # Overwrite this specimen saves only the one in View, even with others marked.
    app.session_state["adjust_marked_bulk-ui_threshold"] = ["leaf_2"]
    app.run(timeout=30)
    assert app.button(key="apply_marked_adjustment_bulk-ui:leaf_1").label == (
        "Overwrite all marked specimens (1)"
    )
    app.button(key="apply_adjustment_bulk-ui:leaf_1").click().run(timeout=30)
    saved = pd.read_csv(results).set_index("sample_id")
    assert saved.loc["leaf_1", "area_cm2"] == 9.0
    assert saved.loc["leaf_2", "area_cm2"] == 2.0

    next(button for button in app.button if button.label == "Mark all").click().run(timeout=30)
    marked_button = app.button(key="apply_marked_adjustment_bulk-ui:leaf_1")
    assert marked_button.label == "Overwrite all marked specimens (2)"
    assert not marked_button.disabled
    assert app.button(key="apply_adjustment_bulk-ui:leaf_1").label == "Overwrite this specimen"
    marked_button.click().run(timeout=30)

    assert not app.exception
    saved = pd.read_csv(results).set_index("sample_id")
    assert saved.loc["leaf_1", "area_cm2"] == 9.0
    assert saved.loc["leaf_2", "area_cm2"] == 2.25


def test_bulk_adjustment_restores_previous_outputs_if_a_save_fails(tmp_path):
    from mats.app import output_adjustment

    results = tmp_path / "results.csv"
    original_csv = (
        "sample_id,area_cm2,width_cm,length_cm\n"
        "leaf_1,1,1,1\nleaf_2,2,2,2\n"
    )
    results.write_text(original_csv)
    pairs = []
    artifacts = []
    for sample_id in ("leaf_1", "leaf_2"):
        target = tmp_path / f"{sample_id}_target_box.png"
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[5:35, 5:35] = 100
        assert cv2.imwrite(str(target), image)
        mask = tmp_path / f"{sample_id}_mask.png"
        assert cv2.imwrite(str(mask), np.zeros((40, 40), dtype=np.uint8))
        pairs.append({"sample_id": sample_id, "target_box": str(target), "mask": str(mask)})
        artifacts.append({"path": str(mask), "sample_id": sample_id,
                          "kind": "mask", "method": "threshold"})
    run = _last_run(
        {"threshold": results}, succeeded=2, failed=0, total=2,
        results_unit="cm", measurement_source="cleaned", artifacts=artifacts,
        scale_axes_by_sample={"threshold": {"leaf_1": (10, 10), "leaf_2": (10, 10)}},
    )
    actual = output_adjustment.apply_threshold_adjustment

    def fail_second(*args, **kwargs):
        if args[1]["sample_id"] == "leaf_2":
            raise OSError("simulated write failure")
        return actual(*args, **kwargs)

    with patch.object(output_adjustment, "apply_threshold_adjustment", side_effect=fail_second):
        with pytest.raises(OSError, match="simulated write failure"):
            output_adjustment.apply_threshold_adjustments(run, pairs, 125)

    assert results.read_text() == original_csv
    assert all(not cv2.imread(pair["mask"], 0).any() for pair in pairs)
    assert not Path(f"{results}.meta.json").exists()
    assert run["artifacts"] == artifacts[:2]
    assert run["threshold_adjustments"] == {}


def test_scale_axes_by_sample_retains_successful_canonical_calibration():
    summary = {
        "by_method": {
            "threshold": {"result_rows": [
                {"sample_id": "leaf_1", "px_per_cm_width": 10, "px_per_cm_height": 12},
                {"sample_id": "failed", "px_per_cm_width": "NA", "px_per_cm_height": "NA"},
            ]},
        },
    }

    assert _scale_axes_by_sample(summary) == {"threshold": {"leaf_1": (10.0, 12.0)}}


def test_specimen_explorer_shows_only_selected_row_and_can_remove_flashfill(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "sample_id,leaf_area_cm2,width_cm,length_cm\n"
        "leaf_1,12.5,2.5,7.0\nleaf_2,8.0,2.0,4.0\n"
    )
    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[20:80, 20:80] = 255
    raw[40:60, 40:60] = 0
    pairs = []
    for sample_id in ("leaf_1", "leaf_2"):
        target = tmp_path / f"{sample_id}_target_box.png"
        raw_path = tmp_path / f"{sample_id}_raw.png"
        mask = tmp_path / f"{sample_id}_mask.png"
        assert cv2.imwrite(str(target), cv2.cvtColor(255 - raw, cv2.COLOR_GRAY2BGR))
        assert cv2.imwrite(str(raw_path), raw)
        assert cv2.imwrite(str(mask), raw)
        pairs.append({
            "sample_id": sample_id, "target_box": str(target),
            "raw_mask": str(raw_path), "mask": str(mask),
        })

    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=2, failed=0, total=2,
        run_id="one-sample", threshold_value=125, measurement_source="cleaned",
    )
    app.session_state["viewer_pairs"] = {"threshold": pairs}
    app.session_state["selected_specimen_one-sample_threshold"] = "leaf_2"
    mounted = []
    preview = patch(
        "mats.app.threshold_preview.show_threshold_preview",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    )
    with preview:
        app.run(timeout=30)

    assert not app.exception
    assert "Sample: leaf_2" in {item.value for item in app.caption}
    assert "Sample: leaf_1" not in {item.value for item in app.caption}
    assert "Browse Output Previews" not in {item.label for item in app.expander}
    # Remove flashfill sits in the explorer under Clean size, not above it.
    assert not [box for box in app.checkbox if "flashfill" in (box.key or "")]
    assert mounted[-1]["fill_toggle"] and mounted[-1]["fill_note"] is None
    assert not mounted[-1]["remove_fill"]

    # Ticking it in the component records it for this specimen only.
    app.session_state["remove_fill_previews"] = {"one-sample:threshold:leaf_2": True}
    with preview:
        app.run(timeout=30)
    assert not app.exception
    assert mounted[-1]["remove_fill"]
    assert "Sample: leaf_1" not in {item.value for item in app.caption}

    # Clicking leaf_1's row in the table records it as the viewed specimen.
    app.session_state["selected_specimen_one-sample_threshold"] = "leaf_1"
    with preview:
        app.run(timeout=30)
    assert "Sample: leaf_1" in {item.value for item in app.caption}
    assert "Sample: leaf_2" not in {item.value for item in app.caption}
    assert not mounted[-1]["remove_fill"]


def test_remove_flashfill_component_value_is_recorded_per_specimen():
    app = AppTest.from_string('''
import streamlit as st
from mats.app.Home import _record_remove_fill, _remove_fill
st.session_state["preview"] = {"remove_fill": True}
_record_remove_fill("preview", "run:threshold:leaf_1")
st.session_state["cleaned"] = _remove_fill({"measurement_source": "cleaned"}, "run:threshold:leaf_1")
st.session_state["raw"] = _remove_fill({"measurement_source": "pre-cleanup"}, "run:threshold:leaf_1")
''').run(timeout=30)

    assert not app.exception
    assert app.session_state["remove_fill_previews"] == {"run:threshold:leaf_1": True}
    assert app.session_state["cleaned"] is True
    assert app.session_state["raw"] is False   # raw-mask runs never fill holes


def test_analyze_selection_opens_same_specimen_in_adjust(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "sample_id,area_cm2,width_cm,length_cm\n"
        "leaf_1,1,1,1\nleaf_2,2,2,2\n"
    )
    pairs = []
    for sample_id in ("leaf_1", "leaf_2"):
        target = tmp_path / f"{sample_id}_target_box.png"
        mask = tmp_path / f"{sample_id}_mask.png"
        assert cv2.imwrite(str(target), np.full((20, 20, 3), 255, dtype=np.uint8))
        assert cv2.imwrite(str(mask), np.zeros((20, 20), dtype=np.uint8))
        pairs.append({"sample_id": sample_id, "target_box": str(target), "mask": str(mask)})
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.session_state["last_run"] = _last_run(
        {"threshold": results}, succeeded=2, failed=0, total=2,
        run_id="carry-selection", results_unit="cm", measurement_source="cleaned",
        threshold_value=125,
    )
    app.session_state["viewer_pairs"] = {"threshold": pairs}
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[1]))
    with patch("streamlit.dataframe", return_value=selected):
        app.run(timeout=30)
        next(button for button in app.button if button.label == "Adjust selected specimen").click().run(timeout=30)

    assert not app.exception
    assert app.session_state[WORKSPACE_TAB_KEY] == "Adjust"
    assert app.session_state["selected_specimen_carry-selection_threshold"] == "leaf_2"
    assert "**Selected Specimen**" not in {item.value for item in app.markdown}
    assert not app.metric
    assert not app.selectbox
    assert not app.multiselect


def _mount_capture(mounted):
    return patch(
        "mats.app.specimen_table.show_specimen_table",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    )


def test_adjust_browser_handles_hundreds_and_keeps_hidden_marks():
    app = AppTest.from_string('''
from mats.app.Home import render_adjust_browser
import streamlit as st
choices = [f"leaf_{index:03d}" for index in range(500)]
st.session_state["browser_result"] = render_adjust_browser(choices, "large", "threshold")
''')
    mounted = []
    with _mount_capture(mounted):
        app.run(timeout=30)
        assert not app.exception
        # One scrollable table: no pages and no per-name widgets.
        assert len(mounted[-1]["rows"]) == 500
        assert mounted[-1]["markable"] and mounted[-1]["view"] == "leaf_000"
        assert not app.checkbox and not app.number_input and not app.selectbox

        # A tick made in the table survives searches that hide it.
        app.session_state["adjust_marked_large_threshold"] = ["leaf_025"]
        app.text_input(key="adjust_search_large_threshold").set_value("LEAF_49").run(timeout=30)
        assert [row["sample_id"] for row in mounted[-1]["rows"]] == [
            f"leaf_{index}" for index in range(490, 500)
        ]
        assert mounted[-1]["marked"] == ["leaf_025"]
        app.button(key="adjust_mark_all_large_threshold").click().run(timeout=30)
        assert len(app.session_state["browser_result"][1]) == 11

        shown = len(mounted)
        app.text_input(key="adjust_search_large_threshold").set_value("no-match").run(timeout=30)
        assert len(mounted) == shown                    # no table, just the notice
        assert any("No specimen names match" in item.value for item in app.info)
        assert len(app.session_state["browser_result"][1]) == 11

        app.text_input(key="adjust_search_large_threshold").set_value("").run(timeout=30)
        assert len(mounted[-1]["marked"]) == 11
        assert "11 marked across all searches." in {item.value for item in app.caption}
        app.button(key="adjust_clear_marks_large_threshold").click().run(timeout=30)
        assert app.session_state["browser_result"][1] == []
        assert mounted[-1]["marked"] == []
    assert not app.exception


def _adjust_table_script(method):
    return f'''
import pandas as pd
import streamlit as st
from mats.app.Home import render_adjust_browser
measurements = pd.DataFrame({{
    "sample_id": ["leaf_04", "leaf_40", "leaf_41", "leaf_50"],
    "leaf_area": [12.5, 8.0, 9.1, 3.0],
    "width": [2.5, 2.0, 2.2, 1.0],
    "length": [7.0, 4.0, 4.4, 3.0],
    "scale_aspect_ratio": [1.0, None, 0.99, 1.0],
}})
st.session_state["browser_result"] = render_adjust_browser(
    measurements["sample_id"].tolist(), "tbl", "{method}", measurements=measurements,
)
'''


def test_adjust_table_shows_measurements_with_view_and_marked_columns():
    app = AppTest.from_string(_adjust_table_script("threshold"))
    app.session_state["selected_specimen_tbl_threshold"] = "leaf_41"
    app.session_state["adjust_marked_tbl_threshold"] = ["leaf_50"]
    mounted = []
    with _mount_capture(mounted):
        app.run(timeout=30)
        assert not app.exception
        table = mounted[-1]
        assert [column["label"] for column in table["columns"]] == [
            "Sample", "Leaf area (cm²)", "Leaf width (cm)", "Leaf length (cm)",
            "Scale axis ratio",
        ]
        assert [row["leaf_area"] for row in table["rows"]] == [12.5, 8.0, 9.1, 3.0]
        assert table["rows"][1]["scale_aspect_ratio"] is None      # blank, not NaN
        assert table["view"] == "leaf_41"
        assert table["markable"] and table["marked"] == ["leaf_50"]
        # The old Mark/Unmark button is gone: the boxes are clickable now.
        assert not [b for b in app.button if b.key.startswith("adjust_toggle_mark_")]

        app.text_input(key="adjust_search_tbl_threshold").set_value("LEAF_4").run(timeout=30)
        assert [row["sample_id"] for row in mounted[-1]["rows"]] == ["leaf_40", "leaf_41"]
        assert "1 marked across all searches." in {item.value for item in app.caption}
    assert app.session_state["browser_result"] == ("leaf_41", ["leaf_50"])

    from mats.app import specimen_table
    assert "'Marked for Adjustment'" in specimen_table._TABLE_JS
    assert "headerCell('View'" in specimen_table._TABLE_JS


def test_specimen_table_clicks_set_the_view_and_marks():
    app = AppTest.from_string('''
import streamlit as st
from mats.app.Home import _record_table_mark, _record_table_view
st.session_state["marks"] = ["leaf_2"]
st.session_state["table"] = {"view": "leaf_3"}
_record_table_view("table", "selected")
st.session_state["table"] = {"mark": {"id": "leaf_1", "marked": True}}
_record_table_mark("table", "marks")
_record_table_mark("table", "marks")            # a repeated tick changes nothing
st.session_state["after_tick"] = list(st.session_state["marks"])
st.session_state["table"] = {"mark": {"id": "leaf_2", "marked": False}}
_record_table_mark("table", "marks")
''').run(timeout=30)

    assert not app.exception
    assert app.session_state["selected"] == "leaf_3"
    assert app.session_state["after_tick"] == ["leaf_1", "leaf_2"]
    assert app.session_state["marks"] == ["leaf_1"]


def test_adjust_table_has_no_marking_for_birefnet():
    app = AppTest.from_string(_adjust_table_script("birefnet"))
    mounted = []
    with _mount_capture(mounted):
        app.run(timeout=30)

    assert not app.exception
    assert not mounted[-1]["markable"]
    assert not [
        button for button in app.button
        if button.key.startswith(("adjust_mark_all_", "adjust_clear_marks_"))
    ]


def test_clean_image_preview_is_saved_by_overwrite(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2,width_cm,length_cm\nleaf_1,4.0,2.0,2.0\n")
    target = tmp_path / "leaf_1_preview_target_box.png"
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image[5:35, 5:35] = 100
    image[10:22, 10:22] = 255                         # hole, inscribed radius 6
    image[2, 2] = 0                                   # 1 px speck beside the leaf
    assert cv2.imwrite(str(target), image)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
        run_id="clean-test", threshold_value=125, measurement_source="cleaned",
        results_unit="cm", scale_axes_by_sample={"threshold": {"leaf_1": (10, 10)}},
    )
    app.session_state["viewer_pairs"] = {
        "threshold": [{"sample_id": "leaf_1", "target_box": str(target), "mask": None}]
    }
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    mounted = []
    with patch("streamlit.dataframe", return_value=selected), patch(
        "mats.app.threshold_preview.show_threshold_preview",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    ):
        app.run(timeout=30)
        # No Clean image checkbox: the clean-size slider starts at 0, already live.
        assert not any("clean_image" in (box.key or "") for box in app.checkbox)
        assert mounted[-1]["clean_radius"] == 0
        assert mounted[-1]["clean_levels_image"] and mounted[-1]["cleaned_image"]
        assert "0 turns Clean image off" in mounted[-1]["clean_help"]
        assert "Overwrite saves" in mounted[-1]["clean_help"]
        assert not app.button(key="apply_adjustment_clean-test:leaf_1").disabled

        # Releasing the slider above 0 records it, and Overwrite saves that mask.
        app.session_state["clean_preview_radii"] = {"clean-test:threshold:leaf_1": 5}
        app.run(timeout=30)
        assert mounted[-1]["clean_radius"] == 5
        assert any("Overwrite saves it" in item.value for item in app.caption)
        assert not app.button(key="apply_adjustment_clean-test:leaf_1").disabled
        app.button(key="apply_adjustment_clean-test:leaf_1").click().run(timeout=30)

    assert not app.exception
    assert any("clean size 5 px" in item.value for item in app.success)
    saved = pd.read_csv(results_path).set_index("sample_id")
    assert saved.loc["leaf_1", "area_cm2"] == pytest.approx((900 - 144) / 100)
    mask = cv2.imread(str(tmp_path / "leaf_1_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[15, 15] == 0 and mask[2, 2] == 0 and mask[30, 30] == 255
    assert app.session_state["last_run"]["threshold_adjustments"]["leaf_1"] == {
        "cutoff": 125, "fill_holes": False, "clean_margin": CLEAN_MARGIN_DEFAULT,
        "stray_gap": STRAY_GAP_DEFAULT, "clean_size": 5,
    }


@pytest.mark.parametrize("saved, expected", [(None, 3), ({"cutoff": 125, "clean_size": 0}, 0)])
def test_clean_size_slider_starts_at_the_saved_or_run_clean_size(tmp_path, saved, expected):
    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2,width_cm,length_cm\nleaf_1,4.0,2.0,2.0\n")
    target = tmp_path / "leaf_1_preview_target_box.png"
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image[5:35, 5:35] = 100
    assert cv2.imwrite(str(target), image)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
        run_id="run-size", threshold_value=125, measurement_source="pre-cleanup",
        clean_size=3, threshold_adjustments={"leaf_1": saved} if saved else {},
    )
    app.session_state["viewer_pairs"] = {"threshold": [{
        "sample_id": "leaf_1", "target_box": str(target), "mask": None,
        "mask_source": "pre-cleanup",
    }]}
    mounted = []
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected), patch(
        "mats.app.threshold_preview.show_threshold_preview",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    ):
        app.run(timeout=30)

    assert not app.exception
    assert mounted[-1]["clean_radius"] == expected


def test_pre_cleanup_explorer_settles_on_the_measured_mask(tmp_path):
    import base64

    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2,width_cm,length_cm\nleaf_1,4.0,2.0,2.0\n")
    target = tmp_path / "leaf_1_preview_target_box.png"
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[0:2, 10:90] = 0                             # printed outline on the edge
    image[30:70, 30:70] = 100
    assert cv2.imwrite(str(target), image)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
        run_id="raw-test", threshold_value=125, measurement_source="pre-cleanup",
        clean_margin=2.0, stray_gap=0.25,
    )
    app.session_state["viewer_pairs"] = {"threshold": [{
        "sample_id": "leaf_1", "target_box": str(target), "mask": None,
        "mask_source": "pre-cleanup",
    }]}
    mounted = []
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected), patch(
        "mats.app.threshold_preview.show_threshold_preview",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    ):
        app.run(timeout=30)

    assert not app.exception
    preview = mounted[-1]
    assert preview["fill_toggle"] and not preview["remove_fill"]
    assert preview["fill_note"] == "This run measured raw masks, so hole filling is already off."
    assert preview["live_margin"] == 2.0
    assert preview["cleaned_cutoff"] == 125
    settled = cv2.imdecode(np.frombuffer(
        base64.b64decode(preview["cleaned_image"].split(",", 1)[1]), dtype=np.uint8,
    ), cv2.IMREAD_GRAYSCALE)
    assert not settled[:2].any() and settled[50, 50] == 255

    # The specimen's own margin re-cleans the preview; 0 leaves the edge line in.
    margin_key = "specimen_clean_margin_raw-test:threshold:leaf_1"
    assert not app.number_input(key=margin_key).disabled
    with patch("streamlit.dataframe", return_value=selected), patch(
        "mats.app.threshold_preview.show_threshold_preview",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    ):
        app.number_input(key=margin_key).set_value(0.0).run(timeout=30)
    assert not app.exception
    assert mounted[-1]["live_margin"] == 0
    unguarded = cv2.imdecode(np.frombuffer(
        base64.b64decode(mounted[-1]["cleaned_image"].split(",", 1)[1]), dtype=np.uint8,
    ), cv2.IMREAD_GRAYSCALE)
    assert unguarded[0, 50] == 0 and unguarded[50, 50] == 255   # a stray edge strip


def test_specimen_cleanup_inputs_follow_flash_fill(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2,width_cm,length_cm\nleaf_1,4.0,2.0,2.0\n")
    target = tmp_path / "leaf_1_preview_target_box.png"
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image[5:35, 5:35] = 100
    assert cv2.imwrite(str(target), image)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
        run_id="flash", threshold_value=125, measurement_source="cleaned",
    )
    app.session_state["viewer_pairs"] = {
        "threshold": [{"sample_id": "leaf_1", "target_box": str(target), "mask": None}]
    }
    margin_key = "specimen_clean_margin_flash:threshold:leaf_1"
    gap_key = "specimen_stray_gap_flash:threshold:leaf_1"
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected):
        app.run(timeout=30)
        assert app.number_input(key=margin_key).disabled       # flash fill is on
        assert app.number_input(key=gap_key).disabled

        app.session_state["remove_fill_previews"] = {"flash:threshold:leaf_1": True}
        app.run(timeout=30)
        assert not app.number_input(key=margin_key).disabled   # margin only
        assert app.number_input(key=gap_key).disabled
        # The Remove flashfill preview above reruns with the new margin.
        app.number_input(key=margin_key).set_value(5.0).run(timeout=30)
        assert not app.exception
        assert app.number_input(key=margin_key).value == 5.0

        app.session_state["clean_preview_radii"] = {"flash:threshold:leaf_1": 3}
        app.run(timeout=30)
        assert not app.number_input(key=margin_key).disabled
        assert not app.number_input(key=gap_key).disabled
    assert not app.exception


def test_birefnet_pre_cleanup_explorer_recleans_the_raw_mask(tmp_path):
    import base64

    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2,width_cm,length_cm\nleaf_1,4.0,2.0,2.0\n")
    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[0:2, 10:90] = 255                             # printed outline on the edge
    raw[30:70, 30:70] = 255
    raw_path = tmp_path / "leaf_1_preview_raw_mask.png"
    assert cv2.imwrite(str(raw_path), raw)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"birefnet": results_path}, succeeded=1, failed=0, total=1,
        run_id="bir-raw", measurement_source="pre-cleanup", clean_margin=2.0,
    )
    app.session_state["viewer_pairs"] = {"birefnet": [{
        "sample_id": "leaf_1", "target_box": None, "mask": None,
        "raw_mask": str(raw_path), "mask_source": "pre-cleanup",
    }]}
    mounted = []
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected), patch(
        "mats.app.threshold_preview.show_threshold_preview",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    ):
        app.run(timeout=30)

    assert not app.exception
    shown = cv2.imdecode(np.frombuffer(
        base64.b64decode(mounted[-1]["mask_image"].split(",", 1)[1]), dtype=np.uint8,
    ), cv2.IMREAD_GRAYSCALE)
    assert not shown[:2].any() and shown[50, 50] == 255
    assert "edge margin cleared" in mounted[-1]["mask_status"]
    assert not app.number_input(key="specimen_stray_gap_bir-raw:birefnet:leaf_1").disabled


def test_birefnet_specimen_gets_a_preview_only_clean_image_explorer(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2,width_cm,length_cm\nleaf_1,4.0,2.0,2.0\n")
    raw = np.zeros((40, 40), dtype=np.uint8)
    raw[5:35, 5:35] = 255
    raw[18:21, 18:21] = 0
    raw[1, 1] = 255
    raw_path = tmp_path / "leaf_1_preview_raw_mask.png"
    mask_path = tmp_path / "leaf_1_mask.png"
    target = tmp_path / "leaf_1_preview_target_box.png"
    assert cv2.imwrite(str(raw_path), raw)
    assert cv2.imwrite(str(mask_path), raw)
    assert cv2.imwrite(str(target), cv2.cvtColor(255 - raw, cv2.COLOR_GRAY2BGR))
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"birefnet": results_path}, succeeded=1, failed=0, total=1,
        run_id="bir-test", measurement_source="cleaned",
    )
    app.session_state["viewer_pairs"] = {"birefnet": [{
        "sample_id": "leaf_1", "target_box": str(target),
        "mask": str(mask_path), "raw_mask": str(raw_path),
    }]}
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected):
        app.run(timeout=30)
        assert not app.exception
        assert "**Explore and adjust output**" in {item.value for item in app.markdown}
        assert any("Drag the clean size above 0" in item.value for item in app.caption)
        app.session_state["clean_preview_radii"] = {"bir-test:birefnet:leaf_1": 4}
        app.run(timeout=30)

    assert not app.exception
    assert any("Clean size is above 0" in item.value for item in app.caption)
    assert not any("apply_adjustment" in (button.key or "") for button in app.button)


def test_birefnet_clean_image_help_says_it_is_preview_only(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2,width_cm,length_cm\nleaf_1,4.0,2.0,2.0\n")
    raw = np.zeros((40, 40), dtype=np.uint8)
    raw[5:35, 5:35] = 255
    raw_path = tmp_path / "leaf_1_preview_raw_mask.png"
    assert cv2.imwrite(str(raw_path), raw)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Adjust"
    app.session_state["last_run"] = _last_run(
        {"birefnet": results_path}, succeeded=1, failed=0, total=1,
        run_id="bir-help", measurement_source="pre-cleanup", clean_size=4,
    )
    app.session_state["viewer_pairs"] = {"birefnet": [{
        "sample_id": "leaf_1", "target_box": None, "mask": None,
        "raw_mask": str(raw_path), "mask_source": "pre-cleanup",
    }]}
    mounted = []
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected), patch(
        "mats.app.threshold_preview.show_threshold_preview",
        side_effect=lambda **kwargs: mounted.append(kwargs),
    ):
        app.run(timeout=30)

    assert not app.exception
    assert mounted[-1]["clean_radius"] == 4                 # the run's clean size
    assert "preview it only" in mounted[-1]["clean_help"]
    assert "Overwrite" not in mounted[-1]["clean_help"]


def test_results_tab_uses_the_completed_run_unit(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,area_in2,width_in,length_in\n"
        "leaf_1,2.5,1.5,3.0\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1, results_unit="in",
    )
    app.run(timeout=30)

    assert not app.exception
    assert app.metric[1].value == "2.50 in²"
    assert app.metric[2].value == "1.50 in"
    assert not app.download_button


def test_export_tab_lists_saved_results_and_downloads(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,leaf_area_cm2,width_cm,length_cm\nleaf_1,12.5,2.5,7.0\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Export"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
        artifacts=[{"path": str(results_path), "kind": "results_csv", "method": "threshold"}],
    )
    app.run(timeout=30)

    assert not app.exception
    assert any(item.value == "Export" for item in app.subheader)
    markdown_values = {item.value for item in app.markdown}
    assert "**Files to include in ZIP**" in markdown_values
    assert "**Measurement CSVs**" in markdown_values
    assert app.download_button[0].label == f"Download {results_path.name}"
    assert app.checkbox(key="export_include_results_csv").value
    assert app.checkbox(key="export_include_overlay").disabled


def test_export_tab_explains_what_to_do_before_a_run():
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Export"
    app.run(timeout=30)

    assert not app.exception
    assert any("Run an analysis" in item.value for item in app.info)
    assert "prepare_zip_export" not in {button.key for button in app.button}


def test_export_tab_prepares_training_dataset_from_current_pairs(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("sample_id,area_cm2\nleaf,1\n")
    image = np.full((16, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[3:13, 4:15] = 255
    image_path = tmp_path / "leaf_target_box.png"
    mask_path = tmp_path / "leaf_mask.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Export"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
        artifacts=[{"path": str(results_path), "kind": "results_csv", "method": "threshold"}],
    )
    app.session_state["viewer_pairs"] = {"threshold": [{
        "sample_id": "leaf", "target_box": str(image_path), "mask": str(mask_path),
    }]}
    app.run(timeout=30)
    assert not app.exception
    assert not app.button(key="prepare_training_dataset").disabled
    app.button(key="prepare_training_dataset").click().run(timeout=30)
    assert not app.exception
    prepared = Path(app.session_state["dataset_zip_path"])
    with zipfile.ZipFile(prepared) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["counts"] == {"train": 1, "val": 0, "test": 0}
        assert archive.read(manifest["samples"][0]["label"])
    assert any(button.key == "download_training_dataset" for button in app.download_button)


def test_export_selection_filters_methods_and_missing_files(tmp_path):
    paths = {
        name: tmp_path / name for name in (
            "leaf_target_box.jpg", "results_threshold.csv", "results_birefnet.csv",
            "leaf_mask_threshold.png", "leaf_mask_birefnet.png",
        )
    }
    for path in paths.values():
        path.write_bytes(path.name.encode())
    artifacts = [
        {"path": str(paths["leaf_target_box.jpg"]), "kind": "target_box", "method": None},
        {"path": str(paths["results_threshold.csv"]), "kind": "results_csv", "method": "threshold"},
        {"path": str(paths["results_birefnet.csv"]), "kind": "results_csv", "method": "birefnet"},
        {"path": str(paths["leaf_mask_threshold.png"]), "kind": "mask", "method": "threshold"},
        {"path": str(paths["leaf_mask_birefnet.png"]), "kind": "mask", "method": "birefnet"},
        {"path": str(tmp_path / "missing_overlay.jpg"), "kind": "overlay", "method": "threshold"},
    ]
    artifacts.append(artifacts[0])  # Shared target box appears only once in the ZIP.

    selected = select_export_files(
        artifacts, ("birefnet",), ("target_box", "results_csv", "mask", "overlay")
    )
    assert [path.name for path in selected] == [
        "leaf_target_box.jpg", "results_birefnet.csv", "leaf_mask_birefnet.png",
    ]
    assert select_export_files(artifacts, (), ("target_box",)) == []
    assert zip_download_name("folder\\leaf_results") == "leaf_results.zip"

    dest = tmp_path / "chosen.zip"
    write_output_zip(selected, dest)
    with zipfile.ZipFile(dest) as archive:
        assert archive.namelist() == [path.name for path in selected]


def test_export_prepared_zip_is_cleared_when_selection_changes(tmp_path):
    results_path = tmp_path / "results.csv"
    mask_path = tmp_path / "leaf_mask.png"
    results_path.write_text("sample_id,area_cm2\nleaf,1\n")
    mask_path.write_bytes(b"mask")
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Export"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=0, total=1,
        run_id="export-test",
        artifacts=[
            {"path": str(results_path), "kind": "results_csv", "method": "threshold"},
            {"path": str(mask_path), "kind": "mask", "method": "threshold"},
        ],
    )
    app.run(timeout=30)
    assert not app.exception
    app.button(key="prepare_zip_export").click().run(timeout=30)
    assert not app.exception
    prepared = Path(app.session_state["export_zip_path"])
    assert prepared.is_file()
    with zipfile.ZipFile(prepared) as archive:
        assert set(archive.namelist()) == {"results.csv", "leaf_mask.png"}

    app.text_input(key="export_zip_name").set_value("selected_leaves").run(timeout=30)
    assert not app.exception
    assert "export_zip_path" not in app.session_state
    assert not prepared.exists()

    app.button(key="prepare_zip_export").click().run(timeout=30)
    prepared = Path(app.session_state["export_zip_path"])
    assert app.download_button[-1].label == "Download ZIP (2 files)"
    app.checkbox(key="export_include_mask").set_value(False).run(timeout=30)
    assert not app.exception
    assert "export_zip_path" not in app.session_state
    assert not prepared.exists()

    app.session_state["last_run"] = {
        **app.session_state["last_run"], "run_id": "next-run",
    }
    app.run(timeout=30)
    assert not app.exception
    assert app.checkbox(key="export_include_mask").value
    assert app.text_input(key="export_zip_name").value == "leaf_morpho_outputs.zip"


def test_export_method_choice_filters_direct_csv_and_zip(tmp_path):
    threshold_csv = tmp_path / "results_threshold.csv"
    birefnet_csv = tmp_path / "results_birefnet.csv"
    target_box = tmp_path / "leaf_target_box.jpg"
    for path in (threshold_csv, birefnet_csv, target_box):
        path.write_bytes(path.name.encode())
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Export"
    app.session_state["last_run"] = _last_run(
        {"threshold": threshold_csv, "birefnet": birefnet_csv},
        succeeded=1, failed=0, total=1, run_id="dual-method-export",
        artifacts=[
            {"path": str(target_box), "kind": "target_box", "method": None},
            {"path": str(threshold_csv), "kind": "results_csv", "method": "threshold"},
            {"path": str(birefnet_csv), "kind": "results_csv", "method": "birefnet"},
        ],
    )
    app.run(timeout=30)
    assert not app.exception
    assert len(app.download_button) == 2

    app.multiselect(key="export_selected_methods").set_value(["birefnet"]).run(timeout=30)
    assert not app.exception
    assert [button.label for button in app.download_button] == [
        f"Download {birefnet_csv.name}"
    ]
    app.button(key="prepare_zip_export").click().run(timeout=30)
    assert not app.exception
    prepared = Path(app.session_state["export_zip_path"])
    try:
        with zipfile.ZipFile(prepared) as archive:
            assert set(archive.namelist()) == {target_box.name, birefnet_csv.name}
    finally:
        prepared.unlink(missing_ok=True)


def test_results_tab_shows_qr_trace_when_full_qr_columns_are_present(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,leaf_area_cm2,width_cm,length_cm,px_per_cm_width,"
        "px_per_cm_height,scale_aspect_ratio,source,qr_opencv,qr_qreader\n"
        "leaf_1,12.5,2.5,7.0,100,100,1.0,0,success,unused\n"
        "leaf_2,NA,NA,NA,NA,NA,NA,QR_READ: QR not found/readable,failed,failed\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.session_state["last_run"] = _last_run(
        {"threshold": results_path}, succeeded=1, failed=1, total=2,
    )
    app.run(timeout=30)

    assert not app.exception
    assert "**QR decoder trace**" in {item.value for item in app.markdown}


def test_results_tab_switches_between_method_results(tmp_path):
    threshold_path = tmp_path / "leaf_morpho_results_threshold.csv"
    birefnet_path = tmp_path / "leaf_morpho_results_birefnet.csv"
    threshold_path.write_text("sample_id,leaf_area_cm2,width_cm,length_cm\nleaf_1,12.5,2.5,7.0\n")
    birefnet_path.write_text("sample_id,leaf_area_cm2,width_cm,length_cm\nleaf_1,11.0,2.4,6.9\n")
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.session_state["last_run"] = _last_run(
        {"threshold": threshold_path, "birefnet": birefnet_path},
        succeeded=1, failed=0, total=1,
    )
    app.run(timeout=30)

    assert not app.exception
    picker = app.segmented_control(key="results_method")
    assert list(picker.options) == ["Classic thresholding", "BiRefNet"]
    assert picker.value == "threshold"
    assert app.metric[1].value == "12.50 cm²"

    picker.set_value("birefnet").run(timeout=30)
    assert not app.exception
    assert app.metric[1].value == "11.00 cm²"


def test_adjust_opens_with_the_method_selected_in_analyze(tmp_path):
    paths = {}
    pairs = {}
    for method in ("threshold", "birefnet"):
        results = tmp_path / f"results_{method}.csv"
        results.write_text(
            "sample_id,area_cm2,width_cm,length_cm\nleaf,4,2,2\n"
        )
        paths[method] = results
        target = tmp_path / f"target_{method}.png"
        mask = tmp_path / f"mask_{method}.png"
        assert cv2.imwrite(str(target), np.full((20, 20, 3), 255, dtype=np.uint8))
        assert cv2.imwrite(str(mask), np.zeros((20, 20), dtype=np.uint8))
        pairs[method] = [{"sample_id": "leaf", "target_box": str(target), "mask": str(mask)}]
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.session_state["last_run"] = _last_run(
        paths, succeeded=1, failed=0, total=1, run_id="method-carry",
        results_unit="cm", measurement_source="cleaned", threshold_value=125,
    )
    app.session_state["viewer_pairs"] = pairs
    selected = SimpleNamespace(selection=SimpleNamespace(rows=[0]))
    with patch("streamlit.dataframe", return_value=selected):
        app.run(timeout=30)
        app.segmented_control(key="results_method").set_value("birefnet").run(timeout=30)
        next(button for button in app.button if button.label == "Adjust selected specimen").click().run(timeout=30)

    assert not app.exception
    assert app.segmented_control(key="results_method").value == "birefnet"


def test_method_switch_updates_table_and_selected_specimen_together(tmp_path):
    paths = {
        method: tmp_path / f"results_{method}.csv"
        for method in ("threshold", "birefnet")
    }
    pairs = {}
    for method, sample_id, area in (
        ("threshold", "classic_leaf", 12.5),
        ("birefnet", "birefnet_leaf", 11.0),
    ):
        paths[method].write_text(
            f"sample_id,leaf_area_cm2,width_cm,length_cm\n"
            f"{sample_id},{area},2.5,7.0\n"
        )
        mask = tmp_path / f"{sample_id}_mask.png"
        assert cv2.imwrite(str(mask), np.full((12, 12), 255, dtype=np.uint8))
        pairs[method] = [{"sample_id": sample_id, "target_box": None, "mask": str(mask)}]

    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.session_state["last_run"] = _last_run(
        paths, succeeded=1, failed=0, total=1, run_id="method-switch-test",
    )
    app.session_state["viewer_pairs"] = pairs
    selected_rows = {"threshold": [0], "birefnet": []}
    displayed_tables = []
    selection_defaults = []

    def selected_table(data, **kwargs):
        key = kwargs["key"]
        method = key.rsplit("_", 1)[-1]
        displayed_tables.append((key, data["sample_id"].tolist()))
        selection_defaults.append((key, kwargs["selection_default"]))
        return SimpleNamespace(selection=SimpleNamespace(rows=selected_rows[method]))

    with patch("streamlit.dataframe", side_effect=selected_table):
        app.run(timeout=30)
        assert "Sample: classic_leaf" in {item.value for item in app.caption}
        assert displayed_tables[-1] == (
            "measurement_table_method-switch-test_threshold", ["classic_leaf"]
        )

        app.segmented_control(key="results_method").set_value("birefnet").run(timeout=30)
        assert not app.exception
        assert displayed_tables[-1] == (
            "measurement_table_method-switch-test_birefnet", ["birefnet_leaf"]
        )
        assert "Sample: classic_leaf" not in {item.value for item in app.caption}
        assert any("Select a row in the measurement table" in item.value for item in app.info)

        selected_rows["birefnet"] = [0]
        app.run(timeout=30)
        assert "Sample: birefnet_leaf" in {item.value for item in app.caption}
        assert "Sample: classic_leaf" not in {item.value for item in app.caption}

        app.segmented_control(key="results_method").set_value("threshold").run(timeout=30)
        assert displayed_tables[-1] == (
            "measurement_table_method-switch-test_threshold", ["classic_leaf"]
        )
        assert selection_defaults[-1] == (
            "measurement_table_method-switch-test_threshold",
            {"selection": {"rows": [0]}},
        )
        assert "Sample: classic_leaf" in {item.value for item in app.caption}
        assert "Sample: birefnet_leaf" not in {item.value for item in app.caption}


def test_cpu_options_page_renders_worker_control():
    app = AppTest.from_file(str(CPU_OPTIONS_PAGE)).run(timeout=30)

    assert not app.exception
    assert [item.label for item in app.number_input] == ["CPU workers"]


def _sheet_size_inputs(app):
    return [item for item in app.number_input if item.key in {"measure_width", "measure_height"}]


def test_home_page_defaults_to_manual_dimensions():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert app.checkbox(key="measure_use_qr").value is False
    width_input, height_input = _sheet_size_inputs(app)
    assert width_input.value == 12.0
    assert height_input.value == 12.0
    assert not width_input.disabled
    assert not height_input.disabled
    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    assert dimensions_bar.value == "12x12in"
    assert not dimensions_bar.disabled
    assert '"12x12in" or "30x30cm"' in dimensions_bar.help


def test_printed_sheet_size_derives_template_creator_calibration_area():
    sheet_dimensions, layout, error = _resolve_sheet_layout(
        "12x12in", parse_template_dimensions
    )

    assert error is None
    assert sheet_dimensions == (12.0, 12.0, "in")
    assert layout.calibration_dimensions == (10.0, 9.5, "in")


def test_analyze_shows_the_sheet_to_calibration_conversion():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    captions = " ".join(item.value for item in app.caption)
    assert "12 × 12 in printed sheet → 10 × 9.5 in calibrated area" in captions


def test_legacy_calibration_is_kept_in_an_explicit_compatibility_control():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    app.toggle(key="measure_use_legacy_calibration").set_value(True).run(timeout=30)

    assert not app.exception
    assert not _sheet_size_inputs(app)
    legacy_input = app.text_input(key="measure_legacy_dimensions_text")
    assert legacy_input.label == "Legacy calibration area"
    assert legacy_input.value == "10x9.5in"


def test_qr_mode_disables_manual_dimension_inputs():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    app.checkbox(key="measure_use_qr").check().run(timeout=30)

    assert not app.exception
    assert all(item.disabled for item in _sheet_size_inputs(app))
    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    assert dimensions_bar.disabled
    assert dimensions_bar.value == "Variable dimensions — QR-derived"


def test_qr_mode_warns_when_robust_fallbacks_are_unavailable(monkeypatch):
    from mats import qr_runtime

    unavailable = qr_runtime.QRRuntimeStatus(
        opencv=qr_runtime.QRBackendStatus(
            "OpenCV", True, "Built-in QR decoder is available."
        ),
        pyzbar=qr_runtime.QRBackendStatus("pyzbar + zbar", False, "Not installed."),
        qreader=qr_runtime.QRBackendStatus("QReader", False, "Not installed."),
    )
    monkeypatch.setattr(qr_runtime, "qr_runtime_status", lambda: unavailable)

    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    app.checkbox(key="measure_use_qr").check().run(timeout=30)

    assert not app.exception
    assert any("Only OpenCV is available" in item.value for item in app.warning)
    targets = {link.proto.page for link in app.get("page_link")}
    assert "Robust_QR_Setup" in targets


def test_unchecking_qr_mode_restores_the_cached_manual_entry():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    dimensions_bar = lambda: [i for i in app.text_input if i.label == "Printed sheet size"][0]

    app.number_input(key="measure_width").set_value(20.0).run(timeout=30)
    app.checkbox(key="measure_use_qr").check().run(timeout=30)
    assert dimensions_bar().value == "Variable dimensions — QR-derived"

    app.checkbox(key="measure_use_qr").uncheck().run(timeout=30)
    assert dimensions_bar().value == "20x12in"


def test_unit_conversion_snaps_selectors_to_nearest_half():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    app.segmented_control(key="measure_unit").set_value("cm").run(timeout=30)

    width_input, height_input = _sheet_size_inputs(app)
    assert width_input.value == pytest.approx(30.5)
    assert height_input.value == pytest.approx(30.5)
    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    assert dimensions_bar.value == "30.5x30.5cm"


def test_off_grid_sheet_size_is_rejected_with_creator_grid_guidance():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    dimensions_bar.set_value("8.27x11.69in").run(timeout=30)

    assert not app.exception
    width_input, height_input = _sheet_size_inputs(app)
    assert width_input.value == 12.0
    assert height_input.value == 12.0
    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    assert dimensions_bar.value == "8.27x11.69in"
    assert any("0.5-unit increments" in item.value for item in app.caption)
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.run(timeout=30)
    assert app.button(key="run_leaf_morphometrics").disabled


def test_invalid_custom_dimensions_block_the_run():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    dimensions_bar.set_value("not-a-size").run(timeout=30)

    assert not app.exception
    app.session_state[WORKSPACE_TAB_KEY] = "Analyze"
    app.run(timeout=30)
    assert app.button(key="run_leaf_morphometrics").disabled
    assert any("Printed sheet size" in item.value for item in app.caption)
