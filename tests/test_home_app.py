from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from mats.app.Home import (
    PENDING_WORKSPACE_TAB_KEY,
    WORKSPACE_TAB_KEY,
    _WORKBENCH_STYLES,
    _resolve_sheet_layout,
    collect_output_pairs,
    merge_viewer_pairs,
    normalize_measurements,
    summarize_measurements,
)
from mats.dimensions import parse_template_dimensions


HOME_PAGE = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "Home.py"
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


def test_home_page_renders_analyze_view_without_worker_control():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert [item.label for item in app.number_input] == ["Sheet width", "Sheet height"]
    subheaders = [item.value for item in app.subheader]
    assert "Analysis setup" in subheaders
    assert "Launch analysis" in subheaders
    assert "**WORKSPACE NAVIGATION**" in [item.value for item in app.markdown]
    assert "**4 · Preflight**" in [item.value for item in app.markdown]
    assert app.segmented_control(key="results_unit").value == "cm"


def test_diagnostics_tab_renders_compute_status_without_worker_control():
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Diagnostics"
    app.run(timeout=30)

    assert not app.exception
    assert [item.value for item in app.subheader if item.value == "Compute status"] == ["Compute status"]


def test_pending_workspace_tab_is_applied_before_navigation():
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[PENDING_WORKSPACE_TAB_KEY] = "Results"
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state[WORKSPACE_TAB_KEY] == "Results"
    assert PENDING_WORKSPACE_TAB_KEY not in app.session_state
    assert any(item.value == "Results" for item in app.subheader)


def test_home_page_offers_input_and_output_folder_pickers():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert app.button(key="choose_input_folder").label == "Choose input folder"
    assert app.button(key="choose_output_folder").label == "Choose output folder"
    assert app.text_input(key="input_directory").value == str(Path.home())
    assert app.text_input(key="output_directory").value == str(
        Path.home() / "mats_outputs"
    )


def test_home_page_has_a_dedicated_launch_section():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

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
    assert app.radio(key="segmentation_method").label == "Select a segmentation method"
    assert {"**1 · Scale**", "**2 · Segmentation**", "**3 · Measurement Output**"} <= {
        item.value for item in app.markdown
    }


def test_blocking_preflight_button_opens_diagnostics():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    app.button(key="open_diagnostics").click().run(timeout=30)

    assert not app.exception
    assert app.session_state[WORKSPACE_TAB_KEY] == "Diagnostics"
    assert any(item.value == "Diagnostics" for item in app.subheader)


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


def test_results_tab_renders_measurement_dashboard(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,leaf_area_cm2,width_cm,length_cm,px_per_cm_width,"
        "px_per_cm_height,scale_aspect_ratio,source\n"
        "leaf_1,12.5,2.5,7.0,100,100,1.0,0\n"
        "leaf_2,8.0,2.0,4.0,100,101,0.99,0\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Results"
    app.session_state["last_run"] = {
        "succeeded": 2,
        "failed": 0,
        "total": 2,
        "workers": 1,
        "worker_reason": "test",
        "execution_device": "cpu",
        "failure_rows": [],
        "failure_overflow": 0,
        "results_path": str(results_path),
        "output_path": str(tmp_path),
        "mask_method": "threshold",
    }
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
        "Leaf Area",
        "Leaf Width",
        "Leaf Length",
    ]
    assert {"**Measurement Table**", "**Selected Specimen**"} <= {
        item.value for item in app.markdown
    }
    assert "**QR decoder trace**" not in {item.value for item in app.markdown}


def test_results_tab_uses_the_completed_run_unit(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,area_in2,width_in,length_in\n"
        "leaf_1,2.5,1.5,3.0\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Results"
    app.session_state["last_run"] = {
        "succeeded": 1,
        "failed": 0,
        "total": 1,
        "workers": 1,
        "worker_reason": "test",
        "execution_device": "cpu",
        "failure_rows": [],
        "failure_overflow": 0,
        "results_path": str(results_path),
        "output_path": str(tmp_path),
        "mask_method": "threshold",
        "results_unit": "in",
    }
    app.run(timeout=30)

    assert not app.exception
    assert app.metric[1].value == "2.50 in²"
    assert app.metric[2].value == "1.50 in"
    assert app.download_button[0].label == "Download results CSV (in)"


def test_results_tab_shows_qr_trace_when_full_qr_columns_are_present(tmp_path):
    results_path = tmp_path / "leaf_morpho_results.csv"
    results_path.write_text(
        "sample_id,leaf_area_cm2,width_cm,length_cm,px_per_cm_width,"
        "px_per_cm_height,scale_aspect_ratio,source,qr_opencv,qr_qreader\n"
        "leaf_1,12.5,2.5,7.0,100,100,1.0,0,success,unused\n"
        "leaf_2,NA,NA,NA,NA,NA,NA,QR_READ: QR not found/readable,failed,failed\n"
    )
    app = AppTest.from_file(str(HOME_PAGE))
    app.session_state[WORKSPACE_TAB_KEY] = "Results"
    app.session_state["last_run"] = {
        "succeeded": 1,
        "failed": 1,
        "total": 2,
        "workers": 1,
        "worker_reason": "test",
        "execution_device": "cpu",
        "failure_rows": [],
        "failure_overflow": 0,
        "results_path": str(results_path),
        "output_path": str(tmp_path),
        "mask_method": "threshold",
    }
    app.run(timeout=30)

    assert not app.exception
    assert "**QR decoder trace**" in {item.value for item in app.markdown}


def test_cpu_options_page_renders_worker_control():
    app = AppTest.from_file(str(CPU_OPTIONS_PAGE)).run(timeout=30)

    assert not app.exception
    assert [item.label for item in app.number_input] == ["CPU workers"]


def test_home_page_defaults_to_manual_dimensions():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert app.checkbox(key="measure_use_qr").value is False
    width_input, height_input = app.number_input
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
    assert not app.number_input
    legacy_input = app.text_input(key="measure_legacy_dimensions_text")
    assert legacy_input.label == "Legacy calibration area"
    assert legacy_input.value == "10x9.5in"


def test_qr_mode_disables_manual_dimension_inputs():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    app.checkbox(key="measure_use_qr").check().run(timeout=30)

    assert not app.exception
    assert all(item.disabled for item in app.number_input)
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

    width_input, height_input = app.number_input
    assert width_input.value == pytest.approx(30.5)
    assert height_input.value == pytest.approx(30.5)
    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    assert dimensions_bar.value == "30.5x30.5cm"


def test_off_grid_sheet_size_is_rejected_with_creator_grid_guidance():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    dimensions_bar.set_value("8.27x11.69in").run(timeout=30)

    assert not app.exception
    width_input, height_input = app.number_input
    assert width_input.value == 12.0
    assert height_input.value == 12.0
    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    assert dimensions_bar.value == "8.27x11.69in"
    assert app.button(key="run_leaf_morphometrics").disabled
    assert any("0.5-unit increments" in item.value for item in app.caption)


def test_invalid_custom_dimensions_block_the_run():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    dimensions_bar = [i for i in app.text_input if i.label == "Printed sheet size"][0]
    dimensions_bar.set_value("not-a-size").run(timeout=30)

    assert not app.exception
    assert app.button(key="run_leaf_morphometrics").disabled
    assert any("Printed sheet size" in item.value for item in app.caption)
