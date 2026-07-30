from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from mats.app.Home import collect_output_pairs, merge_viewer_pairs


HOME_PAGE = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "Home.py"
CPU_OPTIONS_PAGE = HOME_PAGE.parent / "pages" / "3_CPU_Options.py"


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


def test_home_page_renders_compute_status_without_worker_control():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert [item.label for item in app.number_input] == ["Width", "Height"]
    assert [item.value for item in app.subheader if item.value == "Compute status"] == ["Compute status"]


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
    dimensions_bar = [i for i in app.text_input if i.label == "Template dimensions"][0]
    assert dimensions_bar.value == "12x12in"
    assert not dimensions_bar.disabled
    assert 'e.g. "10.5x9.5in" or' in dimensions_bar.help


def test_qr_mode_disables_manual_dimension_inputs():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    app.checkbox(key="measure_use_qr").check().run(timeout=30)

    assert not app.exception
    assert all(item.disabled for item in app.number_input)
    dimensions_bar = [i for i in app.text_input if i.label == "Template dimensions"][0]
    assert dimensions_bar.disabled
    assert dimensions_bar.value == "Variable dimensions — QR-derived"


def test_unchecking_qr_mode_restores_the_cached_manual_entry():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)
    dimensions_bar = lambda: [i for i in app.text_input if i.label == "Template dimensions"][0]

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
    dimensions_bar = [i for i in app.text_input if i.label == "Template dimensions"][0]
    assert dimensions_bar.value == "30.5x30.5cm"


def test_typing_a_custom_size_in_the_bar_does_not_resync_the_selectors():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    dimensions_bar = [i for i in app.text_input if i.label == "Template dimensions"][0]
    dimensions_bar.set_value("8.27x11.69in").run(timeout=30)

    assert not app.exception
    width_input, height_input = app.number_input
    assert width_input.value == 12.0
    assert height_input.value == 12.0
    dimensions_bar = [i for i in app.text_input if i.label == "Template dimensions"][0]
    assert dimensions_bar.value == "8.27x11.69in"


def test_invalid_custom_dimensions_block_the_run():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    dimensions_bar = [i for i in app.text_input if i.label == "Template dimensions"][0]
    dimensions_bar.set_value("not-a-size").run(timeout=30)

    assert not app.exception
    assert any(
        "Template dimensions must look like" in item.value
        for item in app.error
    )
