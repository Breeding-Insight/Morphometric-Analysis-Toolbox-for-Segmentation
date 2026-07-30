from pathlib import Path

import pytest


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


PAGE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mats"
    / "app"
    / "pages"
    / "1_Template_Creator.py"
)


def test_template_creator_has_two_dimensions_and_two_exports():
    pytest.importorskip("reportlab")
    pytest.importorskip("qrcode")
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert [widget.label for widget in app.number_input] == [
        "Template width",
        "Template length",
    ]

    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert [download.label for download in app.get("download_button")] == [
        "Download print PDF",
        "Download editable IDML",
    ]


def test_template_creator_blocks_an_elongated_sheet_and_clears_downloads():
    pytest.importorskip("reportlab")
    pytest.importorskip("qrcode")
    app = AppTest.from_file(str(PAGE)).run(timeout=20)
    app.button[0].click().run(timeout=20)

    app.number_input[1].set_value(5).run(timeout=20)

    assert not app.exception
    assert "2.40:1 ratio" in app.error[0].value
    assert app.button[0].disabled
    assert not app.get("download_button")


def test_template_creator_snaps_inputs_and_unit_switches_to_half_units():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    app.number_input[0].set_value(12.24).run(timeout=20)
    assert app.number_input[0].value == 12

    app.segmented_control[0].set_value("cm").run(timeout=20)
    assert app.number_input[0].value == 30.5
