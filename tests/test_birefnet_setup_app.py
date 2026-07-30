from pathlib import Path
from unittest import mock

import pytest


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


PAGE = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "pages" / "2_BiRefNet_Setup.py"


def test_birefnet_setup_page_renders_download_controls():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert [title.value for title in app.title] == ["BiRefNet setup"]
    assert app.button[0].label == "Download BiRefNet"


def test_birefnet_setup_page_shows_source_picker_when_a_channel_is_available():
    from mats import weights

    hf, lfs = weights.available_sources("birefnet")
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    if hf.available or lfs.available:
        assert len(app.radio) == 1
    else:
        assert any("Neither download source is available" in info.value for info in app.info)


def test_birefnet_setup_page_handles_no_source_available():
    from mats import weights

    with mock.patch.object(weights, "_HF_REPO_ID", None), \
         mock.patch.object(weights, "_REPO_ROOT", Path("/nonexistent-checkout")):
        app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert len(app.radio) == 0
    assert any("Neither download source is available" in info.value for info in app.info)
    assert app.button[0].disabled is True
