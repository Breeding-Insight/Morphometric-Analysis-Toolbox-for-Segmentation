from pathlib import Path
import pytest


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


PAGE = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "pages" / "2_BiRefNet_Setup.py"


def test_birefnet_setup_page_renders_explicit_lfs_fetch_control():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert [title.value for title in app.title] == ["BiRefNet setup"]
    assert app.button[0].label == "Fetch BiRefNet from this repository"


def test_birefnet_setup_page_never_offers_a_hugging_face_source_picker():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert len(app.radio) == 0


def test_birefnet_setup_page_handles_no_lfs_source_available(monkeypatch):
    from mats import weights

    monkeypatch.setattr(weights, "_REPO_ROOT", Path("/nonexistent-checkout"))
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert len(app.radio) == 0
    assert any("Automatic model downloads are disabled" in info.value for info in app.info)
    assert app.button[0].disabled is True
