from pathlib import Path

import pytest


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from mats.app.runtime_paths import current_python, mats_command, python_command

PAGE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mats"
    / "app"
    / "pages"
    / "4_Robust_QR_Setup.py"
)


def test_robust_qr_setup_page_explains_optional_installation():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert [title.value for title in app.title] == ["Robust QR setup"]
    assert {
        "QR decoder status",
        "Do you need robust QR?",
        "Install the optional Python fallbacks",
        "Enable the full pyzbar fallback",
        "Restart and verify",
    } <= {item.value for item in app.subheader}
    assert len(app.button) == 0
    commands = "\n".join(item.value for item in app.code)
    assert "mats-morpho[qr]" in commands
    assert python_command("-m", "pip", "install", "mats-morpho[qr]") in commands
    assert mats_command("app") in commands
    assert "-e ." not in commands
    assert "brew install zbar" in commands
    assert "conda install -c conda-forge zbar" in commands
    captions = "\n".join(item.value for item in app.caption)
    assert str(current_python()) in captions
