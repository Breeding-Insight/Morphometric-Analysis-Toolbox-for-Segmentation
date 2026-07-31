import io
from xml.etree import ElementTree
import zipfile

import pytest

from mats.template_exports import (
    IDML_MIME_TYPE,
    render_template_idml,
    render_template_pdf,
)
from mats.template_layout import build_template_layout


def test_pdf_export_has_a_pdf_header():
    pytest.importorskip("reportlab")
    pytest.importorskip("qrcode")
    layout = build_template_layout(12, 12, "in")

    pdf_bytes = render_template_pdf(layout)

    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1_000


def test_idml_export_is_a_well_formed_editable_package():
    pytest.importorskip("qrcode")
    layout = build_template_layout(12, 12, "in")

    idml_bytes = render_template_idml(layout)

    with zipfile.ZipFile(io.BytesIO(idml_bytes)) as package:
        members = package.infolist()
        assert members[0].filename == "mimetype"
        assert members[0].compress_type == zipfile.ZIP_STORED
        assert package.read("mimetype").decode() == IDML_MIME_TYPE
        assert {
            "META-INF/container.xml",
            "designmap.xml",
            "Resources/Graphic.xml",
            "Resources/Styles.xml",
            "Resources/Fonts.xml",
            "Resources/Preferences.xml",
            "Spreads/Spread_mats.xml",
            "Stories/Story_labels.xml",
        }.issubset(package.namelist())

        for name in package.namelist():
            if name.endswith(".xml"):
                ElementTree.fromstring(package.read(name))

        designmap = package.read("designmap.xml").decode()
        spread = package.read("Spreads/Spread_mats.xml").decode()
        story = package.read("Stories/Story_labels.xml").decode()
        graphics = package.read("Resources/Graphic.xml").decode()

    assert 'Name="Calibration geometry"' in designmap
    assert 'Name="QR and labels"' in designmap
    assert spread.count("<Oval ") == 4
    assert 'Name="Observation box"' in spread
    assert f'Name="QR code ({layout.qr_payload})"' in spread
    assert 'Name="Sheet and calibration information"' in spread
    for label in layout.artwork_labels:
        assert label in story
    assert 'ColorValue="15 100 100 0"' in graphics


def test_idml_export_is_deterministic():
    pytest.importorskip("qrcode")
    layout = build_template_layout(30.48, 30.48, "cm")

    assert render_template_idml(layout) == render_template_idml(layout)
