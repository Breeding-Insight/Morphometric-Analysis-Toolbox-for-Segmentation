from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from mats.app import branding

PAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "pages"
APP_PAGES = sorted(PAGES_DIR.glob("*.py")) + [PAGES_DIR.parent / "Home.py"]


def test_funder_logo_files_ship_with_the_package():
    for filename, _alt, _height in branding.FUNDER_LOGOS:
        path = branding.LOGOS_DIR / filename
        assert path.is_file(), f"missing funder logo {filename}"
        assert path.stat().st_size > 1024


def test_funder_footer_names_all_four_organizations_and_the_funding_statement():
    html = branding.funder_footer_html()
    for alt in (
        "Breeding Insight",
        "USDA Agricultural Research Service",
        "University of Florida Institute of Food and Agricultural Sciences (UF/IFAS)",
        "Cornell University",
    ):
        assert f'alt="{alt}"' in html
    assert html.count("<img ") == 4
    assert "funded by the U.S. Department of Agriculture (USDA)" in html
    assert "Agricultural Research Service (ARS) through University of Florida/IFAS" in html
    assert "Formerly at Cornell University" in html
    assert 'href="https://breedinginsight.org/contact-us/"' in html


@pytest.mark.parametrize("page", APP_PAGES, ids=lambda p: p.stem)
def test_every_page_renders_the_funder_footer(page):
    # Brand requirement: logos and funding statement on every page of the app.
    source = page.read_text(encoding="utf-8")
    assert "branding.render_funder_footer()" in source


def test_help_page_shows_the_funding_statement():
    app = AppTest.from_file(str(PAGES_DIR / "5_Help.py")).run(timeout=20)
    assert not app.exception
    rendered = " ".join(str(node.proto) for node in app.get("html"))
    assert "Formerly at Cornell University" in rendered
