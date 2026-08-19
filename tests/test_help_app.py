from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from mats import samples
from mats.app.runtime_paths import demo_output_directory, display_path, mats_command


PAGE = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "pages" / "5_Help.py"
HOME_PAGE = PAGE.parents[1] / "Home.py"


def test_help_page_renders_the_guided_sections():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert [title.value for title in app.title] == ["Help"]
    assert {
        "Quick start",
        "What a good photo looks like",
        "Which sheet size should I enter?",
        "Choosing a segmentation method",
        "Reading the results CSV",
        "Troubleshooting",
        "More help",
    } <= {item.value for item in app.subheader}


def test_help_page_documents_every_full_schema_column():
    from mats.scaling import COMPACT_RESULTS_FIELDNAMES, QR_TRACE_FIELDNAMES, RESULTS_FIELDNAMES

    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    body = "\n".join(item.value for item in app.markdown)
    for column in (*RESULTS_FIELDNAMES, *QR_TRACE_FIELDNAMES, *COMPACT_RESULTS_FIELDNAMES):
        assert f"`{column}`" in body


def test_help_page_names_each_sample_set_and_its_calibration_dimensions():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    captions = " ".join(item.value for item in app.caption)
    for sample_set in samples.SAMPLE_SETS:
        assert sample_set["calibration_dimensions"] in captions
    # The sheet-to-calibration rule is the single most error-prone instruction.
    body = "\n".join(item.value for item in app.markdown)
    assert "finished outer sheet size" in body
    assert "12 × 12 in sheet becomes a 10 × 9.5 in calibrated area" in body


def test_help_page_renders_the_packaged_sample_images():
    if not samples.samples_installed():
        pytest.skip("sample images are not installed in this environment")

    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    # The comparison gallery shows each set's hero photo; Help may also render
    # standalone diagnostic examples such as the QR-failure photo. The ZIP
    # download intentionally includes the full set.
    rendered_captions = {
        caption
        for image_block in app.image
        for caption in (img.caption for img in image_block.proto.imgs)
    }
    expected_filenames = {
        path.name
        for sample_set in samples.SAMPLE_SETS
        for path in [samples.sample_hero_path(sample_set)]
        if path is not None
    }
    qr_failure_photo = samples.qr_failure_sample_path()
    if qr_failure_photo is not None:
        expected_filenames.add(qr_failure_photo.name)
    assert expected_filenames <= rendered_captions
    assert app.get("download_button")


def test_help_page_degrades_when_the_samples_are_not_installed(monkeypatch):
    monkeypatch.setattr(samples, "SAMPLES_DIR", Path("/nonexistent-samples"))
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    assert len(app.image) == 0
    assert any("sample images are not installed" in info.value for info in app.info)


def test_help_page_run_it_yourself_uses_the_active_runtime_paths():
    if not samples.samples_installed():
        pytest.skip("sample images are not installed in this environment")

    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    code_blocks = "\n".join(item.value for item in app.code)
    # The app resolves paths at runtime, so another local user gets their own
    # interpreter, installed samples, and home-directory output location.
    assert display_path(samples.SAMPLES_DIR) in code_blocks
    assert display_path(demo_output_directory()) in code_blocks
    assert mats_command("run") in code_blocks
    for sample_set in samples.SAMPLE_SETS:
        assert sample_set["directory"] in code_blocks
    body = "\n".join(item.value for item in app.markdown)
    assert "Command line" in body
    assert "In the app" in body
    assert "Upload images" in body


def test_help_page_explains_the_qr_failure_and_its_manual_entry_recovery():
    app = AppTest.from_file(str(PAGE)).run(timeout=20)

    assert not app.exception
    body = "\n".join(item.value for item in app.markdown)
    assert "When the QR can't be read" in body
    assert "QR_READ: QR not found/readable" in body
    assert f"-t {samples.QR_FAILURE_SAMPLE['calibration_dimensions']}" in body
    assert samples.QR_FAILURE_SAMPLE["printed_label"] in body
    # The old, now-false claim that the samples "predate the current QR
    # convention" must not resurface -- both hero photos decode fine.
    assert "predate the current QR" not in body


def test_help_page_links_resolve_inside_the_multipage_app():
    # st.page_link resolves against the *entry* script's directory, so these
    # links can only be verified with Help reached from Home -- the way the
    # real app runs. Running the page standalone exercises the fallback.
    app = AppTest.from_file(str(HOME_PAGE))
    app.switch_page("pages/5_Help.py")
    app.run(timeout=30)

    assert not app.exception
    targets = {link.proto.page for link in app.get("page_link")}
    assert {"Template_Creator", "BiRefNet_Setup", "CPU_Options", "Robust_QR_Setup"} <= targets
