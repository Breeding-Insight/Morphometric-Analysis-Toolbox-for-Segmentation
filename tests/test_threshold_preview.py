"""The interactive preview must agree with the production threshold path."""

import base64

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("streamlit")

from mats.app.threshold_preview import (
    clean_levels_for_mask, cleaned_sample, color_sample, grayscale_sample,
    pre_cleanup_sample, unfilled_sample,
)
from mats.mask_cleanup import clean_levels, clean_raw_mask


def _decode_png(url):
    encoded = base64.b64decode(url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def test_grayscale_preview_uses_exact_pipeline_pixels_and_otsu_cutoff(tmp_path):
    image = np.full((40, 50, 3), (180, 210, 240), dtype=np.uint8)
    image[8:32, 10:40] = (15, 25, 35)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)

    data_url, cutoff = grayscale_sample(str(path))
    expected_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    expected_cutoff, _ = cv2.threshold(
        expected_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    assert np.array_equal(_decode_png(data_url), expected_gray)
    assert cutoff == int(expected_cutoff)


def test_color_preview_matches_the_target_box_geometry_and_colors(tmp_path):
    image = np.full((40, 50, 3), (180, 210, 240), dtype=np.uint8)
    image[8:32, 10:40] = (40, 140, 60)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)

    url = color_sample(str(path))
    assert url.startswith("data:image/jpeg;base64,")
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    # Same pixel grid as the grayscale preview, so the two panels align.
    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(int) - image.astype(int)).mean() < 3


def test_clean_levels_reach_the_browser_in_channel_order(tmp_path):
    mask = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 20, 255, -1)
    cv2.circle(mask, (30, 30), 3, 0, -1)
    mask[2, 2] = 255
    path = tmp_path / "raw.png"
    assert cv2.imwrite(str(path), mask)

    url = clean_levels_for_mask(str(path), path.stat().st_mtime_ns)
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    # The browser reads PNG channels as RGB; OpenCV decodes them as BGR.
    assert np.array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), clean_levels(mask))


def test_clean_levels_use_the_runs_stray_gap(tmp_path):
    mask = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 20, 255, -1)
    mask[30, 4] = 255                                  # 6 px from the leaf
    path = tmp_path / "raw.png"
    assert cv2.imwrite(str(path), mask)

    def levels(stray_gap):
        url = clean_levels_for_mask(str(path), path.stat().st_mtime_ns, stray_gap)
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    assert np.array_equal(levels(0.25), clean_levels(mask, 0.25))
    assert np.array_equal(levels(0), clean_levels(mask, 0))
    assert not np.array_equal(levels(0), levels(0.25))  # separate cache entries


def test_settled_cleaned_preview_uses_production_mask_cleanup(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from mats import core

    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[20:80, 20:80] = 0
    image[43:57, 43:57] = 255
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)

    preview = _decode_png(cleaned_sample(str(path), 125))
    expected = core.clean_leaf_mask(core.threshold_mask(image, 125))
    assert np.array_equal(preview, expected)


def test_remove_flashfill_preserves_holes_but_keeps_other_cleanup(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from mats import core

    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[20:80, 20:80] = 255
    raw[40:60, 40:60] = 0
    raw[4:9, 4:9] = 255
    path = tmp_path / "raw.png"
    assert cv2.imwrite(str(path), raw)

    preview = _decode_png(unfilled_sample(str(path)))
    cleaned = core.clean_leaf_mask(raw)
    assert preview[50, 50] == 0 and cleaned[50, 50] == 255
    assert preview[6, 6] == 0  # The detached speck is still removed.
    assert preview[25, 25] == 255


def test_otsu_zero_cutoff_can_be_previewed(tmp_path):
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    image[5:15, 5:15] = 0
    path = tmp_path / "binary_sample.png"
    assert cv2.imwrite(str(path), image)
    _, cutoff = grayscale_sample(str(path))
    assert cutoff == 0


def _framed_target(tmp_path):
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    image[:4, 30:-30] = image[-4:, 30:-30] = 0         # printed outline on the edge
    image[30:-30, :4] = image[30:-30, -4:] = 0
    image[190:210, 190:210] = 0                          # a leaf smaller than a strip
    path = tmp_path / "framed.png"
    assert cv2.imwrite(str(path), image)
    return path, image


def test_pre_cleanup_preview_is_the_measured_mask(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from mats import core

    path, image = _framed_target(tmp_path)
    settled = _decode_png(pre_cleanup_sample(str(path), 125, 1.0, 0.25))
    expected = clean_raw_mask(core.threshold_mask(image, 125), 1.0, 0.25)
    assert np.array_equal(settled, expected)
    assert not settled[:4].any() and settled[200, 200] == 255


def test_remove_flashfill_previews_clear_the_edge_margin(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from mats import core

    path, image = _framed_target(tmp_path)
    raw = core.threshold_mask(image, 125)
    raw_path = tmp_path / "raw.png"
    assert cv2.imwrite(str(raw_path), raw)
    only_leaf = np.zeros((400, 400), dtype=np.uint8)
    only_leaf[190:210, 190:210] = 255
    assert np.array_equal(_decode_png(unfilled_sample(str(raw_path), 1.0)), only_leaf)
    unfilled = cleaned_sample(str(path), 125, fill_holes=False, clean_margin=1.0)
    assert np.array_equal(_decode_png(unfilled), only_leaf)
