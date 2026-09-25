"""Clean image: drop stray pieces, small specks and small holes, never the leaf itself."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from mats.mask_cleanup import (
    CLEAN_RADIUS_MAX, apply_clean_levels, clean_levels, clean_raw_mask,
    clean_specks_and_holes, clear_margin, drop_stray_pieces, margin_width,
    raw_measurement_mask,
)
from mats.mask_settings import (
    CLEAN_MARGIN_MAX, CLEAN_SIZE_MAX, STRAY_GAP_MAX, checked_clean_size,
)


def _leaf():
    # Bounding box 141 x 141, so the default 0.25 gap limit is ~49.9 px.
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 70, 255, -1)
    return mask


def test_small_radii_leave_the_mask_unchanged():
    mask = _leaf()
    cv2.circle(mask, (80, 80), 3, 0, -1)
    mask[25, 100] = 255                               # 1 px speck beside the leaf
    levels = clean_levels(mask)
    assert np.array_equal(apply_clean_levels(levels, 0), mask)
    assert np.array_equal(apply_clean_levels(levels, 1), mask)


def test_pieces_touching_the_border_are_stray():
    mask = _leaf()
    mask[0:3, 60:140] = 255                           # strip along the top edge
    mask[190:200, 195:200] = 255                      # corner fragment
    kept = drop_stray_pieces(mask)
    assert not kept[0:3].any() and not kept[190:, 195:].any()
    assert np.array_equal(kept, _leaf())


def test_pieces_far_from_the_leaf_are_stray_and_near_ones_stay():
    mask = _leaf()
    cv2.circle(mask, (100, 20), 3, 255, -1)           # gap ~7 px: near
    cv2.circle(mask, (15, 15), 3, 255, -1)            # gap ~47 px: under the limit
    cv2.circle(mask, (185, 185), 5, 255, -1)          # gap ~45 px: under the limit
    far = np.zeros((400, 400), dtype=np.uint8)
    far[:200, :200] = mask
    cv2.circle(far, (300, 300), 5, 255, -1)           # gap ~213 px: far
    kept = drop_stray_pieces(far)
    assert kept[20, 100] == 255 and kept[15, 15] == 255 and kept[185, 185] == 255
    assert kept[300, 300] == 0
    assert drop_stray_pieces(far, 0.1)[15, 15] == 0   # limit ~20 px


def test_the_leaf_stays_even_when_it_crosses_the_border():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:60, 20:80] = 255                           # leaf runs off the top edge
    mask[95:100, 0:10] = 255                          # smaller border piece
    kept = drop_stray_pieces(mask)
    assert kept[0:60, 20:80].min() == 255
    assert not kept[95:, :10].any()


def test_zero_gap_keeps_only_the_leaf():
    mask = _leaf()
    mask[25, 100] = 255
    assert np.array_equal(drop_stray_pieces(mask, 0), _leaf())


def test_drop_stray_pieces_keeps_holes_and_is_idempotent():
    mask = _leaf()
    cv2.circle(mask, (120, 110), 15, 0, -1)
    mask[25, 100] = mask[2, 2] = 255
    kept = drop_stray_pieces(mask)
    assert kept[110, 120] == 0                        # hole is not filled
    assert kept[25, 100] == 255 and kept[2, 2] == 0
    assert np.array_equal(drop_stray_pieces(kept), kept)


@pytest.mark.parametrize("gap", [-0.1, STRAY_GAP_MAX + 1, float("nan"), True, "far"])
def test_invalid_gaps_raise(gap):
    with pytest.raises(ValueError):
        drop_stray_pieces(_leaf(), gap)


def test_clean_levels_drop_stray_pieces_at_every_radius():
    mask = _leaf()
    mask[0:3, 60:140] = 255
    far = np.zeros((300, 300), dtype=np.uint8)
    far[:200, :200] = mask
    cv2.circle(far, (270, 270), 12, 255, 2)           # far ring with a hole inside
    levels = clean_levels(far)
    for radius in (0, 5, CLEAN_RADIUS_MAX):
        cleaned = apply_clean_levels(levels, radius)
        assert not cleaned[0:3].any()
        assert not cleaned[255:285, 255:285].any()    # ring gone, its hole never filled
        assert cleaned[100, 100] == 255
    assert np.array_equal(clean_specks_and_holes(far, 0, max_gap=0)[:200, :200], _leaf())


def test_specks_go_by_size_and_the_leaf_always_stays():
    mask = _leaf()
    mask[5, 5] = 255                                  # 1 px speck
    cv2.circle(mask, (185, 20), 8, 255, -1)           # ~8 px radius speck
    levels = clean_levels(mask)

    small = apply_clean_levels(levels, 2)
    assert small[5, 5] == 0 and small[20, 185] == 255
    large = apply_clean_levels(levels, 9)
    assert large[20, 185] == 0
    assert apply_clean_levels(levels, CLEAN_RADIUS_MAX)[100, 100] == 255


def test_small_enclosed_holes_fill_but_border_background_never_does():
    mask = _leaf()
    cv2.circle(mask, (80, 80), 3, 0, -1)
    cv2.circle(mask, (120, 110), 15, 0, -1)
    cleaned = clean_specks_and_holes(mask, 6)
    assert cleaned[80, 80] == 255
    assert cleaned[110, 120] == 0
    assert cleaned[0, 0] == 0
    assert clean_specks_and_holes(mask, CLEAN_RADIUS_MAX)[0, 0] == 0


def test_removing_a_thin_ring_never_fills_its_hole():
    mask = _leaf()
    cv2.circle(mask, (20, 185), 10, 255, 1)
    for radius in range(2, CLEAN_RADIUS_MAX + 1, 4):
        cleaned = clean_specks_and_holes(mask, radius)
        assert cleaned[185, 20] == 0 and cleaned[185, 30] == 0


def test_an_island_fills_with_its_hole():
    mask = _leaf()
    cv2.circle(mask, (120, 110), 15, 0, -1)
    cv2.circle(mask, (120, 110), 2, 255, -1)
    levels = clean_levels(mask)
    assert apply_clean_levels(levels, 4)[110, 120] == 0      # island removed as a speck
    filled = apply_clean_levels(levels, CLEAN_RADIUS_MAX)
    assert filled[100:121, 110:131].min() == 255              # no black spot remains


def test_empty_and_full_masks():
    empty = np.zeros((20, 20), dtype=np.uint8)
    full = np.full((20, 20), 255, dtype=np.uint8)
    assert not clean_specks_and_holes(empty, 10).any()
    assert np.array_equal(clean_specks_and_holes(full, 10), full)


def _framed(size=400, line=4, corner=30, leaf=20):
    """A small leaf inside the printed box outline, as the target box shows it.

    The outline lands on the box edge as four strips; the corners are white
    where the marker boxes were whited out. Each strip outweighs the leaf.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:line, corner:-corner] = mask[-line:, corner:-corner] = 255
    mask[corner:-corner, :line] = mask[corner:-corner, -line:] = 255
    start = size // 2 - leaf // 2
    only_leaf = np.zeros_like(mask)
    only_leaf[start:start + leaf, start:start + leaf] = 255
    return mask | only_leaf, only_leaf


def test_margin_is_a_percent_of_the_shorter_side():
    mask = np.full((200, 400), 255, dtype=np.uint8)
    assert margin_width(mask.shape, 1) == 2
    cleared = clear_margin(mask, 2.5)                 # 5 px on every edge
    assert not cleared[:5].any() and not cleared[-5:].any()
    assert not cleared[:, :5].any() and not cleared[:, -5:].any()
    assert cleared[5:-5, 5:-5].min() == 255
    assert np.array_equal(clear_margin(mask, 0), mask)


@pytest.mark.parametrize("margin", [-1, CLEAN_MARGIN_MAX + 1, float("nan"), True, "wide"])
def test_invalid_margins_raise(margin):
    with pytest.raises(ValueError):
        clear_margin(_leaf(), margin)


def test_edge_lines_that_outweigh_the_leaf_never_replace_it():
    mask, only_leaf = _framed()
    assert (mask[:4] > 0).sum() > (only_leaf > 0).sum()   # a strip outweighs the leaf
    # Without the margin a strip is the largest piece, and the leaf is dropped.
    assert not drop_stray_pieces(mask)[200, 200]
    assert np.array_equal(clean_raw_mask(mask), only_leaf)


def test_a_line_reaching_past_the_margin_is_still_dropped():
    mask, only_leaf = _framed(line=7)                 # the 1% margin clears only 4 px
    assert np.array_equal(clean_raw_mask(mask), only_leaf)


def test_a_leaf_crossing_into_the_margin_loses_only_the_band():
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[0:100, 150:250] = 255                        # leaf runs off the top edge
    mask[105:108, 200:203] = 255                      # a nearby speck stays
    kept = clean_raw_mask(mask)
    assert not kept[:4].any()
    assert kept[4:100, 150:250].min() == 255
    assert kept[106, 201] == 255


def test_clean_raw_mask_is_idempotent_and_feeds_clean_levels():
    mask, only_leaf = _framed()
    kept = clean_raw_mask(mask)
    assert np.array_equal(clean_raw_mask(kept), kept)
    assert np.array_equal(apply_clean_levels(clean_levels(mask), 0), only_leaf)
    assert np.array_equal(clean_raw_mask(mask, margin=0), drop_stray_pieces(mask))


def _speckled_leaf():
    mask = _leaf()
    cv2.circle(mask, (80, 80), 3, 0, -1)              # small hole in the leaf
    mask[25, 100] = 255                               # 1 px speck beside the leaf
    return mask


def test_raw_measurement_mask_is_clean_raw_mask_at_clean_size_0():
    for mask in (_framed()[0], _speckled_leaf()):
        assert np.array_equal(raw_measurement_mask(mask), clean_raw_mask(mask))
        assert np.array_equal(
            raw_measurement_mask(mask, 0, 0.1, clean_size=0), clean_raw_mask(mask, 0, 0.1)
        )


def test_raw_measurement_mask_above_0_is_the_clean_image_preview():
    mask = _speckled_leaf()
    cleaned = raw_measurement_mask(mask, clean_size=5)
    assert np.array_equal(cleaned, clean_specks_and_holes(mask, 5))
    assert np.array_equal(cleaned, apply_clean_levels(clean_levels(mask), 5))
    assert cleaned[25, 100] == 0 and cleaned[80, 80] == 255
    assert np.array_equal(
        raw_measurement_mask(mask, 2.0, 0.1, clean_size=5),
        clean_specks_and_holes(mask, 5, max_gap=0.1, margin=2.0),
    )


@pytest.mark.parametrize("value", [0, 3, CLEAN_SIZE_MAX, "3", " 7 ", np.int64(4)])
def test_clean_size_accepts_whole_pixels(value):
    assert checked_clean_size(value) == int(value)
    assert type(checked_clean_size(value)) is int


@pytest.mark.parametrize("value", [-1, CLEAN_SIZE_MAX + 1, 2.5, 3.0, True, "3.5", "big", None])
def test_clean_size_rejects_other_values(value):
    with pytest.raises(ValueError, match="clean size"):
        checked_clean_size(value)
    with pytest.raises(ValueError, match="clean size"):
        raw_measurement_mask(_leaf(), clean_size=value)
