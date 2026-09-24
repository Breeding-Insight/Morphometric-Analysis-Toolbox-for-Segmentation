"""Mask cleanup for masks that are never flash-filled.

:func:`clean_raw_mask` is the cleanup every such mask gets. It first clears the
edge margin (:func:`clear_margin`), where the template's printed box outline
lands after perspective correction, so those lines can never outrank the leaf.
It then keeps the leaf -- the largest white component -- and drops every other
piece that touches the cleared margin or lies far from the leaf
(:func:`drop_stray_pieces`). Pre-cleanup measurements use it on every image, and
the Clean image preview applies it before anything else.

Clean image -- a clean size above 0 in the Analyze explorer, an alternative to
:func:`mats.core.clean_leaf_mask` for the preview -- then drops small white
specks and fills small black holes: only pieces smaller than the chosen radius
change, and the leaf is always kept.

Sizes are distance-transform inscribed radii, measured once on the input mask.
:func:`clean_levels` encodes them per pixel, so choosing a radius is a per-pixel
comparison. The browser preview applies the same levels with the same rule, so a
live slider shows exactly what :func:`clean_specks_and_holes` returns.

Imports numpy and OpenCV only -- never torch -- so the offline tests can import it
without :mod:`mats.core`.

The size cleanup is preview-only today. If saving it is added, measure the cleaned
mask under the run's measurement source (the leaf's bounding box for cleaned runs,
the extent of all white pixels for pre-cleanup runs), as the rest of the CSV was.
"""

import math

import cv2
import numpy as np

from .mask_settings import (
    CLEAN_MARGIN_DEFAULT,
    STRAY_GAP_DEFAULT,
    checked_clean_margin,
    checked_stray_gap,
)

# 0 means Clean image is off: the preview shows the run's usual mask.
CLEAN_RADIUS_DEFAULT = 0
CLEAN_RADIUS_MAX = 50

# Level channels. A pixel is foreground at radius r when
#   r < KEEP_BELOW  or  FILL_FROM <= r <= FILL_UNTIL.
KEEP_BELOW, FILL_FROM, FILL_UNTIL = 0, 1, 2
_NEVER_FILL = (255, 0)


def _max_per_label(labels, values, count):
    maxima = np.zeros(count, dtype=np.float32)
    np.maximum.at(maxima, labels.ravel(), values.ravel())
    return np.floor(maxima).astype(np.int32)


def _first_pixels(labels, count):
    """Row and column of each label's first pixel in raster order."""
    first = np.full(count, labels.size, dtype=np.int64)
    np.minimum.at(first, labels.ravel(), np.arange(labels.size))
    return np.divmod(first, labels.shape[1])


def _enclosing(labels, count, other_labels):
    """Label, in ``other_labels``, of the component enclosing each component.

    The pixel directly above a component's first raster pixel lies outside it
    and belongs to the opposite-colour component immediately around it. Row-0
    components touch the border and get -1.
    """
    rows, cols = _first_pixels(labels, count)
    enclosing = np.full(count, -1, dtype=np.int64)
    inside = rows > 0
    enclosing[inside] = other_labels[rows[inside] - 1, cols[inside]]
    return enclosing


def margin_width(shape, margin=CLEAN_MARGIN_DEFAULT):
    """Pixels cleared from each edge: ``margin`` percent of the shorter side."""
    return int(round(min(shape[:2]) * checked_clean_margin(margin) / 100))


def clear_margin(mask, margin=CLEAN_MARGIN_DEFAULT):
    """Set a band ``margin`` percent of the shorter side wide along every edge to 0.

    The template's printed box outline runs through the marker centres, so it
    lies on the edge of the perspective-corrected target box. Returns a 0/255
    uint8 mask; ``margin=0`` changes nothing.
    """
    cleared = np.where(np.asarray(mask) > 127, 255, 0).astype(np.uint8)
    band = margin_width(cleared.shape, margin)
    if band:
        cleared[:band] = cleared[-band:] = 0
        cleared[:, :band] = cleared[:, -band:] = 0
    return cleared


def drop_stray_pieces(mask, max_gap=STRAY_GAP_DEFAULT, border=0):
    """Keep the leaf and the white pieces near it; drop the rest.

    The leaf is the largest 8-connected white component, and it is always kept,
    even where it crosses the image border. Any other piece is stray when it
    comes within ``border`` px of the image edge (touches it, for ``border=0``),
    or when its nearest pixel is more than ``max_gap`` times the leaf's
    bounding-box diagonal from the leaf; ``max_gap=0`` keeps the leaf alone.
    Holes are never filled. Returns a 0/255 uint8 mask.

    A piece at the edge whose inscribed radius is at most ``border`` px -- a
    printed line reaching past a cleared margin -- is never taken as the leaf,
    however large, unless no other piece is left.
    """
    max_gap = checked_stray_gap(max_gap)
    white = (np.asarray(mask) > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if count <= 2:
        return white * 255
    height, width = white.shape
    x, y, w, h = (stats[:, index] for index in range(4))
    at_edge = (
        (x <= border) | (y <= border)
        | (x + w >= width - border) | (y + h >= height - border)
    )
    area = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    if border:
        radius = _max_per_label(labels, cv2.distanceTransform(white, cv2.DIST_L2, 5), count)
        edge_line = at_edge & (radius <= border)
        edge_line[0] = False
        if not edge_line[1:].all():
            area[edge_line] = -1
    leaf = 1 + np.argmax(area[1:])

    gap = cv2.distanceTransform(
        (labels != leaf).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    others = (labels > 0) & (labels != leaf)
    nearest = np.full(count, np.inf, dtype=np.float32)
    np.minimum.at(nearest, labels[others], gap[others])

    stray = at_edge | (nearest > max_gap * math.hypot(w[leaf], h[leaf]))
    stray[0] = True
    stray[leaf] = False
    return np.where(stray[labels], 0, 255).astype(np.uint8)


def clean_raw_mask(mask, margin=CLEAN_MARGIN_DEFAULT, max_gap=STRAY_GAP_DEFAULT):
    """Clear the edge margin, then drop stray pieces: the mask measured without flash fill.

    A piece touching the cleared band counts as touching the border, so a printed
    line that reaches past the margin is still dropped. Idempotent.
    """
    cleared = clear_margin(mask, margin)
    return drop_stray_pieces(cleared, max_gap, border=margin_width(cleared.shape, margin))


def clean_levels(mask, max_gap=STRAY_GAP_DEFAULT, margin=CLEAN_MARGIN_DEFAULT):
    """Per-pixel keep and fill radii for :func:`apply_clean_levels`.

    Returns an ``H x W x 3`` uint8 array with channels ``KEEP_BELOW``,
    ``FILL_FROM`` and ``FILL_UNTIL``:

    - The edge margin and stray pieces (:func:`clean_raw_mask`) are gone at
      every radius, and any hole inside a stray piece never fills.
    - White pixels stay while ``r`` is below their component's inscribed radius
      plus one. The largest white component is always kept.
    - A black hole (4-connected, not touching the border) fills once ``r``
      exceeds its inscribed radius, for as long as its enclosing white component
      is kept, so removing a thin ring never leaves a solid disk behind.
    - A white island inside a hole fills with that hole, so a filled hole shows
      no black spot where a smaller island was removed.
    """
    white = (clean_raw_mask(mask, margin, max_gap) > 127).astype(np.uint8)
    height, width = white.shape
    levels = np.empty((height, width, 3), dtype=np.uint8)
    levels[..., KEEP_BELOW] = 0
    levels[..., FILL_FROM], levels[..., FILL_UNTIL] = _NEVER_FILL

    white_count, white_labels, white_stats, _ = cv2.connectedComponentsWithStats(
        white, connectivity=8
    )
    if white_count == 1:
        return levels
    keep_below = np.minimum(
        _max_per_label(white_labels, cv2.distanceTransform(white, cv2.DIST_L2, 5), white_count)
        + 1,
        254,
    )
    keep_below[0] = 0
    keep_below[1 + np.argmax(white_stats[1:, cv2.CC_STAT_AREA])] = 255
    levels[..., KEEP_BELOW] = keep_below[white_labels]

    black = 1 - white
    black_count, black_labels, black_stats, _ = cv2.connectedComponentsWithStats(
        black, connectivity=4
    )
    x, y, w, h = (black_stats[:, index] for index in range(4))
    is_hole = (x > 0) & (y > 0) & (x + w < width) & (y + h < height)
    is_hole[0] = False
    if not is_hole.any():
        return levels
    hole_radius = _max_per_label(
        black_labels, cv2.distanceTransform(black, cv2.DIST_L2, 5), black_count
    )
    fill_from = np.full(black_count, 255, dtype=np.int32)
    fill_until = np.zeros(black_count, dtype=np.int32)
    enclosing_white = _enclosing(black_labels, black_count, white_labels)
    fill_from[is_hole] = np.minimum(hole_radius[is_hole] + 1, 255)
    fill_until[is_hole] = np.maximum(keep_below[enclosing_white[is_hole]] - 1, 0)

    # Islands take the fill interval of the hole around them.
    enclosing_black = _enclosing(white_labels, white_count, black_labels)
    white_fill_from = np.full(white_count, 255, dtype=np.int32)
    white_fill_until = np.zeros(white_count, dtype=np.int32)
    island = enclosing_black >= 0
    island[island] = is_hole[enclosing_black[island]]
    island[0] = False
    white_fill_from[island] = fill_from[enclosing_black[island]]
    white_fill_until[island] = fill_until[enclosing_black[island]]

    is_white = white.astype(bool)
    levels[..., FILL_FROM] = np.where(
        is_white, white_fill_from[white_labels], fill_from[black_labels]
    )
    levels[..., FILL_UNTIL] = np.where(
        is_white, white_fill_until[white_labels], fill_until[black_labels]
    )
    return levels


def apply_clean_levels(levels, radius):
    """Return the 0/255 mask that :func:`clean_levels` encodes at ``radius``."""
    radius = int(radius)
    keep = levels[..., KEEP_BELOW] > radius
    fill = (levels[..., FILL_FROM] <= radius) & (radius <= levels[..., FILL_UNTIL])
    return np.where(keep | fill, 255, 0).astype(np.uint8)


def clean_specks_and_holes(mask, radius, max_gap=STRAY_GAP_DEFAULT, margin=CLEAN_MARGIN_DEFAULT):
    """Clear the margin and stray pieces, then specks and holes below ``radius`` px."""
    return apply_clean_levels(clean_levels(mask, max_gap, margin), radius)
