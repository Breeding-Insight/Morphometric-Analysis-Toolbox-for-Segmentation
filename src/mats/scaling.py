"""Pixel-to-metric scaling and results-row construction.

Deliberately dependency-light: this module imports only the standard library so
the offline test suite can exercise the scale math without importing
:mod:`mats.core` (which pulls in torch, rfdetr, cv2 and transformers). Keep it
that way -- do not add heavy imports here.

The pipeline calibrates against the *template*, not the leaf: the warped
target-box raster's pixel width/height are compared to the template's known
physical width/height to get pixels-per-cm on each axis independently
(``px_per_cm_width``, ``px_per_cm_height``). Applying each axis's own factor to
its own axis -- and the product of both to area -- makes every measurement
invariant to which marker-quad edge the perspective warp happened to size the
raster from.
"""

RESULTS_FIELDNAMES = [
    "sample_id",
    "leaf_area_cm2",
    "width_cm",
    "length_cm",
    "px_per_cm_width",
    "px_per_cm_height",
    "scale_aspect_ratio",
    "source",
]
QR_TRACE_FIELDNAMES = (
    "qr_opencv",
    "qr_pyzbar_zbar",
    "qr_qreader",
)
COMPACT_RESULTS_FIELDNAMES = ["sample_id", "area_cm2", "width_cm", "length_cm"]
NA_VALUE = "NA"


def full_results_fieldnames(qr_backend_fields=()):
    """Return full-schema columns, optionally including available QR traces."""
    return [*RESULTS_FIELDNAMES, *qr_backend_fields]


def px_per_cm_axes(box_shape, template_width, template_height, unit):
    """Return ``(px_per_cm_width, px_per_cm_height)`` for a target-box raster.

    ``box_shape`` is ``(height, width)`` in pixels (e.g. ``target_box.shape[:2]``).
    ``template_width``/``template_height`` are the template's known physical
    size in ``unit`` (``"cm"`` or ``"in"``). Returns ``None`` if the template
    dimensions are missing or invalid.
    """
    if box_shape is None or template_width is None or template_height is None:
        return None
    if template_width <= 0 or template_height <= 0:
        return None

    img_height, img_width = box_shape
    px_per_cm_width = img_width / template_width
    px_per_cm_height = img_height / template_height
    if unit != "cm":
        px_per_cm_width /= 2.54
        px_per_cm_height /= 2.54
    return px_per_cm_width, px_per_cm_height


def measurement_na_row(sample_id, source):
    return {
        "sample_id": sample_id,
        "leaf_area_cm2": NA_VALUE,
        "width_cm": NA_VALUE,
        "length_cm": NA_VALUE,
        "px_per_cm_width": NA_VALUE,
        "px_per_cm_height": NA_VALUE,
        "scale_aspect_ratio": NA_VALUE,
        "source": source,
    }


def measurement_row(sample_id, white_pixels, bbox_w, bbox_h, px_per_cm_width, px_per_cm_height):
    """Build a results row from a leaf's pixel-space measurements and per-axis scale."""
    return {
        "sample_id": sample_id,
        "leaf_area_cm2": white_pixels / (px_per_cm_width * px_per_cm_height),
        "width_cm": bbox_w / px_per_cm_width,
        "length_cm": bbox_h / px_per_cm_height,
        "px_per_cm_width": px_per_cm_width,
        "px_per_cm_height": px_per_cm_height,
        "scale_aspect_ratio": px_per_cm_width / px_per_cm_height,
        "source": 0,
    }


def compact_measurement_row(row):
    """Return only the UI-facing measurements requested for bulk export."""
    return {
        "sample_id": row.get("sample_id", NA_VALUE),
        "area_cm2": row.get("leaf_area_cm2", NA_VALUE),
        "width_cm": row.get("width_cm", NA_VALUE),
        "length_cm": row.get("length_cm", NA_VALUE),
    }
