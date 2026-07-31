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
RESULT_UNITS = ("mm", "cm", "in")
DEFAULT_RESULTS_UNIT = "cm"
QR_TRACE_FIELDNAMES = (
    "qr_opencv",
    "qr_pyzbar_zbar",
    "qr_qreader",
)
COMPACT_RESULTS_FIELDNAMES = ["sample_id", "area_cm2", "width_cm", "length_cm"]
NA_VALUE = "NA"


def validate_results_unit(results_unit):
    """Return a supported result unit or raise a clear validation error."""
    if results_unit not in RESULT_UNITS:
        raise ValueError(
            f"results_unit must be one of {', '.join(RESULT_UNITS)}"
        )
    return results_unit


def result_measurement_fieldnames(results_unit=DEFAULT_RESULTS_UNIT, compact=False):
    """Return unit-specific measurement columns for one CSV schema."""
    unit = validate_results_unit(results_unit)
    area_name = f"area_{unit}2" if compact else f"leaf_area_{unit}2"
    return [area_name, f"width_{unit}", f"length_{unit}"]


def results_fieldnames(results_unit=DEFAULT_RESULTS_UNIT):
    """Return the full-schema columns for measurements in ``results_unit``."""
    unit = validate_results_unit(results_unit)
    return [
        "sample_id",
        *result_measurement_fieldnames(unit),
        f"px_per_{unit}_width",
        f"px_per_{unit}_height",
        "scale_aspect_ratio",
        "source",
    ]


def compact_results_fieldnames(results_unit=DEFAULT_RESULTS_UNIT):
    """Return the compact-schema columns for measurements in ``results_unit``."""
    unit = validate_results_unit(results_unit)
    return ["sample_id", *result_measurement_fieldnames(unit, compact=True)]


def full_results_fieldnames(qr_backend_fields=(), results_unit=DEFAULT_RESULTS_UNIT):
    """Return full-schema columns, optionally including available QR traces."""
    return [*results_fieldnames(results_unit), *qr_backend_fields]


def length_conversion_factor(results_unit=DEFAULT_RESULTS_UNIT):
    """Return the multiplier that converts a centimeter length to ``results_unit``."""
    unit = validate_results_unit(results_unit)
    return {"mm": 10.0, "cm": 1.0, "in": 1.0 / 2.54}[unit]


def _convert_value(value, factor):
    if value == NA_VALUE:
        return NA_VALUE
    return float(value) * factor


def converted_measurement_row(row, results_unit=DEFAULT_RESULTS_UNIT):
    """Return a canonical centimeter row converted to ``results_unit``.

    Pixel calibration remains internally in pixels per centimeter. The exported
    pixels-per-unit values therefore use the inverse length conversion factor.
    Non-measurement fields, including optional QR trace columns, are retained.
    """
    unit = validate_results_unit(results_unit)
    length_factor = length_conversion_factor(unit)
    converted = {
        "sample_id": row.get("sample_id", NA_VALUE),
        f"leaf_area_{unit}2": _convert_value(row.get("leaf_area_cm2", NA_VALUE), length_factor ** 2),
        f"width_{unit}": _convert_value(row.get("width_cm", NA_VALUE), length_factor),
        f"length_{unit}": _convert_value(row.get("length_cm", NA_VALUE), length_factor),
        f"px_per_{unit}_width": _convert_value(
            row.get("px_per_cm_width", NA_VALUE), 1.0 / length_factor
        ),
        f"px_per_{unit}_height": _convert_value(
            row.get("px_per_cm_height", NA_VALUE), 1.0 / length_factor
        ),
        "scale_aspect_ratio": row.get("scale_aspect_ratio", NA_VALUE),
        "source": row.get("source", NA_VALUE),
    }
    canonical_fields = set(RESULTS_FIELDNAMES)
    converted.update({
        key: value for key, value in row.items() if key not in canonical_fields
    })
    return converted


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


def compact_measurement_row(row, results_unit=DEFAULT_RESULTS_UNIT):
    """Return only the UI-facing measurements requested for bulk export."""
    unit = validate_results_unit(results_unit)
    converted = converted_measurement_row(row, unit)
    return {
        "sample_id": row.get("sample_id", NA_VALUE),
        f"area_{unit}2": converted[f"leaf_area_{unit}2"],
        f"width_{unit}": converted[f"width_{unit}"],
        f"length_{unit}": converted[f"length_{unit}"],
    }
