"""Physical layout policy for generated calibration templates.

This module intentionally uses only the standard library.  Both the Streamlit
page and the document renderers consume the same immutable layout so PDF and
IDML exports cannot silently disagree about marker spacing or QR dimensions.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math


INCHES_PER_CM = 1.0 / 2.54
POINTS_PER_INCH = 72.0

MAX_ASPECT_RATIO = 1.5
MIN_TEMPLATE_EDGE_IN = 4.0
MIN_TEMPLATE_EDGE_CM = 10.0
MAX_TEMPLATE_EDGE_IN = 200.0
MAX_TEMPLATE_EDGE_CM = 508.0

DIMENSION_INCREMENT = 0.5

SMALL_MARKER_DIAMETER_IN = 0.5
SECOND_MARKER_DIAMETER_IN = 0.75
THIRD_MARKER_DIAMETER_IN = 1.0
LARGE_MARKER_DIAMETER_IN = 1.5
SECOND_MARKER_THRESHOLD_IN = 12.0
THIRD_MARKER_THRESHOLD_IN = 18.0
LARGE_MARKER_THRESHOLD_IN = 24.0

SMALL_MARKER_DIAMETER_CM = 1.5
SECOND_MARKER_DIAMETER_CM = 2.0
THIRD_MARKER_DIAMETER_CM = 2.5
LARGE_MARKER_DIAMETER_CM = 4.0
SECOND_MARKER_THRESHOLD_CM = 30.5
THIRD_MARKER_THRESHOLD_CM = 45.5
LARGE_MARKER_THRESHOLD_CM = 61.0

TOP_MARGIN_IN = 1.5
SIDE_MARGIN_IN = 1.0
BOTTOM_MARGIN_IN = 1.0
TOP_MARGIN_CM = 3.5
SIDE_MARGIN_CM = 2.5
BOTTOM_MARGIN_CM = 2.5

QR_SIZE_IN = 1.0
QR_TOP_IN = 0.3
LABEL_TOP_IN = 0.2
LABEL_LEFT_IN = 0.2
LABEL_TO_QR_GAP_IN = 0.1
LABEL_BOTTOM_IN = 0.8


class TemplateLayoutError(ValueError):
    """Raised when requested print-sheet dimensions cannot form a template."""


def to_inches(value, unit):
    """Return ``value`` converted from ``unit`` to inches."""
    if unit == "in":
        return float(value)
    if unit == "cm":
        return float(value) * INCHES_PER_CM
    raise TemplateLayoutError("Unit must be 'in' or 'cm'.")


def from_inches(value, unit):
    """Return inch ``value`` converted to ``unit``."""
    if unit == "in":
        return float(value)
    if unit == "cm":
        return float(value) / INCHES_PER_CM
    raise TemplateLayoutError("Unit must be 'in' or 'cm'.")


def round_to_increment(value, increment=DIMENSION_INCREMENT):
    """Round a numeric value to an increment using conventional half-up rules."""
    decimal_value = Decimal(str(float(value)))
    decimal_increment = Decimal(str(increment))
    return float(
        (decimal_value / decimal_increment).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        * decimal_increment
    )


def minimum_template_edge(unit):
    """Return the unit-native minimum page edge."""
    if unit == "in":
        return MIN_TEMPLATE_EDGE_IN
    if unit == "cm":
        return MIN_TEMPLATE_EDGE_CM
    raise TemplateLayoutError("Unit must be 'in' or 'cm'.")


def maximum_template_edge(unit):
    """Return the unit-native maximum page edge."""
    if unit == "in":
        return MAX_TEMPLATE_EDGE_IN
    if unit == "cm":
        return MAX_TEMPLATE_EDGE_CM
    raise TemplateLayoutError("Unit must be 'in' or 'cm'.")


def margin_profile(unit):
    """Return ``(top, side, bottom)`` fixed margins in the selected unit."""
    if unit == "in":
        return TOP_MARGIN_IN, SIDE_MARGIN_IN, BOTTOM_MARGIN_IN
    if unit == "cm":
        return TOP_MARGIN_CM, SIDE_MARGIN_CM, BOTTOM_MARGIN_CM
    raise TemplateLayoutError("Unit must be 'in' or 'cm'.")


def format_measurement(value):
    """Format a physical measurement compactly without noisy float tails."""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def marker_diameter_for_size(width, length, unit="in"):
    """Choose a unit-native marker tier and return its diameter in inches."""
    longest_edge = max(float(width), float(length))
    if unit == "in":
        if longest_edge < SECOND_MARKER_THRESHOLD_IN:
            diameter = SMALL_MARKER_DIAMETER_IN
        elif longest_edge < THIRD_MARKER_THRESHOLD_IN:
            diameter = SECOND_MARKER_DIAMETER_IN
        elif longest_edge < LARGE_MARKER_THRESHOLD_IN:
            diameter = THIRD_MARKER_DIAMETER_IN
        else:
            diameter = LARGE_MARKER_DIAMETER_IN
        return diameter

    if unit == "cm":
        if longest_edge < SECOND_MARKER_THRESHOLD_CM:
            diameter = SMALL_MARKER_DIAMETER_CM
        elif longest_edge < THIRD_MARKER_THRESHOLD_CM:
            diameter = SECOND_MARKER_DIAMETER_CM
        elif longest_edge < LARGE_MARKER_THRESHOLD_CM:
            diameter = THIRD_MARKER_DIAMETER_CM
        else:
            diameter = LARGE_MARKER_DIAMETER_CM
        return to_inches(diameter, unit)

    raise TemplateLayoutError("Unit must be 'in' or 'cm'.")


@dataclass(frozen=True)
class TemplateLayout:
    """Canonical template geometry, stored in inches."""

    unit: str
    page_width_in: float
    page_length_in: float
    observation_width_in: float
    observation_length_in: float
    marker_diameter_in: float
    side_inset_in: float
    top_inset_in: float
    bottom_inset_in: float

    @property
    def aspect_ratio(self):
        return max(self.page_width_in, self.page_length_in) / min(
            self.page_width_in, self.page_length_in
        )

    def in_unit(self, value_in):
        # Keep converted display values identical to the four-decimal values
        # encoded in the QR payload, avoiding floating-point display tails.
        return float(format_measurement(from_inches(value_in, self.unit)))

    @property
    def page_width(self):
        return self.in_unit(self.page_width_in)

    @property
    def page_length(self):
        return self.in_unit(self.page_length_in)

    @property
    def observation_width(self):
        return self.in_unit(self.observation_width_in)

    @property
    def observation_length(self):
        return self.in_unit(self.observation_length_in)

    @property
    def marker_diameter(self):
        return self.in_unit(self.marker_diameter_in)

    @property
    def qr_payload(self):
        width = format_measurement(self.observation_width)
        length = format_measurement(self.observation_length)
        return f"{width}x{length}{self.unit}"

    @property
    def calibration_dimensions(self):
        """Return the marker-centre dimensions consumed by the measurement pipeline."""
        return self.observation_width, self.observation_length, self.unit

    @property
    def observation_bounds_in(self):
        """Return ``(top, left, bottom, right)`` in top-down page coordinates."""
        return (
            self.top_inset_in,
            self.side_inset_in,
            self.top_inset_in + self.observation_length_in,
            self.side_inset_in + self.observation_width_in,
        )

    @property
    def qr_bounds_in(self):
        left = (self.page_width_in - QR_SIZE_IN) / 2.0
        return (
            QR_TOP_IN,
            left,
            QR_TOP_IN + QR_SIZE_IN,
            left + QR_SIZE_IN,
        )

    @property
    def label_bounds_in(self):
        _, qr_left, _, _ = self.qr_bounds_in
        return (
            LABEL_TOP_IN,
            LABEL_LEFT_IN,
            LABEL_BOTTOM_IN,
            qr_left - LABEL_TO_QR_GAP_IN,
        )

    @property
    def marker_centers_in(self):
        top, left, bottom, right = self.observation_bounds_in
        return (
            (left, top),
            (right, top),
            (left, bottom),
            (right, bottom),
        )

    @property
    def print_sheet_label(self):
        return (
            f"{format_measurement(self.page_width)}{self.unit} x "
            f"{format_measurement(self.page_length)}{self.unit} print sheet"
        )

    @property
    def observation_label(self):
        return (
            f"{format_measurement(self.observation_width)}{self.unit} x "
            f"{format_measurement(self.observation_length)}{self.unit} "
            "observation box"
        )

    @property
    def marker_label(self):
        return (
            f"{format_measurement(self.marker_diameter)}{self.unit} "
            "marker diameter"
        )

    @property
    def artwork_labels(self):
        """Short labels sized to stay left of the centered one-inch QR code."""
        return (
            f"Sheet: {format_measurement(self.page_width)} x "
            f"{format_measurement(self.page_length)} {self.unit}",
            f"Area: {format_measurement(self.observation_width)} x "
            f"{format_measurement(self.observation_length)} {self.unit}",
            f"Marker: {format_measurement(self.marker_diameter)} "
            f"{self.unit} diameter",
        )


def build_template_layout(width, length, unit):
    """Validate a print sheet and derive its observation-box geometry."""
    try:
        width = float(width)
        length = float(length)
    except (TypeError, ValueError) as exc:
        raise TemplateLayoutError("Width and length must be numbers.") from exc

    if not math.isfinite(width) or not math.isfinite(length):
        raise TemplateLayoutError("Width and length must be finite numbers.")
    if width <= 0 or length <= 0:
        raise TemplateLayoutError("Width and length must be greater than zero.")

    if unit not in {"in", "cm"}:
        raise TemplateLayoutError("Unit must be 'in' or 'cm'.")

    width = round_to_increment(width)
    length = round_to_increment(length)
    page_width_in = to_inches(width, unit)
    page_length_in = to_inches(length, unit)
    shortest_in = min(page_width_in, page_length_in)
    longest_in = max(page_width_in, page_length_in)

    minimum_edge = minimum_template_edge(unit)
    maximum_edge = maximum_template_edge(unit)
    if min(width, length) < minimum_edge:
        raise TemplateLayoutError(
            f"Each template edge must be at least {format_measurement(minimum_edge)}{unit} so the QR code, "
            "labels, and markers fit safely."
        )
    if max(width, length) > maximum_edge:
        raise TemplateLayoutError(
            f"Each template edge must be no more than {format_measurement(maximum_edge)}{unit}."
        )

    aspect_ratio = longest_in / shortest_in
    if aspect_ratio > MAX_ASPECT_RATIO:
        minimum_short_in = longest_in / MAX_ASPECT_RATIO
        minimum_short = format_measurement(from_inches(minimum_short_in, unit))
        raise TemplateLayoutError(
            f"This {format_measurement(width)} x {format_measurement(length)}{unit} "
            f"sheet has a {aspect_ratio:.2f}:1 ratio. The maximum is "
            f"{MAX_ASPECT_RATIO:g}:1; increase the shorter edge to at least "
            f"{minimum_short}{unit}."
        )

    marker_diameter_in = marker_diameter_for_size(width, length, unit)
    top_margin, side_margin, bottom_margin = margin_profile(unit)
    top_inset_in = to_inches(top_margin, unit)
    side_inset_in = to_inches(side_margin, unit)
    bottom_inset_in = to_inches(bottom_margin, unit)

    observation_width_in = page_width_in - 2.0 * side_inset_in
    observation_length_in = page_length_in - top_inset_in - bottom_inset_in
    if observation_width_in <= 0 or observation_length_in <= 0:
        raise TemplateLayoutError(
            "The selected sheet is too small for the required marker and header "
            "clearances."
        )

    # The QR payload is limited to four decimal places. Snap the drawn marker
    # spacing to those same values so the encoded calibration and artwork are
    # physically identical, including when centimeters are selected.
    observation_width = float(
        format_measurement(from_inches(observation_width_in, unit))
    )
    observation_length = float(
        format_measurement(from_inches(observation_length_in, unit))
    )
    observation_width_in = to_inches(observation_width, unit)
    observation_length_in = to_inches(observation_length, unit)

    return TemplateLayout(
        unit=unit,
        page_width_in=page_width_in,
        page_length_in=page_length_in,
        observation_width_in=observation_width_in,
        observation_length_in=observation_length_in,
        marker_diameter_in=marker_diameter_in,
        side_inset_in=side_inset_in,
        top_inset_in=top_inset_in,
        bottom_inset_in=bottom_inset_in,
    )
