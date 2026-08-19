import math

import pytest

from mats.dimensions import parse_template_dimensions
from mats.template_layout import (
    MAX_ASPECT_RATIO,
    TemplateLayoutError,
    build_template_layout,
    marker_diameter_for_size,
    round_to_increment,
)


def test_default_layout_is_derived_from_print_sheet():
    layout = build_template_layout(12, 12, "in")

    assert layout.page_width_in == 12
    assert layout.page_length_in == 12
    assert layout.marker_diameter_in == 0.75
    assert layout.observation_width_in == 10
    assert layout.observation_length_in == 9.5
    assert layout.calibration_dimensions == (10.0, 9.5, "in")
    assert layout.qr_payload == "10x9.5in"
    assert parse_template_dimensions(layout.qr_payload) == (
        layout.observation_width,
        layout.observation_length,
        "in",
    )


@pytest.mark.parametrize(
    "longest_edge,expected",
    [
        (4, 0.5),
        (11.9999, 0.5),
        (12, 0.75),
        (17.9999, 0.75),
        (18, 1.0),
        (23.9999, 1.0),
        (24, 1.5),
        (200, 1.5),
    ],
)
def test_marker_tiers_are_inclusive_at_each_inch_threshold(longest_edge, expected):
    assert marker_diameter_for_size(longest_edge, longest_edge) == expected


@pytest.mark.parametrize("width,length", [(12, 8), (8, 12)])
def test_ratio_limit_is_orientation_independent_and_inclusive(width, length):
    layout = build_template_layout(width, length, "in")
    assert layout.aspect_ratio == MAX_ASPECT_RATIO


@pytest.mark.parametrize("width,length", [(10, 5), (5, 10)])
def test_two_to_one_ratio_is_rejected_with_actionable_minimum(width, length):
    with pytest.raises(TemplateLayoutError, match=r"2\.00:1 ratio") as exc_info:
        build_template_layout(width, length, "in")

    assert "increase the shorter edge to at least 6.6667in" in str(exc_info.value)


def test_centimeters_have_clean_native_margins_and_marker_tiers():
    layout = build_template_layout(30.48, 30.48, "cm")

    assert math.isclose(layout.page_width, 30.5)
    assert math.isclose(layout.marker_diameter, 2)
    assert math.isclose(layout.observation_width, 25.5)
    assert math.isclose(layout.observation_length, 24.5)
    assert parse_template_dimensions(layout.qr_payload) == (
        layout.observation_width,
        layout.observation_length,
        "cm",
    )


def test_every_marker_remains_inside_the_page():
    for width, length in ((4, 4), (12, 8), (18, 24), (24, 24)):
        layout = build_template_layout(width, length, "in")
        radius = layout.marker_diameter_in / 2

        for center_x, center_y in layout.marker_centers_in:
            assert center_x - radius >= 0
            assert center_y - radius >= 0
            assert center_x + radius <= layout.page_width_in
            assert center_y + radius <= layout.page_length_in


@pytest.mark.parametrize(
    "width,length,unit,expected_marker",
    [
        (30, 30, "cm", 1.5),
        (30.5, 30.5, "cm", 2),
        (45, 45, "cm", 2),
        (45.5, 45.5, "cm", 2.5),
        (60.5, 60.5, "cm", 2.5),
        (61, 61, "cm", 4),
    ],
)
def test_centimeter_marker_tiers_are_inclusive_at_each_threshold(
    width, length, unit, expected_marker
):
    layout = build_template_layout(width, length, unit)
    assert math.isclose(layout.marker_diameter, expected_marker)


@pytest.mark.parametrize(
    "value,expected",
    [(10.24, 10), (10.25, 10.5), (10.74, 10.5), (10.75, 11)],
)
def test_dimension_rounding_uses_conventional_half_up(value, expected):
    assert round_to_increment(value) == expected


def test_fixed_margins_and_centered_qr_are_derived_from_the_active_unit():
    inch_layout = build_template_layout(12, 12, "in")
    cm_layout = build_template_layout(30, 30, "cm")

    assert (inch_layout.side_inset_in, inch_layout.top_inset_in) == (1, 1.5)
    assert math.isclose(cm_layout.in_unit(cm_layout.side_inset_in), 2.5)
    assert math.isclose(cm_layout.in_unit(cm_layout.top_inset_in), 3.5)

    for layout in (inch_layout, cm_layout):
        qr_top, qr_left, qr_bottom, qr_right = layout.qr_bounds_in
        assert math.isclose((qr_left + qr_right) / 2, layout.page_width_in / 2)
        assert qr_top >= 0
        assert qr_bottom < layout.top_inset_in
        _, label_left, _, label_right = layout.label_bounds_in
        assert label_left < label_right < qr_left


@pytest.mark.parametrize(
    "width,length,unit",
    [(4, 4, "in"), (24, 24, "in"), (10, 10, "cm"), (61, 61, "cm")],
)
def test_markers_fit_inside_fixed_margins_at_small_and_large_tiers(
    width, length, unit
):
    layout = build_template_layout(width, length, unit)
    radius = layout.marker_diameter_in / 2

    for center_x, center_y in layout.marker_centers_in:
        assert center_x - radius >= 0
        assert center_y - radius >= 0
        assert center_x + radius <= layout.page_width_in
        assert center_y + radius <= layout.page_length_in


@pytest.mark.parametrize(
    "width,length,unit,error",
    [
        (3.74, 4, "in", "at least 4in"),
        (4, 201, "in", "no more than 200in"),
        (10, 10, "mm", "Unit must be"),
        (float("nan"), 10, "in", "finite"),
        (0, 10, "in", "greater than zero"),
    ],
)
def test_invalid_physical_dimensions_are_rejected(width, length, unit, error):
    with pytest.raises(TemplateLayoutError, match=error):
        build_template_layout(width, length, unit)
