"""Template-dimension parsing.

This is the contract the Template Creator's QR payloads must round-trip through,
so it is deliberately dependency-light (no torch) and covers the exact formats
the pipeline accepts plus the ones it must reject.
"""

import pytest

from mats.dimensions import parse_template_dimensions


@pytest.mark.parametrize(
    "text,expected",
    [
        ("6x6in", (6.0, 6.0, "in")),
        ("10.5x9.5in", (10.5, 9.5, "in")),
        ("27x24cm", (27.0, 24.0, "cm")),
        ("12X12IN", (12.0, 12.0, "in")),          # case-insensitive
        ("  10x8cm  ", (10.0, 8.0, "cm")),         # surrounding whitespace
        ("10.5 x 9.5in", (10.5, 9.5, "in")),       # spaces around the x
    ],
)
def test_valid(text, expected):
    assert parse_template_dimensions(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "12x12",           # no unit
        "12in x 12in",     # unit on both, wrong shape
        "12x12mm",         # unsupported unit
        "12by12in",        # wrong separator
        "axbin",           # non-numeric
    ],
)
def test_invalid(text):
    assert parse_template_dimensions(text) is None


def test_qr_payload_round_trips():
    # The Template Creator builds "{w:g}x{h:g}{unit}" and asserts it round-trips.
    for w, h, unit in [(10.5, 9.5, "in"), (6, 6, "in"), (27, 24, "cm")]:
        payload = f"{w:g}x{h:g}{unit}"
        assert parse_template_dimensions(payload) == (float(w), float(h), unit)
