"""Threshold-level parsing -- standard library only, so it runs in CI."""

import pytest

from mats.thresholds import (
    CUSTOM_THRESHOLD_LEVEL,
    THRESHOLD_LEVEL_OPTIONS,
    THRESHOLD_LEVELS,
    parse_threshold_level,
    threshold_value_for,
)


def test_presets_are_unchanged():
    # Changing these changes every preset run's measurements.
    assert THRESHOLD_LEVELS == {"auto": None, "low": 100, "medium": 125, "high": 150}


def test_custom_is_an_option_but_never_a_preset():
    # None means Otsu in THRESHOLD_LEVELS; a "custom" key would risk silently running Otsu.
    assert CUSTOM_THRESHOLD_LEVEL not in THRESHOLD_LEVELS
    assert THRESHOLD_LEVEL_OPTIONS == ("auto", "custom", "low", "medium", "high")


@pytest.mark.parametrize("text, expected", [
    ("auto", "auto"),
    ("HIGH", "high"),
    (" Medium ", "medium"),
    ("1", 1),
    ("177", 177),
    ("255", 255),
])
def test_parse_accepts_presets_and_in_range_integers(text, expected):
    assert parse_threshold_level(text) == expected


@pytest.mark.parametrize("text", ["0", "256", "-5", "12.5", "max", "1_77", "", "custom"])
def test_parse_rejects_everything_else(text):
    with pytest.raises(ValueError):
        parse_threshold_level(text)


def test_bare_custom_explains_that_a_number_is_needed():
    with pytest.raises(ValueError, match="integer 1-255"):
        parse_threshold_level("custom")


def test_threshold_value_for_presets_integers_and_custom():
    assert threshold_value_for("auto") is None
    assert threshold_value_for("medium") == 125
    assert threshold_value_for(177) == 177
    assert threshold_value_for("custom", 140) == 140


@pytest.mark.parametrize("custom_value", [None, 0, 256, 140.0, True])
def test_custom_value_must_be_an_in_range_integer(custom_value):
    with pytest.raises(ValueError):
        threshold_value_for("custom", custom_value)
