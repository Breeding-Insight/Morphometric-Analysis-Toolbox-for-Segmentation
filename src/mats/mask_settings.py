"""Settings for masks that are never flash-filled.

The clean margin clears a band along the target-box edge, where the template's
printed box outline lands after perspective correction. The stray gap sets how
far a mask piece may sit from the leaf before it counts as stray.

Deliberately dependency-light: this module imports only the standard library so
that the CLI parser and the Streamlit app can validate the settings without
importing :mod:`mats.mask_cleanup` (numpy, OpenCV) or :mod:`mats.core`. Keep it
that way -- do not add heavy imports here.
"""

import math

# Percent of the target box's shorter side, cleared along every edge.
CLEAN_MARGIN_DEFAULT = 1.0
CLEAN_MARGIN_MAX = 10.0

# A fraction of the leaf's bounding-box diagonal.
STRAY_GAP_DEFAULT = 0.25
STRAY_GAP_MAX = 10.0


def _checked_number(value, name, maximum):
    message = f"{name} must be a number 0-{maximum:g} (got {value!r})"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not (math.isfinite(number) and 0 <= number <= maximum):
        raise ValueError(message)
    return number


def checked_clean_margin(value):
    """Return ``value`` as a float from 0 to ``CLEAN_MARGIN_MAX``, else raise ``ValueError``."""
    return _checked_number(value, "clean margin", CLEAN_MARGIN_MAX)


def checked_stray_gap(value):
    """Return ``value`` as a float from 0 to ``STRAY_GAP_MAX``, else raise ``ValueError``."""
    return _checked_number(value, "stray gap", STRAY_GAP_MAX)
