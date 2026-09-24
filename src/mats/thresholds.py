"""Grayscale threshold levels for classic (non-BiRefNet) segmentation.

Deliberately dependency-light: this module imports only the standard library so
that the CLI parser and the Streamlit app can validate threshold choices without
importing :mod:`mats.core` (which pulls in torch, rfdetr, cv2 and transformers).
Keep it that way -- do not add heavy imports here.
"""

THRESHOLD_LEVELS = {
    "auto": None,   # Otsu's method: threshold computed per-image from histogram
    "low": 100,
    "medium": 125,
    "high": 150,
}
THRESHOLD_MIN = 1
THRESHOLD_MAX = 255
# Deliberately not a THRESHOLD_LEVELS key: None there means Otsu, so a lost
# custom value must fail loudly instead of silently running Otsu.
CUSTOM_THRESHOLD_LEVEL = "custom"
THRESHOLD_LEVEL_OPTIONS = ("auto", CUSTOM_THRESHOLD_LEVEL, "low", "medium", "high")


def _checked_cutoff(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"threshold must be an integer {THRESHOLD_MIN}-{THRESHOLD_MAX} (got {value!r})"
        )
    if not THRESHOLD_MIN <= value <= THRESHOLD_MAX:
        raise ValueError(
            f"threshold must be an integer {THRESHOLD_MIN}-{THRESHOLD_MAX} (got {value})"
        )
    return value


def parse_threshold_level(text):
    """Parse a threshold choice into a preset name or an integer cutoff.

    Preset names (``auto``, ``low``, ``medium``, ``high``) are matched
    case-insensitively and returned lowercased; a plain integer between
    ``THRESHOLD_MIN`` and ``THRESHOLD_MAX`` is returned as an ``int``. Anything
    else -- including a bare ``custom``, which names no cutoff -- raises
    ``ValueError``.
    """
    level = str(text).strip().lower()
    if level in THRESHOLD_LEVELS:
        return level
    if level == CUSTOM_THRESHOLD_LEVEL:
        raise ValueError(
            "custom needs a number: pass the cutoff as an integer "
            f"{THRESHOLD_MIN}-{THRESHOLD_MAX}"
        )
    if not (level.isascii() and level.isdigit()):
        raise ValueError(
            "threshold level must be auto, low, medium, high, or an integer "
            f"{THRESHOLD_MIN}-{THRESHOLD_MAX} (got {text!r})"
        )
    return _checked_cutoff(int(level))


def threshold_value_for(level, custom_value=None):
    """Return the cutoff passed to the pipeline: ``None`` (Otsu) or an ``int``.

    ``level`` is a preset name, an integer cutoff, or ``"custom"`` -- in which
    case ``custom_value`` supplies the cutoff.
    """
    if level == CUSTOM_THRESHOLD_LEVEL:
        return _checked_cutoff(custom_value)
    if isinstance(level, int) and not isinstance(level, bool):
        return _checked_cutoff(level)
    return THRESHOLD_LEVELS[level]
