"""Template-dimension parsing.

Deliberately dependency-light: this module imports only the standard library so
that the Streamlit Template Creator page can parse and validate template sizes
without importing :mod:`mats.core` (which pulls in torch, rfdetr, cv2 and
transformers). Keep it that way -- do not add heavy imports here.
"""

import re

# Matches "<width>x<height><unit>", e.g. "10.5x9.5in" or "27x24cm". The unit
# may also be repeated after the width ("6inx6in") -- a trivial formatting
# variant that should never cost someone a run.
TEMPLATE_DIM_PATTERN = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*(cm|in)?\s*x\s*(\d+(?:\.\d+)?)\s*(cm|in)\s*$',
    re.IGNORECASE,
)


def parse_template_dimensions(dim_str):
    """Parse a template dimension string into ``(width, height, unit)``.

    Returns ``None`` if ``dim_str`` is ``None`` or does not match the expected
    ``<width>x<height><unit>`` format, so callers can validate and re-prompt.
    A unit repeated after the width (``"6inx6in"``) is accepted as long as it
    agrees with the height's unit; a genuine mismatch (``"6cmx6in"``) is not
    guessed at and is rejected.
    """
    if dim_str is None:
        return None
    match = TEMPLATE_DIM_PATTERN.match(dim_str)
    if not match:
        return None
    width, width_unit, height, unit = match.groups()
    if width_unit and width_unit.lower() != unit.lower():
        return None
    return float(width), float(height), unit.lower()
