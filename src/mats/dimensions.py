"""Template-dimension parsing.

Deliberately dependency-light: this module imports only the standard library so
that the Streamlit Template Creator page can parse and validate template sizes
without importing :mod:`mats.core` (which pulls in torch, rfdetr, cv2, pyzbar and
transformers). Keep it that way -- do not add heavy imports here.
"""

import re

# Matches "<width>x<height><unit>", e.g. "10.5x9.5in" or "27x24cm".
TEMPLATE_DIM_PATTERN = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(cm|in)\s*$',
    re.IGNORECASE,
)


def parse_template_dimensions(dim_str):
    """Parse a template dimension string into ``(width, height, unit)``.

    Returns ``None`` if ``dim_str`` is ``None`` or does not match the expected
    ``<width>x<height><unit>`` format, so callers can validate and re-prompt.
    """
    if dim_str is None:
        return None
    match = TEMPLATE_DIM_PATTERN.match(dim_str)
    if not match:
        return None
    width, height, unit = match.groups()
    return float(width), float(height), unit.lower()
