"""MATs -- field morphometric tools.

A two-model pipeline (RF-DETR fiducial-marker detection + BiRefNet leaf
segmentation) that turns a phone photo of leaves on a printed calibration
template into per-leaf area, length and width in real-world units.

Public entry points:
    mats run             batch-measure a folder of images
    mats app             launch the Streamlit GUI
    mats fetch-weights   download the model checkpoints
    mats doctor          report the environment
"""

__version__ = "1.0.0"

# Light-weight re-exports only. Importing heavy pipeline symbols (torch, rfdetr)
# is deferred to `mats.core`, so `import mats` and `mats.dimensions` stay cheap.
from .dimensions import TEMPLATE_DIM_PATTERN, parse_template_dimensions

__all__ = ["__version__", "TEMPLATE_DIM_PATTERN", "parse_template_dimensions"]
