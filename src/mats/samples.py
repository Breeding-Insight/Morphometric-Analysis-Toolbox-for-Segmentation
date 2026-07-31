"""Locations and metadata for the sample images shipped with MATS.

Deliberately dependency-light: this module imports only the standard library so
the CLI, the docs, and the offline tests can resolve the packaged sample images
without importing :mod:`streamlit` or :mod:`mats.core`. Keep it that way.

The images live under the Streamlit app's asset tree so they ship in the wheel
through the ``mats`` package-data globs in ``pyproject.toml``. Everything here
degrades gracefully when they are absent -- an installation without them must
still render the Help page (see :mod:`mats.app.branding` for the same
contract).
"""

from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "app" / "assets" / "samples"

# Ordered easiest-first: the Help page teaches by contrasting these two. Each
# set's "hero" is the one photo its comparison gallery displays; "filenames"
# is the full install manifest (packaging tests, the ZIP export, and the
# `mats run` quickstart iterate every file in the set, not just the hero).
SAMPLE_SETS = (
    {
        "key": "flat_bench",
        "directory": "flat_bench_6x6in",
        "title": "Flat on a bench — the easy case",
        "sheet_dimensions": "8x8in",
        "calibration_dimensions": "6x6in",
        "print_sheet": "8 x 8 in",
        "marker_diameter": "0.5 in",
        "segmentation": "Classic thresholding (Otsu)",
        "hero": "flat_bench_simple_leaf_01.jpg",
        "filenames": ("flat_bench_simple_leaf_01.jpg",),
    },
    {
        "key": "handheld_field",
        "directory": "handheld_field_10.5x9.5in",
        "title": "Hand-held in the field — the hard case",
        "sheet_dimensions": "12x12in",
        "calibration_dimensions": "10.5x9.5in",
        "print_sheet": "12 x 12 in",
        "marker_diameter": "0.5 in",
        "segmentation": "BiRefNet",
        "hero": "handheld_field_compound_leaf_01.jpg",
        "filenames": (
            "handheld_field_compound_leaf_01.jpg",
            "handheld_field_qr_unreadable.jpg",
        ),
    },
)

# A real, reproducible QR failure: this photo's motion blur defeats every
# bundled decoder (OpenCV, pyzbar, QReader). It ships deliberately -- see
# app/assets/samples/README.md -- as the Help page's worked example for why
# manual dimension entry is the most consistent input method. It is also
# declared in the handheld_field set's `filenames` above, so it is packaged,
# tested for presence, and included in the sample ZIP like any other file.
QR_FAILURE_SAMPLE = {
    "directory": "handheld_field_10.5x9.5in",
    "filename": "handheld_field_qr_unreadable.jpg",
    "sheet_dimensions": "12x12in",
    "calibration_dimensions": "10.5x9.5in",
    "printed_label": (
        "12in x 12in template / 10.5in x 9.5in observation box / "
        "0.5in diameter marker"
    ),
}


def sample_set_dir(sample_set):
    """Return the directory for ``sample_set``, which may not exist."""
    return SAMPLES_DIR / sample_set["directory"]


def sample_paths(sample_set):
    """Return the readable image paths for ``sample_set``, in listed order.

    Returns an empty list when the images were not installed, so callers can
    degrade instead of raising.
    """
    directory = sample_set_dir(sample_set)
    return [
        path
        for path in (directory / name for name in sample_set["filenames"])
        if path.is_file()
    ]


def sample_hero_path(sample_set):
    """Return the one photo ``sample_set``'s comparison gallery shows, or None."""
    path = sample_set_dir(sample_set) / sample_set["hero"]
    return path if path.is_file() else None


def qr_failure_sample_path():
    """Return the QR-failure demo photo's path, or None if not installed."""
    path = SAMPLES_DIR / QR_FAILURE_SAMPLE["directory"] / QR_FAILURE_SAMPLE["filename"]
    return path if path.is_file() else None


def samples_installed():
    """True when at least one packaged sample image is readable."""
    return any(sample_paths(sample_set) for sample_set in SAMPLE_SETS)
