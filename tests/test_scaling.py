"""Pixel-to-metric scale math.

Dependency-light (no torch/cv2) so these run in the fast offline suite. See
``mats.scaling`` for why the pipeline scales width/length/area independently
per axis rather than with one averaged scalar.
"""

import pytest

from mats.scaling import (
    COMPACT_RESULTS_FIELDNAMES,
    NA_VALUE,
    RESULTS_FIELDNAMES,
    compact_measurement_row,
    measurement_na_row,
    measurement_row,
    px_per_cm_axes,
)


def test_px_per_cm_axes_cm():
    # 2100x1800 px raster over a 21x18 cm template -> exactly 100 px/cm on both axes.
    assert px_per_cm_axes((1800, 2100), 21.0, 18.0, "cm") == (100.0, 100.0)


def test_px_per_cm_axes_inches():
    sx, sy = px_per_cm_axes((1800, 2100), 10.5, 9.5, "in")
    assert sx == pytest.approx(2100 / 10.5 / 2.54)
    assert sy == pytest.approx(1800 / 9.5 / 2.54)


@pytest.mark.parametrize(
    "box_shape,template_width,template_height",
    [
        (None, 10.5, 9.5),
        ((1800, 2100), None, 9.5),
        ((1800, 2100), 10.5, None),
        ((1800, 2100), 0, 9.5),
        ((1800, 2100), 10.5, -1),
    ],
)
def test_px_per_cm_axes_invalid(box_shape, template_width, template_height):
    assert px_per_cm_axes(box_shape, template_width, template_height, "cm") is None


def test_measurement_row_worked_case():
    # 10.5x9.5in template, 2100x1800 px raster (raster aspect 1.1667 vs. true
    # template aspect 1.1053 -- the ~5.6% mismatch a real perspective warp can
    # introduce), 800x600 px leaf bbox, 310000 white pixels.
    sx, sy = px_per_cm_axes((1800, 2100), 10.5, 9.5, "in")
    row = measurement_row("leaf01", 310000, 800, 600, sx, sy)

    assert row["sample_id"] == "leaf01"
    assert row["leaf_area_cm2"] == pytest.approx(52.7777, abs=1e-3)
    assert row["width_cm"] == pytest.approx(10.16, abs=1e-3)
    assert row["length_cm"] == pytest.approx(8.0433, abs=1e-3)
    assert row["px_per_cm_width"] == pytest.approx(sx)
    assert row["px_per_cm_height"] == pytest.approx(sy)
    assert row["scale_aspect_ratio"] == pytest.approx(sx / sy)
    assert row["source"] == 0


@pytest.mark.parametrize("kx,ky", [(1.0, 1.0), (1.3, 1.0), (1.0, 0.7), (2.0, 0.5), (0.6, 1.8)])
def test_measurement_row_invariant_to_raster_aspect(kx, ky):
    # The warped raster's pixel size is an artifact of which marker-quad edge
    # perspective_transform's max() happened to pick -- not a property of the
    # template. Stretching the raster (and everything measured within it) by
    # independent per-axis factors must not move the calibrated result: each
    # axis's own scale factor is stretched by the same amount and cancels.
    base_sx, base_sy = px_per_cm_axes((1800, 2100), 10.5, 9.5, "in")
    base = measurement_row("s", 310000, 800, 600, base_sx, base_sy)

    stretched_shape = (1800 * ky, 2100 * kx)
    sx, sy = px_per_cm_axes(stretched_shape, 10.5, 9.5, "in")
    stretched = measurement_row(
        "s",
        310000 * kx * ky,
        800 * kx,
        600 * ky,
        sx,
        sy,
    )

    assert stretched["leaf_area_cm2"] == pytest.approx(base["leaf_area_cm2"])
    assert stretched["width_cm"] == pytest.approx(base["width_cm"])
    assert stretched["length_cm"] == pytest.approx(base["length_cm"])


def test_measurement_na_row_all_na_except_sample_and_source():
    row = measurement_na_row("leaf01", "SCALE: physical dimensions unavailable")
    assert row["sample_id"] == "leaf01"
    assert row["source"] == "SCALE: physical dimensions unavailable"
    for field in RESULTS_FIELDNAMES:
        if field in ("sample_id", "source"):
            continue
        assert row[field] == NA_VALUE


def test_results_fieldnames_match_row_keys():
    row = measurement_row("s", 100, 10, 10, 1.0, 1.0)
    assert set(row.keys()) == set(RESULTS_FIELDNAMES)
    assert set(measurement_na_row("s", "x").keys()) == set(RESULTS_FIELDNAMES)


def test_compact_measurement_row_maps_fields():
    row = measurement_row("leaf01", 310000, 800, 600, 78.74, 74.60)
    compact = compact_measurement_row(row)
    assert compact == {
        "sample_id": "leaf01",
        "area_cm2": row["leaf_area_cm2"],
        "width_cm": row["width_cm"],
        "length_cm": row["length_cm"],
    }
    assert set(compact.keys()) == set(COMPACT_RESULTS_FIELDNAMES)


def test_compact_measurement_row_na_passthrough():
    na_row = measurement_na_row("leaf01", "LEAF_MASK: leaf not detected")
    compact = compact_measurement_row(na_row)
    assert compact == {
        "sample_id": "leaf01",
        "area_cm2": NA_VALUE,
        "width_cm": NA_VALUE,
        "length_cm": NA_VALUE,
    }
