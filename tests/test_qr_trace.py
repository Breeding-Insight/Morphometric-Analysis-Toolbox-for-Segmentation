"""Per-decoder QR trace and conditional full-output schema behavior."""

import csv

import pytest


pytest.importorskip("numpy")
from mats import core


class _Detector:
    def __init__(self, result):
        self.result = result

    def detectAndDecodeMulti(self, _image):
        return self.result


def test_qr_trace_stops_after_opencv_success(monkeypatch):
    points = core.np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]])
    monkeypatch.setattr(
        core.cv2,
        "QRCodeDetector",
        lambda: _Detector((True, ("6x6in",), points, None)),
    )
    monkeypatch.setattr(core, "_pyzbar_decode", lambda _image: ())
    monkeypatch.setattr(core, "_QReader", object())

    retval, payload, _points, backend, trace = core._three_pronged_qr_with_trace(
        core.np.zeros((2, 2), dtype=core.np.uint8)
    )

    assert retval is True
    assert payload == ("6x6in",)
    assert backend == "OpenCV"
    assert trace == {
        "qr_opencv": "success",
        "qr_pyzbar_zbar": "unused",
        "qr_qreader": "unused",
    }


def test_qr_trace_records_pyzbar_success_and_skips_qreader(monkeypatch):
    monkeypatch.setattr(
        core.cv2,
        "QRCodeDetector",
        lambda: _Detector((False, (), None, None)),
    )

    class Point:
        x, y = 1, 2

    class Decoded:
        data = b"10.5x9.5in"
        polygon = (Point(), Point(), Point(), Point())

    monkeypatch.setattr(core, "_pyzbar_decode", lambda _image: (Decoded(),))
    monkeypatch.setattr(core, "_QReader", object())

    retval, payload, _points, backend, trace = core._three_pronged_qr_with_trace(
        core.np.zeros((2, 2), dtype=core.np.uint8)
    )

    assert retval is True
    assert payload == "10.5x9.5in"
    assert backend == "pyzbar"
    assert trace == {
        "qr_opencv": "failed",
        "qr_pyzbar_zbar": "success",
        "qr_qreader": "unused",
    }


def test_qr_trace_records_qreader_success_after_other_decoders_fail(monkeypatch):
    monkeypatch.setattr(
        core.cv2,
        "QRCodeDetector",
        lambda: _Detector((False, (), None, None)),
    )
    monkeypatch.setattr(core, "_pyzbar_decode", lambda _image: ())
    monkeypatch.setattr(core, "_QReader", object())

    class QReader:
        def detect_and_decode(self, *, image):
            del image
            return ("10.5x9.5in",)

        def detect(self, *, image):
            del image
            return ({"quad_xy": core.np.array([[1, 2], [3, 4], [5, 6], [7, 8]])},)

    monkeypatch.setattr(core, "_get_qreader", lambda: QReader())

    retval, payload, _points, backend, trace = core._three_pronged_qr_with_trace(
        core.np.zeros((2, 2), dtype=core.np.uint8)
    )

    assert retval is True
    assert payload == "10.5x9.5in"
    assert backend == "qreader"
    assert trace == {
        "qr_opencv": "failed",
        "qr_pyzbar_zbar": "failed",
        "qr_qreader": "success",
    }


def test_missing_optional_packages_do_not_create_trace_columns(monkeypatch):
    monkeypatch.setattr(core, "_pyzbar_decode", None)
    monkeypatch.setattr(core, "_QReader", None)
    monkeypatch.setattr(
        core.cv2,
        "QRCodeDetector",
        lambda: _Detector((False, (), None, None)),
    )

    retval, payload, points, backend, trace = core._three_pronged_qr_with_trace(
        core.np.zeros((2, 2), dtype=core.np.uint8)
    )

    assert (retval, payload, points, backend) == (False, None, None, "FAIL")
    assert trace == {"qr_opencv": "failed"}


def test_full_qr_csv_uses_only_available_backend_columns(monkeypatch, tmp_path):
    fields = ("qr_opencv", "qr_qreader")
    monkeypatch.setattr(core, "available_qr_backend_fields", lambda: fields)

    def fake_process(*args):
        sample_id = args[0]
        backend_fields = args[-1]
        row = core.measurement_na_row(sample_id, "QR_READ: QR not found/readable")
        row.update({field: "failed" for field in backend_fields})
        return sample_id, {"sample_id": sample_id, "status": "ok", "result_row": row}, None

    monkeypatch.setattr(core, "_process_batch_image", fake_process)
    results_path = tmp_path / "full.csv"
    result = core.run_leaf_morpho_batch(
        ["leaf.jpg"],
        str(tmp_path),
        str(results_path),
        template_dimensions=None,
        compact_csv=False,
        workers=1,
    )

    with results_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[-2:] == list(fields)
        assert next(reader)["qr_qreader"] == "failed"
    assert result["qr_backend_fields"] == fields


def test_manual_and_compact_runs_omit_qr_trace_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "available_qr_backend_fields", lambda: ("qr_opencv", "qr_qreader"))
    monkeypatch.setattr(core, "_process_batch_image", lambda *args: (
        args[0],
        {"sample_id": args[0], "status": "ok", "result_row": core.measurement_na_row(args[0], "test")},
        None,
    ))

    for name, dimensions, compact in (
        ("manual.csv", (6.0, 6.0, "in"), False),
        ("compact.csv", None, True),
    ):
        results_path = tmp_path / name
        result = core.run_leaf_morpho_batch(
            ["leaf.jpg"],
            str(tmp_path),
            str(results_path),
            template_dimensions=dimensions,
            compact_csv=compact,
            workers=1,
        )
        with results_path.open(newline="") as handle:
            assert not {"qr_opencv", "qr_qreader"} & set(csv.DictReader(handle).fieldnames)
        assert result["qr_backend_fields"] == ()
