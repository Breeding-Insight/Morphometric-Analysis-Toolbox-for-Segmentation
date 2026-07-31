from mats import qr_runtime


def test_qr_runtime_reports_full_fallback_coverage(monkeypatch):
    monkeypatch.setattr(qr_runtime.importlib, "import_module", lambda _: object())

    status = qr_runtime.qr_runtime_status()

    assert status.opencv.available
    assert status.pyzbar.available
    assert status.qreader.available
    assert status.enhanced_available
    assert status.full_fallback_available


def test_qr_runtime_explains_when_pyzbar_needs_zbar(monkeypatch):
    def fake_import(module_name):
        if module_name == "cv2":
            return object()
        if module_name == "pyzbar.pyzbar":
            raise ImportError("Unable to find zbar shared library")
        raise ModuleNotFoundError("No module named 'qreader'", name="qreader")

    monkeypatch.setattr(qr_runtime.importlib, "import_module", fake_import)

    status = qr_runtime.qr_runtime_status()

    assert status.opencv.available
    assert not status.pyzbar.available
    assert "zbar" in status.pyzbar.detail
    assert not status.qreader.available
    assert not status.enhanced_available
    assert not status.full_fallback_available


def test_opencv_preflight_decodes_known_payload(monkeypatch):
    class Detector:
        def detectAndDecodeMulti(self, image):
            assert image == "test-image"
            return True, (qr_runtime._PREFLIGHT_PAYLOAD,), object(), object()

    class CV2:
        QRCodeDetector = Detector

    monkeypatch.setattr(
        qr_runtime.importlib,
        "import_module",
        lambda name: CV2 if name == "cv2" else object(),
    )
    original = qr_runtime.QRBackendStatus("OpenCV", True, "available")

    probed = qr_runtime._probe_opencv(original, "test-image")

    assert probed.available
    assert "decoded the Preflight test QR" in probed.detail


def test_qreader_preflight_never_constructs_the_reader(monkeypatch):
    status = qr_runtime.QRRuntimeStatus(
        opencv=qr_runtime.QRBackendStatus("OpenCV", False, "missing"),
        pyzbar=qr_runtime.QRBackendStatus("pyzbar + zbar", False, "missing"),
        qreader=qr_runtime.QRBackendStatus("QReader", True, "available"),
    )
    monkeypatch.setattr(qr_runtime, "qr_runtime_status", lambda: status)
    monkeypatch.setattr(qr_runtime, "_preflight_qr_image", lambda: "test-image")

    preflight = qr_runtime.qr_preflight_status()

    assert preflight.qreader.available
    assert "does not initialize QReader" in preflight.qreader.detail
