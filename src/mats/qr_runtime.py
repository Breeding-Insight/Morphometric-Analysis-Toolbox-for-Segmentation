"""Availability checks for MATS' optional robust QR decoders.

These checks import decoder modules but never construct a QReader. Constructing
QReader can download its detector model, which must remain an explicit runtime
operation when a QR code actually needs that fallback.
"""

from dataclasses import dataclass
import importlib


@dataclass(frozen=True)
class QRBackendStatus:
    """One QR decoder's usable state and a user-facing explanation."""

    name: str
    available: bool
    detail: str


@dataclass(frozen=True)
class QRRuntimeStatus:
    """The decoder stack available to the current Python process."""

    opencv: QRBackendStatus
    pyzbar: QRBackendStatus
    qreader: QRBackendStatus

    @property
    def enhanced_available(self) -> bool:
        """Whether at least one fallback beyond OpenCV is ready to use."""
        return self.pyzbar.available or self.qreader.available

    @property
    def full_fallback_available(self) -> bool:
        """Whether both optional fallbacks are ready to use."""
        return self.pyzbar.available and self.qreader.available


def _module_status(module_name, label, available_detail, missing_detail, unusable_detail):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {module_name, module_name.partition(".")[0]}:
            return QRBackendStatus(label, False, missing_detail)
        return QRBackendStatus(
            label,
            False,
            f"Not usable because the Python module `{exc.name}` is missing.",
        )
    except Exception:
        return QRBackendStatus(label, False, unusable_detail)
    return QRBackendStatus(label, True, available_detail)


def qr_runtime_status() -> QRRuntimeStatus:
    """Report which QR decoders are usable without triggering downloads."""
    return QRRuntimeStatus(
        opencv=_module_status(
            "cv2",
            "OpenCV",
            "Built-in QR decoder is available.",
            "OpenCV is not installed.",
            "OpenCV is installed but could not be imported.",
        ),
        pyzbar=_module_status(
            "pyzbar.pyzbar",
            "pyzbar + zbar",
            "Fallback decoder is available.",
            "Not installed. Add the optional `mats-morpho[qr]` extra.",
            "The Python package is installed, but its native zbar library is unavailable.",
        ),
        qreader=_module_status(
            "qreader",
            "QReader",
            "Fallback decoder is available; its detector model downloads on first use.",
            "Not installed. Add the optional `mats-morpho[qr]` extra.",
            "Installed but could not be imported in this Python environment.",
        ),
    )


# The current 12 x 12 in Template Creator sheet encodes its derived
# 10 x 9.5 in marker-centre calibration area in the QR payload.
_PREFLIGHT_PAYLOAD = "10x9.5in"


def _preflight_qr_image():
    """Build a small known QR image using the GUI's Template Creator dependency."""
    try:
        qrcode = importlib.import_module("qrcode")
        np = importlib.import_module("numpy")
        image = qrcode.make(_PREFLIGHT_PAYLOAD).convert("L")
    except Exception:
        return None
    return np.asarray(image)


def _probe_opencv(status, image):
    if not status.available or image is None:
        return status
    try:
        cv2 = importlib.import_module("cv2")
        decoded, values, _points, _straight = cv2.QRCodeDetector().detectAndDecodeMulti(
            image
        )
        passed = decoded and _PREFLIGHT_PAYLOAD in tuple(values or ())
    except Exception:
        passed = False
    return QRBackendStatus(
        status.name,
        bool(passed),
        (
            "Available and decoded the Preflight test QR successfully."
            if passed
            else "Installed, but could not decode the Preflight test QR."
        ),
    )


def _probe_pyzbar(status, image):
    if not status.available or image is None:
        return status
    try:
        module = importlib.import_module("pyzbar.pyzbar")
        passed = any(
            item.data.decode("utf-8") == _PREFLIGHT_PAYLOAD
            for item in module.decode(image)
        )
    except Exception:
        passed = False
    return QRBackendStatus(
        status.name,
        bool(passed),
        (
            "Available and decoded the Preflight test QR successfully."
            if passed
            else "Installed, but could not decode the Preflight test QR."
        ),
    )


def qr_preflight_status() -> QRRuntimeStatus:
    """Verify QR readers without constructing QReader or triggering downloads.

    OpenCV and pyzbar decode a generated, known payload. QReader is import-checked
    only because constructing it may download its detector model on first use.
    """
    status = qr_runtime_status()
    image = _preflight_qr_image()
    qreader = status.qreader
    if qreader.available:
        qreader = QRBackendStatus(
            qreader.name,
            True,
            "Available. Preflight does not initialize QReader because its detector "
            "model may download on first use.",
        )
    return QRRuntimeStatus(
        opencv=_probe_opencv(status.opencv, image),
        pyzbar=_probe_pyzbar(status.pyzbar, image),
        qreader=qreader,
    )
