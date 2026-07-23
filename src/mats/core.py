# flake8: noqa
# pyright: reportMissingImports=false
# Standard library imports
import os
import multiprocessing as mp
import csv
from itertools import combinations
import concurrent.futures
from pathlib import Path

# Some ops (e.g. antialiased bicubic upsampling in RF-DETR preprocessing) are not
# yet implemented for Apple's MPS backend. Enabling this fallback runs only those
# missing ops on CPU. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Third-party libraries
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from pyzbar.pyzbar import decode
from rfdetr import RFDETRLarge
from tqdm import tqdm  # type: ignore[import-not-found]

# Local application/library specific imports
from qreader import QReader

# Template-dimension parsing lives in its own dependency-light module so the
# Template Creator page can import it without pulling in torch/rfdetr/cv2.
# Re-exported here to preserve the historical `core.parse_template_dimensions` API.
from .dimensions import TEMPLATE_DIM_PATTERN, parse_template_dimensions

# Checkpoint resolution and constants live in mats.paths; re-exported for API
# compatibility with callers that read core.RF_DETR_MARKER_CHECKPOINT etc.
from .paths import (
    RF_DETR_MARKER_CHECKPOINT,
    BIREFNET_CHECKPOINT,
    _resolve_checkpoint,
)

RF_DETR_MARKER_RESOLUTION = 1120
RF_DETR_MARKER_CONFIDENCE = 0.5
RF_DETR_MARKER_PAD_COLOR = (0, 0, 0)
BIREFNET_PRETRAINED = "ZhengPeng7/BiRefNet"
BIREFNET_IMAGE_SIZE = 2048
BIREFNET_THRESHOLD = 0.5
BIREFNET_MEAN = [0.485, 0.456, 0.406]
BIREFNET_STD = [0.229, 0.224, 0.225]
TARGET_BOX_SUFFIX = "_target_box"
RESULTS_FIELDNAMES = [
    "sample_id",
    "leaf_area_cm2_meanscale",
    "width_cm_meanscale",
    "length_cm_meanscale",
    "px_per_cm_mean",
    "leaf_area_cm2_widthscale",
    "width_cm_widthscale",
    "length_cm_widthscale",
    "px_per_cm_width",
    "leaf_area_cm2_heightscale",
    "width_cm_heightscale",
    "length_cm_heightscale",
    "px_per_cm_height",
    "source",
]
COMPACT_RESULTS_FIELDNAMES = ["sample_id", "area_cm2", "height_cm", "length_cm"]
NA_VALUE = "NA"
THRESHOLD_LEVELS = {
    "auto": None,   # Otsu's method: threshold computed per-image from histogram
    "low": 100,
    "medium": 125,
    "high": 150,
}
_MARKER_MODEL_V3 = None
_RF_DETR_MARKER_DEVICE = None
_BIREFNET_MODEL = None
_BIREFNET_DEVICE = None

def resolve_rfdetr_device():
    forced_device = os.environ.get("RF_DETR_DEVICE")
    if forced_device:
        return forced_device.lower()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def get_marker_model():
    """Lazy-load the RF-DETR marker detection model once per process."""
    global _MARKER_MODEL_V3, _RF_DETR_MARKER_DEVICE
    if _MARKER_MODEL_V3 is None:
        from . import weights
        checkpoint = weights.ensure_weight("rf-detr")  # resolves or auto-fetches once
        _RF_DETR_MARKER_DEVICE = resolve_rfdetr_device()
        _MARKER_MODEL_V3 = RFDETRLarge(
            resolution=RF_DETR_MARKER_RESOLUTION,
            pretrain_weights=str(checkpoint),
            device=_RF_DETR_MARKER_DEVICE,
        )
        if _RF_DETR_MARKER_DEVICE != "mps":
            try:
                _MARKER_MODEL_V3.optimize_for_inference()
            except Exception as exc:
                print(f"RF-DETR optimize_for_inference() skipped on {_RF_DETR_MARKER_DEVICE}: {exc}")
    return _MARKER_MODEL_V3


def pad_to_square_for_rfdetr(
    pil_img,
    target_size=RF_DETR_MARKER_RESOLUTION,
    fill=RF_DETR_MARKER_PAD_COLOR,
):
    orig_w, orig_h = pil_img.size
    scale = target_size / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2

    canvas = Image.new("RGB", (target_size, target_size), fill)
    canvas.paste(resized, (pad_left, pad_top))

    pad_info = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }
    return canvas, pad_info


def unpad_xyxy(xyxy, pad_info):
    boxes = np.asarray(xyxy, dtype=np.float32)
    single_box = boxes.ndim == 1
    boxes = boxes.reshape(-1, 4).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_info["pad_left"]) / pad_info["scale"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_info["pad_top"]) / pad_info["scale"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, pad_info["orig_w"])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, pad_info["orig_h"])
    return boxes[0] if single_box else boxes


def detect_marker_geometry(image_bgr, confidence=RF_DETR_MARKER_CONFIDENCE):
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    padded_img, pad_info = pad_to_square_for_rfdetr(pil_img)
    detections = get_marker_model().predict(padded_img, threshold=confidence)
    xyxy = np.asarray(detections.xyxy, dtype=np.float32)
    if xyxy.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 4), dtype=np.int32)

    mapped_xyxy = unpad_xyxy(xyxy, pad_info)
    centers = np.column_stack(
        (
            (mapped_xyxy[:, 0] + mapped_xyxy[:, 2]) / 2.0,
            (mapped_xyxy[:, 1] + mapped_xyxy[:, 3]) / 2.0,
        )
    )
    return np.rint(centers).astype(np.int32), np.rint(mapped_xyxy).astype(np.int32)


def detect_marker_centers(image_bgr, confidence=RF_DETR_MARKER_CONFIDENCE):
    centers, _ = detect_marker_geometry(image_bgr, confidence=confidence)
    return centers


def resolve_birefnet_device():
    forced_device = os.environ.get("BIREFNET_DEVICE")
    if forced_device:
        return torch.device(forced_device.lower())
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_birefnet_model():
    """Lazy-load the fine-tuned BiRefNet model once per process."""
    global _BIREFNET_MODEL, _BIREFNET_DEVICE
    if _BIREFNET_MODEL is None:
        from . import weights
        checkpoint = weights.ensure_weight("birefnet")  # resolves or auto-fetches once

        from transformers import AutoModelForImageSegmentation

        _BIREFNET_DEVICE = resolve_birefnet_device()
        model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_PRETRAINED,
            trust_remote_code=True,
        )
        ckpt = torch.load(
            str(checkpoint),
            map_location=_BIREFNET_DEVICE,
            weights_only=False,
        )
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(_BIREFNET_DEVICE)
        model.eval()
        _BIREFNET_MODEL = model
    return _BIREFNET_MODEL


def _preprocess_birefnet_image(image_bgr, image_size=BIREFNET_IMAGE_SIZE):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb).convert("RGB")
    pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(BIREFNET_MEAN).view(3, 1, 1)
    std = torch.tensor(BIREFNET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_birefnet_mask(
    image_bgr,
    image_size=BIREFNET_IMAGE_SIZE,
    threshold=BIREFNET_THRESHOLD,
):
    """Predict a single-channel 0/255 leaf foreground mask for a BGR image."""
    model = get_birefnet_model()
    device = _BIREFNET_DEVICE or resolve_birefnet_device()
    orig_h, orig_w = image_bgr.shape[:2]
    inp = _preprocess_birefnet_image(image_bgr, image_size=image_size).to(device)

    outputs = model(inp)
    pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
    pred = torch.sigmoid(pred)
    pred = F.interpolate(
        pred,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    return (pred[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255

# Initialize QReader
qreader = QReader()

def three_pronged_qr(gray_img):
    # Save retval
    retval = False
    readby = "FAIL"
    # First attempt with OpenCV
    qcd = cv2.QRCodeDetector()
    retval, decoded_info, points, _straight_qrcode = qcd.detectAndDecodeMulti(gray_img)
    if retval:
        points = points.squeeze().astype(np.int64)
        readby = "OpenCV"
        return retval, decoded_info, points, readby

    # If OpenCV fails, try pyzbar
    decoded_objects = decode(gray_img)
    if decoded_objects:
        for obj in decoded_objects:
            decoded_info = obj.data.decode("utf-8")
            points = np.array([[p.x, p.y] for p in obj.polygon])
            retval = True
            readby = "pyzbar"
            return retval, decoded_info, points, readby

    # If pyzbar fails, try qreader
    decoded_info = qreader.detect_and_decode(image=gray_img)
    decoded_info = decoded_info[0]
    decoded_list = qreader.detect(image=gray_img)
    if decoded_list:
        quad_xy_list = [info['quad_xy'] for info in decoded_list]
        points = np.array(quad_xy_list).astype(np.int64).squeeze()
        retval = True
        readby = "qreader"
        return retval, decoded_info, points, readby

    if not retval:
        return retval, None, None, readby

# Order markers in clockwise order, starting with top left
def order_points_clockwise(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right+most points from the sorted
    # x+roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y+coordinates so we can grab the top+left and bottom+left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int32")

# Perspective transform function
def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Convert to numpy array for easier manipulation
        corners = np.array(corners, dtype="float32").squeeze()

        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype="float32")

        # The top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]  # Top-left has the smallest sum
        rect[2] = corners[np.argmax(s)]  # Bottom-right has the largest sum

        # The top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # Top-right has the smallest difference
        rect[3] = corners[np.argmax(diff)]  # Bottom-left has the largest difference
        top_l, top_r, bottom_r, bottom_l = rect[0], rect[1], rect[2], rect[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def apply_affine_transform(points, mat):
    """Apply a 2x3 affine transform (from cv2.getRotationMatrix2D) to Nx2 points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.hstack([pts, ones])
    transformed = hom @ mat.T
    return transformed

def find_best_corner_points(coordinates_array):
    if len(coordinates_array) > 4:
        min_variance = float('inf')

        # Iterate through all combinations of four points
        for combination in combinations(coordinates_array, 4):
            combination = np.array(combination)
            # Calculate the pairwise distances
            dists = [np.linalg.norm(combination[i] - combination[j]) for i in range(4) for j in range(i + 1, 4)]
            variance = np.var(dists)
            if variance < min_variance:
                min_variance = variance
                corner_points = combination
        return order_points_clockwise(corner_points)
    else:
        return order_points_clockwise(coordinates_array)


def get_input_images(input_dir):
    try:
        # Ensure the input directory exists
        if not os.path.isdir(input_dir):
            raise ValueError(f"The directory {input_dir} does not exist or is not a directory.")
        else:
            # List only valid image files in the directory (skip hidden files like .DS_Store)
            valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
            input_images = []
            for f in os.listdir(input_dir):
                if f.startswith('.'):
                    continue
                p = os.path.join(input_dir, f)
                if not os.path.isfile(p):
                    continue
                _, ext = os.path.splitext(f)
                if ext.lower() not in valid_exts:
                    continue
                input_images.append(p)
            return input_images
    except Exception:
        return []


def is_target_box_image(input_image):
    file_stem = os.path.splitext(os.path.basename(input_image))[0]
    return TARGET_BOX_SUFFIX in file_stem


def target_box_sample_id(input_image):
    file_stem = os.path.splitext(os.path.basename(input_image))[0]
    if file_stem.endswith(TARGET_BOX_SUFFIX):
        return file_stem[:-len(TARGET_BOX_SUFFIX)]
    return file_stem.replace(TARGET_BOX_SUFFIX, "", 1)


def threshold_mask(target_box, threshold_value):
    gray_img = cv2.cvtColor(target_box, cv2.COLOR_BGR2GRAY)
    if threshold_value is None:
        _, binary_mask = cv2.threshold(
            gray_img,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
    else:
        _, binary_mask = cv2.threshold(
            gray_img,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV,
        )
    return binary_mask


def white_out_marker_boxes(image_bgr, marker_boxes):
    whitefilled_image = image_bgr.copy()
    if marker_boxes.size == 0:
        return whitefilled_image

    img_h, img_w = whitefilled_image.shape[:2]
    for x1, y1, x2, y2 in marker_boxes.astype(int):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        whitefilled_image[y1:y2 + 1, x1:x2 + 1] = 255
    return whitefilled_image


def clean_leaf_mask(binary_mask):
    # Re-join tiny breaks using morphological closing (dilate then erode).
    h, w = binary_mask.shape[:2]
    k = max(3, min(11, ((min(h, w) // 300) * 2) + 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # Keep only the largest object, which should be the leaf.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        binary_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        cv2.drawContours(binary_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    return binary_mask


def keep_largest_mask_component(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) <= 0:
        return None

    largest_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return largest_mask


def create_leaf_mask(target_box, mask_method, threshold_value):
    if mask_method == "threshold":
        binary_mask = threshold_mask(target_box, threshold_value)
    else:
        binary_mask = predict_birefnet_mask(target_box)

    if not np.any(binary_mask):
        return None

    binary_mask = clean_leaf_mask(binary_mask)
    return binary_mask


def measurement_na_row(sample_id, source):
    return {
        "sample_id": sample_id,
        "leaf_area_cm2_meanscale": NA_VALUE,
        "width_cm_meanscale": NA_VALUE,
        "length_cm_meanscale": NA_VALUE,
        "px_per_cm_mean": NA_VALUE,
        "leaf_area_cm2_widthscale": NA_VALUE,
        "width_cm_widthscale": NA_VALUE,
        "length_cm_widthscale": NA_VALUE,
        "px_per_cm_width": NA_VALUE,
        "leaf_area_cm2_heightscale": NA_VALUE,
        "width_cm_heightscale": NA_VALUE,
        "length_cm_heightscale": NA_VALUE,
        "px_per_cm_height": NA_VALUE,
        "source": source,
    }


def leaf_bounding_rect(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) <= 0:
        return None
    return cv2.boundingRect(largest_contour)


def draw_measurement_axes(target_box, binary_mask):
    rect = leaf_bounding_rect(binary_mask)
    if rect is None:
        return None

    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None

    annotated = target_box.copy()
    x2 = x + w - 1
    y2 = y + h - 1
    center_x = x + (w // 2)
    center_y = y + (h // 2)
    thickness = max(3, min(15, min(target_box.shape[:2]) // 180))

    cv2.line(annotated, (x, center_y), (x2, center_y), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(annotated, (center_x, y), (center_x, y2), (255, 0, 255), thickness, cv2.LINE_AA)
    return annotated


def save_measurement_axes_image(output_dir, file_name, target_box, binary_mask):
    if output_dir is False:
        return

    annotated = draw_measurement_axes(target_box, binary_mask)
    if annotated is None:
        return

    output_path = os.path.join(output_dir, f"{file_name}_measurement_axes.jpg")
    cv2.imwrite(output_path, annotated)


def measurement_row_from_mask(
    sample_id,
    binary_mask,
    px_per_cm_mean,
    px_per_cm_width=None,
    px_per_cm_height=None,
):
    if px_per_cm_mean is None or px_per_cm_mean <= 0:
        return measurement_na_row(sample_id, "SCALE: physical dimensions unavailable")

    white_pixels = int(np.sum(binary_mask == 255))
    if white_pixels == 0:
        return measurement_na_row(sample_id, "LEAF_MASK: leaf not detected")

    rect = leaf_bounding_rect(binary_mask)
    if rect is None:
        return measurement_na_row(sample_id, "LEAF_MASK: leaf contour not detected")

    _x, _y, w, h = rect

    def _metrics_for_scale(scale):
        if scale is None or scale <= 0:
            return NA_VALUE, NA_VALUE, NA_VALUE, NA_VALUE
        return (
            white_pixels / (scale ** 2),
            w / scale,
            h / scale,
            scale,
        )

    mean_area, mean_width, mean_length, mean_px = _metrics_for_scale(px_per_cm_mean)
    width_area, width_width, width_length, width_px = _metrics_for_scale(px_per_cm_width)
    height_area, height_width, height_length, height_px = _metrics_for_scale(px_per_cm_height)
    row = {
        "sample_id": sample_id,
        "leaf_area_cm2_meanscale": mean_area,
        "width_cm_meanscale": mean_width,
        "length_cm_meanscale": mean_length,
        "px_per_cm_mean": mean_px,
        "leaf_area_cm2_widthscale": width_area,
        "width_cm_widthscale": width_width,
        "length_cm_widthscale": width_length,
        "px_per_cm_width": width_px,
        "leaf_area_cm2_heightscale": height_area,
        "width_cm_heightscale": height_width,
        "length_cm_heightscale": height_length,
        "px_per_cm_height": height_px,
        "source": 0,
    }
    return row


def px_per_cm_from_target_box(target_box, template_width, template_height, unit, scale_axis="average"):
    if target_box is None or template_width is None or template_height is None:
        return None
    if template_width <= 0 or template_height <= 0:
        return None

    img_height, img_width = target_box.shape[:2]
    px_per_unit_h = img_height / template_height
    px_per_unit_w = img_width / template_width
    if scale_axis == "width":
        px_per_unit = px_per_unit_w
    elif scale_axis == "height":
        px_per_unit = px_per_unit_h
    else:
        px_per_unit = (px_per_unit_h + px_per_unit_w) / 2.0
    if unit == "cm":
        return px_per_unit
    return px_per_unit / 2.54


def write_results_csv(result_rows, results_path):
    results_dir = os.path.dirname(os.path.abspath(results_path))
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(result_rows)


def compact_measurement_row(row):
    """Return only the UI-facing measurements requested for bulk export."""
    return {
        "sample_id": row.get("sample_id", NA_VALUE),
        "area_cm2": row.get("leaf_area_cm2_meanscale", NA_VALUE),
        # Existing measurement names use target-box orientation: width is x-axis,
        # length is y-axis. The UI exports these as length and height.
        "height_cm": row.get("length_cm_meanscale", NA_VALUE),
        "length_cm": row.get("width_cm_meanscale", NA_VALUE),
    }


def write_compact_results_csv(result_rows, results_path):
    results_dir = os.path.dirname(os.path.abspath(results_path))
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COMPACT_RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(compact_measurement_row(row) for row in result_rows)


def default_worker_count(input_images, output_mode, mask_method):
    cpu_default = max(1, int(mp.cpu_count() / 2))
    all_target_boxes = bool(input_images) and all(is_target_box_image(p) for p in input_images)
    needs_rfdetr = not all_target_boxes
    needs_birefnet = output_mode == "masks" and mask_method == "birefnet"

    if needs_birefnet and resolve_birefnet_device().type in {"cuda", "mps"}:
        return 1, "BiRefNet is using GPU/MPS"

    if needs_rfdetr and resolve_rfdetr_device() in {"cuda", "mps"}:
        return 1, "RF-DETR is using GPU/MPS"

    if output_mode == "masks" and mask_method == "threshold" and all_target_boxes:
        return cpu_default, "thresholding precomputed target boxes on CPU"

    return cpu_default, "CPU default"


def failure_report_row(input_image, result=None, exception=None):
    sample_id = os.path.splitext(os.path.basename(input_image))[0]
    status = ""
    if result:
        sample_id = result.get("sample_id", sample_id)
        status = result.get("status", "")
    if exception is not None:
        status = f"EXCEPTION: {exception}"

    if ":" in status:
        stage, failure_mode = status.split(":", 1)
        stage = stage.strip()
        failure_mode = failure_mode.strip()
    else:
        stage = status.strip()
        failure_mode = status.strip()

    return {
        "sample_id": sample_id,
        "input_image": input_image,
        "stage": stage,
        "failure_mode": failure_mode,
        "status": status,
    }


def warning_report_rows(input_image, result):
    rows = []
    for status in result.get("warnings", []) if result else []:
        rows.append(failure_report_row(
            input_image,
            result={
                "sample_id": result.get("sample_id"),
                "status": status,
            },
        ))
    return rows


def write_failure_report(failure_rows, output_dir):
    if not failure_rows:
        return None

    report_dir = output_dir if output_dir is not False else os.getcwd()
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "leaf_morpho_failures.csv")
    fieldnames = ["sample_id", "input_image", "stage", "failure_mode", "status"]
    with open(report_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)
    return report_path


# Leaf image process function
def leaf_morpho(
    input_image,
    output_dir,
    template_dimensions=None,
    leaf_model_version=None,
    physical_dimensions=None,
    output_mode="masks",
    mask_method="birefnet",
    threshold_value=THRESHOLD_LEVELS["medium"],
    scale_axis="average",
    save_measurement_axes=True,
):
    if physical_dimensions is not None:
        template_dimensions = physical_dimensions
    is_target_box_input = is_target_box_image(input_image)
    if is_target_box_input:
        file_name = target_box_sample_id(input_image)
    else:
        file_name = os.path.splitext(os.path.basename(input_image))[0]

    def _fail(stage: str, msg: str):
        status = f'{stage}: {msg}'
        return {
            'sample_id': file_name,
            'status': status,
            'result_row': measurement_na_row(file_name, status),
        }

    def _ok(result_row, warnings=None):
        result = {'sample_id': file_name, 'status': 'ok', 'result_row': result_row}
        if warnings:
            result['warnings'] = warnings
        return result

    stage = "START"
    warnings = []
    scale_failure = None

    try:
        # Read image
        stage = "READ_IMAGE"
        image = cv2.imread(input_image, cv2.IMREAD_COLOR)
        if image is None:
            return _fail(stage, "failed to read image (unsupported/corrupt)")

        if is_target_box_input:
            target_box = image
            if output_mode == "target-boxes":
                return _ok(measurement_na_row(file_name, "OUTPUT_MODE: target-boxes only"))

            stage = "LEAF_MASK"
            binary_mask = create_leaf_mask(
                target_box,
                mask_method,
                threshold_value,
            )
            if binary_mask is None:
                return _fail(stage, "leaf not detected")

            if output_dir is not False:
                output_path = os.path.join(output_dir, f"{file_name}_mask.png")
                cv2.imwrite(output_path, binary_mask)
                if save_measurement_axes:
                    save_measurement_axes_image(output_dir, file_name, target_box, binary_mask)

            if template_dimensions is None:
                return _ok(
                    measurement_na_row(file_name, "SCALE: target_box input requires --template_dimensions"),
                    warnings,
                )

            width, height, unit = template_dimensions
            px_per_cm_mean = px_per_cm_from_target_box(target_box, width, height, unit, "average")
            px_per_cm_width = px_per_cm_from_target_box(target_box, width, height, unit, "width")
            px_per_cm_height = px_per_cm_from_target_box(target_box, width, height, unit, "height")
            if px_per_cm_mean is None:
                return _ok(measurement_na_row(file_name, "SCALE: invalid template dimensions"))

            return _ok(
                measurement_row_from_mask(
                    file_name,
                    binary_mask,
                    px_per_cm_mean,
                    px_per_cm_width,
                    px_per_cm_height,
                ),
                warnings,
            )

        masked_img = image.copy()

        # Convert the image to grayscale (either masked or full)
        gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # Dimensions: prefer provided template dimensions; only fall back to QR if not provided.
        width = height = unit = None
        qr_points = None
        decoded_info = None

        if template_dimensions is not None:
            width, height, unit = template_dimensions
        else:
            # Find the QR code only when no manual template dimensions are supplied.
            stage = "QR_READ"
            retval, decoded_info, qr_points, _readby = three_pronged_qr(gray_img)

            # Extract QR elements
            if not retval:
                print(f"[leaf_morpho] WARN '{input_image}': QR code not found/readable; continuing without QR rotation")
                warnings.append(f"{stage}: QR not found/readable")
                scale_failure = f"{stage}: QR not found/readable"
                decoded_info = None
                qr_points = None
            elif isinstance(decoded_info, tuple):
                decoded_info = decoded_info[0] if decoded_info else None

            if decoded_info is not None:
                if not isinstance(decoded_info, str) or not decoded_info:
                    print(f"[leaf_morpho] WARN '{input_image}': QR decode empty/invalid; continuing without QR rotation")
                    scale_failure = "QR_READ: QR decode empty/invalid"
                    warnings.append(scale_failure)
                    qr_points = None
                else:
                    qr_parse_ok = True
                    # Use regex to extract all text before the first 'x'
                    match1 = re.match(r'^(\d+\.?\d*)(?=[a-zA-Z])', decoded_info)
                    if match1:
                        width = float(match1.group())
                    else:
                        scale_failure = "QR_PARSE: failed to parse width"
                        warnings.append(scale_failure)
                        qr_parse_ok = False

                    # run the following regex on decoded_info (?<=x)(\d+\.?\d*)(?=[a-zA-Z])
                    match2 = re.search(r'(?<=x)(\d+\.?\d*)(?=[a-zA-Z])', decoded_info)
                    if match2:
                        height = float(match2.group())
                    else:
                        scale_failure = "QR_PARSE: failed to parse height"
                        warnings.append(scale_failure)
                        qr_parse_ok = False

                    # Find units encoded in the QR payload (in or cm). This matters for px/cm scaling.
                    unit_m = re.search(r'(in|cm)', decoded_info)
                    unit = unit_m.group(1) if unit_m else None

                    if not qr_parse_ok:
                        qr_points = None
            else:
                qr_points = None
                if scale_failure is None:
                    scale_failure = "QR_READ: QR not found/readable"

        # Find four markers with local RF-DETR.
        stage = "MARKERS_PREDICT"
        coordinates_array, marker_boxes = detect_marker_geometry(masked_img)

        # If markers are missing, skip (not a valid frame for measurements)
        if coordinates_array.size == 0 or len(coordinates_array) < 4:
            return _fail("MARKERS", "markers not detected (need 4)")

        whitefilled_img = white_out_marker_boxes(masked_img, marker_boxes)

        # Assign points
        center_x = int(np.mean(coordinates_array[:, 0]))
        center_y = int(np.mean(coordinates_array[:, 1]))
        centroid = (center_x, center_y)

        # Estimate rotation. If QR is unavailable (manual template size), skip rotation.
        qr_center = None
        if qr_points is not None:
            x1 = np.mean(qr_points[:, 0]).astype(np.int64)
            y1 = np.mean(qr_points[:, 1]).astype(np.int64)
            qr_center = (x1, y1)

        if qr_center is not None:
            delta_x = centroid[1] - qr_center[1]
            delta_y = qr_center[0] - centroid[0]
            angle_radians = np.arctan2(delta_y, delta_x)
            degrees = np.rad2deg(angle_radians)
        else:
            degrees = 0.0

        # Get rotation matrix
        mat = cv2.getRotationMatrix2D(centroid, degrees, scale=1)

        # Rotate the image
        rotated_img = cv2.warpAffine(whitefilled_img, mat, (whitefilled_img.shape[1], whitefilled_img.shape[0]))

        # Rotate the previously detected marker coordinates instead of rerunning inference
        rotated_coords = apply_affine_transform(coordinates_array, mat)
        if rotated_coords.size == 0:
            return _fail("MARKERS", "markers not detected post-rotation")
        coordinates_array = np.rint(rotated_coords).astype(np.int32)

        # Convert the image to RGB format
        cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)

        # Find best corner points from rotated image
        ordered_corner_points = find_best_corner_points(coordinates_array)

        stage = "TARGET_BOX"
        target_box = perspective_transform(rotated_img, ordered_corner_points)
        if target_box is None or target_box.size == 0:
            return _fail(stage, "failed to compute target box")

        if output_dir is not False:
            # Save the expanded crop to the output directory
            output_path = os.path.join(output_dir, f"{file_name}_target_box.jpg")
            cv2.imwrite(output_path, target_box)

        if output_mode == "target-boxes":
            return _ok(measurement_na_row(file_name, "OUTPUT_MODE: target-boxes only"), warnings)

        stage = "LEAF_MASK"
        binary_mask = create_leaf_mask(
            target_box,
            mask_method,
            threshold_value,
        )
        if binary_mask is None:
            return _fail(stage, "leaf not detected")

        # Save ONLY the refined (most accurate) mask.
        # NOTE: This replaces older outputs:
        # - *_binary_mask.jpg
        # - *_masked.jpg
        # - *_accurately_masked_leaf.jpg
        if output_dir is not False:
            output_path = os.path.join(output_dir, f"{file_name}_mask.png")
            cv2.imwrite(output_path, binary_mask)
            if save_measurement_axes:
                save_measurement_axes_image(output_dir, file_name, target_box, binary_mask)

        px_per_cm_mean = px_per_cm_from_target_box(target_box, width, height, unit, "average")
        px_per_cm_width = px_per_cm_from_target_box(target_box, width, height, unit, "width")
        px_per_cm_height = px_per_cm_from_target_box(target_box, width, height, unit, "height")
        if px_per_cm_mean is None:
            result_row = measurement_na_row(
                file_name,
                scale_failure or "SCALE: physical dimensions unavailable",
            )
        else:
            result_row = measurement_row_from_mask(
                file_name,
                binary_mask,
                px_per_cm_mean,
                px_per_cm_width,
                px_per_cm_height,
            )

        return _ok(result_row, warnings)

    except Exception as e:
        print(f"[leaf_morpho] ERROR '{input_image}' at {stage}: {e}")
        return _fail(stage, f"unexpected error: {e}")


def _process_batch_image(
    input_image,
    output_dir,
    template_dimensions,
    output_mode,
    mask_method,
    threshold_value,
    scale_axis,
    save_measurement_axes,
):
    result = leaf_morpho(
        input_image,
        output_dir,
        template_dimensions,
        None,
        None,
        output_mode,
        mask_method,
        threshold_value,
        scale_axis,
        save_measurement_axes,
    )
    return input_image, result, None


def run_leaf_morpho_batch(
    input_images,
    output_dir,
    results_path,
    template_dimensions=None,
    output_mode="masks",
    mask_method="birefnet",
    threshold_value=None,
    scale_axis="average",
    workers=None,
    progress_callback=None,
    write_failures=False,
    compact_csv=True,
    save_measurement_axes=False,
    csv_update_interval=50,
    serialize_model_inference=True,
):
    """Run the leaf morphometrics pipeline with per-image error isolation.

    ``serialize_model_inference`` forces single-worker execution whenever a
    model-backed path is involved (BiRefNet, or any run that is not purely
    target-box extraction). This is the safe default for the Streamlit app,
    which shares one in-process model across requests. The CLI passes
    ``False`` so an explicit ``--workers`` value is honored on the command line.
    """
    input_images = list(input_images or [])
    if output_dir is not False:
        os.makedirs(output_dir, exist_ok=True)

    if workers is None:
        workers, worker_reason = default_worker_count(input_images, output_mode, mask_method)
    else:
        worker_reason = "user override"
    workers = max(1, int(workers))
    all_target_boxes = bool(input_images) and all(is_target_box_image(p) for p in input_images)
    if serialize_model_inference and workers > 1 and (mask_method == "birefnet" or not all_target_boxes):
        workers = 1
        worker_reason = f"{worker_reason}; serialized for model-backed UI inference"

    succeeded = 0
    failed = 0
    processed = 0
    failure_rows = []
    result_rows = []

    def _record_result(input_image, result, exception=None):
        nonlocal succeeded, failed, processed
        if exception is not None:
            failed += 1
            failure_rows.append(failure_report_row(input_image, exception=exception))
            sample_id = os.path.splitext(os.path.basename(input_image))[0]
            result_rows.append(measurement_na_row(sample_id, f"EXCEPTION: {exception}"))
        elif result and result.get("result_row"):
            result_rows.append(result["result_row"])
            if result.get("status") == "ok":
                succeeded += 1
                failure_rows.extend(warning_report_rows(input_image, result))
            else:
                failed += 1
                failure_rows.append(failure_report_row(input_image, result=result))
                failure_rows.extend(warning_report_rows(input_image, result))
        else:
            failed += 1
            failure_rows.append(failure_report_row(
                input_image,
                result={"status": "UNKNOWN: no result returned"},
            ))
            sample_id = os.path.splitext(os.path.basename(input_image))[0]
            result_rows.append(measurement_na_row(sample_id, "UNKNOWN: no result returned"))

        processed += 1
        if results_path and processed % csv_update_interval == 0:
            if compact_csv:
                write_compact_results_csv(result_rows, results_path)
            else:
                write_results_csv(result_rows, results_path)
        if progress_callback is not None:
            progress_callback({
                "processed": processed,
                "total": len(input_images),
                "succeeded": succeeded,
                "failed": failed,
                "current_image": input_image,
            })

    if workers == 1:
        for input_image in input_images:
            try:
                _, result, _ = _process_batch_image(
                    input_image,
                    output_dir,
                    template_dimensions,
                    output_mode,
                    mask_method,
                    threshold_value,
                    scale_axis,
                    save_measurement_axes,
                )
                _record_result(input_image, result)
            except Exception as exc:
                _record_result(input_image, None, exc)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_batch_image,
                    input_image,
                    output_dir,
                    template_dimensions,
                    output_mode,
                    mask_method,
                    threshold_value,
                    scale_axis,
                    save_measurement_axes,
                ): input_image
                for input_image in input_images
            }

            for future in concurrent.futures.as_completed(futures):
                input_image = futures[future]
                try:
                    _, result, _ = future.result()
                    _record_result(input_image, result)
                except Exception as exc:
                    _record_result(input_image, None, exc)

    if results_path:
        if compact_csv:
            write_compact_results_csv(result_rows, results_path)
        else:
            write_results_csv(result_rows, results_path)

    failure_report_path = None
    if write_failures:
        failure_report_path = write_failure_report(failure_rows, output_dir)

    return {
        "succeeded": succeeded,
        "failed": failed,
        "processed": processed,
        "total": len(input_images),
        "results_path": results_path,
        "failure_report_path": failure_report_path,
        "failure_rows": failure_rows,
        "result_rows": result_rows,
        "compact_rows": [compact_measurement_row(row) for row in result_rows],
        "workers": workers,
        "worker_reason": worker_reason,
    }
