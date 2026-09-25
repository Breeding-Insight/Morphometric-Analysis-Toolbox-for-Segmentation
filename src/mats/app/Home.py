import csv
import tempfile
import traceback
import uuid
import zipfile
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from mats.app import branding
from mats.app.compute import (
    birefnet_parallel_unlocked,
    break_glass_unlocked,
    get_compute_settings,
    reengage_gpu,
    resolve_execution_plan,
)
from mats.app.folder_picker import FolderPickerError, choose_folder
from mats.app.runtime_paths import current_python, display_path
from mats.qr_runtime import qr_preflight_status, qr_runtime_status
from mats.scaling import DEFAULT_RESULTS_UNIT, QR_TRACE_FIELDNAMES, RESULT_UNITS
from mats.mask_settings import (
    CLEAN_MARGIN_DEFAULT,
    CLEAN_MARGIN_MAX,
    CLEAN_SIZE_DEFAULT,
    CLEAN_SIZE_MAX,
    STRAY_GAP_DEFAULT,
    STRAY_GAP_MAX,
)
from mats.template_layout import (
    TemplateLayoutError,
    build_template_layout,
    format_measurement,
    maximum_template_edge,
    minimum_template_edge,
    round_to_increment,
)
from mats.thresholds import (
    CUSTOM_THRESHOLD_LEVEL,
    THRESHOLD_LEVEL_OPTIONS,
    THRESHOLD_LEVELS,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    threshold_value_for,
)


APP_DIR = Path(__file__).resolve().parent
# The app ships inside the installed package, so default the folder pickers to
# the user's home rather than the (possibly read-only) install directory.
DEFAULT_INPUT_DIR = Path.home()
DEFAULT_OUTPUT_DIR = Path.home() / "mats_outputs"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
INPUT_DIR_KEY = "input_directory"
OUTPUT_DIR_KEY = "output_directory"
FOLDER_PICKER_ERROR_KEY = "folder_picker_error"
INPUT_SOURCE_KEY = "input_source"
SEGMENT_THRESHOLD_KEY = "segment_threshold"
SEGMENT_BIREFNET_KEY = "segment_birefnet"
# Measurement methods in output order, with their Setup checkbox keys.
SEGMENTATION_METHOD_KEYS = {"threshold": SEGMENT_THRESHOLD_KEY, "birefnet": SEGMENT_BIREFNET_KEY}
SEGMENTATION_METHOD_LABELS = {"threshold": "Classic thresholding", "birefnet": "BiRefNet"}
RESULTS_METHOD_KEY = "results_method"
THRESHOLD_LEVEL_KEY = "threshold_level"
THRESHOLD_CUSTOM_VALUE_KEY = "threshold_custom_value"
THRESHOLD_MARK_LABELS = {"low": "low", "medium": "med", "high": "high"}
RESULTS_SCHEMA_KEY = "results_schema"
RESULTS_UNIT_KEY = "results_unit"
MEASURE_PRE_CLEANUP_KEY = "measure_pre_cleanup"
STRAY_GAP_KEY = "stray_gap"
CLEAN_MARGIN_KEY = "clean_margin"
CLEAN_SIZE_KEY = "clean_size"
WRITE_FAILURES_KEY = "write_failures"
EXPORT_TARGET_BOXES_KEY = "export_target_boxes"
EXPORT_MASKS_KEY = "export_cleaned_masks"
EXPORT_PRE_CLEANUP_KEY = "export_pre_cleanup"
EXPORT_OVERLAY_KEY = "export_overlay"
EXPORT_CUTOUT_KEY = "export_cutout"
EXPORT_AXES_KEY = "export_axes"
WORKSPACE_TAB_KEY = "analysis_workspace_tab"
PENDING_WORKSPACE_TAB_KEY = "pending_analysis_workspace_tab"
EXPORT_RUN_ID_KEY = "export_run_id"
EXPORT_METHODS_KEY = "export_selected_methods"
EXPORT_ZIP_NAME_KEY = "export_zip_name"
EXPORT_FILE_TYPES = (
    ("results_csv", "Results CSVs"),
    ("results_metadata", "Measurement metadata"),
    ("failure_log", "Failure logs"),
    ("target_box", "Target boxes"),
    ("mask", "Cleaned masks"),
    ("pre_cleanup", "Pre-cleanup masks"),
    ("overlay", "Overlays"),
    ("cutout", "Cutouts"),
    ("axes", "Measurement axes"),
)
RESULTS_DASHBOARD_MAX_ROWS = 5_000
RESULTS_TABLE_MAX_ROWS = 1_000
RESULTS_UNIT_LABELS = {
    "mm": "Millimeters",
    "cm": "Centimeters",
    "in": "Inches",
}
RESULTS_UNIT_SYMBOLS = {
    "mm": "mm",
    "cm": "cm",
    "in": "in",
}

_WORKBENCH_STYLES = """
<style>
.st-key-workspace_navigation_intro {
    border-left: 0.3rem solid #6F9878;
    margin-top: 0.75rem;
    padding: 0.15rem 0 0.15rem 0.85rem;
}
.st-key-workspace_navigation_intro [data-testid="stMarkdownContainer"] p {
    font-size: 1.05rem;
    letter-spacing: 0.045em;
}
/* Streamlit 1.60+ renders tabs with React Aria rather than BaseWeb. */
.st-key-analysis_workspace_tab [role="tablist"] {
    backdrop-filter: blur(0.6rem);
    background: rgba(224, 236, 225, 0.97);
    border: 1px solid #B4C7B8;
    border-radius: 0.85rem;
    box-shadow: 0 0.4rem 1.25rem rgba(45, 73, 54, 0.12);
    gap: 0.5rem;
    padding: 0.5rem;
    position: sticky;
    top: 3.75rem;
    z-index: 900;
}
.st-key-analysis_workspace_tab [data-testid="stTab"] {
    background: #F8FBF8;
    border: 1px solid #B8CABD;
    border-radius: 0.6rem;
    box-shadow: 0 0.14rem 0.3rem rgba(45, 73, 54, 0.10);
    color: #365444;
    flex: 1 1 0;
    font-weight: 650;
    justify-content: center;
    letter-spacing: 0.04em;
    min-height: 3.6rem;
    padding: 0.55rem 1rem;
    text-transform: uppercase;
    transition: background-color 140ms ease, border-color 140ms ease,
        box-shadow 140ms ease, color 140ms ease, transform 140ms ease;
}
.st-key-analysis_workspace_tab [data-testid="stTab"] [data-testid="stMarkdownContainer"] * {
    color: inherit !important;
    font-weight: inherit !important;
}
.st-key-analysis_workspace_tab [data-testid="stTab"]:hover {
    background: #EDF5EE;
    border-color: #78A284;
    box-shadow: 0 0.28rem 0.65rem rgba(45, 73, 54, 0.15);
    color: #214C3B;
    transform: translateY(-1px);
}
.st-key-analysis_workspace_tab [data-testid="stTab"][aria-selected="true"] {
    background: #1F6F5B;
    border-color: #155344;
    box-shadow: 0 0.35rem 0.8rem rgba(31, 111, 91, 0.28);
    color: #FFFFFF;
    transform: translateY(-1px);
}
.st-key-analysis_workspace_tab [data-testid="stTab"][aria-selected="true"]:hover {
    background: #195A4A;
    border-color: #12483B;
    color: #FFFFFF;
}
.st-key-analysis_workspace_tab [data-testid="stTab"]:focus-visible {
    outline: 0.2rem solid #8DB99A;
    outline-offset: 0.15rem;
}
.st-key-analysis_workspace_tab .react-aria-SelectionIndicator,
.st-key-analysis_workspace_tab [role="tablist"]::after {
    display: none;
}
@media (max-width: 900px) {
    .st-key-analysis_workspace_tab [data-testid="stTab"] {
        min-height: 4rem;
        padding: 0.5rem;
    }
}
@media (max-width: 560px) {
    .st-key-analysis_workspace_tab [role="tablist"] {
        gap: 0.25rem;
        padding: 0.3rem;
    }
    .st-key-analysis_workspace_tab [data-testid="stTab"] {
        letter-spacing: 0.01em;
        min-height: 3.4rem;
        padding: 0.35rem 0.2rem;
    }
}
.st-key-launch_analysis {
    background: linear-gradient(135deg, #F5FAF5 0%, #E4F1E7 100%);
    border: 2px solid #9FBEA8;
    border-left: 0.55rem solid #1F6F5B;
    border-radius: 1rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9),
        0 0.65rem 1.6rem rgba(31, 111, 91, 0.14);
    margin-bottom: 1.5rem;
    margin-top: 0.75rem;
    padding: 1.1rem 1.2rem 1.35rem;
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
    border-radius: 0.85rem !important;
    font-size: 1.12rem !important;
    font-weight: 750 !important;
    letter-spacing: 0.06em !important;
    min-height: 4.25rem !important;
    position: relative;
    text-transform: uppercase;
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled) {
    background: linear-gradient(135deg, #1F6F5B 0%, #2C9475 58%, #43B28A 100%) !important;
    border: 2px solid #155344 !important;
    box-shadow: 0 0.42rem 0 #0E4034,
        0 0.82rem 1.35rem rgba(31, 111, 91, 0.28) !important;
    color: #FFFFFF !important;
    text-shadow: 0 1px 0 rgba(0, 0, 0, 0.18);
    transform: translateY(-0.04rem);
    transition: background 130ms ease, box-shadow 130ms ease, transform 130ms ease;
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled) * {
    color: #FFFFFF !important;
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled):hover {
    background: linear-gradient(135deg, #277F69 0%, #37A784 58%, #55C99E 100%) !important;
    box-shadow: 0 0.52rem 0 #0E4034,
        0 1.05rem 1.6rem rgba(31, 111, 91, 0.32) !important;
    transform: translateY(-0.14rem) scale(1.008);
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled):active {
    background: #195A4A !important;
    box-shadow: 0 0.10rem 0 #0E4034,
        0 0.28rem 0.55rem rgba(31, 111, 91, 0.20) !important;
    transform: translateY(0.32rem) scale(0.996);
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:disabled {
    background: linear-gradient(180deg, #E8F0EA 0%, #C9D9CD 100%) !important;
    border: 2px solid #9AB0A0 !important;
    box-shadow: 0 0.28rem 0 #789181,
        0 0.58rem 1rem rgba(45, 73, 54, 0.14) !important;
    color: #526A5A !important;
    opacity: 0.78 !important;
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:disabled * {
    color: #526A5A !important;
}
.st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:focus-visible {
    outline: 0.24rem solid #78A284 !important;
    outline-offset: 0.22rem !important;
}
@media (prefers-reduced-motion: reduce) {
    .st-key-launch_analysis [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:not(:disabled) {
        transition: none;
    }
}
/* Preset cutoffs marked beneath the custom-threshold slider. */
.mats-threshold-marks {
    color: #5F6C66;
    container-type: inline-size;
    font-size: 0.72rem;
    line-height: 1.15;
    margin: -0.4rem 0.5rem 0;
}
.mats-threshold-track {
    height: 2.1rem;
    position: relative;
}
.mats-threshold-mark {
    position: absolute;
    text-align: center;
    transform: translateX(-50%);
    white-space: nowrap;
}
.mats-threshold-mark::before {
    background: #9AB0A0;
    content: "";
    display: block;
    height: 0.35rem;
    margin: 0 auto 0.1rem;
    width: 1px;
}
/* In a narrow column the presets sit ~17 px apart: drop "med" to a second row. */
@container (max-width: 300px) {
    .mats-threshold-track {
        height: 3.5rem;
    }
    .mats-threshold-mark-medium::before {
        height: 1.75rem;
    }
}
</style>
"""


@st.cache_resource
def load_pipeline_module():
    """Import the pipeline library (mats.core), cached for the session.

    Kept as a function so a missing heavy dependency (torch, rfdetr, ...) surfaces
    as a friendly Streamlit message instead of a hard startup crash.
    """
    from mats import core
    return core


def collect_folder_images(folder_path):
    folder = Path(folder_path).expanduser()
    if not folder.is_dir():
        return []
    return [
        str(path)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]


def save_uploaded_images(uploaded_files, destination):
    destination.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in VALID_EXTENSIONS:
            continue
        safe_name = Path(uploaded_file.name).name
        output_path = destination / safe_name
        output_path.write_bytes(uploaded_file.getbuffer())
        image_paths.append(str(output_path))
    return image_paths


# Guardrails so a few-hundred-image run cannot exhaust memory or overwhelm the page.
ZIP_SIZE_WARN_BYTES = 2 * 1024 ** 3  # 2 GB
PREVIEW_IMAGE_WIDTH = 280
LARGE_BATCH_THRESHOLD = 200
OVERLAY_TINT_COLOR = (255, 0, 255)  # BGR magenta -- reads clearly against green foliage
OVERLAY_TINT_ALPHA = 0.4


def build_overlay_image(target_box, binary_mask, color=OVERLAY_TINT_COLOR, alpha=OVERLAY_TINT_ALPHA):
    """Blend a translucent tint + contour outline over the segmented region, for QC review."""
    mask_bool = binary_mask.astype(bool)
    tint = np.full_like(target_box, color, dtype=target_box.dtype)
    blended = cv2.addWeighted(target_box, 1.0 - alpha, tint, alpha, 0)
    overlay = target_box.copy()
    overlay[mask_bool] = blended[mask_bool]
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, min(6, min(target_box.shape[:2]) // 300))
    cv2.drawContours(overlay, contours, -1, color, thickness, cv2.LINE_AA)
    return overlay


def build_cutout_image(target_box, binary_mask):
    """Isolate the segmented leaf pixels; everything outside the mask is black."""
    return cv2.bitwise_and(target_box, target_box, mask=binary_mask)


def generate_export_overlays(output_dir, include_overlay, include_cutout):
    """Materialize {sample}_overlay.jpg / {sample}_cutout.jpg for every mask+target-box pair.

    Always regenerates: output_dir can be reused across runs with a different mask
    method or source images, so a stale derived file from a prior run must not be
    served instead of one matching the current mask.
    """
    if not include_overlay and not include_cutout:
        return
    output_dir = Path(output_dir)
    for mask_path in sorted(output_dir.glob("*_mask.png")):
        sample_id = mask_path.name[: -len("_mask.png")]
        target_box_path = output_dir / f"{sample_id}_target_box.jpg"
        if not target_box_path.is_file():
            continue

        binary_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        target_box = cv2.imread(str(target_box_path), cv2.IMREAD_COLOR)
        if binary_mask is None or target_box is None:
            continue
        if binary_mask.shape[:2] != target_box.shape[:2]:
            continue

        if include_overlay:
            cv2.imwrite(
                str(output_dir / f"{sample_id}_overlay.jpg"),
                build_overlay_image(target_box, binary_mask),
            )
        if include_cutout:
            cv2.imwrite(
                str(output_dir / f"{sample_id}_cutout.jpg"),
                build_cutout_image(target_box, binary_mask),
            )


def gather_output_files(output_dir, results_path, include_overlay=False, include_cutout=False):
    """Return the artifact files an export ZIP should contain."""
    output_dir = Path(output_dir)
    files = []
    results_path = Path(results_path)
    if results_path.is_file():
        files.append(results_path)
    failures_path = output_dir / "leaf_morpho_failures.csv"
    if failures_path.is_file():
        files.append(failures_path)
    files.extend(sorted(output_dir.glob("*_target_box.jpg")))
    files.extend(sorted(output_dir.glob("*_mask.png")))
    if include_overlay:
        files.extend(sorted(output_dir.glob("*_overlay.jpg")))
    if include_cutout:
        files.extend(sorted(output_dir.glob("*_cutout.jpg")))
    return files


def files_from_manifest(artifacts):
    """Select only paths this run successfully wrote, preserving run order."""
    return [Path(item["path"]) for item in artifacts if Path(item["path"]).is_file()]


def pairs_from_manifest(
    artifacts, input_images=(), method=None, measurement_source="cleaned",
    preview_artifacts=(),
):
    """Construct previews from this run's files and accessible pre-cropped inputs.

    Use the mask that produced the measurement; private preview files fill in
    for optional exports. Target boxes are shared by every method. A pre-cleanup
    run measures its raw mask minus stray pieces, which only the preview mask
    holds; its pre-cleanup export is the raw mask.
    """
    if measurement_source not in {"cleaned", "pre-cleanup"}:
        raise ValueError("unknown measurement source")
    by_sample = {}
    raw_kinds = {"preview_raw_mask", "pre_cleanup"}
    mask_kinds = (
        {"preview_mask"} if measurement_source == "pre-cleanup" else {"mask", "preview_mask"}
    )
    # The private PNG target box preserves the exact pixels used for thresholding.
    for item in (*artifacts, *preview_artifacts):
        sample_id = item.get("sample_id")
        kind = item["kind"]
        if sample_id is None or kind not in {
            "target_box", "preview_target_box", *mask_kinds, *raw_kinds,
        }:
            continue
        if method is not None and kind in {
            *mask_kinds, *raw_kinds,
        } and item.get("method") != method:
            continue
        by_sample.setdefault(sample_id, {"sample_id": sample_id, "target_box": None, "mask": None})
        if kind in raw_kinds:
            by_sample[sample_id]["raw_mask"] = item["path"]
        elif kind in mask_kinds:
            by_sample[sample_id]["mask"] = item["path"]
        elif kind in {"target_box", "preview_target_box"}:
            by_sample[sample_id]["target_box"] = item["path"]
    for path in input_images:
        if path.endswith("_target_box.jpg") and Path(path).is_file():
            sample_id = Path(path).stem[:-len("_target_box")]
            if sample_id in by_sample and by_sample[sample_id]["target_box"] is None:
                by_sample[sample_id]["target_box"] = path
    if measurement_source == "pre-cleanup":
        for pair in by_sample.values():
            pair["mask_source"] = measurement_source
    return list(by_sample.values())


def estimate_zip_inputs(files):
    total_bytes = 0
    for path in files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    return len(files), total_bytes


def write_output_zip(files, dest_path):
    """Stream the artifacts into a ZIP on disk (avoids holding it all in RAM)."""
    dest_path = Path(dest_path)
    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=Path(path).name)
    return dest_path


def human_bytes(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def count_csv_rows(results_path):
    with open(results_path, newline="") as csvfile:
        return sum(1 for _ in csv.DictReader(csvfile))


@st.cache_data(max_entries=4, show_spinner=False)
def read_results_dataframe(results_path, modified_ns, limit):
    """Load a bounded, cacheable slice of the CSV for the results dashboard."""
    del modified_ns  # The timestamp is intentionally part of the cache key.
    return pd.read_csv(results_path, nrows=limit)


def _scale_axes_by_sample(summary):
    """Retain canonical calibration for per-specimen output adjustments."""
    axes_by_method = {}
    for method, outcome in summary["by_method"].items():
        axes_by_sample = {}
        for row in outcome["result_rows"]:
            try:
                axes = (
                    float(row["px_per_cm_width"]),
                    float(row["px_per_cm_height"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if all(np.isfinite(value) and value > 0 for value in axes):
                axes_by_sample[str(row["sample_id"])] = axes
        axes_by_method[method] = axes_by_sample
    return axes_by_method


def normalize_measurements(frame, results_unit=DEFAULT_RESULTS_UNIT):
    """Return numeric, consistently named measurements from either CSV schema."""
    if results_unit not in RESULT_UNITS:
        results_unit = DEFAULT_RESULTS_UNIT
    full_area = f"leaf_area_{results_unit}2"
    compact_area = f"area_{results_unit}2"
    width = f"width_{results_unit}"
    length = f"length_{results_unit}"
    measurements = frame.copy()
    if full_area not in measurements and compact_area in measurements:
        measurements[full_area] = measurements[compact_area]

    required_columns = ("sample_id", full_area, width, length)
    if any(column not in measurements for column in required_columns):
        return pd.DataFrame(columns=("sample_id", "leaf_area", "width", "length"))

    measurements = measurements.rename(columns={
        full_area: "leaf_area",
        width: "width",
        length: "length",
    })
    for column in ("leaf_area", "width", "length", "scale_aspect_ratio"):
        if column in measurements:
            measurements[column] = pd.to_numeric(measurements[column], errors="coerce")
    return measurements.dropna(subset=("leaf_area", "width", "length"))


def _unit_symbols(run):
    """Return the length and area unit labels for a run's results."""
    results_unit = run.get("results_unit", DEFAULT_RESULTS_UNIT)
    if results_unit not in RESULT_UNITS:
        results_unit = DEFAULT_RESULTS_UNIT
    unit_symbol = RESULTS_UNIT_SYMBOLS[results_unit]
    return unit_symbol, f"{unit_symbol}²"


def _measurement_columns(unit_symbol="cm", area_symbol="cm²"):
    """(column, label, decimals) shared by the Analyze and Adjust tables; None is text."""
    return [
        ("sample_id", "Sample", None),
        ("leaf_area", f"Leaf area ({area_symbol})", 2),
        ("width", f"Leaf width ({unit_symbol})", 2),
        ("length", f"Leaf length ({unit_symbol})", 2),
        ("scale_aspect_ratio", "Scale axis ratio", 3),
    ]


def _measurement_column_config(unit_symbol="cm", area_symbol="cm²"):
    """Streamlit column config for the Analyze measurement table."""
    return {
        column: (
            st.column_config.TextColumn(label, pinned=True) if decimals is None
            else st.column_config.NumberColumn(label, format=f"%.{decimals}f")
        )
        for column, label, decimals in _measurement_columns(unit_symbol, area_symbol)
    }


def summarize_measurements(measurements):
    """Return dashboard-ready robust summary statistics for leaf measurements."""
    if measurements.empty:
        return None
    return {
        "count": len(measurements),
        "median_area": measurements["leaf_area"].median(),
        "median_width": measurements["width"].median(),
        "median_length": measurements["length"].median(),
    }


def collect_output_pairs(output_dir, sample_ids):
    """Return output pairs for the explicitly supplied sample IDs only."""
    output_dir = Path(output_dir)
    pairs = []
    for sample_id in sorted(set(sample_ids)):
        mask_path = output_dir / f"{sample_id}_mask.png"
        if not mask_path.is_file():
            continue
        target_box = output_dir / f"{sample_id}_target_box.jpg"
        pairs.append({
            "sample_id": sample_id,
            "target_box": str(target_box) if target_box.is_file() else None,
            "mask": str(mask_path),
        })
    return pairs


def merge_viewer_pairs(existing_pairs, new_pairs):
    """Accumulate viewer artifacts within a session without duplicating paths."""
    merged = list(existing_pairs)
    indexes_by_mask = {
        pair["mask"]: index
        for index, pair in enumerate(merged)
    }
    for pair in new_pairs:
        existing_index = indexes_by_mask.get(pair["mask"])
        if existing_index is None:
            indexes_by_mask[pair["mask"]] = len(merged)
            merged.append(pair)
        else:
            merged[existing_index] = pair
    return merged


def find_output_pair(pairs, sample_id):
    """Find this session's output pair for a selected measurement sample."""
    return next(
        (pair for pair in pairs if str(pair["sample_id"]) == str(sample_id)),
        None,
    )


QR_MODE_KEY = "measure_use_qr"
UNIT_KEY = "measure_unit"
PREVIOUS_UNIT_KEY = "measure_previous_unit"
WIDTH_KEY = "measure_width"
HEIGHT_KEY = "measure_height"
DIMENSIONS_TEXT_KEY = "measure_dimensions_text"
MANUAL_TEXT_CACHE_KEY = "measure_manual_dimensions_cache"
USE_LEGACY_DIMENSIONS_KEY = "measure_use_legacy_calibration"
LEGACY_DIMENSIONS_TEXT_KEY = "measure_legacy_dimensions_text"


def _compose_dimensions_text():
    unit = st.session_state[UNIT_KEY]
    return (
        f"{format_measurement(st.session_state[WIDTH_KEY])}"
        f"x{format_measurement(st.session_state[HEIGHT_KEY])}{unit}"
    )


def _measure_dimensions_changed():
    """Keep the Width/Height selectors on the 0.5 grid and mirror them into the bar."""
    for key in (WIDTH_KEY, HEIGHT_KEY):
        st.session_state[key] = round_to_increment(st.session_state[key])
    st.session_state[DIMENSIONS_TEXT_KEY] = _compose_dimensions_text()


def _convert_measure_unit():
    """Convert Width/Height to the new unit, snapped to the nearest 0.5."""
    previous = st.session_state[PREVIOUS_UNIT_KEY]
    current = st.session_state[UNIT_KEY]
    if previous == current:
        return
    factor = 2.54 if previous == "in" and current == "cm" else 1.0 / 2.54
    minimum = minimum_template_edge(current)
    maximum = maximum_template_edge(current)
    for key in (WIDTH_KEY, HEIGHT_KEY):
        converted = round_to_increment(st.session_state[key] * factor)
        st.session_state[key] = min(max(converted, minimum), maximum)
    st.session_state[PREVIOUS_UNIT_KEY] = current
    st.session_state[DIMENSIONS_TEXT_KEY] = _compose_dimensions_text()


def _qr_mode_changed():
    """Swap the sheet-size bar to a QR placeholder, caching the manual entry."""
    if st.session_state[QR_MODE_KEY]:
        st.session_state[MANUAL_TEXT_CACHE_KEY] = st.session_state[DIMENSIONS_TEXT_KEY]
        st.session_state[DIMENSIONS_TEXT_KEY] = "Variable dimensions — QR-derived"
    else:
        cached = st.session_state.pop(MANUAL_TEXT_CACHE_KEY, None)
        st.session_state[DIMENSIONS_TEXT_KEY] = cached or _compose_dimensions_text()


def _resolve_sheet_layout(dimensions_text, parse_dimensions):
    """Parse a finished sheet size and derive its marker-centre calibration area."""
    sheet_dimensions = parse_dimensions(dimensions_text)
    if sheet_dimensions is None:
        return None, None, (
            'Printed sheet size must look like "12x12in" or "30x30cm".'
        )
    try:
        layout = build_template_layout(*sheet_dimensions)
    except TemplateLayoutError as exc:
        return sheet_dimensions, None, str(exc)
    width, height, _unit = sheet_dimensions
    if width != layout.page_width or height != layout.page_length:
        normalized = (
            f"{format_measurement(layout.page_width)}x"
            f"{format_measurement(layout.page_length)}{layout.unit}"
        )
        return sheet_dimensions, None, (
            "Template Creator sheet sizes use 0.5-unit increments; "
            f"use {normalized}."
        )
    return sheet_dimensions, layout, None


def _layout_conversion_label(layout):
    """Describe the user-entered sheet and its derived calibration area."""
    return (
        f"{format_measurement(layout.page_width)} × "
        f"{format_measurement(layout.page_length)} {layout.unit} printed sheet "
        "→ "
        f"{format_measurement(layout.observation_width)} × "
        f"{format_measurement(layout.observation_length)} {layout.unit} "
        "calibrated area"
    )


def _threshold_marks_html():
    """Mark the low/medium/high preset cutoffs beneath the custom-threshold slider."""
    span = THRESHOLD_MAX - THRESHOLD_MIN
    marks = "".join(
        f'<span class="mats-threshold-mark mats-threshold-mark-{name}" '
        f'style="left: {(value - THRESHOLD_MIN) / span * 100:.2f}%">'
        f"{THRESHOLD_MARK_LABELS[name]}<br>{value}</span>"
        for name, value in THRESHOLD_LEVELS.items()
        if value is not None
    )
    return (
        '<div class="mats-threshold-marks">'
        f'<div class="mats-threshold-track">{marks}</div></div>'
    )


def _choose_folder(widget_key, title):
    """Open a local desktop picker and copy the result into a path widget."""
    try:
        selected = choose_folder(title, st.session_state[widget_key])
    except FolderPickerError as exc:
        st.session_state[FOLDER_PICKER_ERROR_KEY] = (
            f"{exc}. Enter the folder path manually instead."
        )
        return

    st.session_state.pop(FOLDER_PICKER_ERROR_KEY, None)
    if selected:
        st.session_state[widget_key] = selected


def _switch_workspace_tab(tab_name):
    """Open a named workspace tab through its stateful Streamlit key."""
    st.session_state[WORKSPACE_TAB_KEY] = tab_name


def _open_export():
    """Open the Export workspace from the sidebar."""
    st.session_state[WORKSPACE_TAB_KEY] = "Export"


def apply_pending_workspace_tab():
    """Apply deferred tab navigation before the tab widget is instantiated."""
    tab_name = st.session_state.pop(PENDING_WORKSPACE_TAB_KEY, None)
    if tab_name is not None:
        st.session_state[WORKSPACE_TAB_KEY] = tab_name


def render_workspace_navigation():
    """Render the persistent primary navigation for the workbench views."""
    with st.container(key="workspace_navigation_intro"):
        st.markdown("**WORKSPACE NAVIGATION**")
        st.caption(
            "Set up, analyze, adjust specimens, then export saved results."
        )
    return st.tabs(
        ["Setup", "Analyze", "Adjust", "Export"],
        key=WORKSPACE_TAB_KEY,
        on_change="rerun",
    )


def render_workspace_context(config, execution_plan):
    """Render the compact run context beneath the Analyze workspace controls."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(
            "Analysis Workspace",
            icon=":material/science:",
            color="blue",
        )
        st.badge(
            config["segmentation_label"],
            icon=(
                ":material/auto_awesome:"
                if config["needs_birefnet"]
                else ":material/contrast:"
            ),
            color="green",
        )
        st.badge(
            execution_plan.label,
            icon=(
                ":material/bolt:"
                if execution_plan.uses_gpu
                else ":material/memory:"
            ),
            color="green" if execution_plan.uses_gpu else "orange",
        )


def render_file_sidebar():
    """Render the persistent file controls and return the active selection."""
    with st.sidebar:
        st.header("Workspace Files")
        input_source = st.radio(
            "Image source",
            ["Local folder", "Upload images"],
            key=INPUT_SOURCE_KEY,
            persist_state="page",
        )
        uploaded_files = []
        input_dir = ""
        if input_source == "Local folder":
            input_dir = st.text_input(
                "Input folder",
                key=INPUT_DIR_KEY,
                icon=":material/folder:",
                persist_state="page",
            )
            st.button(
                "Choose input folder",
                key="choose_input_folder",
                icon=":material/folder_open:",
                width="stretch",
                on_click=_choose_folder,
                args=(INPUT_DIR_KEY, "Choose the folder containing input images"),
            )
        else:
            uploaded_files = st.file_uploader(
                "Upload images",
                type=sorted(ext.lstrip(".") for ext in VALID_EXTENSIONS),
                accept_multiple_files=True,
                key="uploaded_images",
            )

        st.header("Output Destination")
        output_dir = st.text_input(
            "Output folder",
            key=OUTPUT_DIR_KEY,
            icon=":material/folder:",
            persist_state="page",
        )
        st.button(
            "Choose output folder",
            key="choose_output_folder",
            icon=":material/folder_open:",
            width="stretch",
            on_click=_choose_folder,
            args=(OUTPUT_DIR_KEY, "Choose where MATS should write its outputs"),
        )
        folder_picker_error = st.session_state.pop(FOLDER_PICKER_ERROR_KEY, None)
        if folder_picker_error:
            st.warning(folder_picker_error, icon=":material/folder_off:")

        st.caption("Choose image outputs in Setup. Download completed results in Export.")
        st.button(
            "Go to Export",
            key="open_export",
            icon=":material/download:",
            width="stretch",
            disabled=not st.session_state.get("last_run"),
            on_click=_open_export,
        )

    return input_source, uploaded_files, input_dir, output_dir


def render_analysis_settings(lm):
    """Render the run-specific settings in the wider Analyze workspace."""
    st.subheader("Analysis setup", anchor=False)
    scale_column, segmentation_column = st.columns(2, vertical_alignment="top")

    with scale_column.container(border=True):
        st.markdown("**1 · Scale**")
        st.caption("Set the physical size used to convert image pixels into measurements.")
        use_qr = st.checkbox(
            "Variable dimensions, read QR code",
            key=QR_MODE_KEY,
            on_change=_qr_mode_changed,
            help=(
                "OpenCV reads clear QR codes with no extra setup. For glare, blur, "
                "or skew, Robust QR setup adds pyzbar and QReader fallbacks; only "
                "pyzbar needs the native zbar library. Manual dimensions require no "
                "additional software."
            ),
            persist_state="page",
        )
        qr_status = qr_runtime_status()
        if use_qr and not qr_status.enhanced_available:
            st.warning(
                "Only OpenCV is available for QR reading. Clear codes may work, but "
                "glare, blur, skew, or poor contrast can leave an image without a "
                "measurement scale and write `NA` values. Robust QR setup adds "
                "pyzbar and QReader fallbacks; printed-sheet entry needs no installation.",
                icon=":material/qr_code_scanner:",
            )
            st.page_link(
                "pages/4_Robust_QR_Setup.py",
                label="Open Robust QR setup",
                icon=":material/qr_code_scanner:",
                width="content",
            )
        with st.expander("Older or custom template?", expanded=st.session_state[USE_LEGACY_DIMENSIONS_KEY]):
            use_legacy_dimensions = st.toggle(
                "Enter marker-centre calibration dimensions",
                key=USE_LEGACY_DIMENSIONS_KEY,
                disabled=use_qr,
                help=(
                    "For templates not made by the current Template Creator. "
                    "New Template Creator sheets should always use their finished "
                    "printed sheet size instead."
                ),
                persist_state="page",
            )
            if use_legacy_dimensions and not use_qr:
                st.info(
                    "Use this only when the template's marker margins are not the "
                    "current Template Creator margins.",
                    icon=":material/history:",
                )

        if use_legacy_dimensions and not use_qr:
            st.text_input(
                "Legacy calibration area",
                key=LEGACY_DIMENSIONS_TEXT_KEY,
                help=(
                    'Marker centre-to-centre size for an older/custom template, e.g. '
                    '"10.5x9.5in" or "27x24cm".'
                ),
                persist_state="page",
            )
        else:
            st.text_input(
                "Printed sheet size",
                key=DIMENSIONS_TEXT_KEY,
                disabled=use_qr,
                help=(
                    'Finished outer-sheet size in the format "<width>x<height><unit>", '
                    'for example "12x12in" or "30x30cm". MATS uses the same fixed '
                    "marker margins as Template Creator to calculate calibration automatically."
                ),
                persist_state="page",
            )
            measure_unit = st.session_state[UNIT_KEY]
            st.segmented_control(
                "Unit",
                options=["in", "cm"],
                format_func={"in": "Inches", "cm": "Centimeters"}.get,
                required=True,
                key=UNIT_KEY,
                on_change=_convert_measure_unit,
                disabled=use_qr,
                persist_state="page",
            )
            measure_minimum = minimum_template_edge(measure_unit)
            measure_maximum = maximum_template_edge(measure_unit)
            with st.container(horizontal=True):
                st.number_input(
                    "Sheet width",
                    min_value=measure_minimum,
                    max_value=measure_maximum,
                    step=0.5,
                    format="%.1f",
                    key=WIDTH_KEY,
                    disabled=use_qr,
                    icon=":material/width:",
                    help=(
                        "Finished printed-sheet width. Template Creator snaps sheet "
                        "dimensions to the nearest 0.5."
                    ),
                    on_change=_measure_dimensions_changed,
                    persist_state="page",
                )
                st.number_input(
                    "Sheet height",
                    min_value=measure_minimum,
                    max_value=measure_maximum,
                    step=0.5,
                    format="%.1f",
                    key=HEIGHT_KEY,
                    disabled=use_qr,
                    icon=":material/height:",
                    help=(
                        "Finished printed-sheet height. The larger top margin leaves "
                        "space for the QR code and header."
                    ),
                    on_change=_measure_dimensions_changed,
                    persist_state="page",
                )
            if not use_qr:
                _, sheet_layout, sheet_error = _resolve_sheet_layout(
                    st.session_state[DIMENSIONS_TEXT_KEY].strip(),
                    lm.parse_template_dimensions,
                )
                if sheet_layout is not None:
                    st.caption(_layout_conversion_label(sheet_layout))
                    st.caption(
                        "Template Creator places marker centres 1.5 in from the top "
                        "and 1 in from the other edges (3.5 cm top, 2.5 cm elsewhere)."
                    )
                elif sheet_error:
                    st.caption(f"Printed sheet size needs attention: {sheet_error}")

    with segmentation_column.container(border=True):
        st.markdown("**2 · Segmentation**")
        st.caption(
            "Choose the mask method best suited to the image background. Check both "
            "to measure every image with each method and compare them."
        )
        use_threshold = st.checkbox(
            SEGMENTATION_METHOD_LABELS["threshold"],
            key=SEGMENT_THRESHOLD_KEY,
            persist_state="session",
        )
        st.caption("Fast and dependable for clean, well-lit backgrounds.")
        # The threshold level only affects Otsu, so it is hidden without it.
        if use_threshold:
            threshold_level = st.selectbox(
                "Threshold level",
                list(THRESHOLD_LEVEL_OPTIONS),
                key=THRESHOLD_LEVEL_KEY,
                help=(
                    "auto = Otsu (adapts per image). custom = choose the cutoff yourself. "
                    "low/medium/high = fixed 100/125/150."
                ),
                persist_state="page",
            )
            if threshold_level == CUSTOM_THRESHOLD_LEVEL:
                st.slider(
                    "Custom threshold",
                    min_value=THRESHOLD_MIN,
                    max_value=THRESHOLD_MAX,
                    step=1,
                    key=THRESHOLD_CUSTOM_VALUE_KEY,
                    help=(
                        "Grayscale cutoff (0 = black, 255 = white): pixels at or below it "
                        "count as leaf. Raise it to include paler leaf tissue; lower it to "
                        "exclude shadows or a dark background."
                    ),
                    persist_state="page",
                )
                st.html(_threshold_marks_html())
        use_birefnet = st.checkbox(
            SEGMENTATION_METHOD_LABELS["birefnet"],
            key=SEGMENT_BIREFNET_KEY,
            persist_state="session",
        )
        st.caption(
            "Better on cluttered backgrounds; needs the optional local checkpoint "
            "and benefits from a GPU."
        )
        if not use_threshold and not use_birefnet:
            st.warning("Choose at least one segmentation method.", icon=":material/warning:")

    with st.container(border=True):
        st.markdown("**3 · Measurement Output**")
        st.caption("Choose the measurement mask, result units, and CSV schema.")
        measure_pre_cleanup = st.checkbox(
            "Measure from pre-cleanup masks",
            key=MEASURE_PRE_CLEANUP_KEY,
            help=(
                "Measure the raw binary segmentation before gap closing and hole "
                "filling. The edge margin is cleared, the largest object is the "
                "leaf, and pieces touching the margin or beyond the stray-piece "
                "distance are dropped. Specks near the leaf still count toward "
                "area, width, and length."
            ),
            persist_state="page",
        )
        st.caption(
            "Pre-cleanup settings for this run. Each specimen's explorer in Adjust can "
            "change them for its own preview and, for Classic thresholding, overwrite."
        )
        margin_column, gap_column, size_column = st.columns(3)
        with margin_column:
            st.number_input(
                "Edge margin (% of box)",
                min_value=0.0,
                max_value=CLEAN_MARGIN_MAX,
                step=0.25,
                format="%.2f",
                key=CLEAN_MARGIN_KEY,
                disabled=not measure_pre_cleanup,
                help=(
                    "Clears a band this percent of the target box's shorter side along "
                    "every edge, where the template's printed box outline lands. Lay "
                    "leaves inside the box so none of the leaf falls in the band; 0 "
                    "clears nothing."
                ),
                persist_state="page",
            )
        with gap_column:
            st.number_input(
                "Stray-piece distance (× leaf size)",
                min_value=0.0,
                max_value=STRAY_GAP_MAX,
                step=0.05,
                format="%.2f",
                key=STRAY_GAP_KEY,
                disabled=not measure_pre_cleanup,
                help=(
                    "Pieces farther from the leaf than this fraction of its "
                    "bounding-box diagonal are dropped; 0 keeps only the leaf. Raise "
                    "it when a leaf's parts lie apart, such as separated leaflets."
                ),
                persist_state="page",
            )
        with size_column:
            st.number_input(
                "Clean size (px)",
                min_value=0,
                max_value=CLEAN_SIZE_MAX,
                step=1,
                key=CLEAN_SIZE_KEY,
                disabled=not measure_pre_cleanup,
                help=(
                    "After the edge margin and stray pieces are cleared, removes white "
                    "specks and fills enclosed holes whose inscribed radius is below "
                    "this many pixels. The leaf is always kept and nothing is "
                    "flash-filled; 0 turns it off."
                ),
                persist_state="page",
            )
        st.segmented_control(
            "Result units",
            options=list(RESULT_UNITS),
            format_func=RESULTS_UNIT_LABELS.get,
            required=True,
            key=RESULTS_UNIT_KEY,
            help=(
                "Controls the measurements shown in Results and written to the CSV. "
                "It does not change the printed-sheet calibration dimensions."
            ),
            persist_state="page",
        )
        st.selectbox(
            "Results CSV schema",
            [
                "Full research schema (area/width/length + px-per-cm)",
                "Compact (sample_id, area, width, length)",
            ],
            key=RESULTS_SCHEMA_KEY,
            help="Full includes independent axis scales and their ratio for QC.",
            persist_state="page",
        )
        st.checkbox("Failure log · leaf_morpho_failures.csv", key=WRITE_FAILURES_KEY,
                    persist_state="page")
        st.caption(
            "The results CSV is always written. Results appear in Analyze after a run."
        )

    with st.container(border=True):
        st.markdown("**4 · Image output options**")
        st.caption("Choose image files to write during this run. QC images use the selected measurement mask.")
        st.checkbox("Target boxes · {id}_target_box.jpg", key=EXPORT_TARGET_BOXES_KEY,
                    persist_state="page")
        st.checkbox("Cleaned leaf masks · {id}_mask.png", key=EXPORT_MASKS_KEY,
                    persist_state="page")
        st.checkbox("Pre-cleanup masks · {id}_mask_precleanup_{method}.png",
                    key=EXPORT_PRE_CLEANUP_KEY, persist_state="page",
                    help="One per selected segmentation method. Pre-cleanup masks keep "
                         "holes, specks, and small objects.")
        st.checkbox("Overlays · {id}_overlay.jpg", key=EXPORT_OVERLAY_KEY, persist_state="page")
        st.checkbox("Cutouts · {id}_cutout.jpg", key=EXPORT_CUTOUT_KEY, persist_state="page")
        st.checkbox("Measurement axes · {id}_measurement_axes.jpg", key=EXPORT_AXES_KEY,
                    persist_state="page")
        if st.session_state[SEGMENT_THRESHOLD_KEY] and st.session_state[SEGMENT_BIREFNET_KEY]:
            st.caption(
                "With both methods selected, each method writes its own masks, overlays, "
                "cutouts, axes, results CSV, and failure log, with names ending in "
                "_threshold or _birefnet (for example {id}_mask_birefnet.png)."
            )


def current_analysis_config(lm):
    """Resolve the persisted controls into the values needed for one batch."""
    mask_methods = tuple(
        method for method, key in SEGMENTATION_METHOD_KEYS.items() if st.session_state[key]
    )
    if len(mask_methods) == 1:
        segmentation_label = SEGMENTATION_METHOD_LABELS[mask_methods[0]]
    elif mask_methods:
        segmentation_label = "Classic thresholding + BiRefNet"
    else:
        segmentation_label = "No segmentation method"
    threshold_level = st.session_state[THRESHOLD_LEVEL_KEY]
    use_qr = st.session_state[QR_MODE_KEY]
    use_legacy_dimensions = st.session_state[USE_LEGACY_DIMENSIONS_KEY]
    sheet_dimensions = None
    sheet_layout = None
    template_dimensions = None
    template_error = None
    if not use_qr and use_legacy_dimensions:
        template_dimensions = lm.parse_template_dimensions(
            st.session_state[LEGACY_DIMENSIONS_TEXT_KEY].strip()
        )
        if template_dimensions is None:
            template_error = (
                'Legacy calibration area must look like "10.5x9.5in" or "27x24cm".'
            )
    elif not use_qr:
        sheet_dimensions, sheet_layout, template_error = _resolve_sheet_layout(
            st.session_state[DIMENSIONS_TEXT_KEY].strip(),
            lm.parse_template_dimensions,
        )
        if sheet_layout is not None:
            template_dimensions = sheet_layout.calibration_dimensions

    return {
        "segmentation_label": segmentation_label,
        "mask_methods": mask_methods,
        "needs_birefnet": "birefnet" in mask_methods,
        "threshold_value": threshold_value_for(
            threshold_level, st.session_state[THRESHOLD_CUSTOM_VALUE_KEY]
        ),
        "compact_csv": st.session_state[RESULTS_SCHEMA_KEY].startswith("Compact"),
        "results_unit": st.session_state[RESULTS_UNIT_KEY],
        "measurement_source": (
            "pre-cleanup" if st.session_state[MEASURE_PRE_CLEANUP_KEY] else "cleaned"
        ),
        "stray_gap": float(st.session_state[STRAY_GAP_KEY]),
        "clean_margin": float(st.session_state[CLEAN_MARGIN_KEY]),
        # The clean size shapes only pre-cleanup measurements; the pipeline
        # rejects it otherwise, so a grayed-out value never reaches a run.
        "clean_size": (
            int(st.session_state[CLEAN_SIZE_KEY])
            if st.session_state[MEASURE_PRE_CLEANUP_KEY] else CLEAN_SIZE_DEFAULT
        ),
        "write_failures": st.session_state[WRITE_FAILURES_KEY],
        "export_options": {
            "target_boxes": st.session_state[EXPORT_TARGET_BOXES_KEY],
            "cleaned_masks": st.session_state[EXPORT_MASKS_KEY],
            "pre_cleanup_methods": (
                mask_methods if st.session_state[EXPORT_PRE_CLEANUP_KEY] else ()
            ),
            "overlay": st.session_state[EXPORT_OVERLAY_KEY],
            "cutout": st.session_state[EXPORT_CUTOUT_KEY],
            "axes": st.session_state[EXPORT_AXES_KEY],
        },
        "use_qr": use_qr,
        "use_legacy_dimensions": use_legacy_dimensions,
        "sheet_dimensions": sheet_dimensions,
        "sheet_layout": sheet_layout,
        "template_dimensions": template_dimensions,
        "template_error": template_error,
    }


def build_preflight_checks(
    lm,
    config,
    input_source,
    uploaded_files,
    input_dir,
    compute_settings,
    execution_plan,
    worker_risk,
    break_glass_is_unlocked,
):
    """Build the complete preflight report without rendering it."""
    checks = [
        (
            "Pipeline library",
            "success",
            False,
            f"mats.core ({display_path(Path(lm.__file__).parent)})",
            None,
        ),
    ]

    from mats import weights
    from mats.birefnet_runtime import birefnet_runtime_status
    from mats.devices import birefnet_device_report

    rfdetr_status = weights.get_weight_status("rf-detr")
    rfdetr_level = "success" if rfdetr_status.state == "ready" else "error"
    checks.append((
        "RF-DETR checkpoint",
        rfdetr_level,
        rfdetr_level == "error",
        rfdetr_status.detail,
        None,
    ))

    checks.append((
        "Segmentation method",
        "success" if config["mask_methods"] else "error",
        not config["mask_methods"],
        (
            config["segmentation_label"]
            if config["mask_methods"]
            else "No segmentation method is selected. Check Otsu, BiRefNet, or both in Setup."
        ),
        None,
    ))

    needs_birefnet = config["needs_birefnet"]
    if needs_birefnet:
        birefnet_runtime = birefnet_runtime_status()
        runtime_level = "success" if birefnet_runtime.ready else "error"
        checks.append((
            "BiRefNet runtime",
            runtime_level,
            not birefnet_runtime.ready,
            birefnet_runtime.detail,
            None,
        ))

    birefnet_status = weights.get_weight_status("birefnet")
    birefnet_level = "success" if birefnet_status.state == "ready" else "warning"
    checks.append((
        "BiRefNet checkpoint",
        birefnet_level,
        needs_birefnet and birefnet_status.state != "ready",
        (
            birefnet_status.detail
            if birefnet_status.state != "ready"
            else (
                f"Ready locally at {display_path(birefnet_status.path)}; "
                "no download is needed."
            )
        ),
        "pages/2_BiRefNet_Setup.py" if birefnet_status.state != "ready" else None,
    ))

    if worker_risk.requires_break_glass:
        checks.append((
            "Worker safety",
            "error",
            not break_glass_is_unlocked,
            (
                f"Critical load: {execution_plan.workers}/{compute_settings.available_workers} "
                f"workers ({worker_risk.utilization:.0%}). Break-glass acknowledgement "
                "is required for this run."
            ),
            None,
        ))
    else:
        checks.append((
            "Worker safety",
            "success",
            False,
            (
                f"{worker_risk.label}: {execution_plan.workers}/"
                f"{compute_settings.available_workers} workers ({worker_risk.utilization:.0%})."
            ),
            None,
        ))

    if execution_plan.execution_device == "hybrid":
        checks.append((
            "Execution mode",
            "success",
            False,
            (
                "Hybrid mode: RF-DETR on GPU with Otsu and measurements across "
                f"{execution_plan.workers} CPU workers."
            ),
            None,
        ))
    elif execution_plan.execution_device == "cpu":
        checks.append((
            "Execution mode",
            "warning",
            False,
            (
                f"CPU mode with {execution_plan.workers} worker(s); CUDA/MPS is "
                "disabled for this run."
            ),
            None,
        ))
    else:
        checks.append((
            "Execution mode",
            "success",
            False,
            "Serial GPU inference; accelerator selection is automatic.",
            None,
        ))

    if needs_birefnet and execution_plan.execution_device == "cpu":
        birefnet_cpu_detail = (
            "GPU is available but disabled by an advanced CPU-only override."
            if compute_settings.accelerator_available
            else "No usable CUDA or Apple MPS accelerator was detected."
        )
        checks.append((
            "BiRefNet performance",
            "warning",
            False,
            f"BiRefNet is using CPU. {birefnet_cpu_detail}",
            None,
        ))

    device_report = birefnet_device_report()
    checks.append((
        "BiRefNet compute",
        device_report.severity,
        needs_birefnet and device_report.severity == "error",
        device_report.detail,
        None,
    ))

    if config["use_qr"]:
        qr_status = qr_preflight_status()
        qr_backends = (qr_status.opencv, qr_status.pyzbar, qr_status.qreader)
        available_backends = [backend for backend in qr_backends if backend.available]
        template_level = (
            "success"
            if qr_status.enhanced_available
            else "warning" if available_backends else "error"
        )
        template_detail = "Variable calibration — read from each image's QR code"
        if template_level == "warning":
            template_detail += (
                " (OpenCV only; glare, blur, skew, or poor contrast can produce "
                "unreadable codes and `NA` values)"
            )
        elif template_level == "error":
            template_detail += " (no operational QR reader is available)"
        template_setup_page = (
            None if qr_status.enhanced_available else "pages/4_Robust_QR_Setup.py"
        )
        template_blocking = not available_backends
        for backend in qr_backends:
            checks.append((
                f"QR reader — {backend.name}",
                "success" if backend.available else "warning",
                False,
                backend.detail,
                None if backend.available else "pages/4_Robust_QR_Setup.py",
            ))
    elif config["template_error"] is not None:
        template_level = "error"
        template_detail = config["template_error"]
        template_setup_page = None
        template_blocking = True
    elif config["use_legacy_dimensions"]:
        width, height, unit = config["template_dimensions"]
        template_level = "warning"
        template_detail = (
            f"Legacy/custom marker-centre calibration: "
            f"{format_measurement(width)} × {format_measurement(height)} {unit}."
        )
        template_setup_page = None
        template_blocking = False
    else:
        template_level = "success"
        template_detail = _layout_conversion_label(config["sheet_layout"])
        template_setup_page = None
        template_blocking = False
    checks.append((
        "QR calibration" if config["use_qr"] else "Printed sheet size",
        template_level,
        template_blocking,
        template_detail,
        template_setup_page,
    ))

    if input_source == "Local folder":
        image_paths = collect_folder_images(input_dir)
        image_count = len(image_paths)
        checks.append((
            "Input images",
            "success" if image_paths else "error",
            not bool(image_paths),
            f"{image_count} image(s) found",
            None,
        ))
    else:
        image_paths = []
        image_count = sum(
            1
            for uploaded_file in uploaded_files
            if Path(uploaded_file.name).suffix.lower() in VALID_EXTENSIONS
        )
        checks.append((
            "Uploaded images",
            "success" if image_count else "error",
            not bool(image_count),
            f"{image_count} valid image(s) selected",
            None,
        ))

    return checks, image_paths, image_count


def render_check_messages(checks, attention_only=False):
    """Render preflight rows, optionally reducing them to blockers and warnings."""
    for label, severity, blocking, detail, setup_page in checks:
        if attention_only and severity == "success" and not blocking:
            continue
        getattr(st, severity)(f"{label}: {detail}")
        if setup_page:
            if setup_page == "pages/4_Robust_QR_Setup.py":
                setup_label = "Open Robust QR setup"
                setup_icon = ":material/qr_code_scanner:"
            else:
                setup_label = "Open BiRefNet setup"
                setup_icon = ":material/download:"
            st.page_link(
                setup_page,
                label=setup_label,
                icon=setup_icon,
                width="content",
            )


def render_preflight_summary(checks):
    """Show only the details that matter immediately before launch."""
    blockers = [check for check in checks if check[2]]
    attention = [check for check in checks if check[1] != "success" or check[2]]
    with st.container(border=True):
        st.markdown("**5 · Preflight**")
        if blockers:
            st.badge(
                f"{len(blockers)} blocking item(s)",
                icon=":material/error:",
                color="red",
            )
            st.caption(
                "Needs attention: " + ", ".join(check[0] for check in blockers) + "."
            )
            st.page_link(
                "pages/0_Diagnostics.py",
                label="Open Diagnostics",
                icon=":material/troubleshoot:",
                width="content",
            )
        elif attention:
            st.badge(
                f"Ready with {len(attention)} item(s) to review",
                icon=":material/warning:",
                color="orange",
            )
            st.caption("Review in Diagnostics: " + ", ".join(check[0] for check in attention) + ".")
        else:
            st.badge(
                "All preflight checks passed",
                icon=":material/check_circle:",
                color="green",
            )
        st.caption("Diagnostics contains the complete preflight report and compute details.")


def render_compute_status(compute_settings, execution_plan, config):
    """Render the detailed compute card in Diagnostics."""
    with st.container(border=True):
        st.subheader("Compute status", anchor=False)
        st.badge(
            execution_plan.label,
            icon=(
                ":material/check_circle:"
                if execution_plan.uses_gpu
                else ":material/memory:"
            ),
            color="green" if execution_plan.uses_gpu else "orange",
        )
        st.caption(execution_plan.detail)
        if config["needs_birefnet"] and not execution_plan.uses_gpu:
            gpu_notice = (
                "GPU available but disabled. BiRefNet is running on CPU because an "
                "advanced CPU-only override is active."
                if compute_settings.accelerator_available
                else "No usable GPU was detected. BiRefNet is running on CPU."
            )
            st.warning(
                f"{gpu_notice} This may be substantially slower and use significant system memory.",
                icon=":material/warning:",
            )
        if compute_settings.accelerator_available and compute_settings.cpu_active:
            if st.button(
                "Re-engage GPU",
                icon=":material/restart_alt:",
                width="content",
            ):
                reengage_gpu()
                st.rerun()
        st.page_link(
            "pages/3_CPU_Options.py",
            label="Open CPU options",
            icon=":material/tune:",
            width="content",
        )


def render_diagnostics(compute_settings, execution_plan, config, checks):
    st.subheader("Diagnostics", anchor=False)
    render_compute_status(compute_settings, execution_plan, config)
    with st.container(border=True):
        st.markdown("**Preflight Overview**")
        severity_colors = {"success": "green", "warning": "orange", "error": "red"}
        severity_icons = {
            "success": ":material/check_circle:",
            "warning": ":material/warning:",
            "error": ":material/error:",
        }
        for label, severity, _, detail, setup_page in checks:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(
                    label,
                    icon=severity_icons[severity],
                    color=severity_colors[severity],
                )
                st.caption(detail)
                if setup_page:
                    st.page_link(
                        setup_page,
                        label="Setup",
                        icon=":material/open_in_new:",
                        width="content",
                    )
    with st.expander("Detailed Preflight Report", icon=":material/article:"):
        render_check_messages(checks)


def render_launch_card(checks, image_count, config, execution_plan, output_path):
    """Render the prominent primary action after the completed preflight card."""
    return render_launch_surface(
        checks,
        image_count,
        config,
        execution_plan,
        output_path,
    )


def render_launch_surface(checks, image_count, config, execution_plan, output_path):
    """Render the primary run surface and return whether the user launched it."""
    ready = not any(blocking for _, _, blocking, _, _ in checks)
    blocking_count = sum(1 for _, _, blocking, _, _ in checks if blocking)
    with st.container(key="launch_analysis"):
        st.subheader("Launch analysis", anchor=False)
        launch_status = st.empty()
        st.caption(
            f"{image_count} image{'s' if image_count != 1 else ''} · "
            f"{config['segmentation_label']} · {execution_plan.label} · "
            f"Output: `{display_path(output_path)}`"
        )

        large_batch_ok = True
        if ready and image_count > LARGE_BATCH_THRESHOLD:
            st.warning(
                f"This is a large batch ({image_count} images). Processing runs "
                "synchronously and may take a long time; keep the browser tab open.",
                icon=":material/schedule:",
            )
            large_batch_ok = st.checkbox(
                f"I understand and want to process {image_count} images",
                value=False,
                key="large_batch_confirmation",
            )

        launch_enabled = ready and large_batch_ok
        if not ready:
            item_label = "item needs" if blocking_count == 1 else "items need"
            launch_status.badge(
                f"{blocking_count} preflight {item_label} attention",
                icon=":material/error:",
                color="red",
            )
            st.caption("Resolve the blocking preflight messages above to enable analysis.")
        elif not large_batch_ok:
            launch_status.badge(
                "Large-batch confirmation required",
                icon=":material/pending_actions:",
                color="orange",
            )
        else:
            launch_status.badge(
                "Ready to analyze",
                icon=":material/check_circle:",
                color="green",
            )

        return st.button(
            "Run leaf morphometrics",
            key="run_leaf_morphometrics",
            type="primary",
            icon=":material/rocket_launch:",
            disabled=not launch_enabled,
            width="stretch",
            help=(
                "Start marker detection, segmentation, and morphometric measurement "
                "for the selected images."
            ),
        )


def execute_leaf_analysis(
    lm,
    config,
    image_paths,
    uploaded_files,
    input_source,
    output_path,
    results_path,
    execution_plan,
    worker_risk,
    break_glass_is_unlocked,
    birefnet_parallel_is_unlocked,
):
    """Run the batch and return its summary plus output artifacts for this session."""
    output_path.mkdir(parents=True, exist_ok=True)
    preview_cache = tempfile.TemporaryDirectory(prefix="mats_preview_")
    if worker_risk.requires_break_glass:
        st.session_state.pop("break_glass_available_workers", None)
    if (
        config["needs_birefnet"]
        and execution_plan.execution_device == "cpu"
        and execution_plan.workers > 1
    ):
        st.session_state.pop("birefnet_parallel_available_workers", None)

    with tempfile.TemporaryDirectory(prefix="leaf_morpho_uploads_") as tmpdir:
        if input_source == "Upload images":
            image_paths = save_uploaded_images(uploaded_files, Path(tmpdir))

        progress_bar = st.progress(0)
        status_box = st.empty()
        counts_box = st.empty()

        def update_progress(status):
            total = max(status["total"], 1)
            progress_bar.progress(status["processed"] / total)
            counts_box.write(
                f"Processed {status['processed']} / {status['total']} "
                f"| succeeded {status['succeeded']} | failed {status['failed']}"
            )
            status_box.write(f"Current image: `{Path(status['current_image']).name}`")

        try:
            with st.spinner("Processing images..."):
                summary = lm.run_leaf_morpho_batch(
                    image_paths,
                    str(output_path),
                    str(results_path),
                    template_dimensions=config["template_dimensions"],
                    output_mode="masks",
                    mask_method=config["mask_methods"],
                    threshold_value=config["threshold_value"],
                    workers=int(execution_plan.workers),
                    execution_device=execution_plan.execution_device,
                    worker_safety_check=True,
                    break_glass_acknowledged=break_glass_is_unlocked,
                    birefnet_parallel_acknowledged=birefnet_parallel_is_unlocked,
                    progress_callback=update_progress,
                    write_failures=config["write_failures"],
                    compact_csv=config["compact_csv"],
                    results_unit=config["results_unit"],
                    save_measurement_axes=config["export_options"]["axes"],
                    export_options={**config["export_options"], "preview_dir": preview_cache.name},
                    measurement_source=config["measurement_source"],
                    stray_gap=config["stray_gap"],
                    clean_margin=config["clean_margin"],
                    clean_size=config["clean_size"],
                )
        except ValueError as exc:
            preview_cache.cleanup()
            st.error(f"Run prevented by worker safety checks: {exc}")
            return None, {}

    input_images = image_paths if input_source == "Local folder" else ()
    run_pairs = {
        method: pairs_from_manifest(
            summary["artifacts"], input_images, method,
            measurement_source=summary["measurement_source"],
            preview_artifacts=summary["preview_artifacts"],
        )
        for method in summary["methods"]
    }
    previous_cache = st.session_state.get("preview_cache")
    st.session_state["preview_cache"] = preview_cache
    if previous_cache is not None:
        previous_cache.cleanup()
    return summary, run_pairs


def main():
    st.set_page_config(
        page_title="MATS — Morphometric Analysis Toolbox for Segmentation",
        page_icon=branding.page_icon(),
        layout="wide",
    )
    branding.apply_logo()
    st.html(_WORKBENCH_STYLES)
    st.session_state.setdefault("viewer_pairs", {})
    st.session_state.setdefault(UNIT_KEY, "in")
    st.session_state.setdefault(PREVIOUS_UNIT_KEY, "in")
    st.session_state.setdefault(WIDTH_KEY, 12.0)
    st.session_state.setdefault(HEIGHT_KEY, 12.0)
    st.session_state.setdefault(DIMENSIONS_TEXT_KEY, "12x12in")
    st.session_state.setdefault(QR_MODE_KEY, False)
    st.session_state.setdefault(USE_LEGACY_DIMENSIONS_KEY, False)
    st.session_state.setdefault(LEGACY_DIMENSIONS_TEXT_KEY, "10x9.5in")
    st.session_state.setdefault(SEGMENT_THRESHOLD_KEY, True)
    st.session_state.setdefault(SEGMENT_BIREFNET_KEY, False)
    st.session_state.setdefault(THRESHOLD_LEVEL_KEY, "auto")
    st.session_state.setdefault(THRESHOLD_CUSTOM_VALUE_KEY, THRESHOLD_LEVELS["medium"])
    # Streamlit discards a widget's value once it stops rendering. Re-saving
    # keeps the threshold choice while Otsu is unchecked or the level isn't custom.
    for key in (THRESHOLD_LEVEL_KEY, THRESHOLD_CUSTOM_VALUE_KEY):
        st.session_state[key] = st.session_state[key]
    st.session_state.setdefault(
        RESULTS_SCHEMA_KEY,
        "Full research schema (area/width/length + px-per-cm)",
    )
    st.session_state.setdefault(RESULTS_UNIT_KEY, DEFAULT_RESULTS_UNIT)
    st.session_state.setdefault(MEASURE_PRE_CLEANUP_KEY, False)
    st.session_state.setdefault(STRAY_GAP_KEY, STRAY_GAP_DEFAULT)
    st.session_state.setdefault(CLEAN_MARGIN_KEY, CLEAN_MARGIN_DEFAULT)
    st.session_state.setdefault(CLEAN_SIZE_KEY, CLEAN_SIZE_DEFAULT)
    st.session_state.setdefault(WRITE_FAILURES_KEY, True)
    for key, default in ((EXPORT_TARGET_BOXES_KEY, True), (EXPORT_MASKS_KEY, True),
                         (EXPORT_PRE_CLEANUP_KEY, False),
                         (EXPORT_OVERLAY_KEY, False), (EXPORT_CUTOUT_KEY, False),
                         (EXPORT_AXES_KEY, False)):
        st.session_state.setdefault(key, default)
    st.session_state.setdefault(INPUT_DIR_KEY, str(DEFAULT_INPUT_DIR))
    st.session_state.setdefault(OUTPUT_DIR_KEY, str(DEFAULT_OUTPUT_DIR))
    st.session_state.setdefault(INPUT_SOURCE_KEY, "Local folder")

    st.title("MATS Analysis Workbench")
    st.caption(
        "Leaf morphometrics from calibrated images using RF-DETR and Otsu or "
        "BiRefNet segmentation."
    )

    with st.expander("About This Workspace", icon=":material/info:"):
        st.markdown(
            "Choose input and output locations in the sidebar, then configure scale, "
            "segmentation, and output options in Setup. Analyze contains the launch and "
            "measurement visuals; Adjust contains specimen previews and saving; "
            "Export contains saved files and downloads. Diagnostics in the sidebar "
            "contains compute and preflight details. "
            "Template Creator, BiRefNet Setup, CPU Options, Robust QR Setup, and Help "
            "remain available in the app navigation."
        )
        st.page_link(
            "pages/5_Help.py",
            label="Open Help — sample images, settings guide, CSV glossary",
            icon=":material/help:",
            width="content",
        )

    try:
        lm = load_pipeline_module()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        st.error(f"Failed to import the MATS pipeline: missing dependency {missing}.")
        st.info(
            "The Python environment running Streamlit is missing a pipeline "
            f"dependency ({missing}). Install MATS with its dependencies "
            "(rfdetr, transformers, torch, opencv) and relaunch."
        )
        st.caption(f"Current interpreter: `{current_python()}`")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to import the MATS pipeline: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.caption(f"Current interpreter: `{current_python()}`")
        st.stop()

    input_source, uploaded_files, input_dir, output_dir = render_file_sidebar()
    apply_pending_workspace_tab()
    setup_tab, analyze_tab, adjust_tab, export_tab = render_workspace_navigation()

    if setup_tab.open:
        with setup_tab:
            render_analysis_settings(lm)

    config = current_analysis_config(lm)
    compute_settings = get_compute_settings(lm)
    available_workers = compute_settings.available_workers
    birefnet_parallel_is_unlocked = birefnet_parallel_unlocked(available_workers)
    execution_plan = resolve_execution_plan(
        compute_settings,
        "birefnet" if config["needs_birefnet"] else "threshold",
        birefnet_parallel_allowed=birefnet_parallel_is_unlocked,
    )
    worker_risk = lm.worker_risk_report(execution_plan.workers, available_workers)
    break_glass_is_unlocked = break_glass_unlocked(available_workers)
    output_path = Path(output_dir).expanduser()
    results_path = output_path / "leaf_morpho_results.csv"
    checks, image_paths, image_count = build_preflight_checks(
        lm,
        config,
        input_source,
        uploaded_files,
        input_dir,
        compute_settings,
        execution_plan,
        worker_risk,
        break_glass_is_unlocked,
    )
    st.session_state["diagnostics_context"] = config
    st.session_state["diagnostics_inputs"] = (
        input_source, uploaded_files, input_dir
    )

    if analyze_tab.open:
        with analyze_tab:
            render_workspace_context(config, execution_plan)
            render_preflight_summary(checks)
            run_clicked = render_launch_card(
                checks,
                image_count,
                config,
                execution_plan,
                output_path,
            )
            if run_clicked:
                summary, run_pairs = execute_leaf_analysis(
                    lm, config, image_paths, uploaded_files, input_source,
                    output_path, results_path, execution_plan, worker_risk,
                    break_glass_is_unlocked, birefnet_parallel_is_unlocked,
                )
                if summary is not None:
                    st.session_state["last_run"] = {
                        "run_id": uuid.uuid4().hex,
                        "succeeded": summary["succeeded"],
                        "failed": summary["failed"],
                        "total": summary["total"],
                        "workers": summary["workers"],
                        "worker_reason": summary["worker_reason"],
                        "execution_device": summary["execution_device"],
                        "output_path": str(output_path),
                        "mask_methods": summary["methods"],
                        "by_method": {
                            method: {
                                "succeeded": outcome["succeeded"],
                                "failed": outcome["failed"],
                                "results_path": outcome["results_path"],
                                "failure_rows": outcome["failure_rows"][:200],
                                "failure_overflow": max(0, len(outcome["failure_rows"]) - 200),
                            }
                            for method, outcome in summary["by_method"].items()
                        },
                        "pre_cleanup_methods": config["export_options"]["pre_cleanup_methods"],
                        "results_unit": config["results_unit"],
                        "measurement_source": summary["measurement_source"],
                        "stray_gap": summary["stray_gap"],
                        "clean_margin": summary["clean_margin"],
                        "clean_size": summary["clean_size"],
                        "threshold_value": config["threshold_value"],
                        "scale_axes_by_sample": _scale_axes_by_sample(summary),
                        "artifacts": summary["artifacts"],
                        "export_options": dict(config["export_options"]),
                    }
                    st.session_state["viewer_pairs"] = run_pairs
                    _clear_export_zip_cache()
                    st.rerun()
            render_results(lm)
    elif export_tab.open:
        with export_tab:
            render_export_section()
    elif adjust_tab.open:
        with adjust_tab:
            render_adjust_workspace()


def render_results(lm):
    del lm  # The Results tab operates on the artifacts from the completed run.
    run = st.session_state.get("last_run")
    if not run:
        with st.container(border=True):
            st.subheader("Results", anchor=False)
            st.info(
                "Run an analysis to unlock measurement summaries, charts, and specimen inspection.",
                icon=":material/insights:",
            )
            st.button(
                "Open Analyze",
                icon=":material/science:",
                on_click=_switch_workspace_tab,
                args=("Analyze",),
                width="content",
            )
        return

    unit_symbol, area_symbol = _unit_symbols(run)
    methods = tuple(run["mask_methods"])
    st.subheader("Results", anchor=False)
    st.caption(
        "Measurements from "
        + ("pre-cleanup masks" if run.get("measurement_source") == "pre-cleanup"
           else "cleaned masks")
        + "."
    )
    if len(methods) > 1:
        per_method = "; ".join(
            f"{SEGMENTATION_METHOD_LABELS[method]}: {run['by_method'][method]['succeeded']} "
            f"measured, {run['by_method'][method]['failed']} failed"
            for method in methods
        )
        st.success(
            f"Completed {run['total']} input image(s) with each method. {per_method}.",
            icon=":material/check_circle:",
        )
        if st.session_state.get(RESULTS_METHOD_KEY) not in methods:
            st.session_state[RESULTS_METHOD_KEY] = methods[0]
        with st.container(border=True):
            st.markdown("**Compare measurement methods**")
            method = st.segmented_control(
                "Measurement table and specimen view",
                options=list(methods),
                format_func=SEGMENTATION_METHOD_LABELS.get,
                required=True,
                key=RESULTS_METHOD_KEY,
                help="Each method has its own measurements, masks, and failure log.",
            )
            st.caption(
                "The summary, charts, measurement table, and selected specimen below "
                "follow this choice."
            )
    else:
        method = methods[0]
    outcome = run["by_method"][method]
    results_path = Path(outcome["results_path"])
    output_pairs = st.session_state.get("viewer_pairs", {}).get(method, [])
    if len(methods) == 1:
        st.success(
            f"Completed {outcome['succeeded']} measurement(s); {outcome['failed']} failed "
            f"(of {run['total']} input image(s)).",
            icon=":material/check_circle:",
        )
    device_label = {
        "cpu": "CPU only",
        "hybrid": "GPU RF-DETR + CPU Otsu fan-out",
    }.get(run["execution_device"], "automatic accelerator selection")
    st.caption(
        f"Workers used: {run['workers']} ({run['worker_reason']}); compute: {device_label}."
    )

    if outcome["failure_rows"]:
        with st.expander(f"Processing warnings and failures ({outcome['failed']})"):
            for row in outcome["failure_rows"]:
                st.write(f"{row['sample_id']}: {row['status']}")
            if outcome["failure_overflow"]:
                st.write(f"...and {outcome['failure_overflow']} more (see the failures CSV).")

    if not results_path.is_file():
        st.warning(
            f"The results CSV is not available at {display_path(results_path)}. "
            "Other saved files may still be available in Export.",
            icon=":material/folder_off:",
        )
        return

    total_rows = count_csv_rows(results_path)
    try:
        frame = read_results_dataframe(
            str(results_path),
            results_path.stat().st_mtime_ns,
            RESULTS_DASHBOARD_MAX_ROWS,
        )
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not read the results CSV: {exc}")
        return

    measurements = normalize_measurements(frame, run.get("results_unit", DEFAULT_RESULTS_UNIT))
    if measurements.empty:
        st.warning(
            "The CSV contains no complete area, width, and length measurements to visualize.",
            icon=":material/bar_chart_off:",
        )
    else:
        summary = summarize_measurements(measurements)
        with st.container(horizontal=True):
            st.metric(
                "Successful measurements",
                f"{outcome['succeeded']:,}",
                delta=f"{outcome['failed']} failed",
                delta_color="inverse",
                border=True,
            )
            st.metric(
                "Median leaf area",
                f"{summary['median_area']:.2f} {area_symbol}",
                border=True,
            )
            st.metric(
                "Median leaf width",
                f"{summary['median_width']:.2f} {unit_symbol}",
                border=True,
            )
            st.metric(
                "Median leaf length",
                f"{summary['median_length']:.2f} {unit_symbol}",
                border=True,
            )

        if total_rows > len(frame):
            st.caption(
                f"Charts use the first {len(frame):,} of {total_rows:,} CSV rows to "
                "keep the workspace responsive."
            )

        relationship_column, distribution_column = st.columns(2, vertical_alignment="top")
        with relationship_column.container(border=True):
            st.markdown("**Leaf Width and Length**")
            relationship_chart = (
                alt.Chart(measurements)
                .mark_circle(opacity=0.75)
                .encode(
                    x=alt.X("width:Q", title=f"Leaf width ({unit_symbol})"),
                    y=alt.Y("length:Q", title=f"Leaf length ({unit_symbol})"),
                    size=alt.Size(
                        "leaf_area:Q",
                        title=f"Leaf area ({area_symbol})",
                        scale=alt.Scale(range=[30, 900]),
                    ),
                    color=alt.Color(
                        "leaf_area:Q",
                        title=f"Leaf area ({area_symbol})",
                        scale=alt.Scale(scheme="blues"),
                    ),
                    tooltip=[
                        alt.Tooltip("sample_id:N", title="Sample"),
                        alt.Tooltip("leaf_area:Q", title=f"Area ({area_symbol})", format=".2f"),
                        alt.Tooltip("width:Q", title=f"Width ({unit_symbol})", format=".2f"),
                        alt.Tooltip("length:Q", title=f"Length ({unit_symbol})", format=".2f"),
                    ],
                )
                .interactive()
            )
            st.altair_chart(relationship_chart, width="stretch")

        with distribution_column.container(border=True):
            st.markdown("**Leaf Area Distribution**")
            distribution_chart = (
                alt.Chart(measurements)
                .mark_bar(color="#2c7fb8")
                .encode(
                    x=alt.X(
                        "leaf_area:Q",
                        bin=alt.Bin(maxbins=20),
                        title=f"Leaf area ({area_symbol})",
                    ),
                    y=alt.Y("count():Q", title="Leaf count"),
                    tooltip=[
                        alt.Tooltip("count():Q", title="Leaves"),
                    ],
                )
            )
            st.altair_chart(distribution_chart, width="stretch")

        if "scale_aspect_ratio" in measurements:
            scale_ratio = measurements["scale_aspect_ratio"].dropna()
            if not scale_ratio.empty:
                within_tolerance = (scale_ratio.sub(1).abs() <= 0.02).mean()
                with st.container(border=True):
                    st.markdown("**Scale Quality Control**")
                    with st.container(horizontal=True):
                        st.metric(
                            "Median axis-scale ratio",
                            f"{scale_ratio.median():.3f}",
                            border=True,
                        )
                        st.metric(
                            "Within 2% of 1.0",
                            f"{within_tolerance:.0%}",
                            border=True,
                        )
                        st.metric(
                            "Observed range",
                            f"{scale_ratio.min():.3f}–{scale_ratio.max():.3f}",
                            border=True,
                        )
                    st.caption(
                        "Values near 1.0 indicate similar calibration scale on the "
                        "horizontal and vertical axes."
                    )

        table_columns = ["sample_id", "leaf_area", "width", "length"]
        if "scale_aspect_ratio" in measurements:
            table_columns.append("scale_aspect_ratio")
        table_data = measurements.loc[:, table_columns].head(RESULTS_TABLE_MAX_ROWS).copy()
        selected_key = f"selected_specimen_{run.get('run_id', '')}_{method}"
        sample_ids = table_data["sample_id"].astype(str).tolist()
        previous_sample_id = st.session_state.get(selected_key)
        selection_default = (
            {"selection": {"rows": [sample_ids.index(previous_sample_id)]}}
            if previous_sample_id in sample_ids else None
        )
        with st.container(border=True):
            st.markdown("**Measurement Table**")
            st.caption(
                f"{SEGMENTATION_METHOD_LABELS[method]} measurements. Select a row to "
                "inspect its matching target-box image and segmentation mask."
            )
            table_event = st.dataframe(
                table_data,
                column_config=_measurement_column_config(unit_symbol, area_symbol),
                hide_index=True,
                height=440,
                width="stretch",
                key=f"measurement_table_{run.get('run_id', '')}_{method}",
                on_select="rerun",
                selection_mode="single-row",
                selection_default=selection_default,
            )
            if total_rows > len(table_data):
                st.caption(
                    f"Showing the first {len(table_data):,} of {total_rows:,} rows. "
                    "Download the CSV for the full dataset."
                )

        selected_row = None
        if table_event.selection.rows:
            selected_row = table_data.iloc[table_event.selection.rows[0]]
        selected_sample_id = (
            str(selected_row["sample_id"]) if selected_row is not None else None
        )
        if selected_sample_id is None:
            st.session_state.pop(selected_key, None)
        else:
            st.session_state[selected_key] = selected_sample_id
        render_specimen_inspector(
            selected_sample_id,
            selected_row,
            output_pairs,
            unit_symbol,
            area_symbol,
            run=run,
            method=method,
        )
        if selected_sample_id is not None:
            st.button(
                "Adjust selected specimen",
                icon=":material/tune:",
                on_click=_switch_workspace_tab,
                args=("Adjust",),
            )

        qr_trace_columns = [column for column in QR_TRACE_FIELDNAMES if column in frame.columns]
        if qr_trace_columns:
            with st.container(border=True):
                st.markdown("**QR decoder trace**")
                st.caption(
                    "Shown because this run used variable dimensions. A decoder marked "
                    "unused was skipped after an earlier decoder succeeded."
                )
                st.dataframe(
                    frame.loc[:, ["sample_id", "source", *qr_trace_columns]].head(
                        RESULTS_TABLE_MAX_ROWS
                    ),
                    column_config={
                        "sample_id": st.column_config.TextColumn("Sample", pinned=True),
                        "source": st.column_config.TextColumn("Measurement status"),
                        "qr_opencv": st.column_config.TextColumn("OpenCV"),
                        "qr_pyzbar_zbar": st.column_config.TextColumn("pyzbar + zbar"),
                        "qr_qreader": st.column_config.TextColumn("QReader"),
                    },
                    hide_index=True,
                    height=260,
                )


def render_adjust_workspace():
    """Inspect one specimen and optionally save its settings to marked peers."""
    run = st.session_state.get("last_run")
    st.subheader("Adjust", anchor=False)
    if not run:
        st.info("Run an analysis to adjust its specimen masks and measurements.")
        st.button("Open Analyze", on_click=_switch_workspace_tab, args=("Analyze",))
        return
    methods = tuple(run["mask_methods"])
    if st.session_state.get(RESULTS_METHOD_KEY) not in methods:
        st.session_state[RESULTS_METHOD_KEY] = methods[0]
    if len(methods) > 1:
        method = st.segmented_control(
            "Segmentation method", list(methods),
            format_func=SEGMENTATION_METHOD_LABELS.get, required=True,
            key=RESULTS_METHOD_KEY,
        )
    else:
        method = methods[0]
        st.caption(SEGMENTATION_METHOD_LABELS[method])
    results_path = Path(run["by_method"][method]["results_path"])
    if not results_path.is_file():
        st.warning("This method's results CSV is unavailable.")
        return
    try:
        frame = read_results_dataframe(
            str(results_path), results_path.stat().st_mtime_ns, None
        )
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not read results: {exc}")
        return
    measurements = normalize_measurements(frame, run.get("results_unit", DEFAULT_RESULTS_UNIT))
    pairs = st.session_state.get("viewer_pairs", {}).get(method, [])
    pairs_by_id = {str(pair["sample_id"]): pair for pair in pairs}
    choices = [
        str(sample_id) for sample_id in measurements["sample_id"]
        if str(sample_id) in pairs_by_id
        and (method != "threshold" or pairs_by_id[str(sample_id)].get("target_box"))
    ]
    if not choices:
        st.info("No measured specimens with previews are available in this session.")
        return
    unit_symbol, area_symbol = _unit_symbols(run)
    sample_id, marked = render_adjust_browser(
        choices, run.get("run_id", ""), method, measurements=measurements,
        unit_symbol=unit_symbol, area_symbol=area_symbol,
    )
    if method != "threshold":
        st.caption("BiRefNet cleanup controls currently preview changes only.")
    render_adjust_controls(
        pairs_by_id[sample_id], run, method,
        marked_pairs=[pairs_by_id[item] for item in marked],
    )


def _record_table_view(table_key, selected_key):
    sample_id = _component_value(table_key, "view")
    if sample_id is not None:
        st.session_state[selected_key] = str(sample_id)


def _record_table_mark(table_key, marks_key):
    """Apply one Marked for Adjustment tick; setting, not toggling, keeps repeats harmless."""
    change = _component_value(table_key, "mark")
    if not change:
        return
    marked = set(st.session_state.get(marks_key, []))
    if change["marked"]:
        marked.add(str(change["id"]))
    else:
        marked.discard(str(change["id"]))
    st.session_state[marks_key] = sorted(marked)


def _set_marked_specimens(key, sample_ids):
    st.session_state[key] = list(sample_ids)


def render_adjust_browser(
    choices, run_id, method, *, measurements=None, unit_symbol="cm", area_symbol="cm²",
):
    """Search specimens in a measurement table; marks persist across searches."""
    scope = f"{run_id}_{method}"
    selected_key = f"selected_specimen_{scope}"
    marks_key = f"adjust_marked_{scope}"
    search_key = f"adjust_search_{scope}"
    if st.session_state.get(selected_key) not in choices:
        st.session_state[selected_key] = choices[0]
    valid = set(choices)
    marked = set(st.session_state.get(marks_key, [])) & valid
    st.session_state[marks_key] = sorted(marked)
    query = st.text_input("Search specimen names", key=search_key).strip().casefold()
    filtered = [sample_id for sample_id in choices if query in sample_id.casefold()]
    if method == "threshold":
        with st.container(horizontal=True):
            st.button(
                "Mark all matching" if query else "Mark all",
                key=f"adjust_mark_all_{scope}", disabled=not filtered,
                on_click=_set_marked_specimens,
                args=(marks_key, sorted(marked | set(filtered))),
            )
            st.button(
                "Clear marks", key=f"adjust_clear_marks_{scope}", disabled=not marked,
                on_click=_set_marked_specimens, args=(marks_key, []),
            )
        st.caption(f"{len(marked):,} marked across all searches.")
    if not filtered:
        st.info("No specimen names match this search. Clear the search to see all specimens.")
        return st.session_state[selected_key], sorted(marked)
    from mats.app.specimen_table import show_specimen_table

    if measurements is None:
        table = pd.DataFrame({"sample_id": filtered})
    else:
        table = measurements.assign(sample_id=measurements["sample_id"].astype(str))
        table = table.loc[table["sample_id"].isin(set(filtered))]
    columns = [
        {"key": column, "label": label, "decimals": decimals}
        for column, label, decimals in _measurement_columns(unit_symbol, area_symbol)
        if column in table
    ]
    table = table.loc[:, [column["key"] for column in columns]]
    rows = table.astype(object).where(table.notna(), None).to_dict("records")
    markable = method == "threshold"
    st.caption(
        f"{len(filtered):,} matching specimens. Click a row to view it"
        + ("; tick Marked for Adjustment to include it in the marked overwrite." if markable else ".")
    )
    table_key = f"adjust_table_{scope}"
    show_specimen_table(
        key=table_key, columns=columns, rows=rows,
        view=st.session_state[selected_key], marked=sorted(marked), markable=markable,
        on_view_change=lambda: _record_table_view(table_key, selected_key),
        on_mark_change=lambda: _record_table_mark(table_key, marks_key),
    )
    return st.session_state[selected_key], sorted(marked)


def render_adjust_controls(pair, run, method, *, marked_pairs=None):
    """Render only adjustment controls and the live mask/color preview."""
    st.caption(f"Sample: {pair['sample_id']}")
    with st.container(border=True):
        if method == "threshold" and pair.get("target_box"):
            render_threshold_explorer(pair, run, marked_pairs=marked_pairs)
        elif method == "birefnet" and _raw_mask_path(pair):
            render_mask_explorer(pair, run, method)
        else:
            st.info("The original segmentation mask is unavailable for adjustment.")


def render_specimen_inspector(
    sample_id, measurement, pairs, unit_symbol="cm", area_symbol="cm²",
    run=None, method=None,
):
    """Render the selected specimen's measurements and artifacts below the table."""
    with st.container(border=True):
        st.markdown("**Selected Specimen**")
        if sample_id is None:
            st.info(
                "Select a row in the measurement table to inspect its output artifacts.",
                icon=":material/touch_app:",
            )
            return

        st.badge(sample_id, icon=":material/eco:", color="green")
        if measurement is not None:
            with st.container(horizontal=True):
                st.metric("Leaf Area", f"{measurement['leaf_area']:.2f} {area_symbol}", border=True)
                st.metric("Leaf Width", f"{measurement['width']:.2f} {unit_symbol}", border=True)
                st.metric("Leaf Length", f"{measurement['length']:.2f} {unit_symbol}", border=True)
        pair = find_output_pair(pairs, sample_id)
        if pair is None:
            st.info(
                "No output pair is available in this browser session for this sample. "
                "Run the analysis here to inspect its target box and mask.",
                icon=":material/image_not_supported:",
            )
            return
        render_output_pair(pair, width="stretch")



def _component_value(component_key, name):
    component_state = st.session_state.get(component_key)
    value = getattr(component_state, name, None)
    if value is None and isinstance(component_state, dict):
        value = component_state.get(name)
    return value


def _record_threshold_preview(component_key, state_key):
    cutoff = _component_value(component_key, "cutoff")
    if cutoff is not None and 0 <= int(cutoff) <= THRESHOLD_MAX:
        st.session_state.setdefault("threshold_preview_cutoffs", {})[state_key] = int(cutoff)


def _record_clean_radius(component_key, state_key):
    from mats.mask_cleanup import CLEAN_RADIUS_MAX

    radius = _component_value(component_key, "clean_radius")
    if radius is not None and 0 <= int(radius) <= CLEAN_RADIUS_MAX:
        st.session_state.setdefault("clean_preview_radii", {})[state_key] = int(radius)


CLEAN_SIZE_HELP = (
    "0 turns Clean image off. Above 0, Clean image replaces MATS cleanup: the edge "
    "margin is cleared, pieces touching it or far from the leaf are removed, and "
    "white specks and enclosed black holes with an inscribed radius below the "
    "clean size (px) are removed or filled. The leaf is always kept and nothing is "
    "flash-filled."
)
CLEAN_SIZE_SAVE_HELP = CLEAN_SIZE_HELP + " Overwrite saves and measures the mask shown."
CLEAN_SIZE_PREVIEW_HELP = CLEAN_SIZE_HELP + " BiRefNet specimens preview it only."
CLEAN_SIZE_ON_CAPTION = (
    "Clean size is above 0, so this preview shows Clean image instead of MATS cleanup."
)


def _clean_radius(run, method, sample_id):
    """This specimen's clean size: slider edits, else saved, else the run's; 0 is off."""
    saved = run.get("threshold_adjustments", {}).get(sample_id, {}) if method == "threshold" else {}
    state_key = f"{run.get('run_id', '')}:{method}:{sample_id}"
    return st.session_state.setdefault("clean_preview_radii", {}).get(
        state_key, saved.get("clean_size", run.get("clean_size", CLEAN_SIZE_DEFAULT))
    )


REMOVE_FILL_RAW_NOTE = "This run measured raw masks, so hole filling is already off."


def _record_remove_fill(component_key, state_key):
    remove_fill = _component_value(component_key, "remove_fill")
    if remove_fill is not None:
        st.session_state.setdefault("remove_fill_previews", {})[state_key] = bool(remove_fill)


def _remove_fill(run, state_key):
    """Whether Remove flashfill is on for this specimen; only cleaned runs fill holes."""
    if run.get("measurement_source", "cleaned") != "cleaned":
        return False
    return st.session_state.setdefault("remove_fill_previews", {}).get(state_key, False)


def _remove_fill_note(run):
    """Why Remove flashfill is unavailable for this run, or None when it is available."""
    return None if run.get("measurement_source", "cleaned") == "cleaned" else REMOVE_FILL_RAW_NOTE


def _run_stray_gap(run):
    """The stray-piece gap this run measured with; the default for older runs."""
    return run.get("stray_gap", STRAY_GAP_DEFAULT)


def _run_clean_margin(run):
    """The edge margin this run measured with; the default for older runs."""
    return run.get("clean_margin", CLEAN_MARGIN_DEFAULT)


def _specimen_cleanup_keys(run, method, sample_id):
    base = f"{run.get('run_id', '')}:{method}:{sample_id}"
    return f"specimen_clean_margin_{base}", f"specimen_stray_gap_{base}"


def _specimen_cleanup(run, method, sample_id):
    """This specimen's edge margin and stray gap: explorer edits, else saved, else the run's."""
    saved = run.get("threshold_adjustments", {}).get(sample_id, {}) if method == "threshold" else {}
    margin_key, gap_key = _specimen_cleanup_keys(run, method, sample_id)
    margin = st.session_state.get(margin_key, saved.get("clean_margin", _run_clean_margin(run)))
    gap = st.session_state.get(gap_key, saved.get("stray_gap", _run_stray_gap(run)))
    return float(margin), float(gap)


def _specimen_cleanup_inputs(run, method, sample_id, *, margin_active, gap_active):
    """Render this specimen's edge-margin and stray-gap inputs; return their values.

    Each input is enabled only while the view it drives has flash fill off.
    """
    margin_key, gap_key = _specimen_cleanup_keys(run, method, sample_id)
    margin, gap = _specimen_cleanup(run, method, sample_id)
    st.session_state[margin_key], st.session_state[gap_key] = margin, gap
    margin_column, gap_column = st.columns(2)
    with margin_column:
        st.number_input(
            "Edge margin (% of box)",
            min_value=0.0,
            max_value=CLEAN_MARGIN_MAX,
            step=0.25,
            format="%.2f",
            key=margin_key,
            disabled=not margin_active,
            help=(
                "Band cleared along every edge of this specimen's target box, where "
                "the printed box outline lands. Applies to pre-cleanup runs, a clean "
                "size above 0, and Remove flashfill."
            ),
        )
    with gap_column:
        st.number_input(
            "Stray-piece distance (× leaf size)",
            min_value=0.0,
            max_value=STRAY_GAP_MAX,
            step=0.05,
            format="%.2f",
            key=gap_key,
            disabled=not gap_active,
            help=(
                "Pieces farther from the leaf than this fraction of its bounding-box "
                "diagonal are dropped; 0 keeps only the leaf. Applies to pre-cleanup "
                "runs and a clean size above 0."
            ),
        )
    return float(st.session_state[margin_key]), float(st.session_state[gap_key])


def _raw_mask_path(pair):
    """The specimen's mask before MATS cleanup, when this session has it."""
    return pair.get("raw_mask")


def _reset_threshold_preview(state_key):
    st.session_state.setdefault("threshold_preview_cutoffs", {}).pop(state_key, None)


def _use_preview_threshold(cutoff):
    st.session_state[THRESHOLD_LEVEL_KEY] = CUSTOM_THRESHOLD_LEVEL
    st.session_state[THRESHOLD_CUSTOM_VALUE_KEY] = cutoff


@st.fragment
def render_threshold_explorer(pair, run, *, marked_pairs=None):
    """Preview an Otsu cutoff and save it to this specimen or marked specimens."""
    from mats.app.threshold_preview import (
        clean_levels_for_threshold, cleaned_sample, color_sample, grayscale_sample,
        pre_cleanup_sample, show_threshold_preview,
    )

    target_path = pair["target_box"]
    try:
        grayscale_image, otsu_cutoff = grayscale_sample(str(target_path))
    except ValueError as exc:
        st.info(str(exc))
        return
    try:
        color_image = color_sample(str(target_path))
    except ValueError:
        color_image = None  # The mask panel still works without the color panel.

    saved_adjustment = run.get("threshold_adjustments", {}).get(pair["sample_id"], {})
    saved_cutoff = saved_adjustment.get("cutoff", run.get("threshold_value"))
    if saved_cutoff is None:
        saved_cutoff = otsu_cutoff
    state_key = f"{run.get('run_id', '')}:{pair['sample_id']}"
    component_key = f"threshold_preview_{state_key}"
    cutoff = st.session_state.setdefault("threshold_preview_cutoffs", {}).get(
        state_key, saved_cutoff
    )
    clean_key = f"{run.get('run_id', '')}:threshold:{pair['sample_id']}"
    remove_fill = _remove_fill(run, clean_key)

    st.markdown("**Explore and adjust output**")
    clean_radius = _clean_radius(run, "threshold", pair["sample_id"])
    clean_on = clean_radius > 0
    if clean_on:
        st.caption(
            CLEAN_SIZE_ON_CAPTION + " Overwrite saves it."
            + (" Remove flashfill doesn't apply while it is above 0." if remove_fill else "")
        )
    elif run.get("measurement_source") == "cleaned":
        st.caption(
            "Drag to preview a new cutoff; the masked leaf beside the mask shows which "
            "parts of the leaf it keeps. Release to apply MATS cleanup to this "
            "sample. Saved outputs change only when you press Overwrite below."
        )
    else:
        st.caption(
            "Drag to preview a new raw mask for this sample; the masked leaf beside it "
            "shows which parts of the leaf it keeps. Release to see the mask this run "
            "measures, with the edge margin cleared and stray pieces dropped. Nothing "
            "saved changes until you press Overwrite below."
        )
    pre_cleanup = run.get("measurement_source") == "pre-cleanup"
    # Views without flash fill clear the edge margin, including while dragging.
    flash_fill_off = clean_on or pre_cleanup or remove_fill
    margin, gap = _specimen_cleanup_inputs(
        run, "threshold", pair["sample_id"],
        margin_active=flash_fill_off, gap_active=clean_on or pre_cleanup,
    )
    # Both images go to the browser, so the clean-size slider is live from 0.
    try:
        clean_levels_image = clean_levels_for_threshold(str(target_path), cutoff, gap, margin)
        if pre_cleanup:
            cleaned_image = pre_cleanup_sample(str(target_path), cutoff, margin, gap)
        else:
            cleaned_image = cleaned_sample(
                str(target_path), cutoff, fill_holes=not remove_fill, clean_margin=margin,
            )
    except ValueError as exc:
        st.info(str(exc))
        return
    show_threshold_preview(
        key=component_key,
        grayscale_image=grayscale_image,
        cutoff=cutoff,
        measurement_source=run.get("measurement_source", "cleaned"),
        color_image=color_image,
        cleaned_image=cleaned_image,
        cleaned_cutoff=cutoff,
        remove_fill=remove_fill,
        clean_levels_image=clean_levels_image,
        clean_cutoff=cutoff,
        clean_radius=clean_radius,
        clean_help=CLEAN_SIZE_SAVE_HELP,
        fill_toggle=True,
        fill_note=_remove_fill_note(run),
        live_margin=margin if flash_fill_off else 0,
        on_cutoff_change=lambda: _record_threshold_preview(component_key, state_key),
        on_clean_radius_change=lambda: _record_clean_radius(component_key, clean_key),
        on_remove_fill_change=lambda: _record_remove_fill(component_key, clean_key),
    )
    left, right = st.columns(2)
    with left:
        st.button(
            "Reset to saved threshold", key=f"reset_{state_key}",
            on_click=_reset_threshold_preview, args=(state_key,), width="stretch",
        )
    with right:
        st.button(
            "Use threshold for next run", key=f"use_{state_key}",
            disabled=not THRESHOLD_MIN <= cutoff <= THRESHOLD_MAX,
            on_click=_use_preview_threshold, args=(cutoff,), width="stretch",
            help=(
                "Auto selected cutoff 0. Drag to at least 1 to use a custom threshold."
                if cutoff == 0 else
                "Sets Setup to this custom threshold; run analysis again to update the CSV."
            ),
        )

    notice_key = f"threshold_adjustment_notice_{state_key}"
    notice = st.session_state.pop(notice_key, None)
    if notice:
        st.success(notice, icon=":material/check_circle:")
    marked_pairs = marked_pairs or []
    st.caption(
        "Saving replaces the selected mask, CSV measurements, and any saved "
        "overlays, cutouts, and measurement axes."
    )
    unsavable = not THRESHOLD_MIN <= cutoff <= THRESHOLD_MAX
    overwrite_this = st.button(
        "Overwrite this specimen",
        key=f"apply_adjustment_{state_key}",
        type="primary",
        icon=":material/save:",
        disabled=unsavable,
        help="Saves these settings, including the clean size, to the specimen selected in View.",
        width="stretch",
    )
    overwrite_marked = st.button(
        f"Overwrite all marked specimens ({len(marked_pairs)})",
        key=f"apply_marked_adjustment_{state_key}",
        icon=":material/done_all:",
        disabled=unsavable or not marked_pairs,
        help=(
            "Saves these settings, including the clean size, to every specimen "
            "checked in Marked; each keeps its own calibration."
        ),
        width="stretch",
    )
    if overwrite_this or overwrite_marked:
        from mats.app.output_adjustment import (
            apply_threshold_adjustment, apply_threshold_adjustments,
        )

        bulk = overwrite_marked
        count = len(marked_pairs) if bulk else 1
        try:
            if bulk:
                apply_threshold_adjustments(
                    run, marked_pairs, int(cutoff), remove_fill=remove_fill,
                    clean_margin=margin, stray_gap=gap, clean_size=clean_radius,
                )
            else:
                apply_threshold_adjustment(
                    run, pair, int(cutoff), remove_fill=remove_fill,
                    clean_margin=margin, stray_gap=gap, clean_size=clean_radius,
                )
        except (OSError, ValueError) as exc:
            st.error(f"Could not save adjustments: {exc}")
        else:
            read_results_dataframe.clear()
            _clear_export_zip_cache()
            st.session_state[notice_key] = (
                f"Updated {count} specimen(s) at threshold {int(cutoff)}"
                + (f", clean size {clean_radius} px." if clean_on else ".")
            )
            st.rerun()


@st.fragment
def render_mask_explorer(pair, run, method):
    """Preview cleanup on one specimen's raw mask; nothing is saved."""
    from mats.app.threshold_preview import (
        clean_levels_for_mask, color_sample, mask_sample, pre_cleanup_mask_sample,
        show_threshold_preview, unfilled_sample,
    )

    state_key = f"{run.get('run_id', '')}:{method}:{pair['sample_id']}"
    component_key = f"mask_preview_{state_key}"
    remove_fill = _remove_fill(run, state_key)
    pre_cleanup = run.get("measurement_source") == "pre-cleanup"
    st.markdown("**Explore and adjust output**")
    clean_radius = _clean_radius(run, method, pair["sample_id"])
    clean_on = clean_radius > 0
    if clean_on:
        st.caption(CLEAN_SIZE_ON_CAPTION + " Nothing is saved.")
    elif pre_cleanup:
        st.caption(
            "Showing this specimen's pre-cleanup measurement mask. Adjust the edge "
            "margin and stray-piece distance to preview it; nothing is saved."
        )
    elif remove_fill:
        st.caption(
            "Showing the mask with flashfill removed. Adjust the edge margin to "
            "preview it; nothing is saved."
        )
    else:
        st.caption(
            "Showing the saved measurement mask. Drag the clean size above 0 to "
            "preview removing small specks and filling small holes in the raw mask; "
            "nothing is saved."
        )
    margin, gap = _specimen_cleanup_inputs(
        run, method, pair["sample_id"],
        margin_active=clean_on or pre_cleanup or remove_fill,
        gap_active=clean_on or pre_cleanup,
    )
    mask_status = "Saved measurement mask"
    # Both images go to the browser, so the clean-size slider is live from 0.
    try:
        raw_path = Path(_raw_mask_path(pair))
        clean_levels_image = clean_levels_for_mask(
            str(raw_path), raw_path.stat().st_mtime_ns, gap, margin,
        )
        if pre_cleanup:
            mask_image = pre_cleanup_mask_sample(
                str(raw_path), raw_path.stat().st_mtime_ns, margin, gap,
            )
            mask_status = "Pre-cleanup mask: edge margin cleared and stray pieces dropped"
        elif remove_fill:
            mask_image = unfilled_sample(str(raw_path), margin)
            mask_status = "Mask with hole filling removed and the edge margin cleared"
        else:
            shown_path = Path(pair.get("mask") or raw_path)
            mask_image = mask_sample(str(shown_path), shown_path.stat().st_mtime_ns)
    except (OSError, ValueError) as exc:
        st.info(str(exc))
        return
    color_image = None
    if pair.get("target_box"):
        try:
            color_image = color_sample(str(pair["target_box"]))
        except ValueError:
            pass  # The mask panel still works without the color panel.
    show_threshold_preview(
        key=component_key,
        mask_image=mask_image,
        mask_status=mask_status,
        color_image=color_image,
        clean_levels_image=clean_levels_image,
        clean_radius=clean_radius,
        clean_help=CLEAN_SIZE_PREVIEW_HELP,
        remove_fill=remove_fill,
        fill_toggle=True,
        fill_note=_remove_fill_note(run),
        on_clean_radius_change=lambda: _record_clean_radius(component_key, state_key),
        on_remove_fill_change=lambda: _record_remove_fill(component_key, state_key),
    )


def render_output_pair(pair, width=PREVIEW_IMAGE_WIDTH, mask_override=None):
    st.caption(f"Sample: {pair['sample_id']}")
    left, right = st.columns(2)
    with left:
        if pair["target_box"] is not None:
            st.image(
                str(pair["target_box"]),
                caption="Perspective-corrected target box",
                width=width,
            )
        else:
            st.caption("Target box: not available")
    with right:
        if mask_override is not None:
            st.image(mask_override, caption="Preview with flashfill removed", width=width)
        elif pair["mask"] is not None:
            caption = (
                "Pre-cleanup measurement mask"
                if pair.get("mask_source") == "pre-cleanup"
                else "Cleaned measurement mask"
            )
            st.image(str(pair["mask"]), caption=caption, width=width)
        else:
            st.caption("Measurement mask preview unavailable for this sample")


def _clear_export_zip_cache():
    """Discard the prepared ZIP when its run or selection changes."""
    zip_path = st.session_state.pop("export_zip_path", None)
    st.session_state.pop("export_zip_signature", None)
    if zip_path:
        try:
            Path(zip_path).unlink(missing_ok=True)
        except OSError:
            pass


def select_export_files(artifacts, methods, kinds):
    """Select existing files from this run, including shared files just once."""
    selected_methods = set(methods)
    if not selected_methods:
        return []
    selected_kinds = set(kinds)
    files = []
    seen = set()
    for item in artifacts:
        if item.get("kind") not in selected_kinds:
            continue
        method = item.get("method")
        if method is not None and method not in selected_methods:
            continue
        path = Path(item["path"])
        if path not in seen and path.is_file():
            files.append(path)
            seen.add(path)
    return files


def zip_download_name(raw_name):
    """Keep the requested download name a single ZIP filename."""
    name = str(raw_name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if name in {"", ".", ".."}:
        name = "leaf_morpho_outputs"
    return name if name.lower().endswith(".zip") else f"{name}.zip"


def _export_run_token(run):
    return run.get("run_id") or "|".join(
        str(run.get("by_method", {}).get(method, {}).get("results_path", ""))
        for method in run.get("mask_methods", ())
    )


def _export_signature(run_token, files, download_name):
    """Identify the exact ZIP contents, including files changed on disk."""
    return (
        run_token,
        tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files),
        download_name,
    )


def render_export_section():
    """Choose and download files recorded for the completed analysis run."""
    st.subheader("Export", anchor=False)
    run = st.session_state.get("last_run")
    if not run:
        st.info("Run an analysis to see and download its saved files.", icon=":material/folder_open:")
        st.button("Open Analyze", on_click=_switch_workspace_tab, args=("Analyze",),
                  icon=":material/science:")
        return

    methods = tuple(run["mask_methods"])
    artifacts = tuple(run.get("artifacts", ()))
    available = tuple(item for item in artifacts if Path(item["path"]).is_file())
    missing_count = len(artifacts) - len(available)
    all_files = select_export_files(available, methods, (kind for kind, _ in EXPORT_FILE_TYPES))
    _, all_bytes = estimate_zip_inputs(all_files)
    output_path = Path(run["output_path"])
    st.caption(f"Completed run · {len(all_files):,} saved file(s) · {human_bytes(all_bytes)}")
    st.caption(f"Output folder: `{display_path(output_path)}`")
    if missing_count:
        st.warning(
            f"{missing_count} file(s) recorded for this run are no longer available on disk. "
            "They will be left out of downloads.",
            icon=":material/folder_off:",
        )
    if not all_files:
        st.warning("No saved files from this run are available for download.")
        return

    run_token = _export_run_token(run)
    if st.session_state.get(EXPORT_RUN_ID_KEY) != run_token:
        _clear_export_zip_cache()
        st.session_state[EXPORT_RUN_ID_KEY] = run_token
        st.session_state[EXPORT_METHODS_KEY] = list(methods)
        st.session_state[EXPORT_ZIP_NAME_KEY] = "leaf_morpho_outputs.zip"
        available_kinds = {item["kind"] for item in available}
        for kind, _ in EXPORT_FILE_TYPES:
            st.session_state[f"export_include_{kind}"] = kind in available_kinds

    if len(methods) > 1:
        chosen_methods = st.multiselect(
            "Methods to download",
            options=list(methods),
            format_func=SEGMENTATION_METHOD_LABELS.get,
            key=EXPORT_METHODS_KEY,
            persist_state="page",
        )
    else:
        chosen_methods = list(methods)
        st.caption(f"Method: {SEGMENTATION_METHOD_LABELS[methods[0]]}")

    st.markdown("**Files to include in ZIP**")
    st.caption("These choices package existing files. Image output settings for the next run are in Setup.")
    left, right = st.columns(2)
    selected_kinds = []
    for index, (kind, label) in enumerate(EXPORT_FILE_TYPES):
        count = sum(
            item["kind"] == kind and
            (item.get("method") is None or item["method"] in chosen_methods)
            for item in available
        )
        ever_saved = any(item["kind"] == kind for item in available)
        with (left if index < 5 else right):
            checked = st.checkbox(
                f"{label} · {count} file(s)",
                key=f"export_include_{kind}",
                disabled=not ever_saved,
                persist_state="page",
            )
        if checked and count:
            selected_kinds.append(kind)
    st.caption(
        "Unavailable types were not saved in this run. Image types can be enabled "
        "in Setup before the next analysis."
    )

    selected_files = select_export_files(available, chosen_methods, selected_kinds)
    count, total_bytes = estimate_zip_inputs(selected_files)
    st.markdown("**Download**")
    csv_files = select_export_files(available, chosen_methods, ("results_csv",))
    with st.container(border=True):
        st.markdown("**Measurement CSVs**")
        st.caption("Download a CSV directly, regardless of the ZIP file choices above.")
        if csv_files:
            for path in csv_files:
                with open(path, "rb") as csv_file:
                    st.download_button(
                        f"Download {path.name}",
                        data=csv_file,
                        file_name=path.name,
                        mime="text/csv",
                        icon=":material/download:",
                        key=f"download_csv_{path.name}",
                    )
        else:
            st.caption("No results CSV is available for the selected method(s).")

    with st.container(border=True):
        st.markdown("**Selected files (ZIP)**")
        raw_name = st.text_input(
            "ZIP filename",
            key=EXPORT_ZIP_NAME_KEY,
            max_chars=120,
            persist_state="page",
        )
        download_name = zip_download_name(raw_name)
        st.caption(f"{count:,} file(s) · {human_bytes(total_bytes)} uncompressed")
        with st.expander(f"View files in ZIP ({count:,})"):
            for path in selected_files[:200]:
                st.write(path.name)
            if count > 200:
                st.caption(f"Showing the first 200 of {count:,} files.")
        if total_bytes > ZIP_SIZE_WARN_BYTES:
            st.warning(
                f"Outputs total ~{human_bytes(total_bytes)}. Building a ZIP this large can be "
                f"slow and memory-heavy. Consider collecting files directly from "
                f"`{display_path(output_path)}` instead."
            )

        try:
            signature = _export_signature(run_token, selected_files, download_name)
        except OSError:
            st.warning("A selected file changed or disappeared. Refresh Export and try again.")
            _clear_export_zip_cache()
            return
        if st.session_state.get("export_zip_signature") != signature:
            _clear_export_zip_cache()
        if st.button("Prepare ZIP for download", icon=":material/folder_zip:",
                     key="prepare_zip_export", disabled=not selected_files):
            dest = None
            try:
                with st.spinner("Building ZIP..."):
                    with tempfile.NamedTemporaryFile(
                        prefix="mats_outputs_", suffix=".zip", delete=False
                    ) as temp_zip:
                        dest = Path(temp_zip.name)
                    write_output_zip(selected_files, dest)
            except OSError as exc:
                if dest is not None:
                    dest.unlink(missing_ok=True)
                st.error(f"Could not prepare the ZIP: {exc}")
            else:
                st.session_state["export_zip_path"] = str(dest)
                st.session_state["export_zip_signature"] = signature

        zip_path = st.session_state.get("export_zip_path")
        if zip_path and Path(zip_path).is_file():
            with open(zip_path, "rb") as zip_file:
                st.download_button(
                    f"Download ZIP ({count:,} files)",
                    data=zip_file,
                    file_name=download_name,
                    mime="application/zip",
                    icon=":material/download:",
                    key="download_zip_export",
                )


if __name__ == "__main__":
    main()
