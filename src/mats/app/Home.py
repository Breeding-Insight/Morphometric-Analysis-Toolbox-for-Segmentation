import csv
import tempfile
import traceback
import zipfile
from pathlib import Path

import altair as alt
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
from mats.scaling import QR_TRACE_FIELDNAMES
from mats.template_layout import (
    TemplateLayoutError,
    build_template_layout,
    format_measurement,
    maximum_template_edge,
    minimum_template_edge,
    round_to_increment,
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
SEGMENTATION_METHOD_KEY = "segmentation_method"
THRESHOLD_LEVEL_KEY = "threshold_level"
RESULTS_SCHEMA_KEY = "results_schema"
WRITE_FAILURES_KEY = "write_failures"
WORKSPACE_TAB_KEY = "analysis_workspace_tab"
PENDING_WORKSPACE_TAB_KEY = "pending_analysis_workspace_tab"
RESULTS_DASHBOARD_MAX_ROWS = 5_000
RESULTS_TABLE_MAX_ROWS = 1_000

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
PREVIEW_HARD_MAX = 24
PREVIEW_AUTO_HIDE_THRESHOLD = 50
PREVIEW_IMAGE_WIDTH = 280
LARGE_BATCH_THRESHOLD = 200


def gather_output_files(output_dir, results_path):
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
    return files


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


def normalize_measurements(frame):
    """Return numeric, consistently named measurements from either CSV schema."""
    measurements = frame.copy()
    if "leaf_area_cm2" not in measurements and "area_cm2" in measurements:
        measurements["leaf_area_cm2"] = measurements["area_cm2"]

    required_columns = ("sample_id", "leaf_area_cm2", "width_cm", "length_cm")
    if any(column not in measurements for column in required_columns):
        return pd.DataFrame(columns=required_columns)

    for column in ("leaf_area_cm2", "width_cm", "length_cm", "scale_aspect_ratio"):
        if column in measurements:
            measurements[column] = pd.to_numeric(measurements[column], errors="coerce")
    return measurements.dropna(subset=("leaf_area_cm2", "width_cm", "length_cm"))


def summarize_measurements(measurements):
    """Return dashboard-ready robust summary statistics for leaf measurements."""
    if measurements.empty:
        return None
    return {
        "count": len(measurements),
        "median_area_cm2": measurements["leaf_area_cm2"].median(),
        "median_width_cm": measurements["width_cm"].median(),
        "median_length_cm": measurements["length_cm"].median(),
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


def _open_diagnostics():
    """Open the detailed preflight tab on the next Streamlit rerun."""
    st.session_state[WORKSPACE_TAB_KEY] = "Diagnostics"


def _switch_workspace_tab(tab_name):
    """Open a named workspace tab through its stateful Streamlit key."""
    st.session_state[WORKSPACE_TAB_KEY] = tab_name


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
            "Move between analysis setup, measurement results, and system diagnostics."
        )
    return st.tabs(
        ["Analyze", "Results", "Diagnostics"],
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
                ":material/contrast:"
                if config["mask_method"] == "threshold"
                else ":material/auto_awesome:"
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

        st.caption("Scale, segmentation, and export settings are in Analyze.")

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
        st.caption("Choose the mask method best suited to the image background.")
        segmentation_label = st.radio(
            "Select a segmentation method",
            ["Classic thresholding (Otsu)", "BiRefNet"],
            captions=[
                "Fast and dependable for clean, well-lit backgrounds.",
                "Better on cluttered backgrounds; needs the optional local checkpoint "
                "and benefits from a GPU.",
            ],
            key=SEGMENTATION_METHOD_KEY,
            persist_state="session",
            width="stretch",
        )
        if segmentation_label == "Classic thresholding (Otsu)":
            st.selectbox(
                "Threshold level",
                list(lm.THRESHOLD_LEVELS.keys()),
                key=THRESHOLD_LEVEL_KEY,
                help="auto = Otsu (adapts per image). low/medium/high = fixed 100/125/150.",
                persist_state="page",
            )
        else:
            st.caption("BiRefNet uses its calibrated confidence threshold automatically.")

    with st.container(border=True):
        st.markdown("**3 · Measurement Output**")
        st.caption("Choose the export schema and whether to keep a per-image failure report.")
        schema_column, failures_column = st.columns((1.35, 0.65), vertical_alignment="bottom")
        with schema_column:
            st.selectbox(
                "Results CSV schema",
                [
                    "Full research schema (area/width/length + px-per-cm)",
                    "Compact (sample_id, area, width, length)",
                ],
                key=RESULTS_SCHEMA_KEY,
                help=(
                    "Full adds px_per_cm_width, px_per_cm_height, and scale_aspect_ratio "
                    "for QC. Compact is the trimmed UI export."
                ),
                persist_state="page",
            )
        with failures_column:
            st.toggle(
                "Write Failure Log",
                key=WRITE_FAILURES_KEY,
                help="Per-image failure/warning report used by the failure-taxonomy analysis.",
                persist_state="page",
            )
        st.caption(
            "Results, visual summaries, image previews, and downloads appear in the "
            "Results tab after a run."
        )


def current_analysis_config(lm):
    """Resolve the persisted controls into the values needed for one batch."""
    segmentation_label = st.session_state[SEGMENTATION_METHOD_KEY]
    mask_method = (
        "threshold"
        if segmentation_label == "Classic thresholding (Otsu)"
        else "birefnet"
    )
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
        "mask_method": mask_method,
        "threshold_value": (
            lm.THRESHOLD_LEVELS[threshold_level]
            if mask_method == "threshold"
            else lm.BIREFNET_THRESHOLD
        ),
        "compact_csv": st.session_state[RESULTS_SCHEMA_KEY].startswith("Compact"),
        "write_failures": st.session_state[WRITE_FAILURES_KEY],
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

    if config["mask_method"] == "birefnet":
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
        config["mask_method"] == "birefnet" and birefnet_status.state != "ready",
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

    if config["mask_method"] == "birefnet" and execution_plan.execution_device == "cpu":
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
        config["mask_method"] == "birefnet" and device_report.severity == "error",
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
        st.markdown("**4 · Preflight**")
        if blockers:
            st.badge(
                f"{len(blockers)} blocking item(s)",
                icon=":material/error:",
                color="red",
            )
            st.caption(
                "Needs attention: " + ", ".join(check[0] for check in blockers) + "."
            )
            st.button(
                "Open Diagnostics",
                key="open_diagnostics",
                icon=":material/troubleshoot:",
                on_click=_open_diagnostics,
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
        if config["mask_method"] == "birefnet" and not execution_plan.uses_gpu:
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
    run_sample_ids = []
    succeeded_so_far = 0
    if worker_risk.requires_break_glass:
        st.session_state.pop("break_glass_available_workers", None)
    if (
        config["mask_method"] == "birefnet"
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
            nonlocal succeeded_so_far
            if status["succeeded"] > succeeded_so_far:
                current_image = status["current_image"]
                sample_id = (
                    lm.target_box_sample_id(current_image)
                    if lm.is_target_box_image(current_image)
                    else Path(current_image).stem
                )
                run_sample_ids.append(sample_id)
            succeeded_so_far = status["succeeded"]

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
                    mask_method=config["mask_method"],
                    threshold_value=config["threshold_value"],
                    workers=int(execution_plan.workers),
                    execution_device=execution_plan.execution_device,
                    worker_safety_check=True,
                    break_glass_acknowledged=break_glass_is_unlocked,
                    birefnet_parallel_acknowledged=birefnet_parallel_is_unlocked,
                    progress_callback=update_progress,
                    write_failures=config["write_failures"],
                    compact_csv=config["compact_csv"],
                    save_measurement_axes=False,
                )
        except ValueError as exc:
            st.error(f"Run prevented by worker safety checks: {exc}")
            return None, []

    return summary, collect_output_pairs(output_path, run_sample_ids)


def main():
    st.set_page_config(
        page_title="MATS — Morphometric Analysis Toolbox for Segmentation",
        page_icon=branding.page_icon(),
        layout="wide",
    )
    branding.apply_logo()
    st.html(_WORKBENCH_STYLES)
    st.session_state.setdefault("viewer_pairs", [])
    st.session_state.setdefault(UNIT_KEY, "in")
    st.session_state.setdefault(PREVIOUS_UNIT_KEY, "in")
    st.session_state.setdefault(WIDTH_KEY, 12.0)
    st.session_state.setdefault(HEIGHT_KEY, 12.0)
    st.session_state.setdefault(DIMENSIONS_TEXT_KEY, "12x12in")
    st.session_state.setdefault(QR_MODE_KEY, False)
    st.session_state.setdefault(USE_LEGACY_DIMENSIONS_KEY, False)
    st.session_state.setdefault(LEGACY_DIMENSIONS_TEXT_KEY, "10x9.5in")
    st.session_state.setdefault(SEGMENTATION_METHOD_KEY, "Classic thresholding (Otsu)")
    st.session_state.setdefault(THRESHOLD_LEVEL_KEY, "auto")
    st.session_state.setdefault(
        RESULTS_SCHEMA_KEY,
        "Full research schema (area/width/length + px-per-cm)",
    )
    st.session_state.setdefault(WRITE_FAILURES_KEY, True)
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
            "segmentation, and export options in Analyze. Results contains measurement "
            "visuals and downloads; Diagnostics contains compute and preflight details. "
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
    analyze_tab, results_tab, diagnostics_tab = render_workspace_navigation()

    if analyze_tab.open:
        with analyze_tab:
            render_analysis_settings(lm)

    config = current_analysis_config(lm)
    compute_settings = get_compute_settings(lm)
    available_workers = compute_settings.available_workers
    birefnet_parallel_is_unlocked = birefnet_parallel_unlocked(available_workers)
    execution_plan = resolve_execution_plan(
        compute_settings,
        config["mask_method"],
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

    run_clicked = False
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
    elif results_tab.open:
        with results_tab:
            render_results(lm)
    elif diagnostics_tab.open:
        with diagnostics_tab:
            render_diagnostics(compute_settings, execution_plan, config, checks)

    if run_clicked:
        summary, run_pairs = execute_leaf_analysis(
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
        )
        if summary is not None:
            st.session_state["last_run"] = {
                "succeeded": summary["succeeded"],
                "failed": summary["failed"],
                "total": summary["total"],
                "workers": summary["workers"],
                "worker_reason": summary["worker_reason"],
                "execution_device": summary["execution_device"],
                "failure_rows": summary["failure_rows"][:200],
                "failure_overflow": max(0, len(summary["failure_rows"]) - 200),
                "results_path": str(results_path),
                "output_path": str(output_path),
                "mask_method": config["mask_method"],
            }
            st.session_state["viewer_pairs"] = merge_viewer_pairs(
                st.session_state["viewer_pairs"],
                run_pairs,
            )
            st.session_state.pop("export_zip_path", None)
            st.session_state[PENDING_WORKSPACE_TAB_KEY] = "Results"
            st.rerun()


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

    results_path = Path(run["results_path"])
    output_path = Path(run["output_path"])
    st.subheader("Results", anchor=False)
    st.success(
        f"Completed {run['succeeded']} measurement(s); {run['failed']} failed "
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

    if run["failure_rows"]:
        with st.expander(f"Processing warnings and failures ({run['failed']})"):
            for row in run["failure_rows"]:
                st.write(f"{row['sample_id']}: {row['status']}")
            if run["failure_overflow"]:
                st.write(f"...and {run['failure_overflow']} more (see the failures CSV).")

    if not results_path.is_file():
        st.warning(
            f"The results CSV is not available at {display_path(results_path)}. The output preview "
            "and exports may still be available below.",
            icon=":material/folder_off:",
        )
        render_output_preview(st.session_state.get("viewer_pairs", []))
        render_zip_export(results_path, output_path)
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

    measurements = normalize_measurements(frame)
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
                f"{run['succeeded']:,}",
                delta=f"{run['failed']} failed",
                delta_color="inverse",
                border=True,
            )
            st.metric(
                "Median leaf area",
                f"{summary['median_area_cm2']:.2f} cm²",
                border=True,
            )
            st.metric(
                "Median leaf width",
                f"{summary['median_width_cm']:.2f} cm",
                border=True,
            )
            st.metric(
                "Median leaf length",
                f"{summary['median_length_cm']:.2f} cm",
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
                    x=alt.X("width_cm:Q", title="Leaf width (cm)"),
                    y=alt.Y("length_cm:Q", title="Leaf length (cm)"),
                    size=alt.Size(
                        "leaf_area_cm2:Q",
                        title="Leaf area (cm²)",
                        scale=alt.Scale(range=[30, 900]),
                    ),
                    color=alt.Color(
                        "leaf_area_cm2:Q",
                        title="Leaf area (cm²)",
                        scale=alt.Scale(scheme="blues"),
                    ),
                    tooltip=[
                        alt.Tooltip("sample_id:N", title="Sample"),
                        alt.Tooltip("leaf_area_cm2:Q", title="Area (cm²)", format=".2f"),
                        alt.Tooltip("width_cm:Q", title="Width (cm)", format=".2f"),
                        alt.Tooltip("length_cm:Q", title="Length (cm)", format=".2f"),
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
                        "leaf_area_cm2:Q",
                        bin=alt.Bin(maxbins=20),
                        title="Leaf area (cm²)",
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

        table_columns = ["sample_id", "leaf_area_cm2", "width_cm", "length_cm"]
        if "scale_aspect_ratio" in measurements:
            table_columns.append("scale_aspect_ratio")
        table_data = measurements.loc[:, table_columns].head(RESULTS_TABLE_MAX_ROWS).copy()
        output_pairs = st.session_state.get("viewer_pairs", [])
        table_column, inspector_column = st.columns((1.15, 0.85), vertical_alignment="top")
        with table_column.container(border=True):
            st.markdown("**Measurement Table**")
            st.caption("Select a row to inspect the matching target-box image and segmentation mask.")
            table_event = st.dataframe(
                table_data,
                column_config={
                    "sample_id": st.column_config.TextColumn("Sample", pinned=True),
                    "leaf_area_cm2": st.column_config.NumberColumn(
                        "Leaf area (cm²)",
                        format="%.2f",
                    ),
                    "width_cm": st.column_config.NumberColumn("Leaf width (cm)", format="%.2f"),
                    "length_cm": st.column_config.NumberColumn("Leaf length (cm)", format="%.2f"),
                    "scale_aspect_ratio": st.column_config.NumberColumn(
                        "Scale axis ratio",
                        format="%.3f",
                    ),
                },
                hide_index=True,
                height=440,
                key="measurement_table",
                on_select="rerun",
                selection_mode="single-row",
            )
            if total_rows > len(table_data):
                st.caption(
                    f"Showing the first {len(table_data):,} of {total_rows:,} rows. "
                    "Download the CSV for the full dataset."
                )

        selected_row = None
        if table_event.selection.rows:
            selected_row = table_data.iloc[table_event.selection.rows[0]]
        elif not table_data.empty:
            selected_row = table_data.iloc[0]
        selected_sample_id = (
            str(selected_row["sample_id"]) if selected_row is not None else None
        )
        with inspector_column:
            render_specimen_inspector(selected_sample_id, selected_row, output_pairs)
    st.download_button(
        "Download results CSV",
        data=results_path.read_bytes(),
        file_name=results_path.name,
        mime="text/csv",
        icon=":material/download:",
    )

    with st.expander("Browse Output Previews", icon=":material/photo_library:"):
        render_output_preview(st.session_state.get("viewer_pairs", []))
    render_zip_export(results_path, output_path)


def render_specimen_inspector(sample_id, measurement, pairs):
    """Render the selected specimen's measurements beside its generated artifacts."""
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
                st.metric("Leaf Area", f"{measurement['leaf_area_cm2']:.2f} cm²", border=True)
                st.metric("Leaf Width", f"{measurement['width_cm']:.2f} cm", border=True)
                st.metric("Leaf Length", f"{measurement['length_cm']:.2f} cm", border=True)
        pair = find_output_pair(pairs, sample_id)
        if pair is None:
            st.info(
                "No output pair is available in this browser session for this sample. "
                "Run the analysis here to inspect its target box and mask.",
                icon=":material/image_not_supported:",
            )
            return
        render_output_pair(pair, width="stretch")


def render_output_pair(pair, width=PREVIEW_IMAGE_WIDTH):
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
        st.image(
            str(pair["mask"]),
            caption="Leaf segmentation mask",
            width=width,
        )


def render_output_preview(pairs, selected_sample_id=None):
    if not pairs:
        return

    if selected_sample_id is not None:
        selected_pairs = [
            pair for pair in pairs
            if str(pair["sample_id"]) == selected_sample_id
        ]
        st.subheader("Selected output", anchor=False)
        if not selected_pairs:
            st.info(
                "No matched target-box and mask pair is available for the selected row.",
                icon=":material/image_not_supported:",
            )
            return
        render_output_pair(selected_pairs[0])
        return

    st.subheader("Output preview", anchor=False)
    show_default = len(pairs) <= PREVIEW_AUTO_HIDE_THRESHOLD
    show_preview = st.checkbox(
        f"Show image preview ({len(pairs)} output(s))",
        value=show_default,
        key="show_output_preview",
        help="Renders matched target-box / mask pairs. Hidden by default for large batches.",
    )
    if not show_preview:
        return

    max_count = min(PREVIEW_HARD_MAX, len(pairs))
    if len(pairs) == 1:
        count = 1
    else:
        count = st.slider(
            "Pairs to display",
            min_value=1,
            max_value=max_count,
            value=min(6, max_count),
            key="output_preview_count",
        )
    st.caption(f"Showing {count} of {len(pairs)} output pair(s).")
    for pair in pairs[:count]:
        render_output_pair(pair)


def render_zip_export(results_path, output_path):
    files = gather_output_files(output_path, results_path)
    if not files:
        return

    count, total_bytes = estimate_zip_inputs(files)
    st.markdown("**Export**")
    st.caption(f"{count} file(s), ~{human_bytes(total_bytes)} uncompressed.")

    if total_bytes > ZIP_SIZE_WARN_BYTES:
        st.warning(
            f"Outputs total ~{human_bytes(total_bytes)}. Building a ZIP this large can be "
            f"slow and memory-heavy. Consider collecting files directly from "
            f"`{display_path(output_path)}` "
            "instead."
        )

    if st.button("Prepare ZIP for download"):
        with st.spinner("Building ZIP..."):
            dest = Path(tempfile.gettempdir()) / "leaf_morpho_outputs.zip"
            write_output_zip(files, dest)
            st.session_state["export_zip_path"] = str(dest)

    zip_path = st.session_state.get("export_zip_path")
    if zip_path and Path(zip_path).is_file():
        with open(zip_path, "rb") as zf:
            st.download_button(
                "Download ZIP (target boxes, masks, CSV)",
                data=zf,
                file_name="leaf_morpho_outputs.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
