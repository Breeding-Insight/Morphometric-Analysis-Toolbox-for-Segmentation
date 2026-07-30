import csv
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import streamlit as st

from mats.app import branding
from mats.app.compute import (
    birefnet_parallel_unlocked,
    break_glass_unlocked,
    get_compute_settings,
    reengage_gpu,
    resolve_execution_plan,
    selected_mask_method,
)
from mats.template_layout import (
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


def read_csv_preview(results_path, limit=200):
    rows = []
    with open(results_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append(row)
    return rows


def count_csv_rows(results_path):
    with open(results_path, newline="") as csvfile:
        return sum(1 for _ in csv.DictReader(csvfile))


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


QR_MODE_KEY = "measure_use_qr"
UNIT_KEY = "measure_unit"
PREVIOUS_UNIT_KEY = "measure_previous_unit"
WIDTH_KEY = "measure_width"
HEIGHT_KEY = "measure_height"
DIMENSIONS_TEXT_KEY = "measure_dimensions_text"
MANUAL_TEXT_CACHE_KEY = "measure_manual_dimensions_cache"
QR_NOTICE_KEY = "measure_qr_extra_notice_pending"


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
    """Swap the dimensions bar to a QR placeholder, caching the manual entry."""
    if st.session_state[QR_MODE_KEY]:
        st.session_state[QR_NOTICE_KEY] = True
        st.session_state[MANUAL_TEXT_CACHE_KEY] = st.session_state[DIMENSIONS_TEXT_KEY]
        st.session_state[DIMENSIONS_TEXT_KEY] = "Variable dimensions — QR-derived"
    else:
        st.session_state.pop(QR_NOTICE_KEY, None)
        cached = st.session_state.pop(MANUAL_TEXT_CACHE_KEY, None)
        st.session_state[DIMENSIONS_TEXT_KEY] = cached or _compose_dimensions_text()


@st.dialog(
    "Enhanced QR reading is not installed",
    width="medium",
    icon=":material/qr_code_scanner:",
    dismissible=False,
)
def warn_enhanced_qr_missing():
    st.warning(
        "Only OpenCV's QR decoder is available. It misses codes with glare, skew, "
        "or blur, and those images fail with no measurement scale."
    )
    st.markdown(
        'Install the optional extra:\n\n```\npip install "mats-morpho[qr]"\n```\n\n'
        "`pyzbar` also needs the system `zbar` library:\n\n"
        "- Linux: `apt install libzbar0`\n"
        "- macOS: `brew install zbar`\n"
        "- conda: `conda install -c conda-forge zbar`\n\n"
        "Restart the app after installing. You can also untick **Variable "
        "dimensions** and enter width and height manually."
    )
    if st.button("Continue with OpenCV only", type="primary", width="stretch"):
        st.session_state[QR_NOTICE_KEY] = False
        st.rerun()


def main():
    st.set_page_config(
        page_title="MATS — Morphometric Analysis Toolbox for Segmentation",
        page_icon=branding.page_icon(),
        layout="wide",
    )
    branding.apply_logo()
    # The viewer is session-scoped. Unlike the selected output directory, this
    # manifest is discarded when the browser session closes and is never rebuilt
    # by scanning artifacts left by earlier sessions.
    st.session_state.setdefault("viewer_pairs", [])
    st.session_state.setdefault(UNIT_KEY, "in")
    st.session_state.setdefault(PREVIOUS_UNIT_KEY, "in")
    st.session_state.setdefault(WIDTH_KEY, 12.0)
    st.session_state.setdefault(HEIGHT_KEY, 12.0)
    st.session_state.setdefault(DIMENSIONS_TEXT_KEY, "12x12in")
    st.session_state.setdefault(QR_MODE_KEY, False)
    st.session_state.setdefault("segmentation_method", "Classic thresholding (Otsu)")

    st.title("Morphometric Analysis Toolbox for Segmentation")
    st.caption("RF-DETR marker detection with Otsu threshold or BiRefNet segmentation.")

    with st.expander("Usage and setup"):
        st.markdown(
            "- **Launch**: `mats app` (from an environment where MATS is installed).\n"
            "- **Checkpoints**: `mats fetch-weights` fetches RF-DETR (mandatory) by default; "
            "add `--only birefnet` or `--all` to also fetch BiRefNet. Files land in "
            "`~/.cache/mats/weights`; override the location with `MATS_WEIGHTS_DIR`, "
            "or point `RF_DETR_MARKER_CHECKPOINT` / `BIREFNET_CHECKPOINT` at specific files. "
            "Run `mats doctor` to check they resolve.\n"
            "- **Outputs**: per image `{sample_id}_target_box.jpg` and `{sample_id}_mask.png`, "
            "plus `leaf_morpho_results.csv` (and `leaf_morpho_failures.csv` when enabled). "
            "Choose the **Full research schema** to match the downstream analysis scripts.\n"
            "- **HPC**: an Open OnDemand wrapper lives under `deploy/ondemand/`."
        )

    try:
        lm = load_pipeline_module()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        st.error(f"Failed to import the MATS pipeline: missing dependency `{missing}`.")
        st.info(
            "The Python environment running Streamlit is missing a pipeline "
            f"dependency (`{missing}`). Install MATS with its dependencies "
            "(rfdetr, transformers, torch, opencv) and relaunch:\n\n"
            "```\npip install -e .\nmats app\n```"
        )
        st.caption(f"Current interpreter: {sys.executable}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to import the MATS pipeline: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.caption(f"Current interpreter: {sys.executable}")
        st.stop()

    mask_method = selected_mask_method()
    compute_settings = get_compute_settings(lm)
    available_workers = compute_settings.available_workers
    birefnet_parallel_is_unlocked = birefnet_parallel_unlocked(available_workers)
    execution_plan = resolve_execution_plan(
        compute_settings,
        mask_method,
        birefnet_parallel_allowed=birefnet_parallel_is_unlocked,
    )
    workers = execution_plan.workers
    worker_risk = lm.worker_risk_report(workers, available_workers)
    break_glass_is_unlocked = break_glass_unlocked(available_workers)
    execution_device = execution_plan.execution_device

    with st.container(border=True):
        st.subheader("Compute status")
        st.badge(
            execution_plan.label,
            icon=":material/check_circle:" if execution_plan.uses_gpu else ":material/memory:",
            color="green" if execution_plan.uses_gpu else "orange",
        )
        st.caption(execution_plan.detail)
        if mask_method == "birefnet" and not execution_plan.uses_gpu:
            gpu_notice = (
                "GPU available but disabled. BiRefNet is running on CPU because an advanced "
                "CPU-only override is active."
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
            label="Open compute options",
            icon=":material/tune:",
            width="content",
        )

    with st.sidebar:
        st.header("Inputs")
        input_source = st.radio("Image source", ["Local folder", "Upload images"])
        uploaded_files = []
        input_dir = ""
        if input_source == "Local folder":
            input_dir = st.text_input("Input folder", value=str(DEFAULT_INPUT_DIR))
        else:
            uploaded_files = st.file_uploader(
                "Upload images",
                type=sorted(ext.lstrip(".") for ext in VALID_EXTENSIONS),
                accept_multiple_files=True,
            )

        st.header("Outputs")
        output_dir = st.text_input("Output folder", value=str(DEFAULT_OUTPUT_DIR))

        st.header("Measurement")
        use_qr = st.checkbox(
            "Variable dimensions, read QR code",
            key=QR_MODE_KEY,
            on_change=_qr_mode_changed,
            help="Each image carries its own template QR code. Leave unticked to "
                 "measure every image against one manually entered sheet size.",
            persist_state="page",
        )

        st.text_input(
            "Template dimensions",
            key=DIMENSIONS_TEXT_KEY,
            disabled=use_qr,
            help=(
                'Exact format: "<width>x<height><unit>" — e.g. "10.5x9.5in" or '
                '"27x24cm". Only "in" or "cm" are accepted. Type a size directly '
                "here for a custom template that doesn't land on the Width/Height "
                "selectors' 0.5 grid below."
            ),
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
                "Width",
                min_value=measure_minimum,
                max_value=measure_maximum,
                step=0.5,
                format="%.1f",
                key=WIDTH_KEY,
                disabled=use_qr,
                icon=":material/width:",
                help="Horizontal size of the printed sheet, snapped to the nearest 0.5.",
                on_change=_measure_dimensions_changed,
                persist_state="page",
            )
            st.number_input(
                "Height",
                min_value=measure_minimum,
                max_value=measure_maximum,
                step=0.5,
                format="%.1f",
                key=HEIGHT_KEY,
                disabled=use_qr,
                icon=":material/height:",
                help="Vertical size of the printed sheet, snapped to the nearest 0.5.",
                on_change=_measure_dimensions_changed,
                persist_state="page",
            )
        segmentation_label = st.selectbox(
            "Segmentation method",
            ["Classic thresholding (Otsu)", "BiRefNet"],
            key="segmentation_method",
            persist_state="session",
        )
        mask_method = "threshold" if segmentation_label == "Classic thresholding (Otsu)" else "birefnet"

        threshold_level = "auto"
        if mask_method == "threshold":
            threshold_level = st.selectbox(
                "Threshold level",
                list(lm.THRESHOLD_LEVELS.keys()),
                index=list(lm.THRESHOLD_LEVELS.keys()).index("auto"),
                help="auto = Otsu (adapts per image). low/medium/high = fixed 100/125/150.",
            )

        st.header("Output options")
        schema_label = st.selectbox(
            "Results CSV schema",
            ["Full research schema (*_meanscale)", "Compact (sample_id, area, height, length)"],
            help=(
                "Full matches the columns the downstream MATS analysis scripts expect. "
                "Compact is the trimmed UI export."
            ),
        )
        compact_csv = schema_label.startswith("Compact")
        write_failures = st.checkbox(
            "Write failures log (leaf_morpho_failures.csv)",
            value=True,
            help="Per-image failure/warning report used by the failure-taxonomy analysis.",
        )

    if st.session_state.get(QR_NOTICE_KEY) and not lm._enhanced_qr_available():
        warn_enhanced_qr_missing()

    threshold_value = lm.THRESHOLD_LEVELS[threshold_level] if mask_method == "threshold" else lm.BIREFNET_THRESHOLD
    output_path = Path(output_dir).expanduser()
    results_path = output_path / "leaf_morpho_results.csv"

    template_error = None
    if use_qr:
        template_dimensions = None
    else:
        dimensions_text = st.session_state[DIMENSIONS_TEXT_KEY].strip()
        template_dimensions = lm.parse_template_dimensions(dimensions_text)
        if template_dimensions is None:
            template_error = 'Template dimensions must look like "10.5x9.5in" or "27x24cm".'

    st.subheader("Preflight")
    checks = []
    checks.append(("Pipeline library", "success", False, f"mats.core ({Path(lm.__file__).parent})", None))

    from mats import weights
    from mats.devices import birefnet_device_report

    rfdetr_status = weights.get_weight_status("rf-detr")
    rfdetr_level = "success" if rfdetr_status.state == "ready" else "error"
    checks.append(("RF-DETR checkpoint", rfdetr_level, rfdetr_level == "error", rfdetr_status.detail, None))

    birefnet_status = weights.get_weight_status("birefnet")
    birefnet_level = "success" if birefnet_status.state == "ready" else "warning"
    checks.append((
        "BiRefNet checkpoint", birefnet_level, mask_method == "birefnet" and birefnet_status.state != "ready",
        birefnet_status.detail if birefnet_status.state != "ready" else f"Installed at {birefnet_status.path}",
        "pages/2_BiRefNet_Setup.py" if birefnet_status.state != "ready" else None,
    ))

    device_report = birefnet_device_report()
    if worker_risk.requires_break_glass:
        checks.append((
            "Worker safety", "error", not break_glass_is_unlocked,
            (
                f"Critical load: {workers}/{available_workers} workers "
                f"({worker_risk.utilization:.0%}). Break-glass acknowledgement "
                "is required for this run."
            ), None,
        ))
    else:
        checks.append((
            "Worker safety", "success", False,
            f"{worker_risk.label}: {workers}/{available_workers} workers "
            f"({worker_risk.utilization:.0%}).", None,
        ))
    if execution_device == "hybrid":
        checks.append((
            "Execution mode", "success", False,
            f"Hybrid mode: RF-DETR on GPU with Otsu and measurements across {workers} CPU workers.", None,
        ))
    elif execution_device == "cpu":
        checks.append((
            "Execution mode", "warning", False,
            f"CPU mode with {workers} worker(s); CUDA/MPS is disabled for this run.", None,
        ))
    else:
        checks.append((
            "Execution mode", "success", False,
            "Serial GPU inference; accelerator selection is automatic.", None,
        ))
    if mask_method == "birefnet" and execution_device == "cpu":
        birefnet_cpu_detail = (
            "GPU is available but disabled by an advanced CPU-only override."
            if compute_settings.accelerator_available
            else "No usable CUDA or Apple MPS accelerator was detected."
        )
        checks.append((
            "BiRefNet performance", "warning", False,
            f"BiRefNet is using CPU. {birefnet_cpu_detail}", None,
        ))
    checks.append((
        "BiRefNet compute", device_report.severity,
        mask_method == "birefnet" and device_report.severity == "error",
        device_report.detail, None,
    ))
    if use_qr:
        template_level = "success" if lm._enhanced_qr_available() else "warning"
        template_detail = "Variable — read from each image's QR code"
        if template_level == "warning":
            template_detail += ' (OpenCV only; install "mats-morpho[qr]" for tougher codes)'
        template_blocking = False
    elif template_error is not None:
        template_level, template_detail, template_blocking = "error", template_error, True
    else:
        template_level, template_detail, template_blocking = "success", f"Manual: {dimensions_text}", False
    checks.append(("Template dimensions", template_level, template_blocking, template_detail, None))

    if input_source == "Local folder":
        image_paths = collect_folder_images(input_dir)
        checks.append(("Input images", "success" if image_paths else "error", not bool(image_paths),
                       f"{len(image_paths)} image(s) found", None))
    else:
        image_paths = []
        valid_upload_count = sum(
            1 for uploaded_file in uploaded_files
            if Path(uploaded_file.name).suffix.lower() in VALID_EXTENSIONS
        )
        checks.append(("Uploaded images", "success" if valid_upload_count else "error", not bool(valid_upload_count),
                       f"{valid_upload_count} valid image(s) selected", None))

    for label, severity, blocking, detail, setup_page in checks:
        getattr(st, severity)(f"{label}: {detail}")
        if setup_page:
            st.page_link(
                setup_page,
                label="Open BiRefNet setup",
                icon=":material/download:",
                width="content",
            )

    ready = not any(blocking for _, _, blocking, _, _ in checks)

    image_count = len(image_paths) if input_source == "Local folder" else valid_upload_count
    large_batch_ok = True
    if ready and image_count > LARGE_BATCH_THRESHOLD:
        st.warning(
            f"This is a large batch ({image_count} images). Processing runs synchronously "
            "and may take a long time; the browser tab must stay open."
        )
        large_batch_ok = st.checkbox(
            f"I understand and want to process {image_count} images",
            value=False,
        )

    run_clicked = st.button(
        "Run Leaf Morphometrics",
        type="primary",
        disabled=not (ready and large_batch_ok),
    )
    if run_clicked:
        output_path.mkdir(parents=True, exist_ok=True)
        run_sample_ids = []
        succeeded_so_far = 0
        break_glass_acknowledged = break_glass_is_unlocked
        birefnet_parallel_acknowledged = birefnet_parallel_is_unlocked
        if worker_risk.requires_break_glass:
            # An acknowledgement authorizes one high-risk run only.
            st.session_state.pop("break_glass_available_workers", None)
        if mask_method == "birefnet" and execution_device == "cpu" and workers > 1:
            # CPU-parallel BiRefNet needs its own acknowledgement even at a
            # non-critical CPU count.
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
                    if lm.is_target_box_image(current_image):
                        sample_id = lm.target_box_sample_id(current_image)
                    else:
                        sample_id = Path(current_image).stem
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
                        template_dimensions=template_dimensions,
                        output_mode="masks",
                        mask_method=mask_method,
                        threshold_value=threshold_value,
                        workers=int(workers),
                        execution_device=execution_device,
                        worker_safety_check=True,
                        break_glass_acknowledged=break_glass_acknowledged,
                        birefnet_parallel_acknowledged=birefnet_parallel_acknowledged,
                        progress_callback=update_progress,
                        write_failures=write_failures,
                        compact_csv=compact_csv,
                        save_measurement_axes=False,
                    )
            except ValueError as exc:
                st.error(f"Run prevented by worker safety checks: {exc}")
                return

        # Persist only what the results view needs; keep large lists out of session
        # state so the page stays light after big runs.
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
            "mask_method": mask_method,
        }
        run_pairs = collect_output_pairs(output_path, run_sample_ids)
        st.session_state["viewer_pairs"] = merge_viewer_pairs(
            st.session_state["viewer_pairs"],
            run_pairs,
        )
        # A fresh run invalidates any previously prepared ZIP.
        st.session_state.pop("export_zip_path", None)

    render_results(lm)


def render_results(lm):
    run = st.session_state.get("last_run")
    if not run:
        return

    results_path = Path(run["results_path"])
    output_path = Path(run["output_path"])

    st.subheader("Results")
    st.success(
        f"Done. {run['succeeded']} succeeded, {run['failed']} failed "
        f"(of {run['total']}). Results written to `{results_path}`."
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
                st.write(f"`{row['sample_id']}`: {row['status']}")
            if run["failure_overflow"]:
                st.write(f"...and {run['failure_overflow']} more (see the failures CSV).")

    if results_path.is_file():
        total_rows = count_csv_rows(results_path)
        st.markdown("**Measurements**")
        preview_rows = read_csv_preview(results_path, limit=200)
        st.caption(f"Showing {len(preview_rows)} of {total_rows} row(s).")
        st.dataframe(preview_rows, width="stretch")
        st.download_button(
            "Download results CSV",
            data=results_path.read_bytes(),
            file_name=results_path.name,
            mime="text/csv",
        )

    render_output_preview(st.session_state.get("viewer_pairs", []))
    render_zip_export(results_path, output_path)


def render_output_preview(pairs):
    if not pairs:
        return

    st.markdown("**Output preview**")
    show_default = len(pairs) <= PREVIEW_AUTO_HIDE_THRESHOLD
    show_preview = st.checkbox(
        f"Show image preview ({len(pairs)} output(s))",
        value=show_default,
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
        )
    st.caption(f"Showing {count} of {len(pairs)} output pair(s).")

    for pair in pairs[:count]:
        st.markdown(f"`{pair['sample_id']}`")
        left, right = st.columns(2)
        with left:
            if pair["target_box"] is not None:
                st.image(str(pair["target_box"]), caption="target box", width=PREVIEW_IMAGE_WIDTH)
            else:
                st.caption("target box: n/a")
        with right:
            st.image(str(pair["mask"]), caption="mask", width=PREVIEW_IMAGE_WIDTH)


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
            f"slow and memory-heavy. Consider collecting files directly from `{output_path}` "
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
