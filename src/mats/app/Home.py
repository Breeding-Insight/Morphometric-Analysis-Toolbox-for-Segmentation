import csv
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import streamlit as st

from mats.app import branding


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


@st.dialog(
    "Break glass: high-risk CPU allocation",
    width="medium",
    icon=":material/warning:",
    dismissible=False,
)
def confirm_high_risk_workers(available_workers):
    """Require an explicit, single-run acknowledgement for >75% CPU use."""
    dialog_nonce = st.session_state.get("break_glass_dialog_nonce", 0)
    st.error(
        f"You are unlocking up to {available_workers} workers. This can use more than "
        "75% of the CPU allocation and may freeze or crash the computer."
    )
    st.markdown(
        "- Save important work and close unnecessary applications first.\n"
        "- Model-backed runs can consume substantial memory and increase heat/fan activity.\n"
        "- Other users of this shared server may be affected.\n"
        "- CUDA/MPS remains disabled whenever more than one worker is selected."
    )
    saved_work = st.checkbox(
        "I have saved important work and closed unnecessary applications.",
        key=f"break_glass_saved_work_{dialog_nonce}",
    )
    understands_risk = st.checkbox(
        "I understand this run may freeze or crash the computer.",
        key=f"break_glass_understands_risk_{dialog_nonce}",
    )
    if st.button("Keep the safe worker limit", width="stretch"):
        st.session_state["break_glass_dialog_nonce"] = dialog_nonce + 1
        st.rerun()
    if st.button(
        "Break the glass for one run",
        type="primary",
        icon=":material/warning:",
        width="stretch",
        disabled=not (saved_work and understands_risk),
    ):
        st.session_state["break_glass_available_workers"] = available_workers
        st.session_state["break_glass_dialog_nonce"] = dialog_nonce + 1
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
        template_dimensions_text = st.text_input(
            "Template dimensions",
            value="",
            placeholder="10.5x9.5in or 27x24cm",
            help="Leave blank to use the QR fallback when available.",
        )
        segmentation_label = st.selectbox(
            "Segmentation method",
            ["Classic thresholding (Otsu)", "BiRefNet"],
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

        st.header("Bulk Processing")
        if input_source == "Local folder":
            candidate_images = collect_folder_images(input_dir)
        else:
            candidate_images = [
                f.name for f in uploaded_files
                if Path(f.name).suffix.lower() in VALID_EXTENSIONS
            ]
        default_workers, worker_reason = lm.default_worker_count(
            candidate_images if input_source == "Local folder" else [],
            "masks",
            mask_method,
        )
        available_workers = lm.available_cpu_workers()
        worker_capacity = min(available_workers, max(1, len(candidate_images)))
        risk_at_one_worker = lm.worker_risk_report(1, available_workers)
        safe_worker_max = min(worker_capacity, risk_at_one_worker.normal_limit)
        break_glass_key = "break_glass_available_workers"
        if st.session_state.get(break_glass_key) != available_workers:
            st.session_state.pop(break_glass_key, None)
        break_glass_unlocked = st.session_state.get(break_glass_key) == available_workers
        max_workers = worker_capacity if break_glass_unlocked else safe_worker_max
        default_workers = min(max_workers, max(1, int(default_workers)))
        worker_key = "batch_worker_count"
        if worker_key not in st.session_state:
            st.session_state[worker_key] = default_workers
        elif st.session_state[worker_key] > max_workers:
            st.session_state[worker_key] = max_workers

        workers = int(st.number_input(
            "Workers",
            min_value=1,
            max_value=max_workers,
            step=1,
            key=worker_key,
            help=(
                f"{available_workers} CPU worker(s) are available to this app. "
                f"Default: {worker_reason}. Choosing more than one worker runs "
                "RF-DETR and BiRefNet on CPU for this run."
            ),
        ))
        worker_risk = lm.worker_risk_report(workers, available_workers)
        st.badge(
            f"{worker_risk.label} · {workers}/{available_workers} workers · "
            f"{worker_risk.utilization:.0%}",
            icon=(
                ":material/check_circle:" if worker_risk.level == "low"
                else ":material/warning:"
            ),
            color=worker_risk.color,
            width="stretch",
            help=(
                "Green: 25% or less. Yellow: 26–50%. Red: 51–75%. "
                "Counts above 75% require a one-run break-glass acknowledgement."
            ),
        )
        if worker_risk.requires_break_glass:
            st.error(
                "Break-glass mode is active. This worker count may make the computer "
                "unresponsive or crash; save work before running."
            )
        elif worker_capacity > safe_worker_max and not break_glass_unlocked:
            if st.button(
                "Unlock high-risk worker counts",
                icon=":material/warning:",
                width="stretch",
            ):
                confirm_high_risk_workers(available_workers)
        elif break_glass_unlocked:
            st.caption("Break-glass unlock is valid for one high-risk run.")

        execution_device = "cpu" if workers > 1 else "auto"
        if execution_device == "cpu":
            st.warning(
                f"Parallel CPU mode: {workers} workers selected. CUDA/MPS is disabled "
                "for RF-DETR and BiRefNet during this run."
            )
        else:
            st.caption("Single-worker mode: CUDA/MPS will be used when available.")

    threshold_value = lm.THRESHOLD_LEVELS[threshold_level] if mask_method == "threshold" else lm.BIREFNET_THRESHOLD
    output_path = Path(output_dir).expanduser()
    results_path = output_path / "leaf_morpho_results.csv"

    template_dimensions = None
    template_error = None
    if template_dimensions_text.strip():
        template_dimensions = lm.parse_template_dimensions(template_dimensions_text.strip())
        if template_dimensions is None:
            template_error = "Template dimensions must look like 10.5x9.5in or 27x24cm."

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
            "Worker safety", "error", not break_glass_unlocked,
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
    if execution_device == "cpu":
        checks.append((
            "Execution mode", "warning", False,
            f"Parallel CPU mode with {workers} workers; CUDA/MPS disabled for this run.", None,
        ))
    else:
        checks.append((
            "Execution mode", "success", False,
            "Single worker; accelerator selection is automatic.", None,
        ))
    checks.append((
        "BiRefNet compute", device_report.severity,
        mask_method == "birefnet" and device_report.severity == "error",
        device_report.detail, None,
    ))
    checks.append(("Template dimensions", "success" if template_error is None else "error", template_error is not None,
                   template_error or "OK", None))

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
        break_glass_acknowledged = break_glass_unlocked
        if worker_risk.requires_break_glass:
            # An acknowledgement authorizes one high-risk run only.
            st.session_state.pop("break_glass_available_workers", None)

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
    device_label = "CPU only" if run["execution_device"] == "cpu" else "automatic accelerator selection"
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
