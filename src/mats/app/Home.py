import csv
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
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


def collect_output_pairs(output_dir):
    """Match each mask to its target box by sample_id, sorted by sample_id."""
    output_dir = Path(output_dir)
    pairs = []
    for mask_path in sorted(output_dir.glob("*_mask.png")):
        sample_id = mask_path.name[: -len("_mask.png")]
        target_box = output_dir / f"{sample_id}_target_box.jpg"
        pairs.append({
            "sample_id": sample_id,
            "target_box": target_box if target_box.is_file() else None,
            "mask": mask_path,
        })
    return pairs


def main():
    mark_icon = ASSETS_DIR / "mats_mark.svg"
    st.set_page_config(
        page_title="Morphometric Analysis Tools (MATs)",
        page_icon=str(mark_icon) if mark_icon.is_file() else "🌿",
        layout="wide",
    )

    # Brand mark in the app/sidebar header (full lockup + collapsed-state icon).
    logo_wide = ASSETS_DIR / "mats_logo_horizontal.svg"
    if logo_wide.is_file():
        st.logo(
            str(logo_wide),
            icon_image=str(mark_icon) if mark_icon.is_file() else None,
        )

    st.title("Morphometric Analysis Tools (MATs)")
    st.caption("RF-DETR marker detection with BiRefNet or Otsu threshold segmentation.")

    with st.expander("Usage and setup"):
        st.markdown(
            "- **Launch**: `mats app` (from an environment where MATs is installed).\n"
            "- **Checkpoints**: fetched once with `mats fetch-weights` into "
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
        st.error(f"Failed to import the MATs pipeline: missing dependency `{missing}`.")
        st.info(
            "The Python environment running Streamlit is missing a pipeline "
            f"dependency (`{missing}`). Install MATs with its dependencies "
            "(rfdetr, qreader, transformers, torch, opencv, pyzbar) and relaunch:\n\n"
            "```\npip install -e .\nmats app\n```"
        )
        st.caption(f"Current interpreter: {sys.executable}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to import the MATs pipeline: {exc}")
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
            ["BiRefNet", "Classic thresholding (Otsu)"],
        )
        mask_method = "birefnet" if segmentation_label == "BiRefNet" else "threshold"

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
                "Full matches the columns the downstream MATs analysis scripts expect. "
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
        all_target_boxes = bool(candidate_images) and all(
            lm.is_target_box_image(p) for p in candidate_images
        )
        # Model-backed runs use process-global singletons, so the pipeline serializes
        # them to a single worker regardless of this value. Parallelism only helps the
        # CPU-only Otsu path over pre-computed target boxes.
        can_parallelize = all_target_boxes and mask_method == "threshold"
        default_workers, worker_reason = lm.default_worker_count(
            candidate_images if input_source == "Local folder" else [],
            "masks",
            mask_method,
        )
        if can_parallelize:
            workers = st.number_input(
                "Workers",
                min_value=1,
                max_value=32,
                value=max(1, int(default_workers)),
                step=1,
                help=f"Default reason: {worker_reason}",
            )
        else:
            workers = 1
            st.number_input(
                "Workers",
                min_value=1,
                max_value=32,
                value=1,
                step=1,
                disabled=True,
                help="Model-backed runs (RF-DETR / BiRefNet) always run on a single worker.",
            )
            st.caption("Serialized to 1 worker for model-backed inference.")

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
    checks.append(("Pipeline library", True, f"mats.core ({Path(lm.__file__).parent})"))
    checks.append(("RF-DETR checkpoint", lm.RF_DETR_MARKER_CHECKPOINT.is_file(), str(lm.RF_DETR_MARKER_CHECKPOINT)))
    if mask_method == "birefnet":
        checks.append(("BiRefNet checkpoint", lm.BIREFNET_CHECKPOINT.is_file(), str(lm.BIREFNET_CHECKPOINT)))
    checks.append(("Template dimensions", template_error is None, template_error or "OK"))

    if input_source == "Local folder":
        image_paths = collect_folder_images(input_dir)
        checks.append(("Input images", len(image_paths) > 0, f"{len(image_paths)} image(s) found"))
    else:
        image_paths = []
        valid_upload_count = sum(
            1 for uploaded_file in uploaded_files
            if Path(uploaded_file.name).suffix.lower() in VALID_EXTENSIONS
        )
        checks.append(("Uploaded images", valid_upload_count > 0, f"{valid_upload_count} valid image(s) selected"))

    for label, ok, detail in checks:
        if ok:
            st.success(f"{label}: {detail}")
        else:
            st.error(f"{label}: {detail}")

    ready = all(ok for _, ok, _ in checks)

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
                    progress_callback=update_progress,
                    write_failures=write_failures,
                    compact_csv=compact_csv,
                    save_measurement_axes=False,
                )

        # Persist only what the results view needs; keep large lists out of session
        # state so the page stays light after big runs.
        st.session_state["last_run"] = {
            "succeeded": summary["succeeded"],
            "failed": summary["failed"],
            "total": summary["total"],
            "workers": summary["workers"],
            "worker_reason": summary["worker_reason"],
            "failure_rows": summary["failure_rows"][:200],
            "failure_overflow": max(0, len(summary["failure_rows"]) - 200),
            "results_path": str(results_path),
            "output_path": str(output_path),
            "mask_method": mask_method,
        }
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
    st.caption(f"Workers used: {run['workers']} ({run['worker_reason']}).")

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
        st.dataframe(preview_rows, use_container_width=True)
        st.download_button(
            "Download results CSV",
            data=results_path.read_bytes(),
            file_name=results_path.name,
            mime="text/csv",
        )

    render_output_preview(output_path)
    render_zip_export(results_path, output_path)


def render_output_preview(output_path):
    pairs = collect_output_pairs(output_path)
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
