"""Guided help: photographing templates, choosing settings, reading results."""

import io
import zipfile

import streamlit as st

from mats import samples
from mats.app import branding
from mats.app.runtime_paths import (
    demo_output_directory,
    display_path,
    mats_command,
    shell_language,
)
from mats.scaling import COMPACT_RESULTS_FIELDNAMES, QR_TRACE_FIELDNAMES


def _page_link(target, label, icon):
    """Render a cross-page link, degrading to plain text off the main app.

    ``st.page_link`` resolves its target against the *entry* script's
    directory. That works under ``mats app`` (the entry script is Home.py),
    but raises when this page is executed directly -- e.g. by the Streamlit
    AppTest harness. Mirror branding.apply_logo()'s degrade-don't-raise
    contract rather than letting a navigation nicety break the page.
    """
    try:
        st.page_link(target, label=label, icon=icon, width="content")
    except Exception:
        st.caption(f"{label} — open it from the sidebar.")


@st.cache_data(show_spinner=False)
def _samples_archive():
    """Zip every installed sample image, preserving its set subdirectory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for sample_set in samples.SAMPLE_SETS:
            for path in samples.sample_paths(sample_set):
                archive.write(path, f"{sample_set['directory']}/{path.name}")
    return buffer.getvalue()


st.set_page_config(
    page_title="MATS — Help",
    page_icon=branding.page_icon(),
    layout="wide",
)
branding.apply_logo()

st.title("Help")
st.caption(
    "How to photograph a calibration template, what each control means, and how "
    "to read the results CSV. The sample images below ship with MATS — you can "
    "run them as they are."
)

# ---------------------------------------------------------------- quick start
with st.container(border=True):
    st.subheader("Quick start", anchor=False)
    st.markdown(
        "1. **Print a template** at 100% scale — no *fit to page*; a scaled "
        "print breaks the calibration. Template Creator makes one at any size.\n"
        "2. **Photograph** one specimen, flat, inside the box, with all four corner "
        "markers in frame.\n"
        "3. **Pick your folders** in the sidebar: images in, results out.\n"
        "4. **Set the scale** in Analyze — enter the finished printed-sheet width "
        "and height. MATS derives the calibrated marker-centre area using the same "
        "margins as Template Creator. Tick *Variable "
        "dimensions* to read it from each image's QR code instead; if reads are "
        "unreliable, **Robust QR setup** adds sturdier decoders.\n"
        "5. **Run**, then read **Results** and download the CSV."
    )
    _page_link("pages/1_Template_Creator.py", "Open Template Creator", ":material/grid_on:")
    _page_link("pages/4_Robust_QR_Setup.py", "Open Robust QR setup", ":material/qr_code_scanner:")

# ------------------------------------------------------- what a good photo is
st.subheader("What a good photo looks like", anchor=False)
st.caption(
    "These two sets ship with MATS. They are the same pipeline on very "
    "different photos — compare them before you shoot your own."
)

_SET_NOTES = {
    "flat_bench": (
        "- The sheet lies **flat** and the camera is nearly perpendicular, so "
        "there is little perspective for the correction step to undo.\n"
        "- The background is plain and uniform, with nothing leaf-coloured near "
        "the box — brightness alone separates leaf from board, which is exactly "
        "what Otsu thresholding needs.\n"
        "- All four markers are crisp, unobstructed, and well inside the frame.\n"
        "- One specimen, flat, clear of the markers. **Otsu is the right call here.**"
    ),
    "handheld_field": (
        "- The board is **hand-held at an angle**, so the four markers form a "
        "strong trapezoid. This still measures correctly: the marker centroids "
        "define a homography that rectifies the box before anything is measured. "
        "What matters is that all four are sharp and in frame — not that the "
        "sheet is square to the lens.\n"
        "- The background is soil, plastic mulch, a hand and a boot. Brightness "
        "thresholding can't tell a dark leaf from dark soil, so **if Otsu fails, "
        "switch to BiRefNet** — it handles cluttered, low-contrast backgrounds "
        "like this one.\n"
        "- The QR code sits at the top edge. Rotated or skewed codes are fine; "
        "decoding does not require the sheet to face the camera squarely.\n"
        "- A compound leaf is segmented as one specimen, leaflets included."
    ),
}

for sample_set in samples.SAMPLE_SETS:
    photo = samples.sample_hero_path(sample_set)
    with st.container(border=True):
        st.markdown(f"**{sample_set['title']}**")
        if photo:
            st.image(str(photo), caption=photo.name, width=420)
        else:
            st.info(
                "The packaged sample images are not installed in this "
                "environment. The guidance below still applies.",
                icon=":material/image_not_supported:",
            )
        st.markdown(_SET_NOTES[sample_set["key"]])
        st.caption(
            f"Print sheet {sample_set['print_sheet']} · "
            f"Legacy calibration area `{sample_set['calibration_dimensions']}` · "
            f"Marker diameter {sample_set['marker_diameter']} · "
            f"Recommended segmentation: {sample_set['segmentation']}"
        )

with st.container(border=True):
    st.markdown("**Photography checklist**")
    st.markdown(
        "- All four corner markers in frame, sharp, and unobstructed.\n"
        "- Printed at 100% scale — scaling silently breaks the calibration.\n"
        "- Leaf flat and fully inside the box, not overlapping a marker.\n"
        "- Fill the frame with the sheet; avoid hard glare on the markers.\n"
        "- One specimen per photo — MATS measures the largest connected mask.\n"
        "- Moderate tilt is fine. Blur, a missing marker, or a folded sheet is not."
    )

with st.expander("Run these samples yourself", icon=":material/terminal:"):
    if samples.samples_installed():
        st.markdown("**Command line**")
        st.markdown(
            "These photographs use legacy sheets whose margins predate Template "
            "Creator, so these compatibility commands pass their calibration areas "
            "directly. Run the sets separately. The paths are resolved for the "
            "machine currently running MATS:"
        )
        output_root = demo_output_directory()
        st.code(
            "\n".join(
                mats_command(
                    "run",
                    "-i",
                    samples.sample_set_dir(sample_set),
                    "-o",
                    output_root / sample_set["key"],
                    "-t",
                    sample_set["calibration_dimensions"],
                )
                for sample_set in samples.SAMPLE_SETS
            ),
            language=shell_language(),
        )
        st.caption(
            f"Sample images: `{display_path(samples.SAMPLES_DIR)}` · "
            f"Demo output: `{display_path(output_root)}`"
        )
        st.markdown("**Reading dimensions from the QR code instead**")
        field_set = next(
            s for s in samples.SAMPLE_SETS if s["key"] == "handheld_field"
        )
        st.markdown(
            "Drop `-t` and MATS reads each image's own QR code. The hero photo "
            "above decodes; the second field photo doesn't, so that row comes "
            "back `NA` — see *When the QR can't be read* below."
        )
        st.code(
            mats_command(
                "run",
                "-i",
                samples.sample_set_dir(field_set),
                "-o",
                output_root / "qr_demo",
            ),
            language=shell_language(),
        )
        st.markdown("**In the app**")
        st.markdown(
            "Download the ZIP below, unzip it, then in the sidebar set **Image "
            "source** to *Upload images* and upload one photo at a time. Their "
            "readable photos work in QR mode. For the deliberately unreadable QR "
            "example, open **Older or custom template?** and enter its legacy "
            "calibration area."
        )
        st.download_button(
            "Download sample images (ZIP)",
            data=_samples_archive(),
            file_name="mats_sample_images.zip",
            mime="application/zip",
            icon=":material/download:",
            width="content",
        )
    else:
        st.caption("The packaged sample images are not installed in this environment.")

# ---------------------------------------------------------- printed sheet size
st.subheader("Which sheet size should I enter?", anchor=False)
with st.container(border=True):
    st.markdown(
        "Enter the **finished outer sheet size** printed on a current Template "
        "Creator sheet. For example, enter `12x12in` for a 12 × 12 in sheet. "
        "MATS derives the marker-centre calibration area automatically.\n\n"
        "Template Creator places marker centres 1.5 in from the top and 1 in "
        "from the other edges, or 3.5 cm from the top and 2.5 cm elsewhere. "
        "Therefore a 12 × 12 in sheet becomes a 10 × 9.5 in calibrated area. "
        "The top margin is larger to leave space for the QR code and header."
    )
    st.markdown(
        "Format: `<width>x<height><unit>`, unit `in` or `cm` — "
        "e.g. `12x12in`, `30x30cm`. Older sheets with different margins remain "
        "supported under **Older or custom template?**, where their marker-centre "
        "calibration area can be entered directly."
    )
    st.markdown(
        "**Variable dimensions, read QR code** reads the calibration from each "
        "image's own QR code instead. Both hero photos above decode "
        "successfully — but **entering the sheet size is the more consistent "
        "option** for current Template Creator sheets: it needs no optional packages and "
        "never depends on glare, blur, skew, or which decoders happen to be "
        "installed. OpenCV handles clear codes with no extra setup; Robust QR "
        "setup adds pyzbar and QReader fallbacks for tougher ones."
    )
    _page_link("pages/1_Template_Creator.py", "Open Template Creator", ":material/grid_on:")
    _page_link("pages/4_Robust_QR_Setup.py", "Open Robust QR setup", ":material/qr_code_scanner:")

with st.container(border=True):
    st.markdown("**When the QR can't be read**")
    qr_failure_photo = samples.qr_failure_sample_path()
    if qr_failure_photo:
        st.image(str(qr_failure_photo), caption=qr_failure_photo.name, width=420)
    else:
        st.info(
            "The packaged QR-failure sample is not installed in this "
            "environment. The guidance below still applies.",
            icon=":material/image_not_supported:",
        )
    st.markdown(
        "This packaged photo is a real failure case: motion blur, and **no** "
        "bundled decoder — OpenCV, pyzbar, or QReader — can read its QR code. "
        "The run doesn't stop. The image is still marker-detected, "
        "perspective-corrected, and masked; only the scale is missing. Its "
        "row is written with every measurement as `NA` and "
        "`source = QR_READ: QR not found/readable` — and it still counts as a "
        "**successful** run, which is why reading `source` matters.\n\n"
        "The recovery is printed on the sheet itself. Every MATS template "
        f"states its own size — here `{samples.QR_FAILURE_SAMPLE['printed_label']}` "
        "— still legible despite the blur. Because this is a legacy sheet, re-run "
        "it with the compatibility calibration option "
        f"(`-t {samples.QR_FAILURE_SAMPLE['calibration_dimensions']}`, or untick "
        "*Variable dimensions* and use **Older or custom template?** in Analyze)."
    )

# ------------------------------------------------------- segmentation method
st.subheader("Choosing a segmentation method", anchor=False)
otsu_column, birefnet_column = st.columns(2)
with otsu_column:
    with st.container(border=True):
        st.markdown("**Classic thresholding (Otsu)** — the default")
        st.markdown(
            "- Fast, no GPU, no extra download.\n"
            "- Separates leaf from background by brightness.\n"
            "- Right choice for the **bench** samples above.\n"
        )
with birefnet_column:
    with st.container(border=True):
        st.markdown("**BiRefNet** — optional")
        st.markdown(
            "- A learned leaf-foreground segmenter; handles clutter and "
            "low contrast.\n"
            "- Right choice for the **field** samples above *if* Otsu fails.\n"
            "- Needs a one-time ~2.65 GB checkpoint; uses a GPU when present. "
            "CPU works but is slow."
        )
        _page_link("pages/2_BiRefNet_Setup.py", "Open BiRefNet setup", ":material/download:")
st.caption(
    "Start with Otsu. Switch only when the mask shown in Results is visibly "
    "wrong — that is the signal, not the file size or the leaf species."
)

# ---------------------------------------------------------------- csv glossary
st.subheader("Reading the results CSV", anchor=False)
with st.container(border=True):
    st.markdown("**Full research schema** (default)")
    st.markdown(
        "| Column | Meaning |\n"
        "|---|---|\n"
        "| `sample_id` | Input filename without its extension. Matches "
        "`{sample_id}_target_box.jpg` and `{sample_id}_mask.png` in the output "
        "folder. |\n"
        "| `leaf_area_cm2` | Segmented leaf area — mask pixel count divided by "
        "`px_per_cm_width` x `px_per_cm_height`. |\n"
        "| `width_cm` | Horizontal extent of the leaf's bounding box, divided by "
        "`px_per_cm_width`. |\n"
        "| `length_cm` | Vertical extent of the leaf's bounding box, divided by "
        "`px_per_cm_height`. |\n"
        "| `px_per_cm_width` | Horizontal scale: corrected-box pixel width "
        "divided by the template width you supplied. |\n"
        "| `px_per_cm_height` | Vertical scale: corrected-box pixel height "
        "divided by the template height you supplied. |\n"
        "| `scale_aspect_ratio` | `px_per_cm_width / px_per_cm_height`. **A QC "
        "column** — should sit near 1.0. Far from 1.0 usually means a scaled "
        "print, lens distortion, or a non-planar sheet. |\n"
        "| `source` | `0` when the row measured cleanly. Otherwise the stage "
        "and reason the scale could not be established, e.g. "
        "`SCALE: physical dimensions unavailable` or "
        "`QR_READ: QR not found/readable`. |\n"
    )
    st.caption(
        "Each axis is calibrated independently, and area uses the product of "
        "both, so a measurement never depends on which marker-quad edge the "
        "perspective warp happened to size the raster from."
    )
    st.markdown("**QR-mode trace columns**")
    st.markdown(
        "Appended to the full schema only for runs using **Variable dimensions, "
        "read QR code** — they don't appear for `-t` runs, and a column is "
        "included only if that backend is installed:\n\n"
        "| Column | Meaning |\n"
        "|---|---|\n"
        f"| `{QR_TRACE_FIELDNAMES[0]}` | Outcome of the bundled OpenCV decoder: "
        "`success`, `failed`, or `unused`. |\n"
        f"| `{QR_TRACE_FIELDNAMES[1]}` | Outcome of the optional pyzbar/zbar "
        "fallback, if installed. |\n"
        f"| `{QR_TRACE_FIELDNAMES[2]}` | Outcome of the optional QReader "
        "fallback, if installed. |\n"
    )
    st.caption(
        "Backends are tried in order and stop at the first success: `unused` "
        "means a later backend wasn't tried because an earlier one already "
        "succeeded, and `not_reached` means QR decoding wasn't attempted for "
        "that row at all."
    )
    st.markdown("**Compact schema**")
    st.markdown(
        "`" + "`, `".join(COMPACT_RESULTS_FIELDNAMES) + "` — a trimmed export "
        "for spreadsheet use. `area_cm2` here is the same quantity as "
        "`leaf_area_cm2` above; the name differs between schemas."
    )
    st.caption(
        "Unmeasurable values are written as the literal `NA`. Also written per "
        "image: `{sample_id}_target_box.jpg` (perspective-corrected box) and "
        "`{sample_id}_mask.png` (segmentation mask). A failures log, when "
        "enabled, lists `sample_id, input_image, stage, failure_mode, status`."
    )

# -------------------------------------------------------------- troubleshooting
st.subheader("Troubleshooting", anchor=False)
st.info(
    "Start in **Diagnostics** — every preflight check there names the exact "
    "failing component and links to its setup page. This section covers what "
    "preflight cannot see.",
    icon=":material/troubleshoot:",
)
with st.container(border=True):
    st.markdown(
        "- **No markers detected.** Check print scale, marker colour, framing, "
        "and glare — all four must be sharp and unobstructed.\n"
        "- **The mask covers the wrong thing.** Switch segmentation method: "
        "Otsu needs a clean background, BiRefNet handles clutter.\n"
        "- **`scale_aspect_ratio` far from 1.0.** A calibration problem, not a "
        "leaf problem — a scaled print, lens distortion, or a non-planar sheet.\n"
        "- **Measurements are plausible but uniformly wrong.** Check the finished "
        "sheet size, confirm an older/custom sheet is using compatibility mode, "
        "and print at 100% scale rather than *fit to page*.\n"
        "- **A row is all `NA`.** Read its `source` column for the stage and "
        "reason.\n"
        "- **The run is slow.** See CPU Options for worker and device controls."
    )
    _page_link("pages/3_CPU_Options.py", "Open CPU options", ":material/tune:")

# ------------------------------------------------------------------ more help
st.subheader("More help", anchor=False)
with st.container(border=True):
    st.markdown(
        "- **Command-line usage** — `docs/cli.md`\n"
        "- **Printable templates** — `docs/templates.md`\n"
        "- **Model weights and offline setup** — `docs/weights.md`\n"
        "- **Running on shared/HPC systems** — `docs/hpc.md`\n"
        "- **`mats doctor`** — run from a terminal for a full diagnostic of "
        "weights resolution, device detection, and the QR-reading backend.\n"
        "- **Report an issue** — "
        "https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation/issues\n"
        "- **Citing MATS** — see `CITATION.cff` in the repository."
    )
