# Using the app

Launch the GUI:

```bash
mats app
```

Streamlit opens in your browser (default http://localhost:8501). The sidebar
provides the image source and output destination. The MATS Analysis Workbench
has a prominent, sticky **WORKSPACE NAVIGATION** bar with large, button-like
controls for **Analyze**, **Results**, and **Diagnostics**. Its filled active section stays visible while the workbench
scrolls, including after a preflight shortcut opens Diagnostics. Analyze presents a
numbered workflow for scale, segmentation, output, and preflight; its highlighted
launch surface follows Preflight in the normal page flow. The separate sidebar
pages are **Template Creator** (making printable templates), **BiRefNet setup**
(optional model install and hardware diagnostics), **CPU Options**, **Robust QR
setup** (optional QR fallbacks), and **Help** (packaged sample images, a settings
guide, and the results-CSV glossary).

## Home — measuring leaves

1. In the sidebar, choose an **image source**: select a local folder with the
   desktop folder picker (or type its path manually), or upload images. Choose
   the output folder there as well. The picker works on Windows, macOS, and
   Linux where a desktop folder-dialog backend is available.
   Accepted: `.jpg .jpeg .png .tif .tiff .bmp`.
2. In **Analyze**, enter the finished **printed sheet size** — outer-sheet
   **width** and **height** — and choose **in** or **cm** (defaults to 12x12 in).
   MATS uses the Template Creator's fixed margins to derive the marker-centre
   calibration area. Older/custom templates with different margins remain
   available in the collapsed compatibility control.
   Tick **Variable
   dimensions, read QR code** to read each image's size from its own template
   QR code instead. OpenCV handles clear codes with no extra setup. For glare,
   skew, or blur, **Robust QR setup** explains the optional `mats-morpho[qr]`
   fallbacks and the optional native `zbar` library.
3. Choose a **segmentation method**: *Classic thresholding (Otsu)* (fast, default; best
   on clean backgrounds) or *BiRefNet* (more accurate on cluttered backgrounds;
   uses a GPU when available; needs its optional ~2.65 GB local checkpoint).
4. Choose **output options**: pick the **Full research schema** CSV for area/width/length
   plus per-axis `px_per_cm_width`/`px_per_cm_height` and a `scale_aspect_ratio`
   QC column, or **Compact** for a trimmed export. In **Variable dimensions** mode,
   the Full schema also records each installed QR decoder's outcome (`success`,
   `failed`, or `unused`). Optionally write a failures log.
5. **CPU Options** retains the worker controls. The app detects the CPU workers assigned to it (including HPC
   scheduler limits). One worker uses CUDA/MPS when available. Selecting two or
   more workers enables parallel CPU processing and disables CUDA/MPS for that
   run, including RF-DETR and BiRefNet inference. A warning light is green at
   25% or less of the CPU allocation, yellow through 50%, and red through 75%.
   Counts above 75% require a one-run **Break the glass** acknowledgement.
6. The compact **Preflight** in Analyze shows readiness without overwhelming
   the workspace. The **Diagnostics** tab has compute status, a compact
   overview of every check, and a collapsible detailed report. They show green,
   yellow, and red checks for weights, BiRefNet compute availability, printed
   sheet calibration, QR-reader availability, and input images. OpenCV and
   pyzbar decode a known test QR during QR-mode Preflight; QReader is import-
   checked without initialization so Preflight cannot trigger a model download. A missing
   optional BiRefNet checkpoint is yellow and links to **BiRefNet setup**;
   it only blocks a run when BiRefNet segmentation is selected.
7. After the run, MATS opens **Results** for measurement cards and charts, a
   selectable measurements table beside a linked target-box / mask specimen
   inspector, CSV download, and a ZIP of all outputs.

Large batches (>200 images) ask for confirmation and run synchronously — keep
the browser tab open until they finish.

## Preflight failures

- **RF-DETR checkpoint (red)** — run `mats fetch-weights`, or set
  `MATS_WEIGHTS_DIR`. See [weights.md](weights.md).
- **BiRefNet checkpoint (yellow)** — open **BiRefNet setup** in the sidebar to
  explicitly fetch it from this repository, or pre-stage it with
  `MATS_WEIGHTS_DIR` / `BIREFNET_CHECKPOINT`. It is never downloaded when Otsu
  is selected.
  CPU-only BiRefNet is also yellow but remains supported.
- **QR reader (yellow)** — shown for each unavailable or non-operational reader
  in **Variable dimensions** mode. OpenCV may miss glare/skew/blur codes, which
  produces `NA` values rather than stopping the run. Open **Robust QR setup**
  for the optional `mats-morpho[qr]` and `zbar` steps, or enter the finished
  printed sheet size instead.
- **Input images (red)** — the folder has no supported image files.

## Help

The **Help** page in the sidebar is the in-app guide. It carries three packaged
sample photographs — one shot flat on a bench (`6x6in` legacy calibration), one
shot hand-held in the field (`10.5x9.5in` legacy calibration), and a second field photo whose QR
code no bundled decoder can read — used to contrast an easy capture with a
hard one and to show, with a real example, why entering the finished sheet size
is the most consistent option. Also: a photography checklist, an explanation
of how sheet margins become calibration dimensions, an Otsu-vs-BiRefNet
comparison, a column-by-column results-CSV glossary, and troubleshooting for
the symptoms Preflight cannot detect. Preflight-specific failures stay in
[Preflight failures](#preflight-failures) above; Help links back to
Diagnostics rather than repeating them.

The sample images install with the package. To run them from a terminal:

```bash
python -c "from mats import samples; print(samples.SAMPLES_DIR)"
mats run -i <that path>/flat_bench_6x6in          -o ~/mats_demo/bench -t 6x6in
mats run -i <that path>/handheld_field_10.5x9.5in -o ~/mats_demo/field -t 10.5x9.5in
```

The Help page also offers them as a ZIP download.

## Template Creator

See [templates.md](templates.md). It generates a print-ready PDF template at any
supported width and length, with the correct marker color and a QR code the
pipeline can read back. Marker size and the calibrated observation area are
calculated automatically. An editable Adobe InDesign IDML is also available,
while PDF remains the recommended print format.
