# Using the app

Launch the GUI:

```bash
mats app
```

Streamlit opens in your browser (default http://localhost:8501). The app has
three pages, chosen from the sidebar: **Home** (measuring), **Template Creator**
(making printable templates), and **BiRefNet setup** (optional model install and
hardware diagnostics).

## Home — measuring leaves

1. **Image source.** Choose a local folder (type its path) or upload images.
   Accepted: `.jpg .jpeg .png .tif .tiff .bmp`.
2. **Output folder.** Where masks and target boxes are written (defaults to
   `~/mats_outputs`).
3. **Template dimensions.** Enter the observation-box size, e.g. `10.5x9.5in` or
   `27x24cm`. Leave blank to read it from the template's QR code.
4. **Segmentation method.** *Classic thresholding (Otsu)* (fast, default; best
   on clean backgrounds) or *BiRefNet* (more accurate on cluttered backgrounds;
   uses a GPU when available; needs the ~2.65 GB checkpoint).
5. **Output options.** Pick the **Full research schema** CSV to match the
   analysis scripts, or **Compact** for a trimmed export. Optionally write a
   failures log.
6. **Workers.** The app detects the CPU workers assigned to it (including HPC
   scheduler limits). One worker uses CUDA/MPS when available. Selecting two or
   more workers enables parallel CPU processing and disables CUDA/MPS for that
   run, including RF-DETR and BiRefNet inference. A warning light is green at
   25% or less of the CPU allocation, yellow through 50%, and red through 75%.
   Counts above 75% require a one-run **Break the glass** acknowledgement.
7. **Preflight** shows green, yellow, and red checks for weights, BiRefNet
   compute availability, template dimensions, and input images. A missing
   optional BiRefNet checkpoint is yellow and links to **BiRefNet setup**;
   it only blocks a run when BiRefNet segmentation is selected.
8. After the run: preview matched target-box / mask pairs, view and download the
   measurements CSV, or build a ZIP of all outputs.

Large batches (>200 images) ask for confirmation and run synchronously — keep
the browser tab open until they finish.

## Preflight failures

- **RF-DETR checkpoint (red)** — run `mats fetch-weights`, or set
  `MATS_WEIGHTS_DIR`. See [weights.md](weights.md).
- **BiRefNet checkpoint (yellow)** — open **BiRefNet setup** in the sidebar to
  install it, or pre-stage it with `MATS_WEIGHTS_DIR` / `BIREFNET_CHECKPOINT`.
  CPU-only BiRefNet is also yellow but remains supported.
- **Template dimensions (red)** — the text must look like `10.5x9.5in` or
  `27x24cm`.
- **Input images (red)** — the folder has no supported image files.

## Template Creator

See [templates.md](templates.md). It generates a print-ready PDF template at any
supported width and length, with the correct marker color and a QR code the
pipeline can read back. Marker size and the calibrated observation area are
calculated automatically. An editable Adobe InDesign IDML is also available,
while PDF remains the recommended print format.
