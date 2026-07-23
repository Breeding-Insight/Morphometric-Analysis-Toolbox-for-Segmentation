# Using the app

Launch the GUI:

```bash
mats app
```

Streamlit opens in your browser (default http://localhost:8501). The app has two
pages, chosen from the sidebar: **Home** (measuring) and **Template Creator**
(making printable templates).

## Home — measuring leaves

1. **Image source.** Choose a local folder (type its path) or upload images.
   Accepted: `.jpg .jpeg .png .tif .tiff .bmp`.
2. **Output folder.** Where masks and target boxes are written (defaults to
   `~/mats_outputs`).
3. **Template dimensions.** Enter the observation-box size, e.g. `10.5x9.5in` or
   `27x24cm`. Leave blank to read it from the template's QR code.
4. **Segmentation method.** *BiRefNet* (accurate; uses a GPU when available) or
   *Classic thresholding (Otsu)* (fast; best on clean backgrounds).
5. **Output options.** Pick the **Full research schema** CSV to match the
   analysis scripts, or **Compact** for a trimmed export. Optionally write a
   failures log.
6. **Preflight** shows green/red checks for weights, template dimensions, and
   input images. When all pass, click **Run Leaf Morphometrics**.
7. After the run: preview matched target-box / mask pairs, view and download the
   measurements CSV, or build a ZIP of all outputs.

Large batches (>200 images) ask for confirmation and run synchronously — keep
the browser tab open until they finish.

## Preflight failures

- **RF-DETR / BiRefNet checkpoint (red)** — run `mats fetch-weights`, or set
  `MATS_WEIGHTS_DIR`. See [weights.md](weights.md).
- **Template dimensions (red)** — the text must look like `10.5x9.5in` or
  `27x24cm`.
- **Input images (red)** — the folder has no supported image files.

## Template Creator

See [templates.md](templates.md). It generates a print-ready PDF template at any
size, with the correct marker color and a QR code the pipeline can read back.
