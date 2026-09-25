# Using the app

Launch the GUI:

```bash
mats app
```

Streamlit opens in your browser (default http://localhost:8501). The sidebar
provides the image source and output destination. The MATS Analysis Workbench
has a prominent, sticky **WORKSPACE NAVIGATION** bar with large, button-like
controls for **Setup**, **Analyze**, **Adjust**, and **Export**. Its filled active section stays visible while the workbench
scrolls. Setup holds
scale, segmentation, measurement output, and image output options. Analyze holds
Preflight, the launch button, progress, and results. Adjust holds specimen previews
and saving controls. Export holds saved files and downloads. The separate sidebar
pages are **Diagnostics** (compute status and preflight), **Template Creator** (making printable templates), **BiRefNet setup**
(optional model install and hardware diagnostics), **CPU Options**, **Robust QR
setup** (optional QR fallbacks), and **Help** (packaged sample images, a settings
guide, and the results-CSV glossary).
After a run, **Go to Export** in the sidebar opens the Export tab.

## Home — measuring leaves

1. In the sidebar, choose an **image source**: select a local folder with the
   desktop folder picker (or type its path manually), or upload images. Choose
   the output folder there as well. The picker works on Windows, macOS, and
   Linux where a desktop folder-dialog backend is available.
   Accepted: `.jpg .jpeg .png .tif .tiff .bmp`.
2. In **Setup**, enter the finished **printed sheet size** — outer-sheet
   **width** and **height** — and choose **in** or **cm** (defaults to 12x12 in).
   MATS uses the Template Creator's fixed margins to derive the marker-centre
   calibration area. Older/custom templates with different margins remain
   available in the collapsed compatibility control.
   Tick **Variable
   dimensions, read QR code** to read each image's size from its own template
   QR code instead. OpenCV handles clear codes with no extra setup. For glare,
   skew, or blur, **Robust QR setup** explains the optional `mats-morpho[qr]`
   fallbacks and the optional native `zbar` library.
3. Choose one or both **segmentation methods**: *Classic thresholding* (fast,
   default; best on clean backgrounds) and/or *BiRefNet* (more accurate on cluttered
   backgrounds; uses a GPU when available; needs its optional ~2.65 GB local
   checkpoint). Check both to measure every image with each method and compare
   them: each method then gets its own results CSV
   (`leaf_morpho_results_threshold.csv`, `leaf_morpho_results_birefnet.csv`),
   failure log, and masks, while marker detection runs once per image.
   **Threshold level** appears immediately below Classic thresholding, before
   BiRefNet, and sets its cutoff: `auto` (Otsu, the default), the
   `low`/`medium`/`high` presets, or **custom**, which shows a 1–255 slider with
   the presets marked on it.
4. Choose **output options**: optionally check **Measure from pre-cleanup masks**
   to use the raw binary segmentation for area, width, and length. Raw mode
   first clears **Edge margin**, a band along every edge of the target box
   (default 1% of its shorter side), where the template's printed box outline
   lands. It then keeps the leaf (the largest object) and drops pieces that
   touch that band or lie farther from the leaf than **Stray-piece distance**,
   a fraction of the leaf's bounding-box diagonal (default 0.25; 0 keeps only
   the leaf). Specks near the leaf still count toward area, width, and length,
   and holes stay excluded from area, unless **Clean size** (px, default 0) is
   above 0: it then removes white specks and fills enclosed holes whose
   inscribed radius is below that many pixels, without flash-filling the leaf.
   Lay leaves inside the printed box, since any part of a leaf within the
   margin is cleared too. All three settings are grayed out unless the box is
   checked. The default uses the cleaned mask.
   This choice is separate from exporting pre-cleanup mask images, which keep
   every piece. Choose result
   units in **mm**, **cm**, or **in** (separate from the printed-sheet calibration
   unit), then pick the **Full research schema** CSV for area/width/length plus per-axis
   pixels-per-selected-unit and a `scale_aspect_ratio` QC column, or **Compact** for a
   trimmed export. In **Variable dimensions** mode,
   the Full schema also records each installed QR decoder's outcome (`success`,
   `failed`, or `unused`).
   The failure log is also checked by default in **Measurement Output**.
   In **4 · Image output options**, choose target-box and cleaned-mask images
   (checked by default), or optional pre-cleanup masks, overlays, cutouts,
   and measurement axes. Pre-cleanup masks are binary segmentations before
   closing, hole filling, and removal of smaller objects; one is written for
   each checked segmentation method. With both methods checked, masks, overlays,
   cutouts, and axes end in `_threshold` or `_birefnet`
   (for example `{id}_mask_birefnet.png`). Existing target-box inputs are not
   copied. After a run with both methods, **Measurement table and specimen
   view** switches the summary, table, charts, and selected specimen between
   Classic thresholding and BiRefNet. Each method has its own table selection;
   choose a row to see that method's specimen mask.
   Select one row in the Analyze measurement table to inspect that specimen;
   **Adjust selected specimen** opens its controls in Adjust. Adjust lists
   specimens in a measurement table (click a column header to sort); search
   specimen names to narrow it, and click a row, or its **View** button, to open
   its live preview. Adjust contains the controls and live preview,
   while saved images and measurement cards are shown in Analyze. The Analyze sample viewer shows
   the mask used for that run's measurements:
   raw with the edge margin cleared and stray pieces dropped when **Measure
   from pre-cleanup masks** was checked, cleaned otherwise. In a pre-cleanup
   run, the threshold explorer also clears the margin while you drag and shows
   that measured mask when you release. **Edge margin** and **Stray-piece
   distance** also appear in each specimen's Adjust controls, starting from the run's
   values. There they adjust just that specimen's preview, and are enabled
   only while flash fill is off: in pre-cleanup runs, with a **Clean size**
   above 0, or (the margin alone) with **Remove flashfill**. Overwriting a specimen saves
   its mask with the values shown and records them in the `.meta.json`.
   It can show the mask and target box even if their exports were turned off;
   those extra previews last only for the current app session.
   **Remove flashfill from this preview**, under **Clean size** in
   **Explore and adjust output**, leaves enclosed holes unfilled while
   retaining the other cleanup for the selected specimen. It is available for
   cleaned-mask runs and remains a preview until Overwrite is pressed.
   For a selected Otsu sample, **Explore and adjust output** lets you
   drag a cutoff and see the raw mask change immediately, with the masked leaf
   in its original colors beside it so you can see which parts of the leaf the
   cutoff keeps or loses. Releasing the slider
   applies MATS cleanup when the run used cleaned masks, honoring the flashfill
   checkbox. The preview does not change files by itself.
   **Overwrite this specimen** replaces only the mask and CSV row of the
   specimen selected in the table's **View** column. Existing overlays, cutouts, and measurement axes
   for that sample are regenerated; every other sample remains unchanged.
   To save the same cutoff and cleanup settings across specimens, tick their
   **Marked for Adjustment** boxes in the Adjust table (tick again to unmark).
   Marks persist across searches. **Mark all matching** marks every search match;
   **Mark all** with an empty search marks every available specimen.
   Then press **Overwrite all marked specimens (N)**, below **Overwrite this
   specimen**, to save the settings to every specimen ticked in **Marked for
   Adjustment**.
   Each specimen keeps its own scale; all
   marked specimens are validated before any saved output changes.
   **Reset to saved threshold** restores the sample's current saved cutoff;
   **Use threshold for next run** sets a custom cutoff in Setup for the next
   analysis. BiRefNet specimens have no threshold, reset, next-run, or
   overwrite controls. The **Clean size** slider, in the same box for both
   methods, starts at the specimen's saved clean size, else the run's (0
   unless set in Setup); 0 turns Clean image off, and its **?** explains it.
   Above 0 it previews Clean image, a gentler alternative to MATS cleanup:
   it clears the specimen's edge margin, drops pieces that touch it or lie
   beyond the stray-piece distance, then drops disconnected white specks and
   fills enclosed holes smaller than the clean size. It always keeps the leaf
   and never flash-fills. For an Otsu specimen, either Overwrite button saves
   the mask shown, re-measures it under the run's measurement source, and
   records the clean size with the cutoff in the `.meta.json`; in a
   cleaned-mask run, Clean image then replaces MATS cleanup (and Remove
   flashfill) in the saved mask. BiRefNet adjustments remain preview-only.
   Each results CSV has a `.meta.json` companion recording which measurement
   source was used and any per-sample threshold adjustments.
5. **CPU Options** retains the worker controls. The app detects the CPU workers assigned to it (including HPC
   scheduler limits). One worker uses CUDA/MPS when available. Selecting two or
   more workers enables parallel CPU processing and disables CUDA/MPS for that
   run, including RF-DETR and BiRefNet inference. A warning light is green at
   25% or less of the CPU allocation, yellow through 50%, and red through 75%.
   Counts above 75% require a one-run **Break the glass** acknowledgement.
6. The compact **Preflight** in Analyze shows readiness without overwhelming
   the workspace. The **Diagnostics** sidebar page has compute status, a compact
   overview of every check, and a collapsible detailed report. They show green,
   yellow, and red checks for weights, BiRefNet compute availability, printed
   sheet calibration, QR-reader availability, and input images. OpenCV and
   pyzbar decode a known test QR during QR-mode Preflight; QReader is import-
   checked without initialization so Preflight cannot trigger a model download. A missing
   optional BiRefNet checkpoint is yellow and links to **BiRefNet setup**;
   it blocks a run when BiRefNet is needed for measurements or an export.
7. After the run, **Analyze** shows measurement cards and charts, a
   full-width selectable measurements table followed by a full-width linked
   target-box / mask specimen inspector. **Export** lists the files saved for
   that run and its output folder. Choose a method (or both), select file types
   for a ZIP, review its file list and size, and set its download filename.
   All available file types are selected by default.
   Results CSVs can also be downloaded individually. File types not saved during
   the run cannot be added to the ZIP; enable image output options in Setup
   before the next run if needed.
   The **Training dataset** section creates a separate ZIP from the current
   run's aligned target-box images and masks, including saved specimen
   adjustments. Choose one segmentation method, the measured mask or an
   available raw pre-cleanup mask, and image/PNG-mask, YOLO segmentation, YOLO
   detection, or COCO segmentation format. Set train/validation/test percentages
   (default 70/20/10; they must total 100) and a random seed. An optional UTF-8
   CSV with `sample_id,group_id` columns keeps repeated photos of the same
   plant or specimen in one split; group sizes can shift the exact percentages.
   The ZIP contains `manifest.json` with actual counts, exclusions, and any
   conversion notes. MATS generates these labels from its masks, so review
   them before training. YOLO segmentation polygons cannot retain holes in
   masks; COCO segmentation uses run-length masks that retain them. This export
   uses current-session previews when regular image output was turned off, so
   prepare it before ending the app session.

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
