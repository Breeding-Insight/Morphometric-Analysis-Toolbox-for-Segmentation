# Changelog

All notable changes to MATs are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Adjust now lists specimens in a searchable measurement table with sortable
  columns. Clicking a row (its **View** button) previews that specimen; for Otsu
  runs, ticking **Marked for Adjustment** marks it, and marks persist across
  searches. Its duplicate
  measurement cards and saved-image viewer have been removed, leaving the
  controls and live preview.
- **Remove flashfill from this preview** now sits under **Clean size** inside
  **Explore and adjust output**.
- Adjust saves marked specimens with its own **Overwrite all marked specimens
  (N)** button, below **Overwrite this specimen** (the specimen in View),
  replacing the "Apply these adjustments to all marked specimens" checkbox.
- Analyze now makes the Classic thresholding/BiRefNet comparison switch explicit
  and keeps each method's table selection separate, so the specimen viewer
  always follows the active method's table.
- Setup now places threshold level directly under Classic thresholding and
  labels the default method without the Otsu parenthetical.
- The Analyze measurement table and selected-specimen viewer occupy separate
  full-width blocks. Interactive controls now live in the Adjust tab.
- Analyze shows only the specimen selected in the measurement table. Adjust
  carries that selection into its preview and overwrite controls.
- The Analyze sample viewer now shows the raw or cleaned mask used for the
  completed run, even when that image was not selected for export.
- The app's segmentation choice is now two checkboxes. **Threshold level**
  appears only while Otsu is checked, and a single **Pre-cleanup masks** export
  covers every checked method.
- The workbench now uses Setup, Analyze, Adjust, and Export, with Diagnostics
  in the sidebar. Image exports are selected before a run; downloads include
  only files recorded for that run.
- Target boxes and cleaned masks remain default exports but can be disabled.
- Documentation now matches the shipped weights-delivery behavior: Git LFS is
  listed as an install prerequisite (a clone without it yields a 134-byte
  pointer stub, not the model), RF-DETR is documented as arriving *with* the
  clone rather than needing `mats fetch-weights`, and `mats fetch-weights` is
  described as the repair path it is.

### Added
- An interactive threshold preview for the selected Otsu sample in Adjust.
  Dragging updates its raw mask, and a color panel of the masked leaf beside
  it, immediately; releasing applies the existing
  cleanup when cleaned measurements are selected. A chosen cutoff can be used
  for the next run or overwrite the selected specimen's mask, measurement row,
  and dependent images with one button. Marked specimens can receive the same
  settings in one bulk save, each using its own calibration.
- A **Clean size** slider in Adjust's Explore and adjust box, for Otsu and
  BiRefNet specimens. Above 0 it live-previews Clean image, which drops small
  white specks and fills small enclosed holes without flash-filling the leaf.
  For Otsu specimens, **Overwrite this specimen** and **Overwrite all marked
  specimens** save that mask, re-measure it, and record the clean size;
  BiRefNet specimens preview it only.
- A run-wide clean size for pre-cleanup measurements: `--clean-size PX` in the
  CLI, **Clean size (px)** in Setup (default 0, off). It removes specks and
  fills enclosed holes whose inscribed radius is below that many pixels before
  measuring, exactly as the Adjust slider previews, and is recorded in each
  results CSV's `.meta.json`. It requires `--measure-pre-cleanup`.
- Optional measurements from pre-cleanup binary masks in the app and CLI
  (`--measure-pre-cleanup`), with per-CSV metadata recording the source.
  A band along the target-box edge (`--clean-margin`, **Edge margin** in the
  app; default 1% of the box's shorter side) is cleared first, so the
  template's printed box outline can never outweigh and replace a small leaf.
  The largest remaining object then anchors the measurement: pieces touching
  that band, or farther from the leaf than `--stray-gap` (**Stray-piece
  distance**; default 0.25 × the leaf's bounding-box diagonal), are dropped so
  printed box lines and distant debris no longer stretch width and length.
  Clean image (a clean size above 0), Remove flashfill, and the pre-cleanup
  threshold explorer apply
  the same cleanup; the default cleaned measurements are unchanged. In Setup
  these settings are grayed out unless pre-cleanup measurement is checked; each
  specimen's explorer can adjust them for its own preview and overwrite.
- Measure with Otsu and BiRefNet in one run: `--mask-method both`, or check both
  segmentation methods in the app. Markers are detected once per image; each
  method writes its own results CSV and failure log (`_threshold`/`_birefnet`
  suffixes) in the single-method schema, and the Results view can switch
  between them. Single-method runs write the same files as before.
- Separate pre-cleanup binary mask exports for threshold/Otsu and BiRefNet,
  including both methods in one run while measurements use the selected method.
- Optional overlay, cutout, and measurement-axis exports in the CLI and app.
- A custom grayscale threshold: `--threshold-level` accepts an integer cutoff
  `1`–`255`, and the app's **custom** threshold level shows a slider with the
  low/medium/high presets marked.
- A **Robust QR setup** sidebar page that explains the optional pyzbar and
  QReader fallbacks, reports their usable status, and keeps Conda optional.
- A **Help** page in the app (sidebar) with a quick start, a photography guide
  built around three packaged sample images, a printed-sheet calibration explainer,
  an Otsu-vs-BiRefNet comparison, a results-CSV column glossary, and
  troubleshooting.
- Three de-identified sample photographs ship with the wheel under
  `mats/app/assets/samples/`, resolvable via `mats.samples.SAMPLES_DIR`. One
  demonstrates an easy flat capture (`-t 6x6in`), one a hard hand-held field
  capture (`-t 10.5x9.5in`), and the third a real QR-read failure on the same
  field template, kept deliberately to show manual dimension entry recovering it.
- Installable package (`pip install -e .`) with a single `mats` console script:
  `run`, `app`, `fetch-weights`, `doctor`.
- `mats fetch-weights` and `mats doctor` for checkpoint management and
  environment diagnostics.
- Checkpoint resolution via `MATS_WEIGHTS_DIR` / per-checkpoint env vars, with a
  per-user cache default (`~/.cache/mats/weights`).

### Changed
- The Streamlit app now imports the pipeline as the `mats.core` package instead
  of loading a versioned file by path.
- Analyze now accepts the finished Template Creator sheet size and derives the
  marker-centre calibration area from the Creator's fixed margin policy. The
  previous direct calibration-area entry remains in a legacy/custom control.
- QR-mode Preflight lists OpenCV, pyzbar/zbar, and QReader separately and
  exercises a known QR through readers that do not trigger model downloads.
- The CLI adds `--sheet-dimensions`; legacy `-t/--template-dimensions` keeps its
  historical marker-centre meaning for existing scripts.
- The README explains the lightweight initial configuration and the explicit
  BiRefNet and robust-QR upgrade paths.
- The CLI and GUI share one execution path (`run_leaf_morpho_batch`); the CLI
  defaults to the full research CSV schema and writes a failures log.
- The Template Creator imports the dimension parser from `mats.dimensions`
  instead of duplicating the regex.
- Open OnDemand launcher uses `mats app` (no repo-path assumption).
- `--mask-method` / `mask_method` / the GUI segmentation selector now default to
  `threshold` (Otsu) instead of `birefnet`, so a default run needs no GPU and
  no extra checkpoint download. `mats fetch-weights` now fetches RF-DETR only
  by default; `--all` or `--only birefnet` fetches the BiRefNet checkpoint.
- `mats doctor` no longer treats a missing BiRefNet checkpoint as an error
  (exit 1); it's reported as optional, since it's only needed for
  `--mask-method birefnet`.

### Fixed
- `parse_template_dimensions` now accepts a unit repeated after the width
  (e.g. `6inx6in`), not just after the height.
- `weights/birefnet_leaf.pth` is no longer committed to the repository, so a
  plain `git clone` no longer downloads the 2.65 GB checkpoint.
  `weights/rf_detr_marker.pth` (~134 MB) stays committed via Git LFS, since
  it's mandatory for every run and Hugging Face auto-fetch isn't configured
  yet.

### Notes
- First public extraction of the pipeline from the manuscript repository.
- Model weights are hosted on Hugging Face with transparent first-run auto-fetch,
  a shared-filesystem option (`MATS_WEIGHTS_DIR`, e.g. SCINet `/project`), and
  optional Git LFS (see `docs/weights.md`).
