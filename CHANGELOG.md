# Changelog

All notable changes to MATs are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
