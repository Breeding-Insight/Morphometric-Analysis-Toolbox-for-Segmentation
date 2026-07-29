# Changelog

All notable changes to MATs are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Installable package (`pip install -e .`) with a single `mats` console script:
  `run`, `app`, `fetch-weights`, `doctor`.
- `mats fetch-weights` and `mats doctor` for checkpoint management and
  environment diagnostics.
- Checkpoint resolution via `MATS_WEIGHTS_DIR` / per-checkpoint env vars, with a
  per-user cache default (`~/.cache/mats/weights`).

### Changed
- The Streamlit app now imports the pipeline as the `mats.core` package instead
  of loading a versioned file by path.
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
