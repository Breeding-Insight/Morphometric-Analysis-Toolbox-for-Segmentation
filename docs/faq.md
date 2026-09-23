# FAQ

Short answers for people who just cloned MATS. For the full reference see
[README.md](../README.md), [cli.md](cli.md), [gui.md](gui.md),
[templates.md](templates.md), [weights.md](weights.md), and [hpc.md](hpc.md).

---

## Installing

**What do I need?** Python ≥ 3.9 and Git LFS. Apart from Git LFS the install
pulls everything from wheels, with no conda environment and no other system
libraries.

```bash
git lfs install             # FIRST -- see the next question
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
pip install -e ".[app]"     # ".[app]" adds the Streamlit app; drop it for CLI only
mats doctor                 # confirms checkpoints, device, and QR decoders
```

**Why do I need Git LFS?** The RF-DETR marker checkpoint (~134 MB) is stored
with [Git LFS](https://git-lfs.com), and it is required for every run. If you
clone without Git LFS installed, **the clone still succeeds** — but you get a
134-byte placeholder instead of the model, and MATS can't detect markers.

Check it:

```bash
ls -l weights/rf_detr_marker.pth      # ~134 MB = good; ~134 bytes = placeholder
```

Repair an existing clone without re-cloning:

```bash
git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"
```

The `--exclude` keeps the repair to the ~134 MB RF-DETR file. With this
repository's `lfs.fetchexclude`, a bare `git lfs pull` also leaves the 2.65 GB
BiRefNet checkpoint out; fetch it explicitly only when needed:

```bash
git lfs pull -X "" -I "weights/birefnet_leaf.pth"
```

**Do I need to download the model first?** No. The clone brings RF-DETR with it,
so there is no separate download step — run `mats doctor` and you're done. (If
it reports the checkpoint missing, `mats fetch-weights` repairs it.) The large
BiRefNet checkpoint is the opposite: it is never downloaded unless you ask.

**Why doesn't the install just download everything?** So the first install stays
predictable on laptops, managed machines, and clusters. You get fast Otsu
segmentation and OpenCV's QR reader immediately; the ~2.65 GB BiRefNet
checkpoint and the extra QR decoders are added only if your photographs need
them.

**Do I need conda?** No. `conda` works if you already use it
(`environment.yml` is provided for clusters), but nothing requires it.

**Do I need `zbar`?** Only for the optional `pyzbar` QR fallback. OpenCV reads
clear QR codes with no extra setup, and you can always enter the sheet size by
hand instead.

**`mats: command not found`** — the console script landed outside your `PATH`,
usually from installing into a different interpreter. Check with
`python -m pip show mats-morpho`, and use the same Python you installed with:
`python -m mats.cli ...` works as a fallback.

---

## Getting started

**Fastest path from clone to a measurement?** Print a template, photograph
leaves on it, then:

```bash
mats app                                                      # point and click
mats run -i ./images -o ./out -r results.csv --sheet-dimensions 12x12in
```

Both routes run exactly the same pipeline and produce the same numbers.

**I don't have photographs yet.** Three de-identified sample images ship with
the package. Open `mats app` → **Help** in the sidebar: it walks through an easy
flat capture, a hard hand-held field capture, and a real QR-read failure, and
explains the settings each one needs.

**What do I print?** Open **Template Creator** in the app, enter your finished
sheet's width and height, and download the PDF (an editable IDML is also
offered). Rules worth knowing: dimensions move in 0.5-unit steps, the
width-to-length ratio can't exceed 1.5:1, and margins and marker size are
derived for you. See [templates.md](templates.md).

**Print at 100 % scale.** "Fit to page" or "shrink to fit" rescales the markers
and silently corrupts every measurement from that sheet.

**How should I photograph the sheet?** Flat, evenly lit, with all four corner
markers inside the frame and the leaves inside the printed box. A hand-held
photo at a slight angle is fine — perspective correction handles it — but a
curled or folded sheet is not.

---

## Running

**What size do I enter — the sheet or the box?** The **finished printed sheet**
(for example `12x12in`). MATS derives the marker-centre calibration area from
it. Older or custom templates with different margins can still supply that area
directly via `-t/--template_dimensions` in the CLI, or the compatibility control
in the app.

**Can it read the size from the template instead?** Yes — the Template Creator
puts a QR code on the sheet. In the app, tick **Variable dimensions, read QR
code**; in the CLI, leave `--sheet-dimensions` off. Per-image QR codes are what
you want when a batch mixes several template sizes.

**Otsu or BiRefNet?**

| | Otsu threshold (default) | BiRefNet |
|---|---|---|
| Speed | Fast, CPU-friendly | Slow without a GPU |
| Extra download | None | ~2.65 GB checkpoint |
| Best for | Clean, high-contrast backgrounds (a leaf on plain white) | Cluttered or low-contrast backgrounds |

Start with the default. Switch with `--mask-method birefnet` only if the masks
disappoint you.

**How many workers?** `-w/--workers` applies to the threshold path. One worker
uses CUDA/MPS when available; two or more switch to parallel CPU processing and
disable CUDA/MPS for that run. The app shows the CPUs allocated to it and warns
above 75 % usage.

**Does it need a GPU?** No. The default path is CPU-only. A GPU helps only with
BiRefNet.

---

## Model weights

**Where do they come from?** Both checkpoints are tracked in this repository with
Git LFS. A normal `git clone` brings RF-DETR (~134 MB, required for every run).
BiRefNet (~2.65 GB) is meant to stay out of the default clone, so it
arrives only when you ask:

```bash
mats fetch-weights --only birefnet --source lfs
```

**My network blocks huggingface.co.** Use `--source lfs`, which fetches through
the Git remote instead.

**A checkpoint seems corrupt / torch complains about the file.** You probably
have a Git-LFS pointer stub — a ~130-byte text file, not the model. Run
`git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"`. `mats doctor` flags this case.

**Shared filesystem (lab server, HPC).** Stage the checkpoints once and point
everyone at them; nobody needs a personal copy:

```bash
export MATS_WEIGHTS_DIR=/project/your_project/mats_weights
export MATS_NO_AUTO_FETCH=1      # fail fast instead of downloading on a login node
```

Full detail, checksums, and the complete resolution order: [weights.md](weights.md).

---

## Results

**What comes out?** Per image, a perspective-corrected `{sample_id}_target_box.jpg`
and a `{sample_id}_mask.png`, plus one measurements CSV and a
`leaf_morpho_failures.csv` listing anything that warned or failed.

**Which columns?** `--csv-schema full` (the default) gives `sample_id`,
`leaf_area_cm2`, `width_cm`, `length_cm`, `px_per_cm_width`, `px_per_cm_height`,
`scale_aspect_ratio`, `source`, plus a QR trace when sizes are read from codes.
`--csv-schema compact` gives just `sample_id`, area, width, and length. The GUI's
Help page has a glossary for every column.

**Can I get millimetres or inches?** Yes — `--results-unit mm|cm|in` (or
**Result units** in the app). The unit-bearing column names change to match. It
changes the reported units only, never the calibration maths.

**Why isn't `scale_aspect_ratio` exactly 1.0?** Small deviations are normal.
MATS calibrates each axis independently, so this column is your quality check: a
value far from 1.0 means the horizontal and vertical scales disagree, which
usually points to a skewed print, a non-flat sheet, or strong lens distortion.
Re-print or re-photograph before trusting those rows.

**I have CSVs from an older version.** Columns ending `_meanscale`,
`_widthscale`, and `_heightscale` predate the current anisotropic output. The
README's migration note gives the exact conversion — old files stay usable.

**Can I check the length/width axes visually?** Add `--save-axes` to write
overlay images alongside the masks.

---

## Troubleshooting

Run `mats doctor` first; it reports most of these.

| Symptom | Fix |
|---|---|
| No markers detected, on a fresh clone | Check for a Git LFS placeholder first: `ls -l weights/rf_detr_marker.pth` should be ~134 MB. If it's ~134 bytes, run `git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"` |
| "QR code not read" | Pass the size yourself: `--sheet-dimensions 12x12in`. To add decoders: `pip install -e ".[qr]"` (plus the native `zbar` for pyzbar) |
| No markers detected | Get all four markers in frame; print at 100 % scale in the template's marker colour |
| Masks include the background | Try `--threshold-level low/medium/high`, or `--mask-method birefnet` |
| CUDA out of memory | Only with BiRefNet — process fewer images at a time, or use the default `threshold` |
| Very slow run | Use `threshold` and raise `-w/--workers` |
| Blank page in Open OnDemand | Reverse-proxy `baseUrlPath` mismatch — see [deploy/ondemand/mats/README.md](../deploy/ondemand/mats/README.md) |

---

## Clusters and support

**HPC?** MATS runs as an ordinary batch job, and an Open OnDemand app serves the
GUI on a compute node. See [hpc.md](hpc.md).

**Something else is wrong.** Open an issue at
[github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation/issues](https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation/issues)
and include the output of `mats doctor`, the command you ran, and the error.

**Citing MATS.** See [CITATION.cff](../CITATION.cff).
