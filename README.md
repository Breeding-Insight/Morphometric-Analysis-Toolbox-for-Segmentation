# MATS — Morphometric Analysis Toolbox for Segmentation

Measure leaf **area, length, and width** in real-world units from a photo of
leaves laid on a printed calibration template.

MATS has four main steps:
1. Locate four fiducial markers using an RF-DETR detection model.
2. Applies a transform to undo any perspective distortion.
3. Segments each leaf, using either a fast Otsu threshold (default option) or a BiRefNet segmentation model for tougher backgrounds.
4. Writes out a CSV of the measurements.

> Companion code for the manuscript (target journal: *Plant Phenomics*).
> BiRefNet is optional and runs entirely from a locally installed checkpoint (see [Model weights](#model-weights)).

## Table of Contents  

1. [Installation](#installation)  
3. [Model weights](#model-weights)
4. 2. [Setup??](#)

5. [Outputs](#outputs)
6. [Running on a compute cluster](#running-on-a-compute-cluster)
7. [How it works](#how-it-works)
8. [Troubleshooting](#troubleshooting)
9. [Citing](#citing)
10. [License](#license)

---



## Installation

MATS requires Python ≥ 3.9 and **Git LFS**. Apart from Git LFS, the recommended
`pip` installation method pulls all required packages from wheels with **no other
system libraries and no conda required**.

### Step 1 — install Git LFS *before* cloning

The RF-DETR marker checkpoint (~134 MB) is stored with [Git LFS](https://git-lfs.com).
**If you clone without it, you get a 134-byte placeholder file instead of the
model** — the clone appears to succeed, and MATS then can't detect markers.

```bash
# macOS:          brew install git-lfs
# Debian/Ubuntu:  sudo apt install git-lfs
# Conda:          conda install -c conda-forge git-lfs
# Windows:        included with Git for Windows
# RHEL/Fedora:    sudo dnf install git-lfs

git lfs install          # one-time setup, per machine
```

### Step 2 — clone and install

```bash
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
pip install -e ".[app]"                  # ".[app]" adds the Streamlit GUI
```

The clone brings the RF-DETR checkpoint with it. Confirm it is the real file and
not a placeholder — it should be ~134 MB, not ~134 bytes:

```bash
ls -l weights/rf_detr_marker.pth
```

**Already cloned without Git LFS?** No need to start over — install Git LFS as
above, then repair the checkout in place. The `--exclude` keeps this to the
~134 MB RF-DETR file; a bare `git lfs pull` can also fetch the 2.65 GB BiRefNet
checkpoint:

```bash
git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"
```

**Enhanced QR reading (optional).** OpenCV reads QR codes well when they are oriented correctly and clearly
visible. For images with any issues affecting the QR codes (glare, skew, blur) you can add the `pyzbar` + `qreader` fallbacks.
`pyzbar` requires the system library `zbar`:

```bash
pip install -e ".[app,qr]"
```

This enables QReader without Conda. To enable the additional `pyzbar`
fallback, install its native `zbar` library with your existing environment:

```bash
# Linux: apt install libzbar0
# macOS: brew install zbar
# Existing Conda environment only: conda install -c conda-forge zbar
```

If a code can't be read, the pipeline continues — pass the finished sheet size
with `--sheet-dimensions` (for example, `--sheet-dimensions 12x12in`), so
enhanced QR is a convenience rather than a requirement.

Finally, confirm the environment:

```bash
mats doctor             # checks weights, GPU/CPU device, QR backends
```

Run `mats doctor` after installing, this will report MATS operable status.
Note: RF-DETR weights are REQUIRED for marker detection, BiRefNet is OPTIONAL.
```bash
mats fetch-weights                               # repairs a clone made without Git LFS
mats fetch-weights --only birefnet --source lfs  # optional: the ~2.65 GB BiRefNet checkpoint
```

---

## Model weights

The checkpoints for both the marker detection model and the leaf segmentation model are located in this repository, tracked with Git LFS:

| Model | File | Size |
|---|---|---|
| RF-DETR marker detector | `rf_detr_marker.pth` | ~134 MB |
| BiRefNet leaf segmenter | `birefnet_leaf.pth` | ~2.65 GB |

By default, only the RF-DETR model checkpoint will be downloaded: it arrives with every
`git clone` made with Git LFS installed, so a normal checkout is immediately runnable.
The BiRefNet checkpoint is LFS-tracked but excluded from the default clone, so it is
downloaded only through an explicit action:

- **Otsu (default)** — needs no BiRefNet checkpoint and never downloads one.
- **BiRefNet (optional)** — fetch explicitly with
  `mats fetch-weights --only birefnet --source lfs`, or use the setup page.
- **Shared filesystem** — set `MATS_WEIGHTS_DIR` (e.g. a SCINet `/project` path)
  to read weights in place with no per-user copy.

**Cloned without Git LFS?** Both files come through as ~134-byte pointer stubs
rather than models, which MATS detects and reports rather than handing to
PyTorch. Fix it with `git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"`,
or `mats fetch-weights`.

Full details and checksums: [docs/weights.md](docs/weights.md).


### Why the first installation is lightweight

MATS installs in a lightweight **operating configuration** for convenience. The
standard app includes fast Otsu segmentation and OpenCV's built-in QR reader,
but it does not automatically download the optional ~2.65 GB BiRefNet
checkpoint or install the pyzbar/QReader robust-QR fallbacks.

The required ~134 MB RF-DETR marker checkpoint is different: it is mandatory for
every run, so it ships **in the clone** via Git LFS and needs no separate
download step. If it is ever missing — a clone made without Git LFS, or an
install outside a Git checkout — MATS fetches it once on first use and prints
`Fetching weights/rf_detr_marker.pth via Git LFS ...` while it does. Set
`MATS_NO_AUTO_FETCH=1` to turn that off and require pre-staged weights instead
(recommended on HPC login nodes). The app never does this silently: a missing
RF-DETR checkpoint is a blocking Preflight error.

This keeps the initial network and disk footprint predictable, avoids native
`zbar` failures on managed machines, and works better on HPC systems and
restricted networks. Start with the standard path, then add only what the
photographs require:

| Capability | Included initially | Add when needed |
|---|---|---|
| Otsu leaf segmentation | Yes | Nothing |
| Clear QR codes with OpenCV | Yes | Nothing |
| RF-DETR marker detection | Yes — checkpoint ships in the clone (Git LFS) | Nothing |
| BiRefNet segmentation | No checkpoint | `mats fetch-weights --only birefnet --source lfs` |
| Robust QR fallbacks | No | `pip install "mats-morpho[app,qr]"` |

QReader can download its detector model when that fallback is first used.
pyzbar requires the native `zbar` library described above. The app's Preflight
and setup pages show exactly which readers and checkpoints are available before
a run.

---

## Running MATS

- **I want to click buttons →** [Using the app](#using-the-app)
- **I want to script it →** [Using the command line](#using-the-command-line)
- **I have a question →** [FAQ](docs/faq.md)

Both run the exact same pipeline and produce the same measurements.

---

### Using the app

MATS comes with a point-and-click user interface. To open it, simply run:

```bash
mats app
```

This will open the Streamlit app locally in your web browser. From there:

1. **Pick images** — a local folder, or drag-and-drop uploads.
2. **Set the scale** — enter the printed sheet's width, height, and unit (e.g.
   `10.5 x 9.5 in`), or tick **Variable dimensions, read QR code** to read it
   from each image's template QR code automatically.
3. **Choose segmentation** — Otsu threshold (fast, default) or BiRefNet (accurate, must have local model checkpoint installed).
4. **Choose workers** — the app detects the number of CPUs available to it. One worker uses
   CUDA/MPS when available; two or more workers use parallel CPU processing and
   disable CUDA/MPS for that run. A colored warning light shows CPU allocation;
   counts above 75% require a one-run break-glass acknowledgement.
5. **Run**, then preview results and download a CSV or a ZIP of masks + boxes.

**Getting oriented.** The app's **Help** page (sidebar) ships three annotated
sample photographs — including a real QR-read failure that shows why printed
sheet entry is the most consistent option — a photography checklist, and a glossary
for every results-CSV column. See [docs/gui.md](docs/gui.md).

**Printing templates.** The app has a **Template Creator** page (in the sidebar)
that accepts only the finished sheet's width and length, then automatically
sizes the observation area and corner markers. Download the print-ready PDF or
an editable Adobe InDesign IDML; the PDF is recommended for final printing.
Print at 100% scale (no "fit to page"), lay your leaves inside the box, and
photograph it flat. See [docs/templates.md](docs/templates.md).

---

### Using the command line

```bash
mats run -i ./images -o ./out -r results.csv --sheet-dimensions 12x12in
```

Common options (full reference in [docs/cli.md](docs/cli.md)):

| Flag | Meaning | Default |
|---|---|---|
| `-i, --input_dir` | Folder of images to measure | prompt |
| `-o, --output_dir` | Where masks / target boxes are written | prompt |
| `-r, --results_path` | Measurement CSV path | `./leaf_morpho_results.csv` |
| `--sheet-dimensions` | Finished Template Creator sheet size, `<w>x<h><unit>` | read from QR |
| `-t, --template_dimensions` | Legacy/custom marker-centre calibration area | unused |
| `--mask-method` | `birefnet` (accurate, GPU) or `threshold` (fast) | `threshold` |
| `--threshold-level` | `auto` (Otsu) / `low` / `medium` / `high` | `auto` |
| `--csv-schema` | `full` (area/width/length + per-axis pixels-per-selected-unit) or `compact` | `full` |
| `--results-unit` | Measurement-output unit: `mm`, `cm`, or `in` | `cm` |
| `-w, --workers` | Parallel workers (threshold path only) | auto |
| `--save-axes` | Also save length/width overlay images for QC | off |

**Choosing a segmentation method.** `threshold` (Otsu) is the default — fast,
no GPU, no extra download, and good for clean, high-contrast backgrounds where
a leaf sits on plain white. `birefnet` is more accurate on cluttered or
low-contrast backgrounds and uses a GPU when available (CPU works but is
slow), at the cost of the ~2.65 GB checkpoint — fetch it once with
`mats fetch-weights --only birefnet`.

---

## Outputs

Per image, in the output folder:

- `{sample_id}_target_box.jpg` — the perspective-corrected observation box
- `{sample_id}_mask.png` — the leaf segmentation mask

Plus a measurements CSV. Choose `mm`, `cm` (the default), or `in` with
`--results-unit` in the CLI or the **Result units** control in the app. The
selection changes results, dashboard labels, and unit-bearing CSV column names;
it does not change calibration math. Two schemas:

- **full** (default, research schema) — `sample_id, leaf_area_cm2, width_cm,
  length_cm, px_per_cm_width, px_per_cm_height, scale_aspect_ratio, source`.
  When dimensions are read from QR codes, it also appends a trace column for
  OpenCV and each optional decoder installed locally, showing which decoder
  succeeded or whether a fallback failed or was unused.
  Scaling is **anisotropic**: the x-extent (`width_cm`) is divided by
  `px_per_cm_width`, the y-extent (`length_cm`) by `px_per_cm_height`, and area
  by their product — each axis calibrated independently against the template,
  rather than one averaged scalar applied to everything. `scale_aspect_ratio`
  (`px_per_cm_width / px_per_cm_height`) is a QC signal: it should sit near 1.0,
  and a value far from 1.0 flags a calibration problem (skewed template print,
  lens distortion, a non-planar sheet) worth investigating.
- **compact** — `sample_id, area_cm2, width_cm, length_cm` by default. With
  millimeters or inches selected, `cm` is replaced consistently in the
  measurement column names.

A `leaf_morpho_failures.csv` records per-image warnings and failures.

> **Migration note:** earlier versions reported three isotropic scale
> conventions (`*_meanscale`, `*_widthscale`, `*_heightscale`). Old CSVs remain
> usable — the new `leaf_area_cm2` can be recovered from an old row with
> `leaf_area_cm2_widthscale * (px_per_cm_width / px_per_cm_height)`, and the new
> `width_cm`/`length_cm` equal the old `width_cm_widthscale`/`length_cm_heightscale`.

---

## Running on a compute cluster

An Open OnDemand Batch Connect app that serves the GUI on a compute node is in
[deploy/ondemand/mats/](deploy/ondemand/mats/). See its README and
[docs/hpc.md](docs/hpc.md).

---

## How it works

MATS chains two models. **RF-DETR** (fine-tuned, single "Marker" class) detects
the four corner fiducials at 1120×1120 px; their centroids define a homography
that rectifies the observation box. The rectified box's pixel width and height
are compared against the template's known physical size, independently per
axis, to fix `px_per_cm_width` and `px_per_cm_height`. **BiRefNet** (fine-tuned
for leaf foreground) then segments the leaf, from which area (pixel count) and
length/width (bounding dimensions) are computed and converted to centimeters
using their respective axis scale. A classic Otsu threshold is offered as a
fast alternative to BiRefNet. See the manuscript for training and evaluation
detail.

---

## Troubleshooting

Run `mats doctor` first — it reports most of these, and the [FAQ](docs/faq.md)
covers the common questions in more detail.

- **QR code not read / measurements need a scale** — the default OpenCV decoder
  couldn't read the code. Pass the finished sheet size with
  `--sheet-dimensions` (e.g. `--sheet-dimensions 12x12in`), or add enhanced QR
  reading: `pip install -e ".[qr]"` plus the
  `zbar` system lib (Linux: `apt install libzbar0`; macOS: `brew install zbar`;
  conda: `conda install -c conda-forge zbar`).
- **CUDA out of memory** (only relevant with `--mask-method birefnet`) — process
  in smaller batches, or use `--mask-method threshold` (the default).
- **No markers detected / "RF-DETR checkpoint missing"** — first check that the
  checkpoint is a real file and not a Git LFS placeholder:
  `ls -l weights/rf_detr_marker.pth` should show ~134 MB, not ~134 bytes. If it's
  a placeholder, run `git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"`. Otherwise check print
  quality and that the marker color
  matches the template (the Template Creator uses the trained color); make sure
  all four corners are in frame.
- **Blank page on Open OnDemand** — almost always the reverse-proxy
  `baseUrlPath` mismatch; see [deploy/ondemand/mats/README.md](deploy/ondemand/mats/README.md).

---

## Working with an AI assistant

This repository ships agent instructions in [AGENTS.md](AGENTS.md) (with a
companion [CLAUDE.md](CLAUDE.md)), so a coding assistant you point at your clone
— Claude Code, Codex, Cursor, Copilot, Gemini CLI — already knows how MATS is
installed, run, and structured, and can help you troubleshoot a batch.

## Citing

If you use MATS, please cite the manuscript.
See [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE). The pipeline builds on RF-DETR (Apache-2.0) and BiRefNet (MIT);
see [docs/weights.md](docs/weights.md) for model provenance.
