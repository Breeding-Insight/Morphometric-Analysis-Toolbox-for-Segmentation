# MATS — Morphometric Analysis Toolbox

Measure leaf **area, length, and width** in real-world units from a photo of
leaves laid on a printed calibration template.

MATS has four main steps:
1. Locate four fiducial markers using an RF-DETR detection model.
2. Applies a transform to undo any perspective distortion.
3. Segments each leaf, using either a fast Otsu threshold (default option) or a BiRefNet segmentation model for tougher backgrounds.
4. Writes out a CSV of the measurements.

> Companion code for the manuscript (target journal: *Plant Phenomics*).
> BiRefNet is optional and runs entirely from a locally installed checkpoint (see [Model weights](#model-weights)).
> For USDA users the model weights are hosted on Agdatacommons and the pipeline is available on SciNET

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

MATS requires only Python ≥ 3.9; the recommended `pip` installation method
pulls all required packages from wheels with **no system libraries and no conda
required**.

**pip (recommended):**

```bash
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
pip install -e ".[app]"                  # ".[app]" adds the Streamlit GUI
```

**Enhanced QR reading (optional).** OpenCV reads QR codes well when they are oriented correctly and clearly
visible. For images with any issues affecting the QR codes (glare, skew, blur) you can add the `pyzbar` + `qreader` fallbacks.
`pyzbar` requires the system library `zbar`:

```bash
pip install -e ".[app,qr]"
# then install zbar:  Linux: apt install libzbar0
#                     macOS: brew install zbar
#                     conda: conda install -c conda-forge zbar
```

If a code can't be read, the pipeline continues — just pass the template size
manually with `-t` (e.g. `-t 10.5x9.5in`), so enhanced QR is a convenience, not
a requirement.

Finally, fetch the model weights and confirm the environment:

```bash
mats fetch-weights      # fetches the ~134 MB RF-DETR checkpoint (mandatory, default)
mats fetch-weights --only birefnet --source lfs # optional: explicitly fetch the ~2.65 GB BiRefNet checkpoint
mats doctor             # checks weights, GPU/CPU device, QR backends
```
---

## Model weights

The checkpoints for both the marker detection model and the leaf segmentation model are located in this repository:

| Model | File | Size |
|---|---|---|
| RF-DETR marker detector | `rf_detr_marker.pth` | ~134 MB |
| BiRefNet leaf segmenter | `birefnet_leaf.pth` | ~2.65 GB |

By default, only the RF-DETR model checkpoint will be downloaded.
The BiRefNet checkpoint is LFS-tracked but excluded from the default clone, so it is
downloaded only through an explicit action:

- **Otsu (default)** — needs no BiRefNet checkpoint and never downloads one.
- **BiRefNet (optional)** — fetch explicitly with
  `mats fetch-weights --only birefnet --source lfs`, or use the setup page.
- **Shared filesystem** — set `MATS_WEIGHTS_DIR` (e.g. a SCINet `/project` path)
  to read weights in place with no per-user copy.

Full details and checksums: [docs/weights.md](docs/weights.md).


---

## Running MATS

- **I want to click buttons →** [Using the app](#using-the-app)
- **I want to script it →** [Using the command line](#using-the-command-line)

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

**Printing templates.** The app has a **Template Creator** page (in the sidebar)
that accepts only the finished sheet's width and length, then automatically
sizes the observation area and corner markers. Download the print-ready PDF or
an editable Adobe InDesign IDML; the PDF is recommended for final printing.
Print at 100% scale (no "fit to page"), lay your leaves inside the box, and
photograph it flat. See [docs/templates.md](docs/templates.md).

---

### Using the command line

```bash
mats run -i ./images -o ./out -r results.csv -t 10.5x9.5in
```

Common options (full reference in [docs/cli.md](docs/cli.md)):

| Flag | Meaning | Default |
|---|---|---|
| `-i, --input_dir` | Folder of images to measure | prompt |
| `-o, --output_dir` | Where masks / target boxes are written | prompt |
| `-r, --results_path` | Measurement CSV path | `./leaf_morpho_results.csv` |
| `-t, --template_dimensions` | Box size, `<w>x<h><unit>` (else read from QR) | QR fallback |
| `--mask-method` | `birefnet` (accurate, GPU) or `threshold` (fast) | `threshold` |
| `--threshold-level` | `auto` (Otsu) / `low` / `medium` / `high` | `auto` |
| `--csv-schema` | `full` (area/width/length + per-axis px-per-cm) or `compact` | `full` |
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

Plus a measurements CSV. Two schemas:

- **full** (default, research schema) — `sample_id, leaf_area_cm2, width_cm,
  length_cm, px_per_cm_width, px_per_cm_height, scale_aspect_ratio, source`.
  Scaling is **anisotropic**: the x-extent (`width_cm`) is divided by
  `px_per_cm_width`, the y-extent (`length_cm`) by `px_per_cm_height`, and area
  by their product — each axis calibrated independently against the template,
  rather than one averaged scalar applied to everything. `scale_aspect_ratio`
  (`px_per_cm_width / px_per_cm_height`) is a QC signal: it should sit near 1.0,
  and a value far from 1.0 flags a calibration problem (skewed template print,
  lens distortion, a non-planar sheet) worth investigating.
- **compact** — `sample_id, area_cm2, width_cm, length_cm`.

A `leaf_morpho_failures.csv` records per-image warnings and failures.

> **Migration note:** earlier versions reported three isotropic scale
> conventions (`*_meanscale`, `*_widthscale`, `*_heightscale`). Old CSVs remain
> usable — the new `leaf_area_cm2` can be recovered from an old row with
> `leaf_area_cm2_widthscale * (px_per_cm_width / px_per_cm_height)`, and the new
> `width_cm`/`length_cm` equal the old `width_cm_widthscale`/`length_cm_heightscale`.

---



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

Run `MATS doctor` first — it reports most of these.

- **QR code not read / measurements need a scale** — the default OpenCV decoder
  couldn't read the code. Pass the template size manually with `-t` (e.g.
  `-t 10.5x9.5in`), or add enhanced QR reading: `pip install -e ".[qr]"` plus the
  `zbar` system lib (Linux: `apt install libzbar0`; macOS: `brew install zbar`;
  conda: `conda install -c conda-forge zbar`).
- **CUDA out of memory** (only relevant with `--mask-method birefnet`) — process
  in smaller batches, or use `--mask-method threshold` (the default).
- **No markers detected** — check print quality and that the marker color
  matches the template (the Template Creator uses the trained color); make sure
  all four corners are in frame.
- **Blank page on Open OnDemand** — almost always the reverse-proxy
  `baseUrlPath` mismatch; see [deploy/ondemand/mats/README.md](deploy/ondemand/mats/README.md).

---

## Citing

If you use MATS, please cite the manuscript.
See [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE). The pipeline builds on RF-DETR (Apache-2.0) and BiRefNet (MIT);
see [docs/weights.md](docs/weights.md) for model provenance.
