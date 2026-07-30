# MATS — Morphometric Analysis Toolbox

Measure leaf **area, length, and width** in real-world units from a photo of
leaves laid on a printed calibration template. MATS finds four fiducial markers
with RF-DETR, corrects perspective, segments each leaf with a fast Otsu
threshold by default (or the heavier BiRefNet model for tougher backgrounds),
and writes a measurements CSV.

Pipeline in one line: **detect markers → perspective-correct → segment leaf →
measure → CSV**.

> Companion code for the manuscript (target journal: *Plant Phenomics*).
> Model weights are hosted available for download here or usable through hugging face (see [Model weights](#model-weights)).
> For USDA users the model weights are hosted on Agdatacommons and the pipeline is available on SciNET

---

## Install

MATS needs only Python ≥ 3.9 — QR codes are decoded with OpenCV, so the default
install pulls everything from wheels with **no system libraries and no conda
required**.

**pip (recommended):**

```bash
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
pip install -e ".[app]"                  # ".[app]" adds the Streamlit GUI
```

**Enhanced QR reading (optional).** OpenCV decodes clean codes reliably; for
tougher photos (glare, skew, blur) you can add the `pyzbar` + `qreader`
fallbacks. `pyzbar` needs the system library `zbar`:

```bash
pip install -e ".[app,qr]"
# then install zbar:  Linux: apt install libzbar0
#                     macOS: brew install zbar
#                     conda: conda install -c conda-forge zbar
```

If a code can't be read, the pipeline continues — just pass the template size
manually with `-t` (e.g. `-t 10.5x9.5in`), so enhanced QR is a convenience, not
a requirement.

Then fetch the model weights once and confirm the environment:

```bash
mats fetch-weights      # fetches the ~134 MB RF-DETR checkpoint (mandatory, default)
mats fetch-weights --all # also fetches the ~2.65 GB BiRefNet checkpoint
mats doctor             # checks weights, GPU/CPU device, QR backends
```

---

## Choose your path

- **I want to click buttons →** [Using the app](#using-the-app)
- **I want to script it →** [Using the command line](#using-the-command-line)

Both run the exact same pipeline and produce the same measurements.

---

## Using the app

```bash
mats app
```

This opens the Streamlit GUI in your browser. From there:

1. **Pick images** — a local folder, or drag-and-drop uploads.
2. **Set the scale** — enter the printed sheet's width, height, and unit (e.g.
   `10.5 x 9.5 in`), or tick **Variable dimensions, read QR code** to read it
   from each image's template QR code automatically.
3. **Choose segmentation** — Otsu threshold (fast, default) or BiRefNet (accurate).
4. **Choose workers** — the app detects the CPUs assigned to it. One worker uses
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

## Using the command line

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

## Model weights

The checkpoints are too large for GitHub, so they are hosted on Hugging Face and
resolved at runtime:

| Model | File | Size |
|---|---|---|
| RF-DETR marker detector | `rf_detr_marker.pth` | ~134 MB |
| BiRefNet leaf segmenter | `birefnet_leaf.pth` | ~2.65 GB |

RF-DETR is committed to this repo via Git LFS, so a plain `git clone` gets it
automatically — an interim delivery mechanism until Hugging Face hosting is
configured. BiRefNet is **not** committed (that would force every clone to
download 2.65 GB); it's fetched separately, only when you actually use it:

- **Auto-fetch (default)** — `mats fetch-weights` downloads the mandatory
  RF-DETR checkpoint; BiRefNet is fetched only on request (first use of
  `--mask-method birefnet`, or `mats fetch-weights --only birefnet` / `--all`).
- **Shared filesystem** — set `MATS_WEIGHTS_DIR` (e.g. a SCINet `/project` path)
  to read weights in place with no per-user copy.
- **Git LFS** — RF-DETR is committed to this repo (interim, until Hugging Face
  hosting is live); BiRefNet is intentionally not committed.

Set `MATS_NO_AUTO_FETCH=1` to disable the automatic download (e.g. on an HPC login
node). Full detail, DOI, and checksums: [docs/weights.md](docs/weights.md).

---

## On a cluster (HPC / Open OnDemand)

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

If you use MATS, please cite the manuscript and the Hugging Face weights deposit.
See [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE). The pipeline builds on RF-DETR (Apache-2.0) and BiRefNet (MIT);
see [docs/weights.md](docs/weights.md) for model provenance.
