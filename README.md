# MATs — Morphometric Analysis Tools

Measure leaf **area, length, and width** in real-world units from a photo of
leaves laid on a printed calibration template. MATs finds four fiducial markers
with RF-DETR, corrects perspective, segments each leaf with BiRefNet (or a fast
Otsu threshold), and writes a measurements CSV.

Pipeline in one line: **detect markers → perspective-correct → segment leaf →
measure → CSV**.

> Companion code for the manuscript (target journal: *Plant Phenomics*).
> Model weights are hosted on Hugging Face (see [Model weights](#model-weights)).

---

## Install

MATs needs Python ≥ 3.9 and the system library `zbar` (for QR decoding). Conda
handles `zbar` for you and is the recommended route.

**Conda (recommended):**

```bash
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
conda env create -f environment.yml     # env "mats", includes zbar
conda activate mats
pip install -e ".[app]"                  # ".[app]" adds the Streamlit GUI
```

**pip:**

```bash
pip install -e ".[app]"
# then install zbar yourself:  Linux: apt install libzbar0
#                              macOS: brew install zbar
```

Then fetch the model weights once and confirm the environment:

```bash
mats fetch-weights      # downloads ~2.6 GB to ~/.cache/mats/weights
mats doctor             # checks weights, GPU/CPU device, zbar
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
2. **Set the scale** — type the observation-box size (e.g. `10.5x9.5in`) or
   leave it blank to read it from the template's QR code automatically.
3. **Choose segmentation** — BiRefNet (accurate) or Otsu threshold (fast).
4. **Run**, then preview results and download a CSV or a ZIP of masks + boxes.

**Printing templates.** The app has a **Template Creator** page (in the sidebar)
that generates a print-ready PDF at any canvas/box size, with correctly colored
corner markers and a QR code encoding the box dimensions. Print it at 100%
scale (no "fit to page"), lay your leaves inside the box, and photograph it flat.
See [docs/templates.md](docs/templates.md).

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
| `--mask-method` | `birefnet` (accurate, GPU) or `threshold` (fast) | `birefnet` |
| `--threshold-level` | `auto` (Otsu) / `low` / `medium` / `high` | `auto` |
| `--csv-schema` | `full` (all scale conventions) or `compact` | `full` |
| `-w, --workers` | Parallel workers (threshold path only) | auto |
| `--save-axes` | Also save length/width overlay images for QC | off |

**Choosing a segmentation method.** `birefnet` is the accurate default and uses
a GPU when available (CPU works but is slow). `threshold` is much faster and
good for clean, high-contrast backgrounds where a leaf sits on plain white.

---

## Outputs

Per image, in the output folder:

- `{sample_id}_target_box.jpg` — the perspective-corrected observation box
- `{sample_id}_mask.png` — the leaf segmentation mask

Plus a measurements CSV. Two schemas:

- **full** (default, research schema) — leaf area, width, and length under three
  scale calibrations (mean, width-based, height-based) with the pixels-per-cm
  factors. This matches the columns the analysis scripts expect.
- **compact** — `sample_id, area_cm2, height_cm, length_cm`.

A `leaf_morpho_failures.csv` records per-image warnings and failures.

---

## Model weights

The checkpoints are too large for GitHub, so they are hosted on Hugging Face and
resolved at runtime:

| Model | File | Size |
|---|---|---|
| RF-DETR marker detector | `rf_detr_marker.pth` | ~134 MB |
| BiRefNet leaf segmenter | `birefnet_leaf.pth` | ~2.65 GB |

You don't have to fetch them manually — the **first run downloads them once** to
`~/.cache/mats/weights` and caches them. `mats fetch-weights` just does it up
front. Three delivery options, picked automatically:

- **Auto-fetch (default)** — downloads from Hugging Face on first use.
- **Shared filesystem** — set `MATS_WEIGHTS_DIR` (e.g. a SCINet `/project` path)
  to read weights in place with no per-user copy.
- **Git LFS** — optionally keep them in `weights/` in your checkout.

Set `MATS_NO_AUTO_FETCH=1` to disable the automatic download (e.g. on an HPC login
node). Full detail, DOI, and checksums: [docs/weights.md](docs/weights.md).

---

## On a cluster (HPC / Open OnDemand)

An Open OnDemand Batch Connect app that serves the GUI on a compute node is in
[deploy/ondemand/mats/](deploy/ondemand/mats/). See its README and
[docs/hpc.md](docs/hpc.md).

---

## How it works

MATs chains two models. **RF-DETR** (fine-tuned, single "Marker" class) detects
the four corner fiducials at 1120×1120 px; their centroids define a homography
that rectifies the observation box and fixes the pixels-per-cm scale.
**BiRefNet** (fine-tuned for leaf foreground) then segments the leaf, from which
area (pixel count) and length/width (bounding dimensions) are computed and
converted to centimeters. A classic Otsu threshold is offered as a fast
alternative to BiRefNet. See the manuscript for training and evaluation detail.

---

## Troubleshooting

Run `mats doctor` first — it reports most of these.

- **`zbar` / pyzbar not found** — install the system library (conda:
  `conda install -c conda-forge zbar`; Linux: `apt install libzbar0`;
  macOS: `brew install zbar`).
- **CUDA out of memory** — use `--mask-method threshold`, or process in smaller
  batches.
- **No markers detected** — check print quality and that the marker color
  matches the template (the Template Creator uses the trained color); make sure
  all four corners are in frame.
- **Blank page on Open OnDemand** — almost always the reverse-proxy
  `baseUrlPath` mismatch; see [deploy/ondemand/mats/README.md](deploy/ondemand/mats/README.md).

---

## Citing

If you use MATs, please cite the manuscript and the Hugging Face weights deposit.
See [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE). The pipeline builds on RF-DETR (Apache-2.0) and BiRefNet (MIT);
see [docs/weights.md](docs/weights.md) for model provenance.
