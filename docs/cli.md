# CLI reference

MATs installs a single `mats` command with four subcommands:

```
mats run             Batch-measure a folder of images (the default).
mats app             Launch the Streamlit GUI.
mats fetch-weights   Download the model checkpoints.
mats doctor          Report weights, devices and QR decoders.
```

`mats -i IN -o OUT ...` with no subcommand is treated as `mats run -i IN -o OUT ...`.

## `mats run`

Measures every image in a folder and writes a CSV.

```bash
mats run -i ./images -o ./out -r results.csv --sheet-dimensions 12x12in
```

| Flag | Description | Default |
|---|---|---|
| `-i, --input_dir, --input-dir` | Directory of images to analyze. | prompted if omitted (interactive) |
| `-o, --output_dir, --output-dir` | Directory for masks / target boxes. | prompted if omitted (interactive) |
| `-r, --results_path, --results-path` | Measurement CSV path. | `./leaf_morpho_results.csv` |
| `--sheet-dimensions` | Finished Template Creator sheet size as `<w>x<h><unit>`, e.g. `12x12in` or `30x30cm`; MATS derives calibration using Creator margins. | read from QR |
| `-t, --template_dimensions, --template-dimensions` | Legacy/custom marker-centre calibration area. Retained for existing scripts and non-Creator sheets. | unused |
| `--output-mode` | `masks` (segment leaves) or `target-boxes` (only save corrected boxes). | `masks` |
| `--mask-method` | `birefnet` (accurate, GPU), `threshold` (fast), or `both` (measure every image with each method; see below). | `threshold` |
| `--threshold-level` | For `threshold` and threshold pre-cleanup exports: `auto` (Otsu), `low` (100), `medium` (125), `high` (150), or a custom integer cutoff `1`–`255` (e.g. `--threshold-level 140`). Grayscale pixels at or below the cutoff count as leaf. | `auto` |
| `--csv-schema` | `full` (area/width/length + per-axis pixels-per-selected-unit) or `compact`. | `full` |
| `--results-unit` | CSV measurement unit: `mm`, `cm`, or `in`. | `cm` |
| `--measure-pre-cleanup` | Measure from the raw binary segmentation before gap closing and hole filling, after clearing the edge margin and dropping stray pieces (see below). | off (cleaned mask) |
| `--clean-margin` | With `--measure-pre-cleanup`: width of the band cleared along every target-box edge, as a percent of the box's shorter side, `0`–`10`. `0` clears nothing. | `1` |
| `--stray-gap` | With `--measure-pre-cleanup`: drop pieces whose nearest pixel is farther from the leaf than this fraction of the leaf's bounding-box diagonal, `0`–`10`. `0` keeps only the leaf. | `0.25` |
| `--clean-size` | With `--measure-pre-cleanup`: after the margin and stray pieces are cleared, remove white specks and fill enclosed holes whose inscribed radius is below this many pixels, `0`–`50`. The app's **Clean size**. | `0` (off) |
| `-w, --workers` | Parallel workers. Only the CPU `threshold` path over pre-made target boxes parallelizes; model-backed runs use one worker. | auto |
| `--save-axes` | Also write per-image length/width overlay images for QC. | off |
| `--export` | Repeatable: `pre-cleanup`, `overlay`, `cutout`, `axes`. Overlays, cutouts, and axes use the selected measurement mask. | none |
| `--pre-cleanup-methods` | With `--export pre-cleanup`: `selected` (every `--mask-method` method), `threshold`, `birefnet`, or `both`. Requesting BiRefNet runs it for every image and requires its local checkpoint. | `selected` |
| `--no-target-boxes` | Do not save newly rectified target boxes. Existing target-box inputs are never copied. | off |
| `--no-masks` | Do not save cleaned masks. This does not change the measurement source. | off |
| `--no-failure-log` | Do not write `leaf_morpho_failures.csv`. | off |

Pre-cleanup masks are binary segmentations before gap closing, hole filling,
and removal of smaller objects. BiRefNet masks are already thresholded, not
probability maps. `--pre-cleanup-methods both` writes
`{sample_id}_mask_precleanup_threshold.png` and
`{sample_id}_mask_precleanup_birefnet.png`. Only a `--mask-method` method
determines area, width, and length in the CSV. With `--output-mode target-boxes`,
segmentation exports are ignored. The results CSV is always written.

`--measure-pre-cleanup` uses the selected method's raw binary mask for measurements.
First it clears a band `--clean-margin` percent of the box's shorter side wide along
every edge of the target box. The template's printed box outline runs through the
marker centres, so after perspective correction it lies on that edge; clearing it
first means the outline can never outweigh, and replace, a small leaf. The
measurement is then anchored on the leaf, the largest remaining object. Any other
piece that touches the cleared band (the rest of a printed line, marker remnants,
shadows at the sheet edge) is dropped, and so is any piece farther from the leaf
than `--stray-gap` times the leaf's bounding-box diagonal. A thin piece in the band
is never taken as the leaf. Area counts every remaining foreground pixel,
including specks near the leaf, while width and length span the remaining
foreground extent. Holes remain excluded from area. Lay leaves inside the printed
box: any part of a leaf within the margin is cleared too. Raise `--stray-gap` when
a leaf's parts lie apart, such as separated leaflets; lower it to drop specks
closer to the leaf.

`--clean-size` then cleans what is left by size. Each remaining piece and each
enclosed hole is sized by its inscribed radius, the distance from its deepest
pixel to its edge. White specks with a radius below the clean size are removed,
and holes with a radius below it are filled; the leaf is always kept, and nothing
is flash-filled, so a hole at least that large stays excluded from area. It is the
**Clean size** slider in the app, and gives exactly the mask that slider previews.
For example, measure raw masks at the medium threshold, removing specks and
filling holes under 3 px:

```bash
mats run -i ./images -o ./out -r results.csv --sheet-dimensions 12x12in \
  --measure-pre-cleanup --threshold-level medium --clean-size 3
```

Without this flag, measurements use the cleaned mask as before. This setting is
independent of `--export pre-cleanup`, which writes the raw mask exactly as
segmented, stray pieces included. Each results CSV has a `.meta.json` companion
recording its measurement source, segmentation method, unit, and schema, plus
the clean margin, stray gap, and clean size for pre-cleanup runs.

`--mask-method both` detects markers once per image and then measures it with
Otsu and with BiRefNet. Each method gets its own results CSV and failure log,
named from `-r` with a method suffix — `leaf_morpho_results_threshold.csv` and
`leaf_morpho_results_birefnet.csv` — in the same schema as a single-method run.
Each method's cleaned masks, overlays, cutouts, and axes also end in
`_threshold` or `_birefnet` (for example `{sample_id}_mask_birefnet.png`);
target boxes are shared. A run with one method keeps the unsuffixed names.

### Interactive vs non-interactive

If `--input_dir` / `--output_dir` are omitted **and** a terminal is attached,
`mats run` prompts for them. Under a job scheduler or any non-TTY context it
fails fast with a clear message instead of hanging on a prompt — always pass
`-i` and `-o` in scripts.

### Workers

Model-backed inference (RF-DETR, BiRefNet) shares one in-process model, so those
runs execute on a single worker regardless of `-w`. Parallelism helps only when
you re-segment already-extracted `*_target_box` images with `--mask-method
threshold`.

## `mats fetch-weights`

```bash
mats fetch-weights                 # RF-DETR only (default)
mats fetch-weights --all           # both checkpoints
mats fetch-weights --only birefnet --source lfs # explicitly fetch just BiRefNet
mats fetch-weights --force         # re-download even if present
```

Git LFS downloads write to the Git checkout's `weights/` directory.
`MATS_WEIGHTS_DIR` is for pre-staged local or shared checkpoints; a configured
Hugging Face source may use it as its download destination. See
[weights.md](weights.md).

## `mats doctor`

Prints the resolved checkpoint paths and whether they exist, the Torch version
and available devices, and which QR-decoding backends are available (OpenCV by
default; `pyzbar`/`qreader` when the optional `qr` extra is installed). Run it
first whenever something is off.

## `mats app`

Launches the Streamlit GUI. Extra arguments are forwarded verbatim to
`streamlit run`, e.g.:

```bash
mats app --server.port 8502 --server.headless true
```
