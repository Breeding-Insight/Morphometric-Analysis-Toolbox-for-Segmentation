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
mats run -i ./images -o ./out -r results.csv -t 10.5x9.5in
```

| Flag | Description | Default |
|---|---|---|
| `-i, --input_dir, --input-dir` | Directory of images to analyze. | prompted if omitted (interactive) |
| `-o, --output_dir, --output-dir` | Directory for masks / target boxes. | prompted if omitted (interactive) |
| `-r, --results_path, --results-path` | Measurement CSV path. | `./leaf_morpho_results.csv` |
| `-t, --template_dimensions, --template-dimensions` | Observation-box size as `<w>x<h><unit>`, e.g. `10.5x9.5in` or `27x24cm`. | read from the template QR |
| `--output-mode` | `masks` (segment leaves) or `target-boxes` (only save corrected boxes). | `masks` |
| `--mask-method` | `birefnet` (accurate, GPU) or `threshold` (fast). | `birefnet` |
| `--threshold-level` | For `threshold`: `auto` (Otsu), `low` (100), `medium` (125), `high` (150). | `auto` |
| `--csv-schema` | `full` (all three scale conventions) or `compact`. | `full` |
| `-w, --workers` | Parallel workers. Only the CPU `threshold` path over pre-made target boxes parallelizes; model-backed runs use one worker. | auto |
| `--save-axes` | Also write per-image length/width overlay images for QC. | off |
| `--scale-axis` | Deprecated; the CSV always reports mean, width- and height-based conversions. | `average` |

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
mats fetch-weights                 # both checkpoints
mats fetch-weights --only rf-detr  # just one
mats fetch-weights --force         # re-download even if present
```

Downloads to `~/.cache/mats/weights` (or `$MATS_WEIGHTS_DIR`). See
[weights.md](weights.md).

## `mats doctor`

Prints the resolved checkpoint paths and whether they exist, the Torch version
and available devices, and whether `pyzbar`/`zbar` load. Run it first whenever
something is off.

## `mats app`

Launches the Streamlit GUI. Extra arguments are forwarded verbatim to
`streamlit run`, e.g.:

```bash
mats app --server.port 8502 --server.headless true
```
