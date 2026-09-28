# MATS — instructions for AI coding agents

Project instructions for any agent working in this repository (Claude Code,
Codex, Cursor, Copilot, Gemini CLI, …). Humans should start with
[README.md](README.md) and [docs/faq.md](docs/faq.md).

**MATS (Morphometric Analysis Toolbox for Segmentation)** measures leaf area,
length, and width in real-world units from a photograph of leaves on a printed
calibration template:

> detect the four fiducial markers (RF-DETR) → perspective-correct → segment the
> leaf (Otsu by default, BiRefNet optionally) → measure → CSV

It ships as the installable package `mats-morpho` with one console script,
`mats`, and a Streamlit GUI. It is companion code for a scientific manuscript,
so **measurement correctness outranks convenience** in every trade-off.

## Who you are helping

Most people who clone this repo are **researchers measuring leaves**, not
contributors. Work out which one you have before answering:

- **A user** — installing, running a batch, reading the CSV, or fixing bad
  photos. They want working commands and a diagnosis, not a code tour. Jump to
  [Getting a new user running](#getting-a-new-user-running) and
  [Troubleshooting](#troubleshooting-playbook); point them at `docs/faq.md`.
- **A contributor** — changing the pipeline, the GUI, or packaging. See
  [Working on the code](#working-on-the-code).

When a user hits an environment problem, run `mats doctor` (or ask them to)
before theorising: it reports the MATS and Python versions, the interpreter and
environment in use, checkpoint resolution, the compute device, and which QR
decoders are usable. `mats doctor` only exists once MATS is installed. Before
that, run the [preflight](#assume-nothing-is-installed).

## Orientation

| Where | What is in it |
|---|---|
| `README.md` | User-facing entry point: install, both run paths, outputs, troubleshooting |
| `docs/install.md` | From-scratch install for a bare machine, per OS, with an error → fix table |
| `docs/faq.md` | Human FAQ — installation, first run, QR, weights, units, GPU |
| `docs/cli.md` | Full `mats run` flag reference |
| `docs/gui.md` | The Streamlit workbench, page by page |
| `docs/weights.md` | Checkpoints, sha256s, Git LFS, `MATS_WEIGHTS_DIR`, resolution order |
| `docs/templates.md` | Template Creator rules: margins, marker sizes, printing |
| `docs/hpc.md` | Batch jobs and Open OnDemand on a cluster |
| `CHANGELOG.md` | What changed and when |

Package layout (`src/mats/`):

| Module | Role |
|---|---|
| `core.py` | The pipeline: marker detection, homography, segmentation, measurement, `run_leaf_morpho_batch` |
| `cli.py` | The `mats` entry point: `run`, `app`, `fetch-weights`, `doctor` |
| `paths.py` | Checkpoint **resolver** (import-light: no torch at module load) |
| `weights.py` | Checkpoint **delivery**: fetch channels, manifest, `doctor()`, LFS-pointer detection |
| `scaling.py` | Pixels-per-unit maths and unit conversion |
| `dimensions.py` | Parses template dimension strings (`12x12in`, `30x30cm`) |
| `qr_runtime.py` | QR decoding with OpenCV plus the optional pyzbar / QReader fallbacks |
| `devices.py` | CUDA / MPS / CPU selection |
| `birefnet_runtime.py`, `models/birefnet/` | BiRefNet loading and the pinned bundled architecture |
| `template_layout.py`, `template_exports.py` | Template geometry rules and the PDF / IDML exports |
| `samples.py` | Resolver for the packaged sample photos (`SAMPLES_DIR`, `SAMPLE_SETS`) |
| `app/` | Streamlit GUI: `Home.py` plus numbered `pages/N_Name.py` |
| `deploy/ondemand/mats/` | Open OnDemand Batch Connect app (GUI on a compute node) |
| `tests/` | Dependency-light unit tests — no torch, no network |

## Getting a new user running

The install needs **Python 3.10–3.13 (3.13 recommended) and Git LFS**. Nothing
else — no conda, no other system libraries (QR codes are decoded with OpenCV).
[docs/install.md](docs/install.md) is the step-by-step guide for people, from a
bare machine to `mats doctor`. Send users there, and use this section to steer.

### Assume nothing is installed

Many users arrive with no Python, no Homebrew, and no idea what a virtual
environment is. `mats doctor` doesn't exist until MATS is installed, so start
with a preflight and read the answers before suggesting anything:

```bash
python3 --version      # Windows: py --version
git --version
git lfs version
uname -sm              # Windows PowerShell: $env:PROCESSOR_ARCHITECTURE
```

Hard rules. Each one comes from a real failed install:

- **Never use a Python below 3.10.** On macOS, `/usr/bin/python3` is Apple's
  3.9, installed with the Command Line Tools. It can't install MATS
  (`rfdetr==1.5.2` needs 3.10+), and its pip 21 can't do an editable install of
  this package.
- **Always install into a virtual environment** (`python3.13 -m venv .venv`) or
  a conda env, and always call pip as `python -m pip`, upgrading pip first. That
  avoids `pip: command not found`, user-site installs that leave `mats` off
  PATH, and the `externally-managed-environment` error from Homebrew and Debian
  Pythons.
- **Keep `-e`.** `paths.py` finds `<repo>/weights/` from the package's own
  location (`_REPO_ROOT`), so a non-editable install finds the checkpoint only
  when run from the repo folder.
- **Never suggest** `sudo pip`, `pip install --user`, `--break-system-packages`,
  `git lfs install --system` (needs admin, and `git lfs install` is enough), or
  `brew upgrade python` (it fails unless Homebrew already owns a Python; use
  `brew install python@3.13`).
- **Don't send users to "the latest 3.12" on python.org.** python.org stops
  publishing installers once a version leaves bug-fix support. 3.12's last
  installer is 3.12.10, and newer 3.12 entries say "No files for this release".

Pick the route from the preflight (the section names are in `docs/install.md`):

| Situation | Route |
|---|---|
| Python 3.10–3.13 already present | Use it for the virtual environment; skip to Git LFS and install |
| macOS, has the admin password | python.org 3.13 installer, then **Install Certificates.command** |
| macOS, already uses Homebrew | `brew install python@3.13 git-lfs` |
| Windows x64 | python.org installer with "Add python.exe to PATH", then `py -3.13`; Git for Windows includes Git LFS |
| Linux, has sudo, `python3` ≥ 3.10 | Distro packages; Debian/Ubuntu also need `python3-venv` |
| No admin rights, or an old distro | Miniforge: `conda create -n mats -c conda-forge python=3.13 git git-lfs` |
| Intel Mac | Python 3.12 (python.org 3.12.10 or `python@3.12`) plus `"numpy<2"`, because PyTorch stops at 2.2.2 there. Untested |
| Windows on ARM | Not supported: PyTorch publishes no Windows-on-ARM wheels on PyPI |

What has been verified (September 2026, MATS 1.0.0):

- The full sample pipeline has run on Apple-silicon macOS with Python 3.10
  (conda), 3.12 (python.org) and 3.13 (Homebrew), and the results CSVs were
  byte-identical.
- The dependencies install from wheels on Linux x86_64/aarch64 and Windows x64
  for 3.10–3.13, but MATS hasn't been run there.
- Intel Macs install only with `"numpy<2"` and have never been run. Don't claim
  more than this.

How to coach a first-time user:

- Give commands **without** `$` or `%` prompt characters. People paste them
  literally and get `zsh: command not found: $`.
- One block at a time, each with what success looks like. Wait for the output
  before the next step.
- Warn before a step opens a dialog (the Xcode Command Line Tools prompt) or
  asks for a password that shows nothing as it is typed.
- After a Homebrew install, the "Next steps" PATH lines it prints are required.
  `command not found: brew` right after installing means they were skipped.
- When a user pastes a terminal transcript, match each error against
  [Before MATS is installed](#before-mats-is-installed) before theorising.
- Remind them that every new terminal needs `source .venv/bin/activate` (or
  `conda activate mats`) before `mats` works.
- Tell them the rf-detr `[WARNING]` lines and albumentations' SSL version-check
  warning during a run are harmless (see the "safe to ignore" table in
  `docs/install.md`).

> **Check Git LFS before cloning.** The RF-DETR checkpoint
> (~134 MB) is stored in Git LFS and is mandatory for every run. A `git clone`
> on a machine without `git-lfs` **appears to succeed** but writes a 134-byte
> pointer stub in place of the model. This is the single most likely reason a
> fresh checkout fails to detect markers, and it looks like a model problem
> rather than a setup problem. Verify with `ls -l weights/rf_detr_marker.pth`
> (~134 MB, not ~134 bytes; `ls -lh` shows `128M`, which is correct) or
> `mats doctor`; repair with
> `git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"` — the `--exclude` keeps the repair to RF-DETR. With
> `.lfsconfig` active, a bare `git lfs pull` also leaves the 2.65 GB BiRefNet checkpoint out; fetch it only with
> `git lfs pull -X "" -I "weights/birefnet_leaf.pth"` (or `mats fetch-weights --only birefnet --source lfs`).

```bash
git lfs install                     # one-time, per machine, BEFORE cloning
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
python3.13 -m venv .venv            # Windows: py -3.13 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[app]"   # ".[app]" adds the Streamlit GUI
mats doctor                         # versions, environment, checkpoints, device, QR
```

With Miniforge, skip the two virtual-environment lines and run
`conda activate mats` instead.

The clone delivers RF-DETR — there is **no separate download step** for it.
`mats fetch-weights` exists to repair a checkout made without Git LFS, not as
part of a normal install; suggesting it as a routine step misleads users.

Then either path — both run the same code and produce the same numbers:

```bash
mats app                                                      # the GUI
mats run -i ./images -o ./out -r results.csv --sheet-dimensions 12x12in
```

Things worth telling a first-time user, in this order:

1. **They need a printed template.** Measurements come from four corner markers
   of known spacing. The GUI's **Template Creator** page produces a print-ready
   PDF; print at 100 % scale ("fit to page" silently breaks calibration), lay
   leaves inside the box, photograph flat with all four markers in frame.
2. **`--sheet-dimensions` is the finished printed sheet size**, e.g. `12x12in` —
   MATS derives the marker-centre calibration area from it. Alternatively the
   template's QR code can carry it per image.
3. **Otsu is the default** and needs no extra download. BiRefNet is the accurate
   option for cluttered backgrounds and costs a ~2.65 GB checkpoint, fetched only
   on request.
4. **They can try it with no data of their own.** Three de-identified sample
   photographs ship with the package (`mats.samples.SAMPLES_DIR`) and are walked
   through on the GUI's **Help** page — including a real QR-read failure that
   shows why entering the printed sheet size by hand is the most reliable route.

## Answer from the code, not from memory

Model behaviour and flags have changed across versions. Before you state a flag,
a default, or a column name, check the source — `src/mats/cli.py`, `docs/cli.md`,
or `mats run --help`.

Currently true, and worth knowing because users ask:

- Subcommands are exactly `run`, `app`, `fetch-weights`, `doctor`. `mats` with
  bare arguments implies `run`.
- `mats doctor` opens with `MATS:`, `python:` (version and interpreter path),
  and `environment:` (virtual environment, conda environment, or system/user
  Python). Check those first when an install misbehaves.
- `mats app` doesn't stop at Streamlit's first-run "Email:" prompt:
  `server.showEmailPrompt = false` in the packaged
  `src/mats/app/.streamlit/config.toml` turns it off, and the browser still
  opens.
- Long flags accept both spellings where noted in `cli.py` (`--input_dir` and
  `--input-dir`).
- Defaults that surprise people: `--mask-method threshold`, `--threshold-level
  auto`, `--csv-schema full`, `--results-unit cm`, `--output-mode masks`.
- `--threshold-level` takes a preset (`auto`, `low`, `medium`, `high`) or an
  integer cutoff `1`–`255`; the GUI's **custom** level is the same thing.
- `--mask-method both` (two checked methods in the GUI) measures each image with
  Otsu and BiRefNet and writes one CSV and failure log per method
  (`*_threshold.csv`, `*_birefnet.csv`); a single-method run keeps the
  unsuffixed names and identical output.
- `--measure-pre-cleanup` is not "every raw pixel": it clears a `--clean-margin`
  band along the target-box edge (default 1% of the shorter side, where the
  printed box outline lands), keeps the largest remaining object (the leaf),
  and drops pieces that touch the band or lie beyond `--stray-gap` (default
  `0.25` × the leaf's bounding-box diagonal) — see `mask_cleanup.clean_raw_mask`.
  `--clean-size` (pixels, default `0` = off, pre-cleanup only) then removes
  specks and fills holes below that inscribed radius — the app's **Clean size**,
  via `mask_cleanup.raw_measurement_mask`; Adjust's Overwrite saves it for Otsu
  specimens. `--clean-margin` is the edge margin, not the clean size.
  Clean image and Remove flashfill use the same margin. `--export pre-cleanup`
  still writes the untouched raw mask. The default cleaned path is unaffected.
- `mats fetch-weights` with no flag fetches **RF-DETR only**; BiRefNet needs
  `--only birefnet` or `--all`. `--source {auto,hf,lfs}` picks the channel —
  `lfs` is the one to use on networks that block huggingface.co. After a
  successful clone it is a no-op that prints "already present".

The two checkpoints are deliberately asymmetric in code, and it is easy to get
backwards:

| | RF-DETR | BiRefNet |
|---|---|---|
| Loader calls | `weights.ensure_weight("rf-detr")` | `weights.require_local_weight("birefnet")` |
| Missing at run time | Fetched once, announced on stdout | **Never fetched** — raises with instructions |
| In the clone | Yes (Git LFS) | Intended: no — kept out by `lfs.fetchexclude` in `.lfsconfig`. **Verify it is active** with `git lfs env \| grep FetchExclude`; if that is empty, a clone or bare `git lfs pull` fetches BiRefNet (2.65 GB) too |
| In the GUI | Blocking Preflight error, no auto-download | Blocking only when BiRefNet is the selected method |

Do not "fix" BiRefNet by switching it to `ensure_weight`: never starting a
2.65 GB download because someone picked a dropdown option is the intended
behavior, and `tests/test_weights.py` asserts it.

## Measurement semantics — do not paraphrase loosely

Scaling is **anisotropic**: each axis is calibrated independently against the
template. `width_cm` is the x-extent ÷ `px_per_cm_width`, `length_cm` is the
y-extent ÷ `px_per_cm_height`, and area is divided by *both*. There is no single
averaged scale factor — do not describe one.

`scale_aspect_ratio` (`px_per_cm_width / px_per_cm_height`) is a QC signal: it
should sit near 1.0, and a value far from it flags a calibration problem (skewed
print, lens distortion, a non-planar sheet). Older CSVs with `*_meanscale`,
`*_widthscale`, `*_heightscale` columns predate this; the conversion is in the
README's migration note.

**Never invent a measurement, an accuracy figure, or a citation.** If a number
isn't in the output or the docs, say so and run the pipeline to get it.

## Troubleshooting playbook

### Before MATS is installed

Match the exact text. The full table, with the harmless warnings, is section 7
of `docs/install.md`.

| Message | Cause → fix |
|---|---|
| `xcode-select: note: No developer tools were found, requesting install.` | macOS is installing its Command Line Tools (this provides Git) → click Install, wait, rerun |
| `Python 3.9.x` from `python3 --version` | Apple's system Python (or an old distro Python) → install 3.13; never install MATS into it |
| `command not found: brew` right after installing Homebrew | Its "Next steps" PATH lines weren't run → run them |
| `Error: python not installed` from `brew upgrade python` | Homebrew has no Python yet → `brew install python@3.13` |
| `command not found: pip` / `pip3` | No venv active → activate `.venv`, use `python -m pip` |
| `File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode` | Pre-21.3 pip (Apple's Python 3.9 ships 21.2.4) → a 3.13 venv with upgraded pip |
| `Defaulting to user installation because normal site-packages is not writeable` | Installing into a system Python → stop, create the venv |
| `requires a different Python: 3.9.6 not in '>=3.10'` | Python too old → 3.13 venv |
| `No matching distribution found for rfdetr==1.5.2` | Python < 3.10, on copies of MATS older than the `requires-python` fix → 3.13 venv |
| `No matching distribution found for torch` | Intel Mac on Python ≥ 3.13 (→ 3.12 plus `"numpy<2"`), or Windows on ARM (not supported) |
| `error: externally-managed-environment` | Homebrew or Debian Python outside a venv → create the venv |
| `ensurepip is not available` creating the venv | Debian/Ubuntu → `sudo apt install python3-venv` (or the `python3.X-venv` it names) |
| `mats: command not found` | Venv not active in this terminal → activate it |
| `could not lock config file /etc/gitconfig` after `git lfs install --system` | `--system` needs admin and isn't needed → plain `git lfs install` |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell policy → `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or `activate.bat` in cmd |
| `CERTIFICATE_VERIFY_FAILED` | python.org macOS Python → run **Install Certificates.command**; institutional network → proxy settings from IT |
| python.org says "No files for this release" | That version is security-only (3.12 after 3.12.10) → the newest release with an installer, or 3.13 |

### Once installed

`mats doctor` first — it reports most of these. Then, by symptom:

| Symptom | First move |
|---|---|
| QR code not read / no scale | Pass the sheet size explicitly: `--sheet-dimensions 12x12in`. Optional fallbacks: `pip install -e ".[qr]"` (QReader works without conda; pyzbar also needs the native `zbar`: `apt install libzbar0` / `brew install zbar` / `conda install -c conda-forge zbar`) |
| No markers detected | All four markers in frame? Printed at 100 % scale, in the Template Creator's marker colour? See `docs/templates.md` |
| BiRefNet unavailable | `mats fetch-weights --only birefnet --source lfs`, or the GUI's **BiRefNet setup** page |
| A `.pth` "loads" as text / torch errors on the checkpoint | A Git-LFS pointer stub (≤1024 bytes starting `version https://git-lfs.github.com/spec/v1`), not weights. `git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"`. `weights.looks_like_lfs_pointer()` exists so torch is never handed one |
| CUDA out of memory | Only possible with `--mask-method birefnet`: use smaller batches, or the default `threshold` |
| Slow on CPU | Use `threshold`, and raise `-w/--workers`. Multiple workers disable CUDA/MPS for that run by design |
| Blank page on Open OnDemand | Almost always the reverse-proxy `baseUrlPath`; see `deploy/ondemand/mats/README.md` |

Weights resolve in this order (`paths.py`), first hit wins:

1. `RF_DETR_MARKER_CHECKPOINT` / `BIREFNET_CHECKPOINT` — explicit file paths
2. `MATS_WEIGHTS_DIR` — a shared or mounted directory, read in place, no copy
3. `~/.cache/mats/weights` (or `$XDG_CACHE_HOME/mats/weights`)
4. `<repo>/weights/`, then `./weights/` — the Git-LFS checkout

Canonical filenames are `rf_detr_marker.pth` and `birefnet_leaf.pth`. Setting
`MATS_NO_AUTO_FETCH=1` turns off downloading entirely so a misconfigured path
fails fast — use it on HPC login nodes and air-gapped systems.

## Working on the code

```bash
python -m pip install -e ".[app,dev]"   # inside a venv or conda env
python -m pytest                        # fast, offline, no torch required
```

Rules that keep this repo working:

- **Tests stay offline and torch-free.** CI (`.github/workflows/ci.yml`) installs
  with `pip install --no-deps -e .` on Python 3.10 and 3.11, so a new
  module-level import of torch, rfdetr, or streamlit in an imported path breaks
  CI even when it works locally.
- **`paths.py` stays import-light** — standard library only. Models load lazily
  inside `core.py`; `samples.py`, `dimensions.py`, `scaling.py`, and
  `mask_settings.py` follow the same contract.
- **The CLI and the GUI share one execution path** (`run_leaf_morpho_batch`).
  Never fork pipeline logic between them — divergence would mean the two
  interfaces report different measurements.
- **Never commit checkpoints.** `*.pth`, `*.pt`, `*.pkl`, `*.onnx` are gitignored
  except the two LFS-tracked files already in `weights/`.
- Package data in `pyproject.toml` uses per-segment, extension-specific globs; a
  new asset type or nesting level needs a new glob line or it won't ship in the
  wheel.
- Streamlit pages are numbered (`app/pages/N_Name.py`) — the number sets sidebar
  order. Match the surrounding style of whatever file you are editing.

## Guardrails

- Confirm before anything that downloads gigabytes, starts a long GPU run, or
  writes into a user's image directories.
- Treat a user's input images and their results CSV as precious: write outputs to
  the designated output folder, never overwrite inputs.
- Don't fabricate measurements, model accuracy claims, or citations. The
  manuscript is not in this repository.
