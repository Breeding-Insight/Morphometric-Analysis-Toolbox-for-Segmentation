# Model weights

MATs uses two fine-tuned checkpoints. Both are committed to this repository
via Git LFS, and both are also available from Hugging Face — two independent
channels, because our primary collaborators (USDA) cannot reach
huggingface.co from their networks, so Hugging Face alone is not a viable
delivery mechanism for everyone.

A plain `git clone` only fetches RF-DETR (~134 MB, mandatory for every run,
so a checkout should be immediately runnable). BiRefNet (~2.65 GB, optional —
only needed for `--mask-method birefnet`) is excluded from the default clone
by `.lfsconfig` and is fetched on demand: via the **BiRefNet setup** page in
the app, `mats fetch-weights --only birefnet`, or transparently on first use.

| Model | File | Size | sha256 |
|---|---|---|---|
| RF-DETR marker detector | `rf_detr_marker.pth` | ~134 MB | `15896dd7cfaf8ee6e38c1226f6384a908a111f9a405f229bb5fe7db6264b6bca` |
| BiRefNet leaf segmenter | `birefnet_leaf.pth` | ~2.65 GB | `27b9481d18243101177c8a8606b60c01e03bed4926e97ae1e14ceeebb5c91377` |

**Hugging Face:** _TBD_ · **DOI:** _TBD_

## The key idea: PyTorch needs the bytes locally

Inference cannot "read weights remotely" over HTTP — the checkpoint must be on a
local or *mounted* filesystem for `torch.load` to use it. MATs therefore supports
three sources, all resolved the same way, so each audience gets the least-copy
option available to them:

| You are… | Use | What happens |
|---|---|---|
| A general user who cloned the repo | **Git LFS (RF-DETR) + a download for BiRefNet** | RF-DETR is already in the checkout; BiRefNet installs on demand from whichever channel you pick |
| On a network that blocks Hugging Face (e.g. USDA) | **Git LFS** | `mats fetch-weights --source lfs`, or the "This repository (Git LFS)" option on the setup page — same GitHub connection the clone itself used |
| A general user with unrestricted internet | **Hugging Face** | `mats fetch-weights --source hf`, or the default option on the setup page |
| A USDA / SCINet collaborator | **Shared filesystem** | Point `MATS_WEIGHTS_DIR` at a `/project` copy — read in place, **no download at all**, for the whole institution |

## Resolution order

For each checkpoint, MATs uses the first that resolves to a real file:

1. `RF_DETR_MARKER_CHECKPOINT` / `BIREFNET_CHECKPOINT` — an explicit file path.
2. `$MATS_WEIGHTS_DIR/<filename>` — e.g. a shared SCINet `/project` directory.
3. `~/.cache/mats/weights/<filename>` (or `$XDG_CACHE_HOME/...`) — the Hugging
   Face auto-fetch target.
4. `<repo>/weights/<filename>` — the Git LFS checkout location.

Un-smudged Git LFS pointer stubs are ignored at this layer, so a checkout with
an unfetched BiRefNet pointer falls through to the cache tier. Separately,
`mats.weights` also checks the checkout's `weights/` directory directly (not
just this resolution order) so a checkpoint fetched there via Git LFS is
recognized without needing `MATS_WEIGHTS_DIR` or the cache.

## Fetching a checkpoint

```bash
mats fetch-weights                       # RF-DETR only (default; the mandatory one)
mats fetch-weights --only birefnet       # just BiRefNet, auto channel (HF if configured, else LFS)
mats fetch-weights --only birefnet --source lfs   # force the Git LFS channel
mats fetch-weights --only birefnet --source hf    # force the Hugging Face channel
mats fetch-weights --all                 # both checkpoints
mats fetch-weights --force               # re-download even if present
mats doctor                              # show resolved paths, channels, and source
```

The first BiRefNet inference on a fresh machine also downloads it
automatically (same "auto" channel selection as above) — you don't have to
run `fetch-weights` up front.

To disable the automatic download — e.g. on an HPC login node where a 2.65 GB
pull would be antisocial, or in an air-gapped run — set `MATS_NO_AUTO_FETCH=1`.
MATs will then raise a clear error instead of downloading, and you pre-stage the
weights yourself.

### The BiRefNet setup page

The **BiRefNet setup** page in the app uses the same checkpoint service as the
CLI and lets you pick a channel explicitly — Hugging Face or "this repository
(Git LFS)" — with the unavailable one shown greyed out with a reason (e.g. no
`MATS_WEIGHTS_HF_REPO` configured, or not running from a Git checkout). Pick
Git LFS on a network that blocks Hugging Face.

A release build should set `MATS_WEIGHTS_HF_REPO` to the Hugging Face
repository containing the fine-tuned MATS checkpoints (optionally
`MATS_WEIGHTS_HF_REVISION` to a pinned tag/commit) to enable that channel.
MATS deliberately does not substitute the upstream BiRefNet repository,
because that is not the fine-tuned leaf model.

## Shared filesystem (SCINet and other HPC)

Download once to a shared, readable location and point everyone at it — the
only option that needs no per-user download at all:

```bash
export MATS_WEIGHTS_DIR=/project/<your_project>/mats_weights
mats fetch-weights --all    # populates it once (from a data-transfer node)
mats doctor                 # confirm it resolves
```

On SCINet, `/project` is a mounted filesystem, so compute jobs read the weights
directly — no per-user copy. For external collaborators without SCINet accounts,
a **Globus guest collection** on that directory lets them pull the files (they
need a free Globus login).

## Git LFS

Both checkpoints are committed to this repo via Git LFS. `weights/rf_detr_marker.pth`
is fetched on every `git clone`. `weights/birefnet_leaf.pth` is committed but
excluded from the default clone and from a bare `git lfs pull` by `.lfsconfig`
(`lfs.fetchexclude`) — a fresh checkout gets a 134-byte pointer stub in its
place, not the 2.65 GB file. Pull it explicitly:

```bash
git lfs pull --include="weights/birefnet_leaf.pth"
```

or `mats fetch-weights --only birefnet --source lfs`, or the setup page.

This exclusion exists because committing BiRefNet without it would force
*every* `git clone` to download 2.65 GB and spend the repository's Git LFS
bandwidth quota. Every LFS-channel download (this quota, not storage) is
metered against the org's plan — for repeat/institutional use, `MATS_WEIGHTS_DIR`
(above) is the better answer since it downloads once, not once per user.

If your Git LFS version predates the exclusion behavior (needs the
`.lfsconfig` fetchexclude to be read from the repo index/HEAD during the
initial clone — true for modern Git LFS), a clone could pull BiRefNet anyway.
`GIT_LFS_SKIP_SMUDGE=1 git clone ...` is a guaranteed way to skip *all* LFS
content on clone if you want to be certain, then `git lfs pull --include=...`
each file you actually need.

## Manual / air-gapped

Download the two files from the Hugging Face repo (or copy them from a
colleague's Git LFS checkout), place them in your weights directory under the
names above, and verify:

```bash
shasum -a 256 rf_detr_marker.pth birefnet_leaf.pth
```

Compare against the checksums in the table.

## Provenance and licensing

- **RF-DETR** marker detector — fine-tuned from RF-DETR (Apache-2.0).
- **BiRefNet** leaf segmenter — fine-tuned from
  [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) (MIT).

The redistributed checkpoints are derivative works of those base models; their
upstream licenses apply. State this on the Hugging Face model card.

---

> **Maintainer note (remove before release):**
> 1. Create the Hugging Face weights repo; upload `rf_detr_marker.pth` and
>    `birefnet_leaf.pth` (`huggingface-cli upload <repo> <file>`).
> 2. Set `_DEFAULT_HF_REPO_ID` (and pin `_HF_REVISION`) in `src/mats/weights.py`,
>    or set `MATS_WEIGHTS_HF_REPO` per deployment.
> 3. ~~Fill the two `sha256` values in `_MANIFEST`~~ — done; the table above matches.
> 4. Mint the DOI on the HF repo (Settings → DataCite) and fill the table above.
> Until the Hugging Face repo id is set, that channel is reported unavailable
> and Git LFS is the only working automatic path (see the "auto" channel
> selection above).
