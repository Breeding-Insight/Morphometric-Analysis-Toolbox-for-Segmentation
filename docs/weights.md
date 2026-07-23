# Model weights

MATs uses two fine-tuned checkpoints. They are too large to ship in the repo, so
they are hosted separately and resolved at runtime.

| Model | File | Size | sha256 |
|---|---|---|---|
| RF-DETR marker detector | `rf_detr_marker.pth` | ~134 MB | _TBD_ |
| BiRefNet leaf segmenter | `birefnet_leaf.pth` | ~2.65 GB | _TBD_ |

**Hugging Face:** _TBD_ · **DOI:** _TBD_

## The key idea: PyTorch needs the bytes locally

Inference cannot "read weights remotely" over HTTP — the checkpoint must be on a
local or *mounted* filesystem for `torch.load` to use it. MATs therefore supports
three sources, all resolved the same way, so each audience gets the least-copy
option available to them:

| You are… | Use | What happens |
|---|---|---|
| A general user who cloned the repo | **Auto-fetch (Hugging Face)** | Weights download **once** to `~/.cache/mats/weights` on first run, then cache |
| A USDA / SCINet collaborator | **Shared filesystem** | Point `MATS_WEIGHTS_DIR` at a `/project` copy — read in place, **no download** |
| A user who wants weights in the checkout | **Git LFS** | Place them in `weights/` (optional, off by default) |

## Resolution order

For each checkpoint, MATs uses the first that resolves to a real file:

1. `RF_DETR_MARKER_CHECKPOINT` / `BIREFNET_CHECKPOINT` — an explicit file path.
2. `$MATS_WEIGHTS_DIR/<filename>` — e.g. a shared SCINet `/project` directory.
3. `~/.cache/mats/weights/<filename>` (or `$XDG_CACHE_HOME/...`) — the auto-fetch
   target.
4. `<repo>/weights/<filename>` — a Git LFS or manual placement in the checkout.

Un-smudged Git LFS pointer stubs are ignored, so a checkout without `git lfs pull`
falls through to auto-fetch instead of handing PyTorch a text stub.

## Auto-fetch (Hugging Face)

The first inference on a fresh machine downloads the weights automatically. You
can also do it up front:

```bash
mats fetch-weights                 # both checkpoints
mats fetch-weights --only rf-detr  # just one
mats fetch-weights --force         # re-download
mats doctor                        # show resolved paths + source
```

To disable the automatic download — e.g. on an HPC login node where a 2.65 GB
pull would be antisocial, or in an air-gapped run — set `MATS_NO_AUTO_FETCH=1`.
MATs will then raise a clear error instead of downloading, and you pre-stage the
weights yourself.

## Shared filesystem (SCINet and other HPC)

Download once to a shared, readable location and point everyone at it:

```bash
export MATS_WEIGHTS_DIR=/project/<your_project>/mats_weights
mats fetch-weights          # populates it once (from a data-transfer node)
mats doctor                 # confirm it resolves
```

On SCINet, `/project` is a mounted filesystem, so compute jobs read the weights
directly — no per-user copy. For external collaborators without SCINet accounts,
a **Globus guest collection** on that directory lets them pull the files (they
need a free Globus login).

## Git LFS (optional)

The repo ships a `.gitattributes` rule for `weights/*.pth`, but does **not**
commit the checkpoints — that would force every `git clone` to download 2.65 GB
and spend the repo's LFS bandwidth quota. If you maintain a private mirror and
want the weights in the checkout, place them under `weights/` and commit; a
`git lfs pull` in a fresh clone then populates them and MATs resolves from there.

## Manual / air-gapped

Download the two files from the Hugging Face repo, place them in your weights
directory under the names above, and verify:

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
> 2. Set `_HF_REPO_ID` (and pin `_HF_REVISION`) in `src/mats/weights.py`.
> 3. Fill the two `sha256` values in `_MANIFEST`.
> 4. Mint the DOI on the HF repo (Settings → DataCite) and fill the table above.
> Until `_HF_REPO_ID` is set, `mats fetch-weights` prints manual instructions.
