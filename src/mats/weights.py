"""Download, verify and resolve the MATs model checkpoints.

The weights are large (RF-DETR ~134 MB, BiRefNet ~2.65 GB) and are not shipped
inside the package or committed to the repository. They are delivered through
three interchangeable sources, all resolved by :mod:`mats.paths`:

1. **Hugging Face Hub** -- the canonical public host. ``mats fetch-weights`` and
   the transparent first-run auto-fetch pull from here into the per-user cache.
2. **A shared/mounted filesystem** (e.g. USDA SCINet ``/project``) -- point
   ``MATS_WEIGHTS_DIR`` at it and the weights are read in place, no copy.
3. **Git LFS** -- an optional ``weights/`` directory in a checkout.

Before the first public release, set ``_HF_REPO_ID`` (and ideally the per-file
``sha256`` values) below. Until then, ``mats fetch-weights`` prints manual
instructions rather than guessing a location.
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path

from .paths import (
    WEIGHTS_DIR,
    RF_DETR_MARKER_CHECKPOINT,
    BIREFNET_CHECKPOINT,
    RF_DETR_MARKER_FILENAME,
    BIREFNET_FILENAME,
    looks_like_lfs_pointer,
)

# TODO(release): create the Hugging Face weights repo and set its id, e.g.
# "USDA-ARS/mats-weights". Pin _HF_REVISION to a tag or commit for reproducibility.
_HF_REPO_ID = None
_HF_REVISION = "main"

# name -> {filename, target, size_bytes, sha256}. sha256 "" means "not yet pinned".
_MANIFEST = {
    "rf-detr": {
        "filename": RF_DETR_MARKER_FILENAME,
        "target": RF_DETR_MARKER_CHECKPOINT,
        "size_bytes": 134_658_641,
        "sha256": "",  # TODO(release): shasum -a 256 rf_detr_marker.pth
    },
    "birefnet": {
        "filename": BIREFNET_FILENAME,
        "target": BIREFNET_CHECKPOINT,
        "size_bytes": 2_647_028_978,
        "sha256": "",  # TODO(release): shasum -a 256 birefnet_leaf.pth
    },
}

_AUTO_FETCH_DISABLED = "MATS_NO_AUTO_FETCH"


def _hf_available():
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _free_bytes(path):
    return shutil.disk_usage(path).free


def _is_present(target):
    """True only if the target is a real file, not a missing path or LFS stub."""
    target = Path(target)
    return target.is_file() and not looks_like_lfs_pointer(target)


def _manual_instructions():
    print(
        "Automatic download is not configured in this build "
        "(the Hugging Face repo id is not set).\n\n"
        "Get the checkpoints one of these ways:\n"
        f"  - Download them and place them here:\n"
        f"      {WEIGHTS_DIR / RF_DETR_MARKER_FILENAME}\n"
        f"      {WEIGHTS_DIR / BIREFNET_FILENAME}\n"
        "  - Or set MATS_WEIGHTS_DIR to a directory that already contains them\n"
        "    (e.g. a shared SCINet /project path).\n"
        "  - Or set RF_DETR_MARKER_CHECKPOINT / BIREFNET_CHECKPOINT to specific files.\n\n"
        "See docs/weights.md.",
        file=sys.stderr,
    )
    return 1


def _download_from_hf(spec, force):
    """Fetch one checkpoint from the Hugging Face Hub into WEIGHTS_DIR."""
    from huggingface_hub import hf_hub_download

    target = Path(spec["target"])
    target.parent.mkdir(parents=True, exist_ok=True)

    # Preflight free space so a 2.65 GB fetch fails early, not half-written.
    size = spec.get("size_bytes") or 0
    if size and _free_bytes(target.parent) < size * 1.1:
        print(f"error: not enough free space in {target.parent} for {target.name} "
              f"(~{size / 1e9:.1f} GB needed).", file=sys.stderr)
        return False

    print(f"Downloading {spec['filename']} from {_HF_REPO_ID} -> {WEIGHTS_DIR}")
    local = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=spec["filename"],
        revision=_HF_REVISION,
        local_dir=str(WEIGHTS_DIR),
        force_download=force,
    )
    local = Path(local)

    expected = spec.get("sha256")
    if expected:
        actual = _sha256(local)
        if actual != expected:
            print(f"error: checksum mismatch for {local.name}\n"
                  f"  expected {expected}\n  got      {actual}", file=sys.stderr)
            return False
    else:
        print(f"  (no pinned checksum for {local.name}; skipping verification)")

    print(f"  saved {local}")
    return True


def fetch(only=None, force=False):
    """Download one or both checkpoints from Hugging Face. Returns an exit code."""
    if not _HF_REPO_ID or not _hf_available():
        if _HF_REPO_ID and not _hf_available():
            print("error: huggingface_hub is not installed. "
                  "Install it with `pip install huggingface_hub` "
                  "(or `pip install -e .`).", file=sys.stderr)
            return 1
        return _manual_instructions()

    names = [only] if only else list(_MANIFEST)
    ok = True
    for name in names:
        spec = _MANIFEST[name]
        if _is_present(spec["target"]) and not force:
            print(f"{spec['filename']} already present at {spec['target']} "
                  f"(use --force to re-download).")
            continue
        ok = _download_from_hf(spec, force) and ok
    return 0 if ok else 1


def ensure_weight(name):
    """Return a resolved checkpoint Path, auto-fetching once if needed.

    Called lazily by the model loaders so the first inference on a fresh machine
    downloads the weights transparently. Set MATS_NO_AUTO_FETCH=1 to disable the
    download (e.g. on an HPC login node) and require the weights to be pre-staged.
    """
    spec = _MANIFEST[name]
    target = Path(spec["target"])
    if _is_present(target):
        return target

    if os.environ.get(_AUTO_FETCH_DISABLED):
        raise FileNotFoundError(
            f"{spec['filename']} not found at {target} and auto-fetch is disabled "
            f"({_AUTO_FETCH_DISABLED} is set). Pre-stage the weights or run "
            f"`mats fetch-weights`."
        )

    if looks_like_lfs_pointer(target):
        raise FileNotFoundError(
            f"{target} is a Git LFS pointer, not the weight file. "
            f"Run `git lfs pull`, or `mats fetch-weights`, or unset the checkout "
            f"so weights resolve from the cache."
        )

    code = fetch(only=name)
    if code != 0 or not _is_present(target):
        raise FileNotFoundError(
            f"Could not obtain {spec['filename']}. Run `mats fetch-weights` or see "
            f"docs/weights.md."
        )
    return target


def _source_of(target):
    """Human-readable description of where a resolved checkpoint came from."""
    target = Path(target)
    if not target.exists():
        return "not found"
    if looks_like_lfs_pointer(target):
        return "Git LFS pointer -- not fetched (run `git lfs pull`)"
    try:
        if target.parent == Path(WEIGHTS_DIR):
            if os.environ.get("MATS_WEIGHTS_DIR"):
                return "MATS_WEIGHTS_DIR (shared filesystem)"
            return "cache / Hugging Face auto-fetch"
    except Exception:
        pass
    return "explicit path / checkout"


def doctor():
    """Print resolved paths, sources, devices and QR decoder availability."""
    print("MATs environment check")
    print("=" * 60)

    print(f"weights dir:        {WEIGHTS_DIR}")
    print(f"HF weights repo:    {_HF_REPO_ID or '(not configured)'}")
    print(f"huggingface_hub:    {'ok' if _hf_available() else 'NOT installed'}")
    print(f"auto-fetch:         {'disabled (MATS_NO_AUTO_FETCH set)' if os.environ.get(_AUTO_FETCH_DISABLED) else 'enabled'}")

    all_present = True
    for label, path in (("RF-DETR checkpoint", RF_DETR_MARKER_CHECKPOINT),
                        ("BiRefNet checkpoint", BIREFNET_CHECKPOINT)):
        present = _is_present(path)
        all_present = all_present and present
        mark = "ok" if present else "MISSING"
        print(f"{label:<20} [{mark}] {path}")
        print(f"{'':<20}  source: {_source_of(path)}")

    try:
        import torch
        devices = []
        if torch.cuda.is_available():
            devices.append(f"cuda ({torch.cuda.get_device_name(0)})")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")
        print(f"torch:              {torch.__version__}")
        print(f"available devices:  {', '.join(devices)}")
    except Exception as exc:
        print(f"torch:              NOT importable ({exc})")

    try:
        import pyzbar.pyzbar  # noqa: F401
        print("pyzbar/zbar:        ok")
    except Exception as exc:
        print(f"pyzbar/zbar:        NOT importable ({exc}) -- install system 'zbar'")

    if not all_present:
        print("\nSome weights are missing. Run:  mats fetch-weights")
    return 0 if all_present else 1
