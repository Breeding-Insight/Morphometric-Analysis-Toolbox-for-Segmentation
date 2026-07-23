"""Checkpoint path resolution for the MATs pipeline.

The model weights are large (RF-DETR ~134 MB, BiRefNet ~2.65 GB) and are not
shipped inside the package. They are delivered through three interchangeable
sources; this module resolves whichever is available.

Resolution order for each checkpoint (first real file wins):

1. The per-checkpoint environment variable (``RF_DETR_MARKER_CHECKPOINT`` /
   ``BIREFNET_CHECKPOINT``) -- an explicit absolute path.
2. ``$MATS_WEIGHTS_DIR/<filename>`` -- e.g. a shared, mounted SCINet ``/project``
   directory read in place (no per-user copy).
3. The default user cache: ``$XDG_CACHE_HOME/mats/weights`` or
   ``~/.cache/mats/weights`` -- where ``mats fetch-weights`` downloads to.
4. A ``weights/`` directory in the repo checkout (Git LFS or manual placement).

Un-smudged Git LFS pointer stubs are ignored, so a checkout without ``git lfs
pull`` falls through to the cache / auto-fetch rather than handing PyTorch a
130-byte text file.

These constants are resolved at import time, so any environment variable you
rely on must be set *before* ``import mats``.
"""

import os
from pathlib import Path

RF_DETR_MARKER_FILENAME = "rf_detr_marker.pth"
BIREFNET_FILENAME = "birefnet_leaf.pth"

# Repo root when running from a checkout: src/mats/paths.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def looks_like_lfs_pointer(path):
    """True if ``path`` is an un-smudged Git LFS pointer stub, not real content.

    LFS pointer files are small text files beginning with a fixed version line.
    A checkout without ``git lfs pull`` leaves these in place of the weights.
    """
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(len(_LFS_POINTER_MAGIC)) == _LFS_POINTER_MAGIC
    except OSError:
        return False


def _weights_dir() -> Path:
    """Return the directory MATs downloads and looks for weights in."""
    override = os.environ.get("MATS_WEIGHTS_DIR")
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "mats" / "weights"


WEIGHTS_DIR = _weights_dir()


def _resolve_checkpoint(env_var, *candidates):
    """Pick a checkpoint path: env override first, then the first real file.

    Skips Git LFS pointer stubs. Falls back to the first candidate so error
    messages report a sensible path when nothing is found on disk.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    resolved = [Path(c).expanduser() for c in candidates]
    for path in resolved:
        if path.is_file() and not looks_like_lfs_pointer(path):
            return path
    return resolved[0]


RF_DETR_MARKER_CHECKPOINT = _resolve_checkpoint(
    "RF_DETR_MARKER_CHECKPOINT",
    WEIGHTS_DIR / RF_DETR_MARKER_FILENAME,
    _REPO_ROOT / "weights" / RF_DETR_MARKER_FILENAME,
    Path.cwd() / "weights" / RF_DETR_MARKER_FILENAME,
)
BIREFNET_CHECKPOINT = _resolve_checkpoint(
    "BIREFNET_CHECKPOINT",
    WEIGHTS_DIR / BIREFNET_FILENAME,
    _REPO_ROOT / "weights" / BIREFNET_FILENAME,
    Path.cwd() / "weights" / BIREFNET_FILENAME,
)
