"""Install, verify and resolve the MATs model checkpoints.

The checkpoints are large (RF-DETR ~134 MB, BiRefNet ~2.65 GB) and are
delivered through two independent channels, plus a shared-filesystem escape
hatch -- :mod:`mats.paths` resolves whichever produced a real file:

1. **Hugging Face Hub** -- the default public host. Free, no account needed,
   but unreachable on some institutional networks (notably USDA's).
2. **Git LFS, in this repository** -- RF-DETR is fetched on every
   ``git clone`` (mandatory for every run). BiRefNet is committed too, but
   excluded from the default clone/fetch via ``.lfsconfig``
   (``lfs.fetchexclude``), so a plain clone stays small; it's pulled
   explicitly through this module (or the BiRefNet setup page) when needed.
   This channel exists because Hugging Face is not reachable from every
   collaborator's network.
3. **A shared/mounted filesystem** (e.g. USDA SCINet ``/project``) -- point
   ``MATS_WEIGHTS_DIR`` at it and the weights are read in place, no download,
   for anyone who can mount it.

Set ``_DEFAULT_HF_REPO_ID`` (or export ``MATS_WEIGHTS_HF_REPO``) to enable the
Hugging Face channel; the Git LFS channel needs no configuration beyond
``git`` and ``git-lfs`` being installed.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import (
    WEIGHTS_DIR,
    RF_DETR_MARKER_CHECKPOINT,
    BIREFNET_CHECKPOINT,
    RF_DETR_MARKER_FILENAME,
    BIREFNET_FILENAME,
    looks_like_lfs_pointer,
    _REPO_ROOT,
)

# TODO(release): set the Hugging Face repo id, e.g. "Breeding-Insight/mats-weights".
# Pin _HF_REVISION to a tag or commit for reproducibility. Can also be set per
# deployment via MATS_WEIGHTS_HF_REPO without touching package code. We
# deliberately do not guess a public source for the fine-tuned checkpoint:
# downloading upstream BiRefNet would not provide the leaf model.
_DEFAULT_HF_REPO_ID = None
_HF_REPO_ID = os.environ.get("MATS_WEIGHTS_HF_REPO") or _DEFAULT_HF_REPO_ID
_HF_REVISION = os.environ.get("MATS_WEIGHTS_HF_REVISION", "main")

# name -> {filename, target, size_bytes, sha256}.
_MANIFEST = {
    "rf-detr": {
        "filename": RF_DETR_MARKER_FILENAME,
        "target": RF_DETR_MARKER_CHECKPOINT,
        "size_bytes": 134_658_641,
        "sha256": "15896dd7cfaf8ee6e38c1226f6384a908a111f9a405f229bb5fe7db6264b6bca",
    },
    "birefnet": {
        "filename": BIREFNET_FILENAME,
        "target": BIREFNET_CHECKPOINT,
        "size_bytes": 2_647_028_978,
        "sha256": "27b9481d18243101177c8a8606b60c01e03bed4926e97ae1e14ceeebb5c91377",
    },
}

_AUTO_FETCH_DISABLED = "MATS_NO_AUTO_FETCH"


@dataclass(frozen=True)
class WeightSource:
    id: str
    label: str
    available: bool
    reason: str


@dataclass(frozen=True)
class WeightStatus:
    name: str
    path: Path
    state: str
    detail: str
    size_bytes: int
    expected_size_bytes: int
    sources: tuple


class WeightDownloadError(RuntimeError):
    """A user-facing, retryable checkpoint installation error."""


def _hf_available():
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False


def _git_lfs_installed():
    return shutil.which("git") is not None and shutil.which("git-lfs") is not None


def _is_git_checkout():
    return (_REPO_ROOT / ".git").exists()


def available_sources(name):
    """Return the (Hugging Face, Git LFS) download options for a checkpoint."""
    if name == "birefnet":
        hf_reason = "BiRefNet checkpoints are local-only in this build."
    elif _HF_REPO_ID:
        hf_reason = "" if _hf_available() else "huggingface_hub is not installed."
    else:
        hf_reason = "No Hugging Face repository is configured for this build."
    hf = WeightSource("hf", "Hugging Face", hf_reason == "", hf_reason)

    if not _is_git_checkout():
        lfs_reason = "Not running from a Git checkout (e.g. installed from a wheel)."
    elif not _git_lfs_installed():
        lfs_reason = "git and/or git-lfs are not installed."
    else:
        lfs_reason = ""
    lfs = WeightSource("lfs", "This repository (Git LFS)", lfs_reason == "", lfs_reason)
    return (hf, lfs)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def free_bytes(path):
    """Free space, in bytes, on the filesystem containing path."""
    return shutil.disk_usage(path).free


def _is_present(target):
    """True only if the target is a real file, not a missing path or LFS stub."""
    target = Path(target)
    return target.is_file() and not looks_like_lfs_pointer(target)


def _checkout_target(name):
    """The checkout's weights/<filename> path, or None outside a checkout."""
    if not _is_git_checkout():
        return None
    return _REPO_ROOT / "weights" / _MANIFEST[name]["filename"]


def _resolved_present_path(name):
    """A real, already-downloaded path for this checkpoint, if one exists.

    Checks the resolved target (paths.py's env/cache/checkout search) and,
    separately, the checkout's weights/ directory. paths.py deliberately
    skips LFS pointer stubs when resolving the module-level constant, so a
    checkpoint fetched into the checkout by _download_from_lfs would
    otherwise look "not present" for the rest of the process's lifetime.
    """
    target = Path(_MANIFEST[name]["target"])
    if _is_present(target):
        return target
    checkout = _checkout_target(name)
    if checkout is not None and _is_present(checkout):
        return checkout
    return None


def _download_target(name):
    """Return the path a new Hugging Face download writes to.

    A pre-existing checkpoint may resolve from a checkout (see
    _resolved_present_path); new HF downloads always go to an explicit
    per-model override or the configured cache/shared directory, never the
    checkout -- Git LFS owns the checkout copy.
    """
    if name == "rf-detr":
        override = os.environ.get("RF_DETR_MARKER_CHECKPOINT")
        filename = RF_DETR_MARKER_FILENAME
    else:
        override = os.environ.get("BIREFNET_CHECKPOINT")
        filename = BIREFNET_FILENAME
    return Path(override).expanduser() if override else Path(WEIGHTS_DIR) / filename


def get_weight_status(name):
    """Return a cheap, UI-safe description of a checkpoint's current state."""
    if name not in _MANIFEST:
        raise KeyError(f"Unknown checkpoint: {name}")
    spec = _MANIFEST[name]
    sources = available_sources(name)

    present = _resolved_present_path(name)
    if present is not None:
        size = present.stat().st_size
        if size != spec["size_bytes"]:
            return WeightStatus(name, present, "invalid", f"Unexpected size ({size} bytes).",
                                 size, spec["size_bytes"], sources)
        return WeightStatus(name, present, "ready", "Installed.", size, spec["size_bytes"], sources)

    checkout = _checkout_target(name)
    if checkout is not None and looks_like_lfs_pointer(checkout):
        detail = "Excluded from `git clone` by design -- fetch it via Git LFS or Hugging Face."
        return WeightStatus(name, checkout, "missing", detail, 0, spec["size_bytes"], sources)

    target = _download_target(name)
    detail = "Not installed."
    if not any(s.available for s in sources):
        detail += " No download source is configured for this build."
    return WeightStatus(name, target, "missing", detail, 0, spec["size_bytes"], sources)


def _emit(progress_callback, phase, completed, total):
    if progress_callback:
        progress_callback(phase, completed, total)


def _manual_instructions():
    print(
        "No automatic download source is available in this build.\n\n"
        "Get the checkpoints one of these ways:\n"
        f"  - Download them and place them here:\n"
        f"      {WEIGHTS_DIR / RF_DETR_MARKER_FILENAME}\n"
        f"      {WEIGHTS_DIR / BIREFNET_FILENAME}\n"
        "  - Or set MATS_WEIGHTS_DIR to a directory that already contains them\n"
        "    (e.g. a shared SCINet /project path).\n"
        "  - Or set RF_DETR_MARKER_CHECKPOINT / BIREFNET_CHECKPOINT to specific files.\n"
        "  - Or, from a Git checkout with Git LFS installed:\n"
        "      git lfs pull --include=\"weights/birefnet_leaf.pth\"\n\n"
        "See docs/weights.md.",
        file=sys.stderr,
    )
    return 1


def _download_from_hf(spec, force, progress_callback=None):
    """Fetch one checkpoint from the Hugging Face Hub into WEIGHTS_DIR."""
    from huggingface_hub import hf_hub_download

    name = next(key for key, value in _MANIFEST.items() if value is spec)
    target = _download_target(name)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Preflight free space so a 2.65 GB fetch fails early, not half-written.
    size = spec.get("size_bytes") or 0
    if size and free_bytes(target.parent) < size * 1.1:
        print(f"error: not enough free space in {target.parent} for {target.name} "
              f"(~{size / 1e9:.1f} GB needed).", file=sys.stderr)
        return False

    _emit(progress_callback, "preparing", 0, size)
    print(f"Downloading {spec['filename']} from {_HF_REPO_ID} -> {target.parent}")
    try:
        from tqdm.auto import tqdm

        class StreamlitProgressTqdm(tqdm):
            def update(self, n=1):
                result = super().update(n)
                _emit(progress_callback, "downloading", self.n, self.total or size)
                return result
    except Exception:  # pragma: no cover - tqdm is an existing dependency
        StreamlitProgressTqdm = None
    download_kwargs = {
        "repo_id": _HF_REPO_ID,
        "filename": spec["filename"],
        "revision": _HF_REVISION,
        "local_dir": str(target.parent),
        "force_download": force,
    }
    if StreamlitProgressTqdm:
        try:
            local = hf_hub_download(**download_kwargs, tqdm_class=StreamlitProgressTqdm)
        except TypeError:
            # huggingface_hub 0.23 lacks tqdm_class. It still downloads safely;
            # the UI stays on the preparation state until verification begins.
            local = hf_hub_download(**download_kwargs)
    else:
        local = hf_hub_download(**download_kwargs)
    local = Path(local)

    # A per-file override can name a file differently from the hub artifact.
    if local != target:
        local.replace(target)
        local = target

    expected = spec.get("sha256")
    if expected:
        _emit(progress_callback, "verifying", 0, size)
        actual = _sha256(local)
        if actual != expected:
            print(f"error: checksum mismatch for {local.name}\n"
                  f"  expected {expected}\n  got      {actual}", file=sys.stderr)
            return False
    else:
        print(f"  (no pinned checksum for {local.name}; skipping verification)")

    _emit(progress_callback, "complete", size, size)
    print(f"  saved {local}")
    return True


def _parse_lfs_progress_line(line):
    """Parse one `git lfs pull` GIT_LFS_PROGRESS line: '<dir> n/N done/total path'."""
    parts = line.split()
    if len(parts) < 3 or "/" not in parts[2]:
        return None
    try:
        done, total = parts[2].split("/")
        return int(done), int(total)
    except ValueError:
        return None


def _emit_lfs_progress(progress_path, progress_callback, fallback_total, last_done):
    try:
        lines = Path(progress_path).read_text().splitlines()
    except OSError:
        return last_done
    if not lines:
        return last_done
    parsed = _parse_lfs_progress_line(lines[-1])
    if not parsed:
        return last_done
    done, total = parsed
    _emit(progress_callback, "downloading", done, total or fallback_total)
    return done


def _download_from_lfs(name, progress_callback=None):
    """Fetch one checkpoint via `git lfs pull --include`, overriding this file's
    .lfsconfig fetchexclude for just this invocation.

    Writes into the checkout's weights/ directory -- that's where Git LFS
    smudges content, and it's tier 3 of paths.py's resolution order, so the
    file is picked up with no extra configuration once it's there.
    """
    spec = _MANIFEST[name]
    target = _checkout_target(name)
    if target is None:
        print("error: not running from a Git checkout; the Git LFS channel is unavailable.",
              file=sys.stderr)
        return False
    if not _git_lfs_installed():
        print("error: git and/or git-lfs are not installed.", file=sys.stderr)
        return False

    size = spec.get("size_bytes") or 0
    if size and free_bytes(target.parent) < size * 1.1:
        print(f"error: not enough free space near {target.parent} for {target.name} "
              f"(~{size / 1e9:.1f} GB needed).", file=sys.stderr)
        return False

    _emit(progress_callback, "preparing", 0, size)
    rel_path = f"weights/{spec['filename']}"
    print(f"Fetching {rel_path} via Git LFS -> {target}")

    with tempfile.TemporaryDirectory() as tmp:
        progress_path = Path(tmp) / "progress"
        log_path = Path(tmp) / "output.log"
        env = {**os.environ, "GIT_LFS_PROGRESS": str(progress_path)}
        # Output goes to a file, not a pipe: `git lfs pull` runs for minutes on
        # a 2.65 GB fetch, and a PIPE we only read after poll() returns risks
        # deadlocking the child if it writes enough to fill the OS pipe buffer.
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                ["git", "lfs", "pull", "--include", rel_path],
                cwd=_REPO_ROOT, env=env,
                stdout=log_file, stderr=subprocess.STDOUT,
            )
            last_done = 0
            while proc.poll() is None:
                time.sleep(0.2)
                last_done = _emit_lfs_progress(progress_path, progress_callback, size, last_done)
            _emit_lfs_progress(progress_path, progress_callback, size, last_done)
        output = log_path.read_text() if log_path.exists() else ""

    if proc.returncode != 0:
        print(f"error: `git lfs pull` failed:\n{output.strip()}", file=sys.stderr)
        return False

    _emit(progress_callback, "verifying", 0, size)
    if not target.is_file() or looks_like_lfs_pointer(target):
        print(f"error: {target} is still not a real file after `git lfs pull`.", file=sys.stderr)
        return False

    actual_size = target.stat().st_size
    expected = spec.get("sha256")
    if expected:
        actual = _sha256(target)
        if actual != expected:
            print(f"error: checksum mismatch for {target.name}\n"
                  f"  expected {expected}\n  got      {actual}", file=sys.stderr)
            return False
    elif actual_size != spec["size_bytes"]:
        print(f"error: unexpected size for {target.name}: {actual_size} bytes.", file=sys.stderr)
        return False

    _emit(progress_callback, "complete", size, size)
    print(f"  saved {target}")
    return True


def install_weight(name, source="auto", force=False, progress_callback=None):
    """Explicitly install one checkpoint for the GUI or API callers.

    `source` is "auto" (Hugging Face if configured, else Git LFS), "hf", or
    "lfs". This deliberately bypasses MATS_NO_AUTO_FETCH: a button click or
    explicit call is user consent, unlike transparent first-inference
    fetching.
    """
    if name not in _MANIFEST:
        raise WeightDownloadError(f"Unknown checkpoint: {name}")
    if source not in ("auto", "hf", "lfs"):
        raise WeightDownloadError(f"Unknown source: {source!r}")

    status = get_weight_status(name)
    if status.state == "ready" and not force:
        _emit(progress_callback, "complete", status.size_bytes, status.expected_size_bytes)
        return status

    by_id = {s.id: s for s in status.sources}
    if source == "auto":
        chosen = next((s.id for s in status.sources if s.available), None)
        if chosen is None:
            raise WeightDownloadError("No download source is configured for this build.")
    else:
        chosen = source
        if not by_id[chosen].available:
            raise WeightDownloadError(by_id[chosen].reason or f"{by_id[chosen].label} is unavailable.")

    try:
        if chosen == "hf":
            if not _hf_available():
                raise WeightDownloadError("huggingface_hub is not installed.")
            ok = _download_from_hf(_MANIFEST[name], force, progress_callback)
        else:
            ok = _download_from_lfs(name, progress_callback)
    except WeightDownloadError:
        raise
    except Exception as exc:
        raise WeightDownloadError(str(exc)) from exc
    if not ok:
        raise WeightDownloadError(f"Could not install {name} from {by_id[chosen].label}.")

    status = get_weight_status(name)
    if status.state != "ready":
        raise WeightDownloadError(status.detail)
    return status


def fetch(only=None, force=False, source="auto"):
    """Download one or both checkpoints. Returns an exit code."""
    if source not in ("auto", "hf", "lfs"):
        print(f"error: unknown source {source!r} (expected auto, hf, or lfs).", file=sys.stderr)
        return 1

    names = [only] if only else list(_MANIFEST)
    ok = True
    for name in names:
        if _resolved_present_path(name) is not None and not force:
            print(f"{_MANIFEST[name]['filename']} already present "
                  f"(use --force to re-download).")
            continue

        by_id = {s.id: s for s in available_sources(name)}
        if source == "auto":
            chosen = next((sid for sid in ("hf", "lfs") if by_id[sid].available), None)
        else:
            chosen = source if by_id[source].available else None

        if chosen is None:
            if source != "auto":
                print(f"error: {by_id[source].label} is unavailable: {by_id[source].reason}",
                      file=sys.stderr)
            else:
                _manual_instructions()
            ok = False
            continue

        if chosen == "hf":
            if not _hf_available():
                print("error: huggingface_hub is not installed. "
                      "Install it with `pip install huggingface_hub` "
                      "(or `pip install -e .`).", file=sys.stderr)
                ok = False
                continue
            ok = _download_from_hf(_MANIFEST[name], force) and ok
        else:
            ok = _download_from_lfs(name) and ok
    return 0 if ok else 1


def ensure_weight(name):
    """Return a resolved checkpoint Path, auto-fetching once if needed.

    Called lazily by the model loaders so the first inference on a fresh
    machine downloads the weights transparently (checked once per process --
    see the _MARKER_MODELS / _BIREFNET_MODELS caches in core.py). Set
    MATS_NO_AUTO_FETCH=1 to disable the download (e.g. on an HPC login node)
    and require the weights to be pre-staged.
    """
    spec = _MANIFEST[name]
    present = _resolved_present_path(name)
    if present is not None:
        return present

    if os.environ.get(_AUTO_FETCH_DISABLED):
        raise FileNotFoundError(
            f"{spec['filename']} not found and auto-fetch is disabled "
            f"({_AUTO_FETCH_DISABLED} is set). Pre-stage the weights, or run "
            f"`mats fetch-weights --only {name}` after unsetting {_AUTO_FETCH_DISABLED} "
            f"(from a Git checkout, `git lfs pull --include=\"weights/{spec['filename']}\"` "
            f"also works)."
        )

    # A checkout excludes BiRefNet from the default clone (.lfsconfig); a
    # pointer stub here means "not yet pulled", not an error -- fetch it.
    checkout = _checkout_target(name)
    if checkout is not None and looks_like_lfs_pointer(checkout) and _download_from_lfs(name):
        return checkout

    code = fetch(only=name)
    present = _resolved_present_path(name)
    if code != 0 or present is None:
        raise FileNotFoundError(
            f"Could not obtain {spec['filename']}. Run `mats fetch-weights` or see "
            f"docs/weights.md."
        )
    return present


def require_local_weight(name):
    """Return a ready local checkpoint without fetching it.

    BiRefNet is optional and large, so inference must never start a download
    merely because a user selected it.  Explicit setup actions may still use
    :func:`install_weight` or :func:`fetch`.
    """
    if name not in _MANIFEST:
        raise KeyError(f"Unknown checkpoint: {name}")

    status = get_weight_status(name)
    if status.state == "ready":
        return status.path
    if status.state == "invalid":
        raise FileNotFoundError(
            f"{_MANIFEST[name]['filename']} is invalid at {status.path}: {status.detail}"
        )
    raise FileNotFoundError(
        f"{_MANIFEST[name]['filename']} is not installed locally. "
        f"BiRefNet is optional; install it explicitly with "
        f"`mats fetch-weights --only {name} --source lfs`, or place it at {status.path}."
    )


def _source_of(name, target):
    """Human-readable description of where a resolved checkpoint came from."""
    target = Path(target)
    if not target.exists():
        checkout = _checkout_target(name)
        if checkout is not None and looks_like_lfs_pointer(checkout):
            return "Git LFS pointer -- excluded from clone (fetch via the app or `mats fetch-weights`)"
        return "not found"
    if looks_like_lfs_pointer(target):
        return "Git LFS pointer -- not fetched"
    checkout = _checkout_target(name)
    if checkout is not None and target == checkout:
        return "checkout (Git LFS)"
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
    print(f"git-lfs channel:    {'ok' if _git_lfs_installed() else 'unavailable (git/git-lfs not on PATH)'}")
    print(f"auto-fetch:         {'disabled (MATS_NO_AUTO_FETCH set)' if os.environ.get(_AUTO_FETCH_DISABLED) else 'enabled'}")

    # RF-DETR is mandatory for every run; BiRefNet is optional (only needed for
    # --mask-method birefnet), so its absence isn't a failure -- just report it.
    all_present = True
    for name, label, required in (
        ("rf-detr", "RF-DETR checkpoint", True),
        ("birefnet", "BiRefNet checkpoint", False),
    ):
        present = _resolved_present_path(name)
        if required and present is None:
            all_present = False
        if present is not None:
            mark = "ok"
        elif required:
            mark = "MISSING"
        else:
            mark = "not fetched (optional -- only needed for --mask-method birefnet)"
        path = present if present is not None else _MANIFEST[name]["target"]
        print(f"{label:<20} [{mark}] {path}")
        print(f"{'':<20}  source: {_source_of(name, path)}")
        if present is None:
            channels = ", ".join(
                s.label if s.available else f"{s.label} (unavailable: {s.reason})"
                for s in available_sources(name)
            )
            print(f"{'':<20}  download via: {channels}")

    from .devices import birefnet_device_report
    device = birefnet_device_report()
    print(f"torch:              {device.torch_version or 'NOT importable'}")
    print(f"BiRefNet device:    {device.device} ({device.detail})")

    from .qr_runtime import qr_runtime_status

    qr_status = qr_runtime_status()
    backends = [
        backend.name.lower()
        for backend in (qr_status.opencv, qr_status.pyzbar, qr_status.qreader)
        if backend.available
    ]
    suffix = "" if qr_status.enhanced_available else "  (OpenCV only)"
    print(f"QR decoding:        {', '.join(backends) or 'NONE'}{suffix}")

    if not all_present:
        print("\nA required weight is missing. Run:  mats fetch-weights")
    return 0 if all_present else 1
