"""Shared BiRefNet device discovery for the CLI, UI, and inference runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Must be set before any module imports torch on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass(frozen=True)
class BiRefNetDeviceReport:
    device: str
    severity: str
    detail: str
    accelerated: bool
    forced: bool
    error: str | None = None
    torch_version: str | None = None


@dataclass(frozen=True)
class WorkerRiskReport:
    """Describe the safety band for a requested CPU worker count."""

    selected_workers: int
    available_workers: int
    utilization: float
    normal_limit: int
    level: str
    label: str
    color: str
    requires_break_glass: bool


def available_cpu_workers() -> int:
    """Return the CPU capacity available to this process.

    Schedulers commonly communicate a smaller allocation than the host's total
    CPU count.  Linux CPU affinity is the most reliable source when available;
    Slurm and common scheduler environment variables provide useful fallbacks.
    The smallest reported positive value prevents the UI from offering workers
    that would oversubscribe an interactive HPC job.
    """
    candidates = []

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            candidates.append(len(affinity(0)))
        except OSError:
            pass

    for name in ("SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NSLOTS"):
        value = os.environ.get(name)
        if not value:
            continue
        try:
            count = int(value)
        except ValueError:
            continue
        if count > 0:
            candidates.append(count)

    host_count = os.cpu_count()
    if host_count and host_count > 0:
        candidates.append(host_count)

    return max(1, min(candidates)) if candidates else 1


def worker_risk_report(selected_workers: int, available_workers: int) -> WorkerRiskReport:
    """Classify worker utilization and identify counts requiring an override."""
    available = max(1, int(available_workers))
    selected = max(1, int(selected_workers))
    normal_limit = max(1, int(available * 0.75))
    utilization = selected / available

    # A one-CPU allocation has no lower parallelism option, so it must remain
    # usable without an acknowledgement despite representing 100% utilization.
    if available == 1:
        return WorkerRiskReport(
            selected, available, utilization, normal_limit,
            "limited", "Limited CPU capacity", "yellow", False,
        )
    if utilization <= 0.25:
        level, label, color = "low", "Low load", "green"
    elif utilization <= 0.50:
        level, label, color = "moderate", "Moderate load", "yellow"
    elif utilization <= 0.75:
        level, label, color = "high", "High load", "red"
    else:
        level, label, color = "critical", "Critical load", "red"

    return WorkerRiskReport(
        selected,
        available,
        utilization,
        normal_limit,
        level,
        label,
        color,
        selected > normal_limit,
    )


def _probe(torch, device: str) -> str | None:
    """Return an error message when a backend cannot perform basic tensor work."""
    try:
        value = torch.ones(1, device=device)
        _ = (value + 1).sum().item()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    except Exception as exc:  # pragma: no cover - depends on local hardware
        return str(exc)
    return None


def birefnet_device_report() -> BiRefNetDeviceReport:
    """Describe the effective BiRefNet device without loading the model.

    Green means a CUDA or MPS backend passed a small accessibility probe. CPU is
    supported but intentionally reported as a yellow, slower fallback.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency/environment specific
        return BiRefNetDeviceReport(
            device="cpu", severity="error", detail="PyTorch is not importable.",
            accelerated=False, forced=False, error=str(exc), torch_version=None,
        )

    forced_value = os.environ.get("BIREFNET_DEVICE")
    candidates: list[str]
    if forced_value:
        forced = forced_value.strip().lower()
        if forced == "cpu":
            return BiRefNetDeviceReport(
                device="cpu", severity="warning", detail="CPU was forced by BIREFNET_DEVICE.",
                accelerated=False, forced=True, torch_version=torch.__version__,
            )
        if forced.startswith("cuda"):
            candidates = [forced]
        elif forced == "mps":
            candidates = ["mps"]
        else:
            return BiRefNetDeviceReport(
                device="cpu", severity="error",
                detail=f"Unsupported BIREFNET_DEVICE value: {forced_value!r}.",
                accelerated=False, forced=True, error="invalid forced device",
                torch_version=torch.__version__,
            )
    else:
        candidates = []
        if torch.cuda.is_available():
            candidates.append("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_built() and mps.is_available():
            candidates.append("mps")

    for candidate in candidates:
        if candidate.startswith("cuda") and not torch.cuda.is_available():
            error = "CUDA is not available to this Python process."
        elif candidate == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            error = "Apple MPS is not available to this Python process."
        else:
            error = _probe(torch, candidate)
        if error is None:
            name = "CUDA GPU"
            if candidate.startswith("cuda"):
                try:
                    name = torch.cuda.get_device_name(candidate)
                except Exception:
                    pass
            elif candidate == "mps":
                name = "Apple MPS"
            return BiRefNetDeviceReport(
                device=candidate, severity="success", detail=f"BiRefNet acceleration available: {name}.",
                accelerated=True, forced=bool(forced_value), torch_version=torch.__version__,
            )
        if forced_value:
            return BiRefNetDeviceReport(
                device="cpu", severity="error", detail=f"Forced {candidate} is unusable: {error}",
                accelerated=False, forced=True, error=error, torch_version=torch.__version__,
            )

    return BiRefNetDeviceReport(
        device="cpu", severity="warning",
        detail="No usable CUDA or Apple MPS device was found; BiRefNet will run on CPU.",
        accelerated=False, forced=False, torch_version=torch.__version__,
    )
