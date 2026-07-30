"""Dependency checks for the optional, locally packaged BiRefNet model."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


BIREFNET_DEPENDENCIES = ("einops", "kornia", "timm")


@dataclass(frozen=True)
class BiRefNetRuntimeStatus:
    """Whether optional BiRefNet Python dependencies can be imported."""

    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def detail(self) -> str:
        if self.ready:
            return "Packaged BiRefNet runtime is ready."
        packages = ", ".join(self.missing)
        return (
            f"BiRefNet runtime is incomplete: missing {packages}. "
            'Reinstall MATS with `python -m pip install -U -e ".[app]"`.'
        )


def birefnet_runtime_status() -> BiRefNetRuntimeStatus:
    """Check imports without importing the heavy model definition itself."""
    missing = []
    for package in BIREFNET_DEPENDENCIES:
        try:
            importlib.import_module(package)
        except Exception:
            missing.append(package)
    return BiRefNetRuntimeStatus(tuple(missing))


def require_birefnet_dependencies() -> None:
    """Raise one actionable error before attempting to load a checkpoint."""
    status = birefnet_runtime_status()
    if not status.ready:
        raise RuntimeError(status.detail)
