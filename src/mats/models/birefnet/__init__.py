"""Pinned local BiRefNet architecture used by the MATS leaf checkpoint."""

from .BiRefNet_config import BiRefNetConfig
from .birefnet import BiRefNet


def create_birefnet_model() -> BiRefNet:
    """Build the checkpoint-compatible architecture without network access."""
    return BiRefNet(config=BiRefNetConfig(bb_pretrained=False))


__all__ = ["BiRefNet", "BiRefNetConfig", "create_birefnet_model"]
