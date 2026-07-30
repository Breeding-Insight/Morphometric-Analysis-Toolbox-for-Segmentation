"""Pinned BiRefNet configuration bundled for offline MATS inference.

Source: ZhengPeng7/BiRefNet snapshot e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
(MIT; see ``LICENSE`` in this directory).
"""

from transformers import PretrainedConfig

class BiRefNetConfig(PretrainedConfig):
    model_type = "SegformerForSemanticSegmentation"
    def __init__(
        self,
        bb_pretrained=False,
        **kwargs
    ):
        self.bb_pretrained = bb_pretrained
        super().__init__(**kwargs)
