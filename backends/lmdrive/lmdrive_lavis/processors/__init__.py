"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from backends.lmdrive.lmdrive_lavis.processors.base_processor import BaseProcessor

from backends.lmdrive.lmdrive_lavis.processors.alpro_processors import (
    AlproVideoTrainProcessor,
    AlproVideoEvalProcessor,
)
from backends.lmdrive.lmdrive_lavis.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from backends.lmdrive.lmdrive_lavis.processors.blip_diffusion_processors import (
    BlipDiffusionInputImageProcessor,
    BlipDiffusionTargetImageProcessor,
)
from backends.lmdrive.lmdrive_lavis.processors.gpt_processors import (
    GPTVideoFeatureProcessor,
    GPTDialogueProcessor,
)
from backends.lmdrive.lmdrive_lavis.processors.clip_processors import ClipImageTrainProcessor

from backends.lmdrive.lmdrive_lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    # ALPRO
    "AlproVideoTrainProcessor",
    "AlproVideoEvalProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    # BLIP-Diffusion
    "BlipDiffusionInputImageProcessor",
    "BlipDiffusionTargetImageProcessor",
    # CLIP
    "ClipImageTrainProcessor",
    # GPT
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
