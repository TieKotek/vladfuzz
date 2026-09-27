import os

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Minimal base class for BEVDriver runtime models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_checkpoint(self, url_or_filename):
        if not os.path.isfile(url_or_filename):
            raise RuntimeError(f"checkpoint path is invalid: {url_or_filename}")

        checkpoint = torch.load(url_or_filename, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        return self.load_state_dict(state_dict, strict=False)
