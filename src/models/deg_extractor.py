import argparse
import logging

import torch
import torch.nn as nn

from src.networks.degnet import DegNet_DINO

logger = logging.getLogger(__name__)


class DegFeatExtractor(nn.Module):
    def __init__(
        self,
        inner_dim: int,
        num_deg_types: int,
        weight_dtype: torch.dtype,
        args: argparse.Namespace,
        deg_embedding: nn.Parameter | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self._log_counter = 0
        if deg_embedding is not None:
            self.deg_embedding = deg_embedding
        else:
            self.deg_embedding = nn.Parameter(torch.randn(num_deg_types, inner_dim))
            nn.init.orthogonal_(self.deg_embedding)

        self.weight_dtype = weight_dtype
        self.deg_classifier = DegNet_DINO(
            dino_type=args.dino_type,
            num_types=num_deg_types,
        )
        state_dict = torch.load(args.degradation_classifier_path, map_location="cpu")
        self.deg_classifier.load_state_dict(state_dict, strict=False)
        self.deg_classifier.requires_grad_(False).eval()
        self.deg_classifier.to(device=device or torch.device("cpu"))

    def forward(self, lq_images: torch.Tensor) -> torch.Tensor:
        logits = self.deg_classifier(lq_images)
        deg_probs = torch.softmax(logits, dim=-1)[:, :, 0].to(dtype=self.weight_dtype)
        embedding = self.deg_embedding.to(device=lq_images.device, dtype=self.weight_dtype)

        if self._log_counter < 3:
            logger.info(
                f"[DegFeatExtractor] step={self._log_counter}  "
                f"deg_probs  min={deg_probs.min().item():.4f}  "
                f"max={deg_probs.max().item():.4f}  "
                f"mean={deg_probs.mean().item():.4f}  "
                f"shape={list(deg_probs.shape)}"
            )
            self._log_counter += 1

        return deg_probs @ embedding
