from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiLabelFocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Reference: https://arxiv.org/abs/1708.02002

    Shape:
        - logits: (batch_size, num_classes) - raw unnormalized scores
        - targets: (batch_size, num_classes) - binary labels (0 or 1)
    """

    def __init__(
        self,
        alpha: Optional[Sequence] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """Constructor.

        Args:
            alpha (Sequence or Tensor, optional): Weights for each class,
                shape (num_classes,). Defaults to None.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            label_smoothing (float, optional): Label smoothing factor.
                Defaults to 0.0 (no smoothing).
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0.0, 1.0)")

        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Convert Sequence to Tensor if needed
        if alpha is not None and not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # logits: (N, C) - raw scores
        # targets: (N, C) - binary labels

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        # Compute sigmoid probabilities
        probs = torch.sigmoid(logits)

        # Binary cross entropy: -[y*log(p) + (1-y)*log(1-p)]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal weight: (1 - p_t)^gamma
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_expanded = alpha.unsqueeze(0).expand_as(targets)
            focal_weight = focal_weight * alpha_expanded

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
