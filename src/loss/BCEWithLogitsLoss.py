from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Reduction

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class BCEWithLogitsLoss(_Loss):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0.0, 1.0)")
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: Tensor | None
        self.pos_weight: Tensor | None
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / 2
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )