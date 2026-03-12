"""Loss functions for multi-label classification."""

from torch import nn

from .focalloss import MultiLabelFocalLoss
from .BCEWithLogitsLoss import BCEWithLogitsLoss

__all__ = ["MultiLabelFocalLoss", "BCEWithLogitsLoss"]

# Mapping from loss type to its class
LOSS_FACTORY = {
    "MultiLabelFocalLoss": MultiLabelFocalLoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
}


def build_loss(config, device="cpu"):
    """Build loss function from config.

    Args:
        config: Loss config with 'type' and optional parameters
        device: Device to move tensors to

    Returns:
        Loss function
    """
    loss_type = config.get("type")
    if not loss_type:
        raise ValueError("loss.type must be specified in config")

    # Check if it's a custom loss
    if loss_type in LOSS_FACTORY:
        loss_cls = LOSS_FACTORY[loss_type]
        # Extract params except 'type'
        params = {k: v for k, v in config.items() if k != "type"}
        # Move alpha tensor to device if provided
        if "alpha" in params and params["alpha"] is not None:
            if hasattr(params["alpha"], "to"):
                params["alpha"] = params["alpha"].to(device)
        return loss_cls(**params)

    # Fallback to torch.nn built-in losses
    return getattr(nn, loss_type)()
