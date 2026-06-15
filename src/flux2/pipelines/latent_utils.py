import torch


def patchify_latents(latents):
    """16x16 patchify: (B, C, H, W) -> (B, C*4, H//2, W//2)."""
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def pack_latents(latents):
    """Pack: (B, C, H, W) -> (B, H*W, C)."""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


def unpack_latents_with_ids(
    x: torch.Tensor, x_ids: torch.Tensor, height: int | None = None, width: int | None = None
):
    """Unpack (B, seq, C) tokens back to (B, C, H, W) using position IDs."""
    x_list = []
    for data, pos in zip(x, x_ids):
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = height if height is not None else torch.max(h_ids) + 1
        w = width if width is not None else torch.max(w_ids) + 1
        flat_ids = h_ids * w + w_ids
        _, ch = data.shape
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)


def prepare_latent_ids(latents: torch.Tensor):
    """Generate 4D position coordinates (T, H, W, L) for latent tensors.

    Args:
        latents: (B, C, H, W)
    Returns:
        (B, H*W, 4) position IDs: T=0, H=[0..H-1], W=[0..W-1], L=0
    """
    batch_size, _, height, width = latents.shape
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids
