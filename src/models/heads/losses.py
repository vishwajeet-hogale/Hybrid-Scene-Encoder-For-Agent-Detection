from __future__ import annotations
import torch


def sigmoid_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, alpha=2.0, beta=4.0):
    """
    CenterNet-style focal loss.
    pred_logits: [B,C,H,W]
    target:      [B,C,H,W] in [0,1] (gaussian peaks)
    """
    pred = torch.sigmoid(pred_logits)
    pos = target.eq(1.0).float()
    neg = (target.lt(1.0)).float()

    pos_loss = -torch.log(torch.clamp(pred, 1e-6, 1.0)) * torch.pow(1 - pred, alpha) * pos
    neg_weight = torch.pow(1 - target, beta)
    neg_loss = -torch.log(torch.clamp(1 - pred, 1e-6, 1.0)) * torch.pow(pred, alpha) * neg_weight * neg

    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, channel_weights: torch.Tensor = None):
    """
    pred/target: [B, D, H, W]
    mask:        [B, 1, H, W] (per-cell weights)
    channel_weights: optional [D] multiplier per regression channel (e.g. a higher weight for sin/cos yaw)
    """
    diff = (pred - target).abs() * mask
    if channel_weights is not None:
        diff = diff * channel_weights.view(1, -1, 1, 1)
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def heading_loss(pred_sincos: torch.Tensor, target_sincos: torch.Tensor, mask: torch.Tensor):
    """
    Direct heading error: 1 - cos(predicted yaw - target yaw), only the angle counts (the predicted (sin, cos)
    vector is normalized). 0 = correct, 1 = 90 deg off, 2 = 180 deg off.
    pred_sincos/target_sincos: [B, 2, H, W] (sin, cos); mask: [B, 1, H, W] (per-cell weights)
    """
    norm = pred_sincos.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos_diff = (pred_sincos * target_sincos).sum(dim=1, keepdim=True) / norm
    return ((1.0 - cos_diff) * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, channel_weights: torch.Tensor = None):
    """Like masked_l1 but squared error: large errors cost much more, errors below 1 cost less."""
    diff = (pred - target).pow(2) * mask
    if channel_weights is not None:
        diff = diff * channel_weights.view(1, -1, 1, 1)
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom
