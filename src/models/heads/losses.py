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


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    pred/target: [B, D, H, W]
    mask:        [B, 1, H, W]
    """
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom
