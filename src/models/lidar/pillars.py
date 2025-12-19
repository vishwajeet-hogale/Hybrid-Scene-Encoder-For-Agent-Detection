from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PillarConfig:
    x_range: Tuple[float, float] = (-50.0, 50.0)
    y_range: Tuple[float, float] = (-50.0, 50.0)
    z_range: Tuple[float, float] = (-5.0, 3.0)
    voxel_size: Tuple[float, float] = (0.25, 0.25)  # (vx, vy)
    max_pillars: int = 12000
    max_points_per_pillar: int = 32


def pillarize(points_ego: torch.Tensor, cfg: PillarConfig, H: int, W: int):
    """
    points_ego: [N,4] (x,y,z,intensity) in EGO frame

    Returns:
      pillar_points: [P, T, 4]
      coords:        [P, 2]  int64  (iy, ix)
      mask:          [P, T]  bool
    """
    assert points_ego.ndim == 2 and points_ego.shape[1] == 4
    device = points_ego.device

    x, y, z, r = points_ego[:, 0], points_ego[:, 1], points_ego[:, 2], points_ego[:, 3]

    # 1) Filter to ROI bounds
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range

    keep = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max) & (z >= z_min) & (z < z_max)
    points = points_ego[keep]
    if points.shape[0] == 0:
        P = 0
        pillar_points = torch.zeros((0, cfg.max_points_per_pillar, 4), device=device, dtype=torch.float32)
        coords = torch.zeros((0, 2), device=device, dtype=torch.long)
        mask = torch.zeros((0, cfg.max_points_per_pillar), device=device, dtype=torch.bool)
        return pillar_points, coords, mask

    # 2) Compute pillar indices
    vx, vy = cfg.voxel_size
    ix = torch.floor((points[:, 0] - x_min) / vx).long()
    iy = torch.floor((points[:, 1] - y_min) / vy).long()

    # safety clamp (should be redundant due to keep-mask)
    ix = ix.clamp(0, W - 1)
    iy = iy.clamp(0, H - 1)

    # 3) Group by unique pillars
    # coords as a single int64 per point to avoid torch.unique(..., dim=0) on MPS
    lin = iy.to(torch.long) * W + ix.to(torch.long)          # [N]
    uniq_lin, inv = torch.unique(lin, return_inverse=True)   # uniq_lin: [P], inv: [N]

    # recover (iy, ix)
    uniq_iy = torch.div(uniq_lin, W, rounding_mode="floor")
    uniq_ix = uniq_lin % W
    uniq = torch.stack([uniq_iy, uniq_ix], dim=1)            # [P,2]

    # Limit pillars
    if uniq.shape[0] > cfg.max_pillars:
        uniq = uniq[: cfg.max_pillars]
        valid_pillar = inv < cfg.max_pillars
        points = points[valid_pillar]
        inv = inv[valid_pillar]

    P = uniq.shape[0]
    T = cfg.max_points_per_pillar

    # 4) Allocate outputs
    pillar_points = torch.zeros((P, T, 4), device=device, dtype=torch.float32)
    mask = torch.zeros((P, T), device=device, dtype=torch.bool)

    # 5) Fill each pillar up to T points
    # Simple (correct) implementation: loop over pillars
    for p in range(P):
        idxs = torch.nonzero(inv == p, as_tuple=False).squeeze(1)
        if idxs.numel() == 0:
            continue
        idxs = idxs[:T]
        pillar_points[p, : idxs.numel(), :] = points[idxs]
        mask[p, : idxs.numel()] = True

    coords = uniq.long()  # [P,2] (iy, ix)
    return pillar_points, coords, mask


def scatter_to_bev(pillar_feats: torch.Tensor, coords: torch.Tensor, H: int, W: int):
    """
    pillar_feats: [P, C]
    coords:       [P, 2] (iy, ix)
    Returns:
      bev: [C, H, W]
    """
    assert pillar_feats.ndim == 2
    device = pillar_feats.device
    P, C = pillar_feats.shape

    bev = torch.zeros((C, H, W), device=device, dtype=pillar_feats.dtype)
    if P == 0:
        return bev

    iy = coords[:, 0].long()
    ix = coords[:, 1].long()

    # If multiple pillars map to same cell (rare with unique), last write wins.
    bev[:, iy, ix] = pillar_feats.T
    return bev
