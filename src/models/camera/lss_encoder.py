from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.models.lidar.pillars import PillarConfig


class CameraBEVEncoder(nn.Module):
    """
    Camera -> BEV (BEVFusion camera branch, Lift-Splat-Shoot style) with LiDAR-guided depth.

    Per camera feature cell:
      1. reduce the frozen DINOv2 feature (384 -> mid)
      2. predict a categorical depth distribution over D bins; the depth net also sees the sparse LiDAR depth
         projected into that cell (value + valid mask), and is supervised with it (see depth_loss)
      3. "lift": outer product depth-probability x feature -> a weighted feature at D points along the pixel ray
      4. "splat": move those points into the ego frame (cam2ego, augmentation included) and sum them into the
         same BEV grid as the LiDAR branch, with one fixed-size index_add (padding goes to a trash cell)

    Inputs (batch): cam_feats [B,N,384,h,w], cam_depth [B,N,h,w] (0 = no LiDAR), cam2ego [B,N,4,4], cam_K [B,N,3,3]
    Output: bev [B,C,H,W], depth logits [B,N,D,h,w]
    """
    def __init__(
        self,
        cfg: PillarConfig,
        in_channels: int = 384,
        mid_channels: int = 128,
        out_channels: int = 32,
        depth_min: float = 1.0,
        depth_max: float = 61.0,
        depth_step: float = 1.25,
    ):
        super().__init__()
        self.cfg = cfg
        self.H = int((cfg.y_range[1] - cfg.y_range[0]) / cfg.voxel_size[1])
        self.W = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.voxel_size[0])
        self.depth_min, self.depth_max, self.depth_step = depth_min, depth_max, depth_step
        D = int(round((depth_max - depth_min) / depth_step))
        self.register_buffer("depth_centers", depth_min + (torch.arange(D, dtype=torch.float32) + 0.5) * depth_step)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_net = nn.Sequential(
            nn.Conv2d(mid_channels + 2, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, D, 1),
        )
        self.feat_net = nn.Conv2d(mid_channels, out_channels, 1)
        self.out_channels = out_channels

    def depth_targets(self, sparse_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """sparse LiDAR depth [...] -> (bin index [...] long, valid mask [...] bool)"""
        valid = (sparse_depth >= self.depth_min) & (sparse_depth < self.depth_max)
        idx = ((sparse_depth - self.depth_min) / self.depth_step).floor().clamp(0, len(self.depth_centers) - 1).long()
        return idx, valid

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = batch["cam_feats"].float()                               # [B,N,Cin,h,w]
        B, N, _, h, w = feats.shape
        D = len(self.depth_centers)

        x = self.reduce(feats.flatten(0, 1))                             # [BN,mid,h,w]
        sd = batch["cam_depth"].flatten(0, 1).unsqueeze(1)               # [BN,1,h,w]
        guide = torch.cat([sd / self.depth_max, (sd > 0).to(sd.dtype)], dim=1)
        depth_logits = self.depth_net(torch.cat([x, guide], dim=1))      # [BN,D,h,w]
        depth_prob = depth_logits.softmax(dim=1)
        img = self.feat_net(x)                                           # [BN,C,h,w]

        # lift: [BN,D,C,h,w] -> rows [B*N*D*h*w, C]
        frustum = depth_prob.unsqueeze(2) * img.unsqueeze(1)
        values = frustum.permute(0, 1, 3, 4, 2).reshape(-1, self.out_channels)

        # frustum geometry in the ego frame: pixel (cell center) -> ray -> D points -> cam2ego
        device = feats.device
        v, u = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32) + 0.5,
                              torch.arange(w, device=device, dtype=torch.float32) + 0.5, indexing="ij")
        pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)            # [h,w,3]
        K_inv = torch.linalg.inv(batch["cam_K"].float().cpu()).to(device)  # small 3x3 inverses (CPU: robust everywhere)
        rays = torch.einsum("bnij,hwj->bnhwi", K_inv, pix)                # [B,N,h,w,3], z = 1
        pts = rays.unsqueeze(2) * self.depth_centers.view(1, 1, D, 1, 1, 1)   # [B,N,D,h,w,3]
        T = batch["cam2ego"].float()
        ego = torch.einsum("bnij,bndhwj->bndhwi", T[..., :3, :3], pts) + T[..., :3, 3].view(B, N, 1, 1, 1, 3)

        # splat: sum into the BEV grid; points outside the grid / z-range go to the trash cell HW
        c = self.cfg
        ix = torch.floor((ego[..., 0] - c.x_range[0]) / c.voxel_size[0]).long()
        iy = torch.floor((ego[..., 1] - c.y_range[0]) / c.voxel_size[1]).long()
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H) & \
                 (ego[..., 2] >= c.z_range[0]) & (ego[..., 2] < c.z_range[1])
        HW = self.H * self.W
        flat = torch.where(inside, iy * self.W + ix, torch.full_like(ix, HW))
        flat = flat + torch.arange(B, device=device).view(B, 1, 1, 1, 1) * (HW + 1)

        bev = torch.zeros((B * (HW + 1), self.out_channels), device=device, dtype=values.dtype)
        bev = bev.index_add(0, flat.reshape(-1), values)
        bev = bev.reshape(B, HW + 1, self.out_channels)[:, :HW].permute(0, 2, 1).reshape(B, self.out_channels, self.H, self.W)
        return bev.contiguous(), depth_logits.view(B, N, D, h, w)

    def depth_loss(self, depth_logits: torch.Tensor, sparse_depth: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of the predicted depth distribution against the LiDAR depth bin, on cells that have LiDAR."""
        target, valid = self.depth_targets(sparse_depth)                 # [B,N,h,w]
        logits = depth_logits.permute(0, 1, 3, 4, 2)                     # [B,N,h,w,D]
        ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
        valid = valid.reshape(-1).to(ce.dtype)
        return (ce * valid).sum() / valid.sum().clamp(min=1.0)
