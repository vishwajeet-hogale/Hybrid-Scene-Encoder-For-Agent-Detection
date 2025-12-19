from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn

from src.models.lidar.pillars import PillarConfig, pillarize, scatter_to_bev


class PillarFeatureNet(nn.Module):
    """
    Simple PFN: per-point MLP + masked max pooling.
    Input points: [P, T, 4] (x,y,z,intensity)
    Output:       [P, C]
    """
    def __init__(self, in_dim: int = 4, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pillar_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.mlp(pillar_points)                  # [P,T,C]
        x = x * mask.unsqueeze(-1).to(x.dtype)       # mask padded points
        x, _ = x.max(dim=1)                          # [P,C]
        return x


class BEVBackbone(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointPillarsEncoder(nn.Module):
    """
    Input:  list of B point clouds, each [Ni,4] ego frame
    Output: BEV features [B, C, H, W]
    """
    def __init__(
        self,
        cfg: PillarConfig,
        pillar_feat_dim: int = 64,
        bev_feat_dim: int = 128,
    ):
        super().__init__()
        self.cfg = cfg
        self.pfn = PillarFeatureNet(4, pillar_feat_dim)
        self.backbone = BEVBackbone(pillar_feat_dim, bev_feat_dim)

        self.H = int((cfg.y_range[1] - cfg.y_range[0]) / cfg.voxel_size[1])
        self.W = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.voxel_size[0])

    def forward(self, batch_points: List[torch.Tensor]) -> torch.Tensor:
        bev_feats = []
        for pts in batch_points:
            pillars, coords, mask = pillarize(pts, self.cfg, self.H, self.W)  # [P,T,4], [P,2], [P,T]
            pillar_feats = self.pfn(pillars, mask)                             # [P,Cp]
            bev = scatter_to_bev(pillar_feats, coords, self.H, self.W)         # [Cp,H,W]
            bev_feats.append(bev)

        bev = torch.stack(bev_feats, dim=0)  # [B,Cp,H,W]
        bev = self.backbone(bev)             # [B,C,H,W]
        return bev
