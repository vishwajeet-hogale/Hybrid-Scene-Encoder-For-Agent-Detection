from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class ConvFuser(nn.Module):
    """
    BEVFusion-style conv fuser, written as a residual on the LiDAR BEV:
        out = lidar_bev + f(concat(lidar_bev, camera_bev))
    The last conv of f starts at zero, so a model initialized from LiDAR-only weights initially computes exactly
    the LiDAR-only prediction and learns how much camera information to add.
    """
    def __init__(self, lidar_channels: int, camera_channels: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(lidar_channels + camera_channels, lidar_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(lidar_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(lidar_channels, lidar_channels, 1),
        )
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, outs: Dict[str, torch.Tensor]) -> torch.Tensor:
        lidar, camera = outs["bev_lidar"], outs["bev_camera"]
        return lidar + self.f(torch.cat([lidar, camera], dim=1))
