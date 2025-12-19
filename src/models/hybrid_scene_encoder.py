from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class HybridSceneEncoder(nn.Module):
    """
    Runs enabled modality encoders and returns a unified BEV feature map.
    For now: LiDAR -> BEV. Later: add map/camera encoders + fusion.
    """
    def __init__(self, encoders: nn.ModuleDict, fusion: Optional[nn.Module] = None):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = {}

        if "lidar" in self.encoders:
            outs["bev_lidar"] = self.encoders["lidar"](batch["lidar_points"])  # [B,C,H,W]

        if self.fusion is None:
            if "bev_lidar" not in outs:
                raise ValueError("No fusion module and no lidar encoder output.")
            outs["bev"] = outs["bev_lidar"]
        else:
            outs["bev"] = self.fusion(outs)

        return outs
