from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class HybridSceneEncoder(nn.Module):
    """
    Runs enabled modality encoders and returns a unified BEV feature map.
    LiDAR only: the LiDAR encoder includes its BEV backbone.
    LiDAR + camera: both encoders produce BEV maps on the same grid -> fusion -> shared BEV backbone.
    """
    def __init__(self, encoders: nn.ModuleDict, fusion: Optional[nn.Module] = None, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.backbone = backbone

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = {}

        if "lidar" in self.encoders:
            outs["bev_lidar"] = self.encoders["lidar"](batch["lidar_points"])  # [B,C,H,W]
        if "camera" in self.encoders:
            outs["bev_camera"], outs["cam_depth_logits"] = self.encoders["camera"](batch)

        if self.fusion is None:
            if "bev_lidar" not in outs:
                raise ValueError("No fusion module and no lidar encoder output.")
            bev = outs["bev_lidar"]
        else:
            bev = self.fusion(outs)

        outs["bev"] = self.backbone(bev) if self.backbone is not None else bev
        return outs
