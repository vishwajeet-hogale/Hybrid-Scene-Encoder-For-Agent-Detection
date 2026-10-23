from __future__ import annotations
from typing import Any, Dict, List
import torch


def hybrid_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "sample_token": [b["sample_token"] for b in batch],
        "scene_token": [b["scene_token"] for b in batch],
        "lidar_points": [b["lidar_points"] for b in batch],  # variable Ni
        "gt_boxes": [b["gt_boxes"] for b in batch],          # variable N
        "gt_labels": [b["gt_labels"] for b in batch],        # variable N
        "bev": torch.stack([b["bev"] for b in batch], 0),
        "map_polylines": torch.stack([b["map_polylines"] for b in batch], 0),
        "map_mask": torch.stack([b["map_mask"] for b in batch], 0),
    }
    for k in ("cam_feats", "cam2ego", "cam_K", "cam_depth"):  # camera fusion inputs (fixed shapes)
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], 0)
    return out
