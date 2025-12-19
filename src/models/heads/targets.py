from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import math


@dataclass
class GridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    voxel_size_x: float
    voxel_size_y: float

    @property
    def W(self) -> int:
        return int((self.x_max - self.x_min) / self.voxel_size_x)

    @property
    def H(self) -> int:
        return int((self.y_max - self.y_min) / self.voxel_size_y)


def _gaussian2d(shape: Tuple[int, int], sigma: float, device):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(
        torch.arange(-m, m + 1, device=device),
        torch.arange(-n, n + 1, device=device),
        indexing="ij",
    )
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h


def draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], sigma: float):
    """
    heatmap: [H,W]
    center: (cx, cy) integer indices
    """
    H, W = heatmap.shape
    cx, cy = center
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return heatmap

    # radius ~ 3*sigma
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    g = _gaussian2d((diameter, diameter), sigma=sigma, device=heatmap.device)

    left = min(cx, radius)
    right = min(W - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(H - cy - 1, radius)

    masked_hm = heatmap[cy - top: cy + bottom + 1, cx - left: cx + right + 1]
    masked_g = g[radius - top: radius + bottom + 1, radius - left: radius + right + 1]

    heatmap[cy - top: cy + bottom + 1, cx - left: cx + right + 1] = torch.maximum(masked_hm, masked_g)
    return heatmap


def build_targets(
    gt_boxes_list: List[torch.Tensor],   # list of [N,7] (x,y,z,w,l,h,yaw) ego
    gt_labels_list: List[torch.Tensor],  # list of [N] in {0..C-1}
    grid: GridSpec,
    num_classes: int,
    sigma: float = 1.5,
):
    """
    Returns:
      heatmap: [B, C, H, W]
      reg_tgt: [B, 8, H, W]
      reg_mask: [B, 1, H, W] (1 at positive centers)
    """
    B = len(gt_boxes_list)
    H, W = grid.H, grid.W

    heatmap = torch.zeros((B, num_classes, H, W), dtype=torch.float32, device=gt_boxes_list[0].device)
    reg_tgt = torch.zeros((B, 8, H, W), dtype=torch.float32, device=gt_boxes_list[0].device)
    reg_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=gt_boxes_list[0].device)

    for b in range(B):
        boxes = gt_boxes_list[b]
        labels = gt_labels_list[b]
        if boxes.numel() == 0:
            continue

        for i in range(boxes.shape[0]):
            x, y, z, w, l, h, yaw = boxes[i]

            # map metric (x,y) -> grid indices
            fx = (x - grid.x_min) / grid.voxel_size_x
            fy = (y - grid.y_min) / grid.voxel_size_y

            ix = int(torch.floor(fx).item())
            iy = int(torch.floor(fy).item())

            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue

            cls = int(labels[i].item())

            # heatmap gaussian at center
            heatmap[b, cls] = draw_gaussian(heatmap[b, cls], (ix, iy), sigma=sigma)

            # regression targets at center cell
            dx = fx - ix
            dy = fy - iy

            reg_tgt[b, 0, iy, ix] = dx
            reg_tgt[b, 1, iy, ix] = dy
            reg_tgt[b, 2, iy, ix] = z
            reg_tgt[b, 3, iy, ix] = torch.log(torch.clamp(w, min=1e-2))
            reg_tgt[b, 4, iy, ix] = torch.log(torch.clamp(l, min=1e-2))
            reg_tgt[b, 5, iy, ix] = torch.log(torch.clamp(h, min=1e-2))
            reg_tgt[b, 6, iy, ix] = torch.sin(yaw)
            reg_tgt[b, 7, iy, ix] = torch.cos(yaw)

            reg_mask[b, 0, iy, ix] = 1.0

    return heatmap, reg_tgt, reg_mask
