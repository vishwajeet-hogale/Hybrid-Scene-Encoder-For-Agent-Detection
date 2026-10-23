from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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


@dataclass
class BoxPrior:
    """
    Per-class box statistics used to normalize the regression targets:
      z target    = z - z_mean[c]
      size target = log(size / size_mean[c])   (size = w, l, h)
    Defaults: mean z and geometric-mean (w, l, h) of v1.0-mini mini_train, classes (vehicle, pedestrian, cyclist).
    """
    z_mean: Sequence[float] = (1.014, 1.048, 1.196)
    size_mean: Sequence[Sequence[float]] = ((2.076, 5.076, 1.861), (0.693, 0.735, 1.759), (0.678, 1.789, 1.490))

    def tensors(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.z_mean, dtype=torch.float32, device=device),
                torch.tensor(self.size_mean, dtype=torch.float32, device=device))


def gaussian_radius(height: float, width: float, min_overlap: float = 0.1) -> float:
    """
    CenterNet/CenterPoint radius (in cells): the largest center shift that keeps IoU >= min_overlap
    for a box of the given size, taken as the min over the three corner cases.
    """
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def _gaussian2d(shape: Tuple[int, int], sigma: float, device):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(
        torch.arange(-m, m + 1, device=device),
        torch.arange(-n, n + 1, device=device),
        indexing="ij",
    )
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h


def draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int):
    """
    heatmap: [H,W]
    center: (cx, cy) integer indices
    radius: gaussian radius in cells (sigma = diameter / 6)
    """
    H, W = heatmap.shape
    cx, cy = center
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return heatmap

    diameter = 2 * radius + 1
    g = _gaussian2d((diameter, diameter), sigma=diameter / 6, device=heatmap.device)

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
    prior: BoxPrior,
    min_overlap: float = 0.1,
    min_radius: int = 2,
):
    """
    Returns:
      heatmap: [B, C, H, W]
      reg_tgt: [B, 8, H, W]  (dx, dy, z - z_mean, log(w/w_mean), log(l/l_mean), log(h/h_mean), sin yaw, cos yaw)
      reg_mask: [B, 1, H, W] (1 at positive centers)
    """
    B = len(gt_boxes_list)
    H, W = grid.H, grid.W
    device = gt_boxes_list[0].device

    heatmap = torch.zeros((B, num_classes, H, W), dtype=torch.float32, device=device)
    reg_tgt = torch.zeros((B, 8, H, W), dtype=torch.float32, device=device)
    reg_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    z_mean, size_mean = prior.tensors(device)

    for b in range(B):
        boxes = gt_boxes_list[b]
        labels = gt_labels_list[b]
        if boxes.numel() == 0:
            continue

        # one host sync per sample for the integer bookkeeping
        boxes_cpu = boxes.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for i in range(boxes.shape[0]):
            x, y, z, w, l, h, yaw = boxes_cpu[i].tolist()

            # map metric (x,y) -> grid indices
            fx = (x - grid.x_min) / grid.voxel_size_x
            fy = (y - grid.y_min) / grid.voxel_size_y

            ix = int(math.floor(fx))
            iy = int(math.floor(fy))

            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue

            cls = int(labels_cpu[i])

            # heatmap gaussian at center, radius from the box footprint in cells
            radius = gaussian_radius(l / grid.voxel_size_x, w / grid.voxel_size_y, min_overlap)
            radius = max(min_radius, int(radius))
            heatmap[b, cls] = draw_gaussian(heatmap[b, cls], (ix, iy), radius)

            # regression targets at center cell
            reg_tgt[b, :, iy, ix] = torch.tensor([
                fx - ix,
                fy - iy,
                z - float(z_mean[cls]),
                math.log(max(w, 1e-2) / float(size_mean[cls, 0])),
                math.log(max(l, 1e-2) / float(size_mean[cls, 1])),
                math.log(max(h, 1e-2) / float(size_mean[cls, 2])),
                math.sin(yaw),
                math.cos(yaw),
            ], device=device)

            reg_mask[b, 0, iy, ix] = 1.0

    return heatmap, reg_tgt, reg_mask
