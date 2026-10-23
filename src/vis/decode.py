from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class GridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    voxel_size_x: float
    voxel_size_y: float

def topk_heatmap(hm: torch.Tensor, K: int = 100):
    """
    hm: [B, C, H, W] after sigmoid
    returns:
      scores: [B, K]
      clses:  [B, K]
      ys:     [B, K]
      xs:     [B, K]
    """
    B, C, H, W = hm.shape
    hm = hm.view(B, C, -1)  # [B,C,H*W]
    topk_scores, topk_inds = torch.topk(hm, K)  # [B,C,K]

    # flatten across classes
    topk_scores2, topk_cls_inds = torch.topk(topk_scores.view(B, -1), K)  # [B,K]
    topk_clses = topk_cls_inds // K  # [B,K] class index
    topk_inds2 = topk_inds.view(B, -1).gather(1, topk_cls_inds)  # [B,K] flattened index in H*W

    ys = topk_inds2 // W
    xs = topk_inds2 % W
    return topk_scores2, topk_clses, ys, xs

def decode_centers(
    hm_logits: torch.Tensor,
    reg: torch.Tensor,
    grid: GridSpec,
    K: int = 100,
    score_thresh: float = 0.3,
    nms_kernel: int = 3,
    z_mean: torch.Tensor = None,
    size_mean: torch.Tensor = None,
):
    """
    hm_logits: [B,C,H,W] (logits)
    reg:       [B,8,H,W]
    nms_kernel: keep only local maxima in a kernel x kernel window (CenterNet-style NMS); 1 disables it
    z_mean [C], size_mean [C,3]: per-class priors the z / log-size targets were normalized with (see BoxPrior)
    returns list (len B) of dict with:
      boxes:  [M,7] (x,y,z,w,l,h,yaw) in ego meters
      scores: [M]
      labels: [M]
    """
    B, C, H, W = hm_logits.shape
    hm = torch.sigmoid(hm_logits)
    if nms_kernel > 1:
        hmax = torch.nn.functional.max_pool2d(hm, nms_kernel, stride=1, padding=nms_kernel // 2)
        hm = hm * (hmax == hm).float()

    scores, labels, ys, xs = topk_heatmap(hm, K=K)  # [B,K] each
    out = []

    for b in range(B):
        keep = scores[b] >= score_thresh
        if keep.sum() == 0:
            out.append({"boxes": torch.zeros((0,7), device=hm.device),
                        "scores": torch.zeros((0,), device=hm.device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=hm.device)})
            continue

        s = scores[b][keep]
        lab = labels[b][keep].long()
        y = ys[b][keep].long()
        x = xs[b][keep].long()

        # gather regression at those cells
        # reg[b]: [8,H,W] -> [8,M]
        r = reg[b, :, y, x]  # [8, M]

        dx, dy = r[0], r[1]
        z = r[2]
        w = torch.exp(r[3]).clamp(min=0.1)
        l = torch.exp(r[4]).clamp(min=0.1)
        h = torch.exp(r[5]).clamp(min=0.1)
        if z_mean is not None:
            z = z + z_mean[lab]
        if size_mean is not None:
            w, l, h = w * size_mean[lab, 0], l * size_mean[lab, 1], h * size_mean[lab, 2]
        sin_yaw, cos_yaw = r[6], r[7]
        yaw = torch.atan2(sin_yaw, cos_yaw)

        # cell -> metric
        fx = x.float() + dx
        fy = y.float() + dy
        X = grid.x_min + fx * grid.voxel_size_x
        Y = grid.y_min + fy * grid.voxel_size_y

        boxes = torch.stack([X, Y, z, w, l, h, yaw], dim=1)  # [M,7]
        out.append({"boxes": boxes, "scores": s, "labels": lab})

    return out


def circle_nms(det: dict, radius_per_class=(2.0, 0.5, 1.0)) -> dict:
    """
    CenterPoint-style circle NMS on one decoded sample: greedily keep the highest-scoring center and drop
    same-class centers within that class's radius (meters, BEV). det: {"boxes": [M,7], "scores": [M], "labels": [M]}
    """
    boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
    if boxes.shape[0] == 0:
        return det
    order = torch.argsort(scores, descending=True)
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    radius = torch.tensor(radius_per_class, device=boxes.device)[labels]  # [M]
    dist = torch.cdist(boxes[:, :2], boxes[:, :2])                         # [M,M]
    suppress = (dist < radius[:, None]) & (labels[:, None] == labels[None, :])

    keep = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
    for i in range(boxes.shape[0]):
        if keep[i]:
            s = suppress[i].clone()
            s[: i + 1] = False
            keep &= ~s
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
